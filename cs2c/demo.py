"""Read CS2 demos into clean per-map tables.

A map can be split over several demo files (p1, p2, ...) when a server
restarted mid-map. Ticks restart in each part, but `game_time` keeps running,
so every table keeps `part`, `tick` and `game_time`.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from demoparser2 import DemoParser

TICK_RATE = 64
ROUND_OTHER = ["game_time", "total_rounds_played", "is_warmup_period"]
# Kills this long after round_end still belong to that round (exit frags).
# The next round starts 5 s (ESL) to 7 s (PW) after round_end.
POST_ROUND_TICKS = 7 * TICK_RATE
SIDES = {"CT": "CT", "TERRORIST": "T"}

# Parser weapon codes -> the game's own display names, which is also how
# active_weapon_name reports them. Caster slang (Deagle, Scout, Dualies) is
# applied later, when input frames are written. Knives map to "Knife" separately.
WEAPONS = {
    "ak47": "AK-47", "m4a1": "M4A4", "m4a1_silencer": "M4A1-S", "awp": "AWP", "galilar": "Galil AR",
    "famas": "FAMAS", "aug": "AUG", "sg556": "SG 553", "ssg08": "SSG 08", "g3sg1": "G3SG1", "scar20": "SCAR-20",
    "deagle": "Desert Eagle", "usp_silencer": "USP-S", "hkp2000": "P2000", "glock": "Glock-18", "p250": "P250",
    "fiveseven": "Five-SeveN", "tec9": "Tec-9", "cz75a": "CZ75-Auto", "elite": "Dual Berettas", "revolver": "R8 Revolver",
    "mp9": "MP9", "mac10": "MAC-10", "mp7": "MP7", "mp5sd": "MP5-SD", "ump45": "UMP-45", "p90": "P90", "bizon": "PP-Bizon",
    "nova": "Nova", "xm1014": "XM1014", "mag7": "MAG-7", "sawedoff": "Sawed-Off", "negev": "Negev", "m249": "M249",
    "hegrenade": "High Explosive Grenade", "inferno": "Molotov", "molotov": "Molotov", "incgrenade": "Incendiary Grenade",
    "flashbang": "Flashbang", "smokegrenade": "Smoke Grenade", "decoy": "Decoy Grenade", "taser": "Zeus x27",
    "planted_c4": "C4 Explosive", "world": "World",
}


KNIFE_WORDS = ("knife", "bayonet", "karambit", "daggers")


def weapon_name(code) -> str:
    """A parser weapon code, or a display name from active_weapon_name, as one display name."""
    if not isinstance(code, str):
        return ""
    if any(w in code.lower() for w in KNIFE_WORDS):
        return "Knife"
    return WEAPONS.get(code, code)


@dataclass
class DemoPart:
    index: int
    path: Path
    parser: DemoParser
    header: dict
    events: set[str]


def open_parts(paths: list[Path]) -> list[DemoPart]:
    parts = []
    for i, path in enumerate(paths):
        parser = DemoParser(str(path))
        parts.append(DemoPart(i, path, parser, parser.parse_header(), set(parser.list_game_events())))
    return parts


def event(part: DemoPart, name: str, **kw) -> pd.DataFrame:
    """parse_event, but an event the demo never recorded gives an empty table, not a list."""
    d = part.parser.parse_event(name, **kw)
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


def _live(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    return d[~d.is_warmup_period.astype(bool)].sort_values("tick")


def part_rounds(part: DemoPart) -> pd.DataFrame:
    """Rounds that finished inside this demo part.

    Only a round_end with a winner counts: ESL demos open with a winnerless
    round_end at tick 1, and a restored part restarts the interrupted round.
    The round number is total_rounds_played at round_end; the event's own
    `round` field is off by one or two in some demos.
    """
    ends = _live(event(part, "round_end", other=ROUND_OTHER))
    ends = ends[ends.winner.notna()]
    starts = _live(event(part, "round_start", other=ROUND_OTHER))
    freezes = _live(event(part, "round_freeze_end", other=ROUND_OTHER))

    rows, prev_end = [], -1
    for e in ends.itertuples():
        live = freezes[(freezes.tick > prev_end) & (freezes.tick < e.tick)].tail(1)
        start_limit = live.tick.iloc[0] if len(live) else e.tick
        start = starts[(starts.tick > prev_end) & (starts.tick <= start_limit)].tail(1)
        rows.append({
            "part": part.index,
            "round": int(e.total_rounds_played),
            "start_tick": int(start.tick.iloc[0]) if len(start) else pd.NA,
            "live_tick": int(live.tick.iloc[0]) if len(live) else pd.NA,
            "end_tick": int(e.tick),
            "start_time": float(start.game_time.iloc[0]) if len(start) else float("nan"),
            "live_time": float(live.game_time.iloc[0]) if len(live) else float("nan"),
            "end_time": float(e.game_time),
            "winner_side": "CT" if e.winner == "CT" else "T",
            "reason": e.reason,
        })
        prev_end = e.tick
    return pd.DataFrame(rows)


def _sides(part: DemoPart, ticks: list[int], names: dict[str, str]) -> pd.DataFrame:
    """Which team is on CT and which on T at each tick."""
    t = part.parser.parse_ticks(["team_name", "team_clan_name"], ticks=ticks)
    unknown = sorted(set(t.team_clan_name.dropna()) - set(names))
    if unknown:
        raise KeyError(f"team names missing from data/meta/team_names.tsv: {unknown}")
    t["team"] = t.team_clan_name.map(names)
    side = t.groupby(["tick", "team_name"]).team.first().unstack()
    return side.rename(columns={"CT": "ct_team", "TERRORIST": "t_team"})[["ct_team", "t_team"]]


def build_rounds(parts: list[DemoPart], names: dict[str, str]) -> pd.DataFrame:
    rounds = pd.concat([part_rounds(p) for p in parts], ignore_index=True)
    # If a round finished in two parts, the later part's copy is the one that counted.
    rounds = rounds.drop_duplicates("round", keep="last").sort_values("round").reset_index(drop=True)

    sides = []
    for p in parts:
        r = rounds[rounds.part == p.index]
        if r.empty:
            continue
        ticks = [int(x) for x in r.live_tick.fillna(r.end_tick)]
        s = _sides(p, ticks, names).reindex(ticks)
        s.index = r.index
        sides.append(s)
    rounds = rounds.join(pd.concat(sides))

    rounds["winner_team"] = rounds.ct_team.where(rounds.winner_side == "CT", rounds.t_team)
    teams = [rounds.ct_team.iloc[0], rounds.t_team.iloc[0]]  # team_a starts on CT
    for col, team in zip(("team_a", "team_b"), teams):
        won = (rounds.winner_team == team).astype(int)
        rounds[f"{col}_score_after"] = won.cumsum()
        rounds[f"{col}_score_before"] = rounds[f"{col}_score_after"] - won
    rounds["freeze_seconds"] = rounds.live_time - rounds.start_time
    rounds["live_seconds"] = rounds.end_time - rounds.live_time
    rounds["is_last"] = rounds["round"] == rounds["round"].max()
    return rounds


def _alive_at(part: DemoPart, ticks: list[int]) -> pd.DataFrame:
    """Players alive on each side at each tick."""
    t = part.parser.parse_ticks(["is_alive", "team_name"], ticks=ticks)
    t = t[t.team_name.isin(SIDES)]
    alive = t[t.is_alive.astype(bool)].groupby(["tick", "team_name"]).size().unstack(fill_value=0)
    return alive.reindex(index=ticks, columns=list(SIDES), fill_value=0).rename(columns=SIDES)


def _in_rounds(d: pd.DataFrame, r: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep rows that fall in a counted round of this part, and say where in the round they are.

    A round runs from its live tick (not round_start: a backup restore kills
    everyone during freeze time, Dust2 p2 tick 4) to POST_ROUND_TICKS after round_end.
    Returns the kept rows and how many were dropped.
    """
    d = d.sort_values("tick").copy()
    d["round"] = pd.NA
    r = r.sort_values("round")
    # The window after round_end stops where the next round starts (5 s later on ESL
    # servers), or the next round's spawn bomb pickup would land in this round.
    next_start = r.start_tick.shift(-1)
    for row, nxt in zip(r.itertuples(), next_start):
        stop = row.end_tick + POST_ROUND_TICKS if pd.isna(nxt) else min(row.end_tick + POST_ROUND_TICKS, nxt - 1)
        if row.is_last:  # map over: players dying to "world" for fun afterwards don't count
            stop = row.end_tick
        inside = (d.tick >= row.live_tick) & (d.tick <= stop)
        d.loc[inside, "round"] = row.round
    dropped = int(d["round"].isna().sum())
    d = d[d["round"].notna()].copy()
    d["round"] = d["round"].astype(int)
    d["seconds_into_round"] = d.game_time - d["round"].map(dict(zip(r["round"], r.live_time)))
    d["after_round_end"] = d.tick > d["round"].map(dict(zip(r["round"], r.end_tick)))
    return d, dropped


def _kind(k: pd.DataFrame) -> pd.Series:
    kind = pd.Series("kill", index=k.index)
    kind[k.attacker_team_name == k.user_team_name] = "teamkill"
    kind[k.attacker_name == k.user_name] = "suicide"
    kind[k.weapon == "world"] = "world"
    kind[k.weapon == "planted_c4"] = "bomb"
    return kind


def build_kills(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every death in a counted round, with the facts known at that moment.

    Deaths outside counted rounds (warmup, freeze time, a round interrupted by
    a server restart, the restart itself) are dropped and returned separately.
    Freeze-time deaths are not kills: a backup restore kills everyone (Dust2 p2,
    tick 4) and an admin can slay a player during a technical pause (MongolZ vs
    Liquid round 13). The server counts the second kind as a death, not the first.
    """
    props = ["team_name", "team_clan_name", "last_place_name", "X", "Y", "Z", "health", "active_weapon_name"]
    out, dropped = [], []
    for p in parts:
        r = rounds[rounds.part == p.index]
        k = event(p, "player_death", player=props, other=["game_time", "is_warmup_period"])
        if k.empty or r.empty:
            continue
        in_round = k.index.isin(_in_rounds(k, r)[0].index)
        warm = k.is_warmup_period.astype(bool)
        freeze = pd.Series(False, index=k.index)
        for row in r.itertuples():
            freeze |= (k.tick >= row.start_tick) & (k.tick < row.live_tick)
        dropped.append(pd.DataFrame({"part": p.index, "tick": k.tick, "victim": k.user_name, "weapon": k.weapon,
                                     "during": ["freeze" if f else "warmup" if w else "outside rounds"
                                                for f, w in zip(freeze, warm)]})[~in_round])
        k, _ = _in_rounds(k, r)

        # Count down players alive per side, starting from the live tick.
        start = _alive_at(p, [int(x) for x in r.live_tick])
        start.index = r["round"].tolist()
        k["ct_alive_after"] = k["t_alive_after"] = 0
        for rnd, idx in k.groupby("round").groups.items():
            ct, t = int(start.loc[rnd, "CT"]), int(start.loc[rnd, "T"])
            for i in idx:
                side = SIDES.get(k.at[i, "user_team_name"])
                ct -= side == "CT"
                t -= side == "T"
                k.at[i, "ct_alive_after"], k.at[i, "t_alive_after"] = ct, t

        out.append(pd.DataFrame({
            "part": p.index,
            "round": k["round"],
            "tick": k.tick,
            "game_time": k.game_time,
            "seconds_into_round": k.seconds_into_round,
            "after_round_end": k.after_round_end,
            "kind": _kind(k),
            "attacker": k.attacker_name,
            "attacker_team": k.attacker_team_clan_name.map(names),
            "attacker_side": k.attacker_team_name.map(SIDES),
            "attacker_place": k.attacker_last_place_name,
            "attacker_x": k.attacker_X, "attacker_y": k.attacker_Y, "attacker_z": k.attacker_Z,
            "attacker_hp": k.attacker_health,
            "victim": k.user_name,
            "victim_weapon": k.user_active_weapon_name.map(weapon_name),
            "victim_team": k.user_team_clan_name.map(names),
            "victim_side": k.user_team_name.map(SIDES),
            "victim_place": k.user_last_place_name,
            "victim_x": k.user_X, "victim_y": k.user_Y, "victim_z": k.user_Z,
            "assister": k.assister_name,
            "flash_assist": k.assistedflash.astype(bool),
            "weapon": k.weapon.map(weapon_name),
            "weapon_code": k.weapon,
            "headshot": k.headshot.astype(bool),
            "wallbang": k.penetrated > 0,
            "through_smoke": k.thrusmoke.astype(bool),
            "noscope": k.noscope.astype(bool),
            "attacker_blind": k.attackerblind.astype(bool),
            "attacker_in_air": k.attackerinair.astype(bool),
            "distance_m": k.distance,
            "ct_alive_after": k.ct_alive_after.astype(int),
            "t_alive_after": k.t_alive_after.astype(int),
        }))
    return pd.concat(out, ignore_index=True), pd.concat(dropped, ignore_index=True)


BOMB_EVENTS = {
    "bomb_beginplant": "plant_start", "bomb_planted": "planted", "bomb_begindefuse": "defuse_start",
    "bomb_defused": "defused", "bomb_exploded": "exploded", "bomb_dropped": "dropped", "bomb_pickup": "pickup",
}


def build_bomb(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """Bomb events with site A/B and players alive at that moment.

    plant_start and defuse_start exist only in some organisers' demos.
    """
    props = ["team_name", "team_clan_name", "last_place_name", "X", "Y"]
    out = []
    for p in parts:
        r = rounds[rounds.part == p.index]
        frames = []
        for name, label in BOMB_EVENTS.items():
            d = event(p, name, player=props, other=["game_time"])
            if not d.empty:
                frames.append(d.assign(event=label))
        if not frames or r.empty:
            continue
        d, _ = _in_rounds(pd.concat(frames, ignore_index=True), r)
        alive = _alive_at(p, [int(x) for x in d.tick])
        d["ct_alive"], d["t_alive"] = alive["CT"].to_numpy(), alive["T"].to_numpy()
        out.append(d.assign(part=p.index))
    d = pd.concat(out, ignore_index=True)

    # `site` is a map entity number. Planters stand in BombsiteA or BombsiteB,
    # so learn number -> letter from the plants and apply it to every event.
    plants = d[(d.event == "planted") & d.user_last_place_name.str.startswith("Bombsite", na=False)]
    letters = plants.groupby("site").user_last_place_name.agg(lambda s: s.mode().iloc[0][-1]).to_dict()
    site = d.site.map(letters) if "site" in d else pd.Series(pd.NA, index=d.index)
    # Events without a site of their own (defuse_start) take the round's plant site.
    planted_site = site[d.event == "planted"].groupby(d["round"]).first()
    on_site = d.event.isin(["plant_start", "planted", "defuse_start", "defused", "exploded"])
    site = site.fillna(d["round"].map(planted_site)).where(on_site)
    return pd.DataFrame({
        "part": d.part,
        "round": d["round"],
        "tick": d.tick,
        "game_time": d.game_time,
        "seconds_into_round": d.seconds_into_round,
        "after_round_end": d.after_round_end,
        "event": d.event,
        "player": d.user_name,
        "team": d.user_team_clan_name.map(names),
        "side": d.user_team_name.map(SIDES),
        "place": d.user_last_place_name,
        "x": d.user_X, "y": d.user_Y,
        "site": site,
        "has_kit": d.haskit if "haskit" in d else pd.NA,
        "ct_alive": d.ct_alive,
        "t_alive": d.t_alive,
    })


def build_damage(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """Every hit, with real damage (capped at the HP the victim had) and HP left."""
    out = []
    for p in parts:
        r = rounds[rounds.part == p.index]
        d = event(p, "player_hurt", player=["team_name", "team_clan_name"], other=["game_time"])
        if d.empty or r.empty:
            continue
        d, _ = _in_rounds(d, r)
        # dmg_health can exceed the HP left (a 103 headshot on 40 HP), so take the drop in HP.
        # Everyone spawns with 100 HP and nothing heals in competitive.
        d["hp_before"] = d.groupby(["round", "user_name"]).health.shift(1).fillna(100)
        kind = pd.Series("enemy", index=d.index)
        kind[d.attacker_team_name == d.user_team_name] = "team"
        kind[d.attacker_name == d.user_name] = "self"
        kind[d.attacker_name.isna()] = "world"
        kind[d.weapon == "planted_c4"] = "bomb"
        out.append(pd.DataFrame({
            "part": p.index,
            "round": d["round"],
            "tick": d.tick,
            "game_time": d.game_time,
            "seconds_into_round": d.seconds_into_round,
            "after_round_end": d.after_round_end,
            "kind": kind,
            "attacker": d.attacker_name,
            "attacker_team": d.attacker_team_clan_name.map(names),
            "attacker_side": d.attacker_team_name.map(SIDES),
            "victim": d.user_name,
            "victim_team": d.user_team_clan_name.map(names),
            "victim_side": d.user_team_name.map(SIDES),
            "weapon": d.weapon.map(weapon_name),
            "hitgroup": d.hitgroup,
            "damage": (d.hp_before - d.health).astype(int),
            "damage_raw": d.dmg_health,
            "armor_damage": d.dmg_armor,
            "hp_after": d.health,
            "armor_after": d.armor,
        }))
    return pd.concat(out, ignore_index=True)


THROWS = {
    "weapon_smokegrenade": "smoke", "weapon_flashbang": "flash", "weapon_hegrenade": "he",
    "weapon_molotov": "molotov", "weapon_incgrenade": "incendiary", "weapon_decoy": "decoy",
}
# Detonation event -> (start type key, end event or None)
GRENADE_EVENTS = {
    "smokegrenade_detonate": ("smoke", "smokegrenade_expired"),
    "flashbang_detonate": ("flash", None),
    "hegrenade_detonate": ("he", None),
    "inferno_startburn": ("fire", "inferno_expire"),
}
FIRE_WEAPONS = {"Molotov", "Incendiary Grenade"}


def _throws(p: DemoPart) -> pd.DataFrame:
    w = event(p, "weapon_fire", player=["team_name", "team_clan_name", "last_place_name", "X", "Y"], other=["game_time"])
    w = w[w.weapon.isin(THROWS)].copy()
    w["type"] = w.weapon.map(THROWS)
    w["key"] = w.type.replace({"molotov": "fire", "incendiary": "fire"})
    return w.sort_values("tick").reset_index(drop=True)


def _match_throws(g: pd.DataFrame, throws: pd.DataFrame, max_flight: int = 30 * TICK_RATE) -> pd.Series:
    """For each detonation, the index of its throw in `throws` (or NA), first thrown = first to go off.

    Matching each detonation to the latest earlier throw gets double smokes wrong:
    both detonations take the second throw and the first looks like it never went off.
    """
    out = pd.Series(pd.NA, index=g.index, dtype="object")
    for (who, key), det in g.groupby(["user_name", "key"]):
        queue = throws[(throws.user_name == who) & (throws.key == key)].tick
        queue = list(zip(queue.index, queue))
        for i, tick in det.tick.sort_values().items():
            queue = [(j, t) for j, t in queue if t >= tick - max_flight]  # older ones never went off
            if queue and queue[0][1] <= tick:
                out[i] = queue.pop(0)[0]
    return out


def build_utility(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str],
                  damage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per grenade (where it was thrown from, where it landed, how long it lasted,
    the damage and blinding it did) plus one row per player blinded.

    Blinds exist only in demos that record player_blind (PW yes, ESL no); there the
    blind columns are empty rather than zero.
    """
    grenades, blinds = [], []
    for p in parts:
        r = rounds[rounds.part == p.index]
        if r.empty:
            continue
        throws = _throws(p)
        went_off = []
        for name, (key, end_name) in GRENADE_EVENTS.items():
            g = event(p, name, player=["team_name", "team_clan_name"], other=["game_time"])
            if g.empty:
                continue
            g = g.sort_values("tick").reset_index(drop=True)
            g["key"] = key
            # The throw, matched in order per player and grenade kind.
            g["_throw"] = _match_throws(g, throws)
            cols = ["tick", "type", "game_time", "user_last_place_name", "user_X", "user_Y"]
            th = throws[cols].rename(columns={"tick": "throw_tick", "game_time": "throw_time"})
            g = g.join(th, on="_throw")
            # The end: the next expire of the same entity id (ids are reused across rounds).
            if end_name:
                e = event(p, end_name).sort_values("tick")[["tick", "entityid"]].rename(columns={"tick": "end_tick"})
                g = pd.merge_asof(g, e, left_on="tick", right_on="end_tick", by="entityid", direction="forward")
            else:
                g["end_tick"] = pd.NA
            went_off.append(g.assign(part=p.index, detonated=True))
        grenades += went_off

        # Throws that never went off: a molotov that can't catch fire still hits a
        # player for 1 HP in flight (Spinx on NiKo, Falcons vs MOUZ Nuke round 20).
        used = set(pd.concat(went_off)["_throw"].dropna()) if went_off else set()
        lost = throws[~throws.index.isin(used)]
        if len(lost):
            grenades.append(pd.DataFrame({
                "tick": lost.tick, "user_name": lost.user_name, "user_team_name": lost.user_team_name,
                "user_team_clan_name": lost.user_team_clan_name, "game_time": lost.game_time, "key": lost.key,
                "type": lost.type, "throw_tick": lost.tick, "throw_time": lost.game_time,
                "user_last_place_name": lost.user_last_place_name, "user_X": lost.user_X, "user_Y": lost.user_Y,
                "x": float("nan"), "y": float("nan"), "z": float("nan"), "entityid": pd.NA, "end_tick": pd.NA,
                "detonated": lost.type.ne("decoy").map({True: False, False: pd.NA}), "part": p.index,
            }))

        b = event(p, "player_blind", player=["team_name", "team_clan_name"], other=["game_time"])
        if not b.empty:
            b, _ = _in_rounds(b, r)
            blinds.append(b.assign(part=p.index))

    g = pd.concat(grenades, ignore_index=True)
    g = pd.concat([_in_rounds(x, rounds[rounds.part == pi])[0] for pi, x in g.groupby("part")], ignore_index=True)
    g["type"] = g.type.fillna(g.key.replace({"fire": "molotov"}))
    util = pd.DataFrame({
        "part": g.part, "round": g["round"], "type": g.type,
        "thrower": g.user_name, "team": g.user_team_clan_name.map(names), "side": g.user_team_name.map(SIDES),
        "throw_tick": g.throw_tick, "throw_time": g.throw_time,
        "throw_place": g.user_last_place_name, "throw_x": g.user_X, "throw_y": g.user_Y,
        "tick": g.tick, "game_time": g.game_time,
        "seconds_into_round": g.seconds_into_round, "after_round_end": g.after_round_end,
        "detonated": g.detonated,
        "x": g.x, "y": g.y, "z": g.z,
        "end_tick": g.end_tick,
        "duration_s": (pd.to_numeric(g.end_tick) - g.tick) / TICK_RATE,
        "entityid": g.entityid,
    }).sort_values(["part", "tick"]).reset_index(drop=True)

    # Damage done: from the throw (a grenade hitting a player in flight does 1 HP)
    # to the detonation tick for HE, or until the fire goes out.
    # Each hit goes to one grenade: the most recent one whose window holds it.
    util["enemy_damage"] = util["team_damage"] = 0
    dmg = util[util.type.isin(["he", "molotov", "incendiary"])].copy()
    dmg["win_start"] = dmg.throw_tick.fillna(dmg.tick - 3 * TICK_RATE)
    burn_end = dmg.end_tick.fillna(dmg.tick + 8 * TICK_RATE) + TICK_RATE
    dmg["win_end"] = dmg.tick.where(dmg.type == "he", burn_end) + 2
    hits = damage[damage.weapon.isin(["High Explosive Grenade", *FIRE_WEAPONS]) & damage.kind.isin(["enemy", "team", "self"])]
    for h in hits.itertuples():
        is_he = h.weapon == "High Explosive Grenade"
        c = dmg[(dmg.part == h.part) & (dmg.thrower == h.attacker) & ((dmg.type == "he") == is_he)
                & (dmg.win_start <= h.tick) & (dmg.win_end >= h.tick)]
        if len(c):
            col = "enemy_damage" if h.kind == "enemy" else "team_damage"
            util.at[c.win_start.idxmax(), col] += int(h.damage)

    if blinds:
        b = pd.concat(blinds, ignore_index=True)
        kind = pd.Series("enemy", index=b.index)
        kind[b.attacker_team_name == b.user_team_name] = "team"
        kind[b.attacker_name == b.user_name] = "self"
        bl = pd.DataFrame({
            "part": b.part, "round": b["round"], "tick": b.tick, "game_time": b.game_time,
            "seconds_into_round": b.seconds_into_round,
            "flasher": b.attacker_name, "flasher_team": b.attacker_team_clan_name.map(names),
            "victim": b.user_name, "victim_team": b.user_team_clan_name.map(names), "victim_side": b.user_team_name.map(SIDES),
            "duration_s": b.blind_duration, "kind": kind, "entityid": b.entityid,
        })
        flashes = util.type == "flash"
        per = (bl.groupby(["part", "entityid", "round", "kind"]).agg(n=("victim", "size"), s=("duration_s", "sum"))
               .unstack("kind").fillna(0))
        key = list(zip(util.part, util.entityid, util["round"]))
        get = lambda col, k: [per[(col, k)].get(x, 0) if (col, k) in per.columns else 0 for x in key]
        util["enemies_blinded"] = pd.Series(get("n", "enemy"), index=util.index).where(flashes)
        util["teammates_blinded"] = pd.Series(get("n", "team"), index=util.index).where(flashes)
        util["enemy_blind_s"] = pd.Series(get("s", "enemy"), index=util.index).where(flashes)
    else:
        bl = pd.DataFrame()
        util["enemies_blinded"] = util["teammates_blinded"] = util["enemy_blind_s"] = pd.NA
    return util, bl


def add_blinds_to_kills(kills: pd.DataFrame, blinds: pd.DataFrame) -> pd.DataFrame:
    """Was the victim blind when killed, by whom, and was it a teammate's flash?

    Empty (not False) when the demo has no player_blind events.
    """
    kills = kills.copy()
    kills["victim_blind_s"] = pd.NA
    kills["victim_blinded_by"] = pd.NA
    kills["victim_blinded_by_teammate"] = pd.NA
    if blinds.empty:
        return kills
    kills["victim_blind_s"] = 0.0
    kills["victim_blinded_by_teammate"] = False
    for i, k in kills.iterrows():
        b = blinds[(blinds.part == k.part) & (blinds.victim == k.victim) & (blinds.tick <= k.tick)]
        b = b[b.tick + b.duration_s * TICK_RATE >= k.tick]
        if len(b):
            last = b.iloc[-1]
            kills.at[i, "victim_blind_s"] = round((last.tick + last.duration_s * TICK_RATE - k.tick) / TICK_RATE, 2)
            kills.at[i, "victim_blinded_by"] = last.flasher
            kills.at[i, "victim_blinded_by_teammate"] = last.kind == "team"
    return kills


ECON_PROPS = ["team_name", "team_clan_name", "balance", "cash_spent_this_round", "current_equip_value",
              "round_start_equip_value", "armor_value", "has_helmet", "has_defuser", "inventory", "is_alive",
              "kills_total", "mvps"]


def _stat_ticks(p: DemoPart, r: pd.DataFrame) -> list[int]:
    """Where to read a round's running totals: just before the next round starts, after any exit frags.

    The last round of a part has no next start, so read 1 s after round_end.
    """
    starts = event(p, "round_start").tick.sort_values()
    out = []
    for end in r.end_tick:
        later = starts[starts > end]
        out.append(int(later.iloc[0]) - 1 if len(later) else int(end) + TICK_RATE)
    return out


def build_player_rounds(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str], kills: pd.DataFrame,
                        damage: pd.DataFrame, blinds: pd.DataFrame) -> pd.DataFrame:
    """One row per player per round: economy from the game, performance from our own tables.

    The game's damage_total lags round_end, so kills, damage and assists come from
    the kills and damage tables (checked against the server's final totals). The
    game's running kills_total is kept to check per-round kills, and mvps to find the MVP.
    """
    rows = []
    for p in parts:
        r = rounds[rounds.part == p.index]
        if r.empty:
            continue
        stat = _stat_ticks(p, r)
        # Buying goes on for BUY_SECONDS after the round goes live, so the buy is read at buy-time end.
        buy_end = [min(int(lt) + BUY_SECONDS * TICK_RATE, int(et)) for lt, et in zip(r.live_tick, r.end_tick)]
        moments = {"start": r.start_tick.astype(int).tolist(), "live": r.live_tick.astype(int).tolist(),
                   "buy_end": buy_end, "end": r.end_tick.astype(int).tolist(), "stat": stat}
        t = p.parser.parse_ticks(ECON_PROPS, ticks=sorted({x for v in moments.values() for x in v}))
        t = t[t.team_name.isin(SIDES)].set_index(["tick", "name"])
        got = pd.Index(sorted(t.index.get_level_values("tick").unique()))
        # A requested tick can fall in a recording gap; use the nearest recorded tick before it.
        near = lambda tick: got[max(got.searchsorted(tick, side="right") - 1, 0)]
        for i, rnd in enumerate(r["round"]):
            at = {m: t.loc[near(moments[m][i])] for m in moments}
            for name, live in at["live"].iterrows():
                rows.append({
                    "round": int(rnd), "player": name,
                    "team": names[live.team_clan_name], "side": SIDES[live.team_name],
                    # Money available for this buy. Read at the live tick: a reward can land
                    # just after round_start (e.g. for a kill right after the previous round ended).
                    "money_start": int(live.balance) + int(live.cash_spent_this_round),
                    "equip_carried": int(at["start"].at[name, "round_start_equip_value"]),
                    "money_after_buy": int(at["buy_end"].at[name, "balance"]),
                    "spent": int(at["buy_end"].at[name, "cash_spent_this_round"]),
                    "spent_round": int(at["stat"].at[name, "cash_spent_this_round"]),
                    "equip_value": int(at["buy_end"].at[name, "current_equip_value"]),
                    "armor": int(at["buy_end"].at[name, "armor_value"]),
                    "helmet": bool(at["buy_end"].at[name, "has_helmet"]),
                    "defuser": bool(at["buy_end"].at[name, "has_defuser"]),
                    "inventory": list(at["buy_end"].at[name, "inventory"]),
                    "survived": bool(at["end"].at[name, "is_alive"]),
                    "money_end": int(at["stat"].at[name, "balance"]),
                    "_kills_total": int(at["stat"].at[name, "kills_total"]),
                    "_mvps": int(at["stat"].at[name, "mvps"]),
                })
    pr = pd.DataFrame(rows).sort_values(["player", "round"]).reset_index(drop=True)

    # Per-round MVP and server kills from the running totals.
    for col in ("_kills_total", "_mvps"):
        pr[col.replace("_total", "") + "_round"] = pr.groupby("player")[col].diff().fillna(pr[col])
    pr["mvp"] = pr.pop("_mvps_round") == 1
    pr = pr.rename(columns={"_kills_round": "server_kills"}).drop(columns=["_kills_total", "_mvps"])

    k = kills[kills.kind == "kill"]
    count = lambda df, who: df.groupby(["round", who]).size()
    ene = damage[damage.kind == "enemy"]
    util_hits = ene[ene.weapon.isin(["High Explosive Grenade", *FIRE_WEAPONS])]
    key = list(zip(pr["round"], pr.player))
    look = lambda s: [int(s.get(x, 0)) for x in key]
    pr["kills"] = look(count(k, "attacker"))
    pr["headshot_kills"] = look(count(k[k.headshot], "attacker"))
    pr["assists"] = look(count(k[k.assister.notna()], "assister"))
    pr["deaths"] = look(count(kills, "victim"))
    pr["damage"] = look(ene.groupby(["round", "attacker"]).damage.sum())
    pr["utility_damage"] = look(util_hits.groupby(["round", "attacker"]).damage.sum())
    if blinds.empty:
        pr["enemies_flashed"] = pd.NA
    else:
        pr["enemies_flashed"] = look(count(blinds[blinds.kind == "enemy"], "flasher"))
    return pr.sort_values(["round", "team", "player"]).reset_index(drop=True)


BUY_SECONDS = 20  # buy time after the round goes live (PW's buytime_ended confirms 20 s)
TIMEOUT_FLAGS = {"is_terrorist_timeout": "T", "is_ct_timeout": "CT", "is_technical_timeout": "technical"}


def add_timeouts(parts: list[DemoPart], rounds: pd.DataFrame) -> pd.DataFrame:
    """Mark rounds whose freeze time held a timeout, which team called it, and when it started."""
    rounds = rounds.copy()
    rounds["timeout"] = rounds["timeout_team"] = None
    rounds["timeout_tick"] = pd.NA
    for p in parts:
        r = rounds[rounds.part == p.index]
        for i, row in r.iterrows():
            if pd.isna(row.start_tick) or pd.isna(row.live_tick):
                continue
            ticks = list(range(int(row.start_tick), int(row.live_tick), TICK_RATE))
            t = p.parser.parse_ticks(list(TIMEOUT_FLAGS), ticks=ticks).groupby("tick")[list(TIMEOUT_FLAGS)].first()
            for flag, who in TIMEOUT_FLAGS.items():
                on = t.index[t[flag].astype(bool)]
                if len(on):
                    rounds.at[i, "timeout"] = "technical" if who == "technical" else "tactical"
                    rounds.at[i, "timeout_team"] = None if who == "technical" else (row.ct_team if who == "CT" else row.t_team)
                    rounds.at[i, "timeout_tick"] = int(on[0])
                    break
    return rounds


def build_pickups(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """Weapons and kits picked up off the ground during a round.

    item_pickup also fires for purchases (before buy time ends), for grenades
    moving between slots (one player "picks up" 4 smokes in 7 s), and for the
    next round's spawn pistols (just after round_end), so all three are left out.
    """
    out = []
    for p in parts:
        r = rounds[rounds.part == p.index]
        d = event(p, "item_pickup", player=["team_name", "team_clan_name", "last_place_name"], other=["game_time"])
        if d.empty or r.empty:
            continue
        d, _ = _in_rounds(d, r)
        buy_end = d["round"].map(dict(zip(r["round"], r.live_tick + BUY_SECONDS * TICK_RATE)))
        noise = ["c4", "knife", "knife_t", "flashbang", "smokegrenade", "hegrenade", "molotov", "incgrenade", "decoy"]
        d = d[(d.tick > buy_end) & ~d.after_round_end & ~d.item.isin(noise)]
        out.append(pd.DataFrame({
            "part": p.index, "round": d["round"], "tick": d.tick, "game_time": d.game_time,
            "seconds_into_round": d.seconds_into_round, "after_round_end": d.after_round_end,
            "player": d.user_name, "team": d.user_team_clan_name.map(names), "side": d.user_team_name.map(SIDES),
            "place": d.user_last_place_name,
            "item": d.item.map(lambda x: {"defuser": "Defuse Kit", "vest": "Kevlar", "vesthelm": "Kevlar + Helmet"}.get(x, weapon_name(x))),
        }))
    cols = ["part", "round", "tick", "game_time", "seconds_into_round", "after_round_end", "player", "team", "side", "place", "item"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


TIMELINE_COLS = ["part", "round", "tick", "game_time", "seconds_into_round", "type",
                 "actor", "actor_team", "target", "target_team", "summary", "in_hltv"]


def build_timeline(rounds: pd.DataFrame, kills: pd.DataFrame, bomb: pd.DataFrame, util: pd.DataFrame,
                   pickups: pd.DataFrame) -> pd.DataFrame:
    """Every event of the map in time order, one row each, with a short readable summary.

    Built from the other tables, nothing new is parsed. Damage hits stay out
    (HP comes from states). `in_hltv` marks the event kinds HLTV's live log sends.
    """
    rows = []
    a, b = rounds.ct_team.iloc[0], rounds.t_team.iloc[0]
    for r in rounds.itertuples():
        sb, sa = f"{a} {r.team_a_score_before}-{r.team_b_score_before} {b}", f"{a} {r.team_a_score_after}-{r.team_b_score_after} {b}"
        rows.append(dict(part=r.part, round=r.round, tick=r.start_tick, game_time=r.start_time, type="round_start",
                         summary=f"Round {r.round} buy phase · {sb} · {r.ct_team} CT, {r.t_team} T", in_hltv=True))
        if r.timeout:
            who = r.timeout_team or "technical"
            at = r.start_time + (r.timeout_tick - r.start_tick) / TICK_RATE
            rows.append(dict(part=r.part, round=r.round, tick=r.timeout_tick, game_time=at, type="timeout",
                             actor_team=r.timeout_team, summary=f"{r.timeout.capitalize()} timeout ({who})", in_hltv=False))
        rows.append(dict(part=r.part, round=r.round, tick=r.live_tick, game_time=r.live_time, type="round_live",
                         summary=f"Round {r.round} live", in_hltv=False))
        rows.append(dict(part=r.part, round=r.round, tick=r.end_tick, game_time=r.end_time, type="round_end",
                         actor_team=r.winner_team, summary=f"{r.winner_team} win round {r.round} ({r.reason}) · {sa}", in_hltv=True))
    for k in kills.itertuples():
        extra = [x for x, on in (("headshot", k.headshot), ("wallbang", k.wallbang), ("through smoke", k.through_smoke),
                                 ("noscope", k.noscope)) if on]
        if k.kind == "kill" and isinstance(k.assister, str):
            extra.append(("flash assist " if k.flash_assist else "assist ") + k.assister)
        who = f"{k.attacker} ({k.weapon})" if isinstance(k.attacker, str) and k.kind != "bomb" else k.weapon
        text = {"suicide": f"{k.victim} died ({k.weapon})", "world": f"{k.victim} died (world)"}.get(
            k.kind, f"{who} {'team-killed' if k.kind == 'teamkill' else 'killed'} {k.victim}")
        rows.append(dict(part=k.part, round=k.round, tick=k.tick, game_time=k.game_time, type=k.kind,
                         actor=k.attacker, actor_team=k.attacker_team, target=k.victim, target_team=k.victim_team,
                         summary=" · ".join([text, *extra, f"{k.ct_alive_after}v{k.t_alive_after} (CT v T)"]),
                         in_hltv=k.kind in ("kill", "teamkill", "suicide")))
    for b in bomb.itertuples():
        site = f" {b.site}" if isinstance(b.site, str) else ""
        rows.append(dict(part=b.part, round=b.round, tick=b.tick, game_time=b.game_time, type=f"bomb_{b.event}",
                         actor=b.player, actor_team=b.team,
                         summary=f"{b.player} bomb {b.event.replace('_', ' ')}{site} · {b.ct_alive}v{b.t_alive} (CT v T)",
                         in_hltv=b.event in ("planted", "defused")))
    for u in util.itertuples():
        dud = " (never went off)" if u.detonated is False else ""
        rows.append(dict(part=u.part, round=u.round, tick=u.tick, game_time=u.game_time, type=u.type,
                         actor=u.thrower, actor_team=u.team,
                         summary=f"{u.thrower} {u.type} from {u.throw_place}{dud}", in_hltv=False))
    for x in pickups.itertuples():
        rows.append(dict(part=x.part, round=x.round, tick=x.tick, game_time=x.game_time, type="pickup",
                         actor=x.player, actor_team=x.team, summary=f"{x.player} picks up {x.item}", in_hltv=False))

    t = pd.DataFrame(rows).reindex(columns=TIMELINE_COLS)
    live = dict(zip(zip(rounds.part, rounds["round"]), rounds.live_time))
    t["seconds_into_round"] = t.game_time.astype(float) - [live[(p, r)] for p, r in zip(t.part, t["round"])]
    # On a shared tick, the round's last kill or the explosion comes before round_end.
    order = {"round_start": 0, "timeout": 1, "round_live": 2, "round_end": 4}
    t["_o"] = t.type.map(order).fillna(3)
    return t.sort_values(["part", "tick", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)


STATE_PROPS = ["team_name", "team_clan_name", "is_alive", "health", "armor_value", "has_helmet", "has_defuser",
               "balance", "active_weapon_name", "last_place_name", "X", "Y", "Z", "inventory"]


def build_states(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str],
                 kills: pd.DataFrame) -> pd.DataFrame:
    """Every player, once a second, from each round's live tick to its end.

    Frames are built from this. HLTV's slower HP updates are simulated later,
    when frames are written, not here. `kills` corrects rows carried over a
    recording gap: a player who died inside the gap is dead from that tick on.
    """
    out = []
    for p in parts:
        r = rounds[rounds.part == p.index]
        if r.empty:
            continue
        ticks, rnd = [], {}
        for row in r.itertuples():
            for t in range(int(row.live_tick), int(row.end_tick) + 1, TICK_RATE):
                ticks.append(t)
                rnd[t] = (row.round, row.live_time)
        s = p.parser.parse_ticks(STATE_PROPS, ticks=ticks)
        s = s[s.team_name.isin(SIDES)]
        # Some demos skip stretches of 2-3 s (MongolZ vs Liquid round 20). Carry each
        # player's last known state into the missing seconds and mark those rows.
        # Filling stays within a round, so one round's state never leaks into the next.
        have = set(s.tick)
        full = pd.MultiIndex.from_product([ticks, sorted(s.name.unique())], names=["tick", "name"])
        s = s.drop(columns=["steamid"], errors="ignore").set_index(["tick", "name"]).reindex(full).reset_index()
        s["filled"] = ~s.tick.isin(have)
        s["_round"] = s.tick.map(lambda t: rnd[t][0])
        cols = [c for c in s.columns if c not in ("tick", "name", "_round", "filled")]
        s[cols] = s.groupby(["name", "_round"])[cols].transform(lambda x: x.ffill().bfill())
        s = s[s.team_name.isin(SIDES)].drop(columns="_round")
        out.append(pd.DataFrame({
            "part": p.index,
            "round": s.tick.map(lambda t: rnd[t][0]),
            "tick": s.tick,
            "second": ((s.tick - s.tick.map(lambda t: r.set_index("round").live_tick[rnd[t][0]])) // TICK_RATE).astype(int),
            "player": s.name,
            "team": s.team_clan_name.map(names),
            "side": s.team_name.map(SIDES),
            "alive": s.is_alive.astype(bool),
            "hp": s.health,
            "armor": s.armor_value,
            "helmet": s.has_helmet.astype(bool),
            "defuser": s.has_defuser.astype(bool),
            "money": s.balance,
            "weapon": s.active_weapon_name.map(weapon_name),
            # Casters track the bomb ("the bomb is next to NiKo"); carrier from the inventory.
            "has_bomb": s.inventory.map(lambda inv: isinstance(inv, (list, tuple)) and "C4 Explosive" in inv)
                        & s.is_alive.astype(bool),
            "place": s.last_place_name,
            "x": s.X, "y": s.Y, "z": s.Z,
            "filled": s.filled.astype(bool),
        }))
    st = pd.concat(out, ignore_index=True)
    died = kills.groupby(["part", "round", "victim"]).tick.min()
    death_tick = pd.Series([died.get((p, r, n)) for p, r, n in zip(st.part, st["round"], st.player)], index=st.index)
    dead = st.filled & death_tick.notna() & (st.tick >= death_tick.fillna(0))
    st.loc[dead, "alive"] = False
    st.loc[dead, "hp"] = 0
    return st.sort_values(["round", "second", "team", "player"]).reset_index(drop=True)


def server_player_totals(parts: list[DemoPart], rounds: pd.DataFrame) -> pd.DataFrame:
    """The server scoreboard's totals per player after the last round, for checking."""
    last = rounds.iloc[-1]
    p = parts[int(last.part)]
    props = ["kills_total", "deaths_total", "damage_total"]
    t = p.parser.parse_ticks(props, ticks=[int(last.end_tick) + TICK_RATE])
    if t.empty:  # that tick fell in a recording gap
        t = p.parser.parse_ticks(props, ticks=list(range(int(last.end_tick), int(last.end_tick) + 3 * TICK_RATE)))
        t = t[t.tick == t.tick.max()]
    return t.set_index("name")[props]


def rosters(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str]) -> dict[str, list[dict]]:
    """Players of each team, read at the first live round."""
    first = rounds.iloc[0]
    p = parts[int(first.part)]
    t = p.parser.parse_ticks(["team_clan_name"], ticks=[int(first.live_tick)])
    out: dict[str, list[dict]] = {}
    for r in t.sort_values("name").itertuples():
        out.setdefault(names[r.team_clan_name], []).append({"name": r.name, "steamid": str(r.steamid)})
    return out


def server_final_score(parts: list[DemoPart], rounds: pd.DataFrame, names: dict[str, str]) -> dict[str, int]:
    """The server's own team round totals just after the last round, for checking."""
    last = rounds.iloc[-1]
    p = parts[int(last.part)]
    t = p.parser.parse_ticks(["team_clan_name", "team_rounds_total"], ticks=[int(last.end_tick) + TICK_RATE])
    if t.empty:
        t = p.parser.parse_ticks(["team_clan_name", "team_rounds_total"], ticks=[int(last.end_tick)])
    return {names[k]: int(v) for k, v in t.groupby("team_clan_name").team_rounds_total.max().items()}
