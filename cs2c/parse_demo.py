"""Parse pilot demos into data/interim/<map_id>/.

    python -m cs2c.parse_demo                 # all pilot maps
    python -m cs2c.parse_demo cac-2394228-nuke
"""
import json
import sys

import pandas as pd

from . import demo
from .paths import INTERIM, MapSpec, pilot_maps, team_names


def final_score(rounds) -> dict[str, int]:
    last = rounds.iloc[-1]
    return {rounds.ct_team.iloc[0]: int(last.team_a_score_after), rounds.t_team.iloc[0]: int(last.team_b_score_after)}


def check_rounds(rounds, server_score) -> list[tuple[str, bool, str]]:
    ours = final_score(rounds)
    numbers = rounds["round"].tolist()
    low, high = sorted(ours.values())
    swaps = [i + 1 for i in rounds.index[rounds.ct_team != rounds.ct_team.shift()].tolist()[1:]]
    return [
        ("rounds numbered 1..N with no gaps", numbers == list(range(1, len(numbers) + 1)), f"{len(numbers)} rounds"),
        ("score matches the server's own totals", ours == server_score, f"ours {ours}, server {server_score}"),
        ("round count equals total score", len(numbers) == low + high, f"{low}+{high}"),
        ("winner reached 13 (more in overtime)", high >= 13, f"winner {high}"),
        ("every round has a live tick", bool(rounds.live_tick.notna().all()), ""),
        ("first side swap at round 13", swaps[:1] == [13] if len(numbers) > 12 else True, f"sides swap at rounds {swaps}"),
    ]


def check_kills(kills, rounds, server, dropped) -> list[tuple[str, bool, str]]:
    ours = pd.DataFrame({
        "kills": kills[kills.kind == "kill"].groupby("attacker").size(),
        "deaths": kills.groupby("victim").size(),
    }).fillna(0).astype(int).reindex(server.index, fill_value=0)
    # The server may or may not count a freeze-time death (admin slay yes, backup restore no).
    frozen = dropped[dropped.during == "freeze"].groupby("victim").size().reindex(server.index, fill_value=0)
    deaths_ok = (ours.deaths == server.deaths_total) | (ours.deaths + frozen == server.deaths_total)
    bad = ours[(ours.kills != server.kills_total) | ~deaths_ok]
    bad_detail = "; ".join(f"{n} ours {r.kills}/{r.deaths} server {server.at[n, 'kills_total']}/{server.at[n, 'deaths_total']}"
                           for n, r in bad.iterrows())

    # A round won by elimination must end with the losing side at 0 alive.
    in_round = kills[~kills.after_round_end].groupby("round")[["ct_alive_after", "t_alive_after"]].min()
    elim = rounds[rounds.reason.isin(["ct_killed", "t_killed"])].set_index("round")
    wrong = [int(r) for r, row in elim.iterrows()
             if in_round.loc[r, "ct_alive_after" if row.reason == "ct_killed" else "t_alive_after"] != 0]
    kinds = kills.kind.value_counts().to_dict()
    return [
        ("kills and deaths per player match the server scoreboard", bad.empty, bad_detail or f"{len(server)} players"),
        ("elimination rounds end with the losing side at 0 alive", not wrong, f"wrong rounds {wrong}" if wrong else f"{len(elim)} rounds"),
        ("alive counts never negative", bool((kills[["ct_alive_after", "t_alive_after"]] >= 0).all().all()), str(kinds)),
    ]


def check_bomb(bomb, rounds) -> list[tuple[str, bool, str]]:
    planted = bomb[bomb.event == "planted"]
    per_round = planted.groupby("round").size()
    need = rounds[rounds.reason.isin(["bomb_exploded", "bomb_defused"])]
    missing = [int(r) for r in need["round"] if r not in per_round.index]
    timer = (bomb[bomb.event == "exploded"].set_index("round").game_time
             - planted.set_index("round").game_time).dropna()
    return [
        ("exploded/defused rounds have a plant", not missing, f"missing in rounds {missing}" if missing else f"{len(need)} rounds"),
        ("at most one plant per round", bool((per_round <= 1).all()), f"{len(planted)} plants"),
        ("every plant has site A or B", bool(planted.site.isin(["A", "B"]).all()), str(planted.site.value_counts().to_dict())),
        ("bomb explodes about 40 s after the plant", bool(timer.between(39, 41.5).all()),
         f"{timer.min():.1f}-{timer.max():.1f} s" if len(timer) else "no explosions"),
    ]


def check_damage(damage, kills, server) -> list[tuple[str, bool, str]]:
    ours = damage[damage.kind == "enemy"].groupby("attacker").damage.sum().reindex(server.index, fill_value=0)
    raw = damage[damage.kind == "enemy"].groupby("attacker").damage_raw.sum().reindex(server.index, fill_value=0)
    diff = (ours - server.damage_total).abs()
    lethal = int((damage.hp_after == 0).sum())
    return [
        ("damage per player matches the server's damage_total", bool((diff == 0).all()),
         f"max gap {int(diff.max())}; uncapped sums would be off by up to {int((raw - server.damage_total).abs().max())}"),
        ("hits that leave 0 HP = deaths", lethal == len(kills), f"{lethal} vs {len(kills)}"),
        ("HP never rises within a round", bool((damage.damage >= 0).all()), ""),
    ]


def check_utility(util, blinds, damage, kills) -> list[tuple[str, bool, str]]:
    landed = util[util.detonated.eq(True)]
    thrown = landed.throw_tick.notna().mean()
    duds = util[util.detonated.eq(False)].type.value_counts().to_dict()
    fire = util[util.type.isin(["molotov", "incendiary"])].duration_s.median()
    smoke = util[util.type == "smoke"].duration_s.median()
    hit = damage[damage.weapon.isin(["High Explosive Grenade", *demo.FIRE_WEAPONS]) & (damage.kind == "enemy")]
    attributed = util.enemy_damage.sum() / max(hit.damage.sum(), 1)
    # A few grenades are never thrown: dropped with the pin pulled when the holder dies.
    checks = [
        ("grenades linked to their throw (99%+)", thrown >= 0.99,
         f"{thrown:.1%} of {len(landed)}; {landed.type.value_counts().to_dict()}; thrown but never went off {duds}"),
        ("smokes last about 18-22 s, fires about 5-7 s", 18 <= smoke <= 23 and 4 <= fire <= 8, f"median smoke {smoke:.1f} s, fire {fire:.1f} s"),
        ("all enemy utility damage tied to a grenade", abs(attributed - 1) < 0.001,
         f"{int(util.enemy_damage.sum())} of {int(hit.damage.sum())} HP"),
    ]
    if blinds.empty:
        checks.append(("flash assists match a blind (no player_blind in this demo)", True, "skipped"))
    else:
        # The game gives a flash assist for a recent flash, even if the blind wore off just before the kill.
        fa = kills[kills.flash_assist & (kills.kind == "kill")]
        recent = [((blinds.part == k.part) & (blinds.victim == k.victim) & (blinds.flasher == k.assister)
                   & blinds.tick.between(k.tick - 6 * demo.TICK_RATE, k.tick)).any() for k in fa.itertuples()]
        ok = sum(recent) / len(fa) if len(fa) else 1.0
        still = (fa.victim_blinded_by == fa.assister).mean() if len(fa) else 0
        checks.append(("flash assists match a flash by the assister in the last 6 s", ok == 1,
                       f"{ok:.0%} of {len(fa)}; victim still blind at the kill in {still:.0%}"))
    return checks


def check_player_rounds(pr, rounds) -> list[tuple[str, bool, str]]:
    sizes = pr.groupby(["round", "team"]).size()
    kills_gap = pr[pr.kills != pr.server_kills]
    mvps = pr.groupby("round").mvp.sum()
    died_unseen = pr[~pr.survived & (pr.deaths == 0)]
    return [
        ("5 players per team every round", bool((sizes == 5).all()) and len(sizes) == 2 * len(rounds), f"{len(pr)} rows"),
        ("per-round kills match the server's running totals", kills_gap.empty,
         "" if kills_gap.empty else kills_gap[["round", "player", "kills", "server_kills"]].head(4).to_dict("records")),
        ("one MVP every round", bool((mvps == 1).all()), "" if (mvps == 1).all() else f"rounds {mvps[mvps != 1].to_dict()}"),
        ("money left after the buy is never negative, and round spend never shrinks",
         bool((pr.money_after_buy >= 0).all() and (pr.spent_round >= pr.spent).all()),
         f"{int((pr.spent_round > pr.spent).sum())} player-rounds spent more after the buy phase"),
        ("everyone not alive at round end has a death", died_unseen.empty, f"{len(died_unseen)} rows"),
    ]


def check_states(states, kills, rounds) -> list[tuple[str, bool, str]]:
    per = states.groupby(["round", "second"]).size()
    expected = int((rounds.live_seconds.astype(int) + 1).sum())
    # Players alive per side at each snapshot tick must match the kills table at that tick.
    alive = states[states.alive].groupby(["part", "round", "tick", "side"]).size().unstack(fill_value=0)
    wrong = 0
    for (part, rnd, tick), row in alive.iterrows():
        before = kills[(kills.part == part) & (kills["round"] == rnd) & (kills.tick <= tick)]
        if len(before):
            last = before.iloc[-1]
            if (row.get("CT", 0), row.get("T", 0)) != (last.ct_alive_after, last.t_alive_after):
                wrong += 1
    hp_ok = states[~states.alive].hp.eq(0).all() and states[states.alive].hp.between(1, 100).all()
    return [
        ("10 players every second of every live round", bool((per == 10).all()),
         f"{len(per)} seconds, {len(states)} rows, {states.filled.mean():.1%} carried over a recording gap"),
        ("one snapshot per live second", abs(len(per) - expected) <= len(rounds), f"{len(per)} vs about {expected}"),
        ("alive counts agree with the kills table", wrong == 0, f"{wrong} of {len(alive)} snapshots differ"),
        ("HP is 0 when dead and 1-100 when alive", bool(hp_ok), ""),
        ("at most one bomb carrier, always on T side",
         bool((states[states.has_bomb].groupby(["round", "second"]).size() <= 1).all() and (states[states.has_bomb].side == "T").all()),
         f"carried in {states[states.has_bomb].groupby(['round', 'second']).ngroups} of {states.groupby(['round', 'second']).ngroups} seconds"),
    ]


def check_timeline(timeline, sources: dict) -> list[tuple[str, bool, str]]:
    counts = timeline.type.value_counts()
    want = {"kills": len(sources["kills"]), "bomb": len(sources["bomb"]), "utility": len(sources["utility"]),
            "pickups": len(sources["pickups"]), "rounds": 3 * len(sources["rounds"]) + int(sources["rounds"].timeout.notna().sum())}
    got = {"kills": int(counts.reindex(["kill", "teamkill", "suicide", "bomb", "world"]).fillna(0).sum()),
           "bomb": int(counts[counts.index.str.startswith("bomb_")].sum()),
           "utility": int(counts.reindex(["smoke", "flash", "he", "molotov", "incendiary", "decoy"]).fillna(0).sum()),
           "pickups": int(counts.get("pickup", 0)),
           "rounds": int(counts.reindex(["round_start", "round_live", "round_end", "timeout"]).fillna(0).sum())}
    # Ordered by tick; on a shared tick round_end must come after everything else in that tick.
    in_order = all((g.tick.astype(int).diff().dropna() >= 0).all() for _, g in timeline.groupby("part"))
    t = timeline.reset_index(drop=True)
    end_first = [i for i in t.index[t.type == "round_end"]
                 if i + 1 < len(t) and t.at[i + 1, "tick"] == t.at[i, "tick"] and t.at[i + 1, "part"] == t.at[i, "part"]]
    in_order = in_order and not end_first
    r = sources["rounds"]
    return [
        ("timeline holds every event from every table", want == got, f"{got}"),
        ("timeline is in time order", bool(in_order), f"{len(timeline)} events"),
        ("timeouts found", True, f"{r[r.timeout.notna()][['round', 'timeout', 'timeout_team']].values.tolist()}"),
    ]


def parse_map(spec: MapSpec) -> bool:
    names = team_names()
    parts = demo.open_parts(spec.demo_files)
    rounds = demo.add_timeouts(parts, demo.build_rounds(parts, names))
    kills, dropped = demo.build_kills(parts, rounds, names)
    bomb = demo.build_bomb(parts, rounds, names)
    damage = demo.build_damage(parts, rounds, names)
    util, blinds = demo.build_utility(parts, rounds, names, damage)
    kills = demo.add_blinds_to_kills(kills, blinds)
    player_rounds = demo.build_player_rounds(parts, rounds, names, kills, damage, blinds)
    states = demo.build_states(parts, rounds, names, kills)
    pickups = demo.build_pickups(parts, rounds, names)
    timeline = demo.build_timeline(rounds, kills, bomb, util, pickups)
    server = demo.server_player_totals(parts, rounds)
    checks = check_rounds(rounds, demo.server_final_score(parts, rounds, names))
    checks += check_kills(kills, rounds, server, dropped)
    checks += check_bomb(bomb, rounds)
    checks += check_damage(damage, kills, server)
    checks += check_utility(util, blinds, damage, kills)
    checks += check_player_rounds(player_rounds, rounds)
    checks += check_states(states, kills, rounds)
    checks += check_timeline(timeline, {"kills": kills, "bomb": bomb, "utility": util, "pickups": pickups, "rounds": rounds})

    out = INTERIM / spec.map_id
    out.mkdir(parents=True, exist_ok=True)
    rounds.to_parquet(out / "rounds.parquet", index=False)
    kills.to_parquet(out / "kills.parquet", index=False)
    bomb.to_parquet(out / "bomb.parquet", index=False)
    damage.to_parquet(out / "damage.parquet", index=False)
    util.to_parquet(out / "utility.parquet", index=False)
    player_rounds.to_parquet(out / "player_rounds.parquet", index=False)
    states.to_parquet(out / "states.parquet", index=False)
    pickups.to_parquet(out / "pickups.parquet", index=False)
    timeline.to_parquet(out / "timeline.parquet", index=False)
    if not blinds.empty:
        blinds.to_parquet(out / "blinds.parquet", index=False)
    score = final_score(rounds)
    team_a, team_b = rounds.ct_team.iloc[0], rounds.t_team.iloc[0]
    info = {
        "map_id": spec.map_id,
        "map_name": parts[0].header.get("map_name"),
        "server_name": parts[0].header.get("server_name"),
        "event": spec.event,
        "stage": spec.stage,
        "hltv_match_id": spec.hltv_match_id,
        "vod_id": spec.vod_id,
        "parts": [{"file": str(p.path.relative_to(INTERIM.parent)), "events": sorted(p.events)} for p in parts],
        "team_a": team_a,
        "team_b": team_b,
        "final_score": score,
        "rosters": demo.rosters(parts, rounds, names),
        "deaths_dropped_outside_rounds": dropped.during.value_counts().to_dict(),
        "checks": {name: bool(ok) for name, ok, _ in checks},
    }
    (out / "map_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))

    print(f"\n{spec.map_id}: {team_a} {score[team_a]}-{score[team_b]} {team_b} on {info['map_name']}, "
          f"{len(rounds)} rounds, {len(kills)} deaths (dropped outside rounds: {dropped.during.value_counts().to_dict()})")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return all(ok for _, ok, _ in checks)


def main(argv: list[str]) -> int:
    maps = pilot_maps()
    results = [parse_map(maps[i]) for i in (argv or list(maps))]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
