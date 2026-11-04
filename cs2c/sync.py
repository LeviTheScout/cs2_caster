"""Sync each map's demo clock to its VOD and cut the map's captions out of the VOD.

    python -m cs2c.sync [map_id or vod_id ...]

VOD time = demo game_time + offset. The evidence is the casters naming players
around kills: every caption word tagged with a player's steam id (see
captions.clean) supports the offsets that put it just after a kill by or on
that player. Found on the 20 pilot maps (2026-11-04):
- each map has one clear peak, 1.4-3x the next best offset;
- on ESL VODs the offset grows steadily, 2-5.5 s per 1000 s of demo (up to
  17 s over a map), so it is fitted as a line; PGL VODs are nearly flat;
- there are no jumps at pauses, half-time or between demo parts. A step is
  still allowed, at the cost of STEP_COST kills of evidence, so a cut VOD shows up.
Each map uses a constant or a line, whichever predicts held-out rounds better.

The offset says when casters react, not the exact frame a kill is shown: at
the fitted offset, mentions of a kill peak 2-6 s after it.

Needs the maps parsed and the captions cleaned first. Writes, per map, under
data/interim/<map_id>/:
  sync.json         offset model, per-round offsets and evidence, checks
  captions.parquet  the caption words inside the map, with round, phase
                    (freeze, live, over) and demo time
"""
import collections
import json
import sys

import numpy as np
import pandas as pd

from . import captions
from .paths import INTERIM, pilot_maps

# How much a mention counts by its delay after the kill: from 3 s before, full at 2-6 s after, gone by 11 s.
DELAY_S = [-3, 2, 6, 11]
DELAY_W = [0, 1, 1, 0]
NEAR = 25            # line fit: seconds either side of the coarse peak
STEP = 0.25
SLOPES = np.arange(-4, 10.001, 0.25) / 1000  # drift, seconds per second of demo
STEP_COST = 4.0      # evidence, in kills, a step in the offset must add
STEP_RANGE = 60
POST_MAP = 20        # seconds of talk kept after the last round ends
CHANCE_SHIFTS = np.r_[-600:-59:37, 60:601:37]  # offsets this far off are chance
# Words checked against bomb events, never used in the fit.
BOMB_WORDS = {"planted": ("plant",), "defused": ("defus",)}

MIN_PEAK_RATIO = 1.25  # coarse peak over the best offset at least 60 s away
MIN_CHANCE_RATIO = 3   # held-out evidence over evidence at chance offsets
MIN_ANCHORS = 5        # count tags or bomb events needed to judge them


def weight(lag):
    return np.interp(lag, DELAY_S, DELAY_W, left=0, right=0)


def roster_ids(map_id: str) -> dict[str, str]:
    info = json.loads((INTERIM / map_id / "map_info.json").read_text())
    return {p["name"]: p["steamid"] for ps in info["rosters"].values() for p in ps}


def kill_mentions(kills: pd.DataFrame, words: pd.DataFrame, ids: dict[str, str]) -> list[np.ndarray]:
    """VOD times of sure mentions of each kill's killer or victim."""
    tags = words[(words.kind == "player") & words.sure]
    by = {sid: g.t.to_numpy() for sid, g in tags.groupby("steamid")}
    empty = np.array([])
    return [np.concatenate([by.get(ids.get(a), empty), by.get(ids.get(v), empty)])
            for a, v in zip(kills.attacker, kills.victim)]


def coarse_peak(kt: np.ndarray, mentions: list[np.ndarray], vod_end: float) -> dict:
    """Kills with any mention in the delay window, for every whole-second offset over the VOD."""
    offs = np.arange(-kt.max() - 60, vod_end + 60, 1.0)
    score = np.zeros(len(offs), int)
    for t, ms in zip(kt, mentions):
        d = ms - t
        lo = np.searchsorted(offs, d - DELAY_S[-1])
        hi = np.searchsorted(offs, d - DELAY_S[0], side="right")
        mark = np.zeros(len(offs) + 1, int)
        np.add.at(mark, lo, 1)
        np.add.at(mark, hi, -1)
        score += np.cumsum(mark)[:-1] > 0
    best = int(score.argmax())
    far = score.copy()
    far[np.abs(offs - offs[best]) < 60] = 0
    return {"offset": float(offs[best]), "kills": int(score[best]), "runner_up": int(far.max()),
            "runner_up_offset": float(offs[far.argmax()]), "chance": float(np.median(score))}


def line_scores(kt, mentions, a, tmid) -> np.ndarray:
    """Evidence from each kill for each line offset = a + slope * (t - tmid): shape (kills, len(a), len(SLOPES))."""
    out = np.zeros((len(kt), len(a), len(SLOPES)))
    reach = np.abs(SLOPES).max() * np.abs(kt - tmid).max() + DELAY_S[-1] - DELAY_S[0]
    for i, (t, ms) in enumerate(zip(kt, mentions)):
        ms = ms[(ms > t + a[0] - reach) & (ms < t + a[-1] + reach)]
        if len(ms):
            off = a[:, None] + SLOPES[None, :] * (t - tmid)
            out[i] = weight(ms - (t + off)[..., None]).max(-1)
    return out


def fit_line(P: np.ndarray, rnd: np.ndarray) -> dict:
    """Best constant and best line, each also scored on rounds left out of its own fit."""
    zero = int(np.argmin(np.abs(SLOPES)))
    total = P.sum(0)
    fits = {}
    for name, cols in (("constant", [zero]), ("line", list(range(len(SLOPES))))):
        T = total[:, cols]
        ia, ib = np.unravel_index(T.argmax(), T.shape)
        held = 0.0
        for r in np.unique(rnd):
            m = rnd == r
            part = P[m][:, :, cols].sum(0)
            ja, jb = np.unravel_index((T - part).argmax(), T.shape)
            held += part[ja, jb]
        fits[name] = {"ia": int(ia), "ib": int(cols[ib]), "fit": float(T[ia, ib]), "held_out": float(held)}
    return fits


def find_steps(S: np.ndarray) -> np.ndarray:
    """Per-round extra offset (index into the residual grid), constant except where a step pays STEP_COST."""
    score, back = S[0].copy(), []
    for row in S[1:]:
        jump = score.max() - STEP_COST
        back.append(np.where(score >= jump, np.arange(len(score)), score.argmax()))
        score = np.maximum(score, jump) + row
    path = [int(score.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return np.array(path[::-1])


def alive_spans(kills: pd.DataFrame, rounds: pd.DataFrame) -> dict[int, list[tuple[float, int, int]]]:
    """Round -> [(demo time, CT alive, T alive)] from the round start, changing at each death."""
    out = {}
    for r in rounds.itertuples():
        k = kills[(kills["round"] == r.round) & ~kills.after_round_end].sort_values("game_time")
        out[r.round] = [(r.start_time, 5, 5)] + list(zip(k.game_time, k.ct_alive_after, k.t_alive_after))
    return out


def count_agreement(words, spans, rounds, demo_time) -> tuple[int, float, float]:
    """Count tags ('1v3') in the map whose numbers were the alive counts at some point from 20 s before to 5 s after."""
    tags = words[(words.kind == "count") & (words.entity != "5v5")]
    if not len(tags):
        return 0, float("nan"), float("nan")
    starts, ends = rounds.start_time.to_numpy(), rounds.end_time.to_numpy() + 7

    def rate(shift):
        ok = 0
        for t, ent in zip(tags.t, tags.entity):
            d = demo_time(t - shift)
            i = np.searchsorted(starts, d, side="right") - 1
            if i < 0 or d > ends[i]:
                continue
            x, y = (int(n) for n in ent.split("v"))
            span = spans[rounds["round"].iat[i]]
            # each state lasts until the next death; keep those overlapping [d - 20, d + 5]
            seen = [(c, tt) for j, (when, c, tt) in enumerate(span)
                    if when <= d + 5 and (j + 1 == len(span) or span[j + 1][0] >= d - 20)]
            ok += any((x, y) in ((c, tt), (tt, c)) for c, tt in seen)
        return ok / len(tags)
    return len(tags), rate(0), float(np.mean([rate(s) for s in CHANCE_SHIFTS]))


def bomb_agreement(words, bomb, vod_time) -> tuple[int, float, float]:
    """Plant and defuse events with a matching word from 3 s before to 12 s after, by VOD time."""
    keys = words.raw.map(lambda w: captions.key(captions.split_token(w)[1]))
    events = bomb[bomb.event.isin(list(BOMB_WORDS))]
    if not len(events):
        return 0, float("nan"), float("nan")
    times = {ev: np.sort(words.t[keys.str.startswith(BOMB_WORDS[ev])].to_numpy()) for ev in BOMB_WORDS}

    def rate(shift):
        ok = 0
        for ev, t in zip(events.event, events.game_time):
            v = vod_time(t) + shift
            ts = times[ev]
            i = np.searchsorted(ts, v - 3)
            ok += i < len(ts) and ts[i] <= v + 12
        return ok / len(events)
    return len(events), rate(0), float(np.mean([rate(s) for s in CHANCE_SHIFTS]))


def sync_map(spec, words: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    folder = INTERIM / spec.map_id
    rounds = pd.read_parquet(folder / "rounds.parquet").sort_values("round").reset_index(drop=True)
    all_kills = pd.read_parquet(folder / "kills.parquet")
    kills = all_kills[all_kills.kind == "kill"].reset_index(drop=True)
    ids = roster_ids(spec.map_id)
    kt = kills.game_time.to_numpy(float)
    rnd = kills["round"].to_numpy()
    mentions = kill_mentions(kills, words, ids)
    vod_end = float(words.t.max())

    peak = coarse_peak(kt, mentions, vod_end)
    tmid = float((rounds.live_time.iloc[0] + rounds.end_time.iloc[-1]) / 2)
    a = np.arange(peak["offset"] - NEAR, peak["offset"] + NEAR + STEP / 2, STEP)
    P = line_scores(kt, mentions, a, tmid)
    fits = fit_line(P, rnd)
    model = "line" if fits["line"]["held_out"] > fits["constant"]["held_out"] else "constant"
    at, slope = float(a[fits[model]["ia"]]), float(SLOPES[fits[model]["ib"]])

    # Steps on top of the line, round by round. A tiny pull toward 0 breaks ties.
    res = np.arange(-STEP_RANGE, STEP_RANGE + STEP / 2, 0.5)
    S = np.zeros((len(rounds), len(res))) - np.abs(res) * 1e-6
    row_of = {r: i for i, r in enumerate(rounds["round"])}
    for t, r, ms in zip(kt, rnd, mentions):
        if len(ms):
            S[row_of[r]] += weight(ms[None, :] - (t + at + slope * (t - tmid) + res[:, None])).max(1)
    path = find_steps(S)
    extra = res[path] if len(set(path)) > 1 else np.zeros(len(rounds))

    def offset(d, i):
        return at + slope * (d - tmid) + extra[i]

    starts = rounds.start_time.to_numpy()
    vstart = np.array([s + offset(s, i) for i, s in enumerate(starts)])
    last = len(rounds) - 1
    vend = rounds.end_time.iat[last] + offset(rounds.end_time.iat[last], last) + POST_MAP

    def round_index(d):
        return int(np.clip(np.searchsorted(starts, d, side="right") - 1, 0, last))

    def vod_time(d):
        return d + offset(d, round_index(d))

    def demo_time(v):
        i = int(np.clip(np.searchsorted(vstart, v, side="right") - 1, 0, last))
        return (v - at - extra[i] + slope * tmid) / (1 + slope)

    # Per round: the offset at the live tick and the evidence for it.
    ev = np.array([weight(ms - vod_time(t)).max() if len(ms) else 0.0 for t, ms in zip(kt, mentions)])
    per_round = []
    for i, r in enumerate(rounds.itertuples()):
        m = rnd == r.round
        per_round.append({"round": int(r.round), "part": int(r.part), "offset": round(offset(r.live_time, i), 2),
                          "step": float(extra[i]), "kills": int(m.sum()), "kills_mentioned": int((ev[m] > 0).sum()),
                          "evidence": round(float(ev[m].sum()), 2)})

    # The map's caption words: round, phase and demo time for each.
    inside = words[(words.t >= vstart[0]) & (words.t <= vend)].copy()
    d = np.array([demo_time(v) for v in inside.t])
    i = np.clip(np.searchsorted(vstart, inside.t.to_numpy(), side="right") - 1, 0, last)
    r = rounds.iloc[i]
    inside["round"] = r["round"].to_numpy()
    inside["demo_time"] = d.round(3)
    inside["seconds_into_round"] = (d - r.live_time.to_numpy()).round(3)
    inside["phase"] = np.select([d < r.live_time.to_numpy(), d < r.end_time.to_numpy()], ["freeze", "live"], "over")
    inside = inside.reset_index(drop=True)

    chance = np.mean([sum(weight(ms - vod_time(t) - s).max() if len(ms) else 0.0 for t, ms in zip(kt, mentions))
                      for s in CHANCE_SHIFTS])
    held = fits[model]["held_out"]
    n_count, count_rate, count_chance = count_agreement(inside, alive_spans(all_kills, rounds), rounds, demo_time)
    bomb = pd.read_parquet(folder / "bomb.parquet")
    n_bomb, bomb_rate, bomb_chance = bomb_agreement(words, bomb, vod_time)

    def beats(n, rate, base):
        return None if n < MIN_ANCHORS else bool(rate > 2 * base)

    checks = {
        "one clear peak": peak["kills"] >= MIN_PEAK_RATIO * peak["runner_up"],
        "held-out rounds beat chance": bool(held >= MIN_CHANCE_RATIO * chance),
        "drift inside the search range": bool(SLOPES[1] <= slope <= SLOPES[-2]) or model == "constant",
        "no steps": bool(not extra.any()),
        # from the first kill to the last round's end: some VODs start during round 1's buy or first seconds
        "map inside the VOD": bool(vod_time(kt.min()) >= words.t.min() and vend - POST_MAP <= vod_end),
        # None: too few to judge
        "count tags agree": beats(n_count, count_rate, count_chance),
        "bomb words agree": beats(n_bomb, bomb_rate, bomb_chance),
    }
    sync = {
        "map_id": spec.map_id, "vod_id": spec.vod_id,
        "model": model, "offset_at": round(at, 3), "t_mid": round(tmid, 3),
        "drift_per_1000s": round(slope * 1000, 2),
        "formula": "vod_time = game_time + offset_at + drift_per_1000s / 1000 * (game_time - t_mid) + round step",
        "vod_start": round(float(vstart[0]), 2), "vod_end": round(float(vend), 2),
        "fits": {k: {"fit": round(v["fit"], 1), "held_out": round(v["held_out"], 1)} for k, v in fits.items()},
        "evidence": {"kills": len(kills), "chance": round(float(chance), 1), "peak": peak,
                     "count_tags": {"n": n_count, "agree": round(count_rate, 3), "chance": round(count_chance, 3)},
                     "bomb_words": {"n": n_bomb, "agree": round(bomb_rate, 3), "chance": round(bomb_chance, 3)}},
        "checks": checks,
        "rounds": per_round,
    }
    return sync, inside


def main(argv: list[str]) -> int:
    maps = pilot_maps()
    by_vod = collections.defaultdict(list)
    for spec in maps.values():
        by_vod[spec.vod_id].append(spec)
    wanted = [s for a in argv for s in ([maps[a]] if a in maps else by_vod.get(a, []))] if argv else list(maps.values())
    unknown = [a for a in argv if a not in maps and a not in by_vod]
    if unknown:
        raise SystemExit(f"not a map_id or vod_id in pilot.tsv: {unknown}")

    spans = collections.defaultdict(list)
    failed = []
    for spec in wanted:
        for need, how in ((INTERIM / spec.map_id / "map_info.json", f"python -m cs2c.parse_demo {spec.map_id}"),
                          (INTERIM / "captions" / f"{spec.vod_id}.parquet", f"python -m cs2c.clean_captions {spec.vod_id}")):
            if not need.exists():
                raise SystemExit(f"{spec.map_id}: missing {need}, run {how}")
        words = pd.read_parquet(INTERIM / "captions" / f"{spec.vod_id}.parquet")
        sync, inside = sync_map(spec, words)
        spans[spec.vod_id].append((sync["vod_start"], sync["vod_end"], spec.map_id))
        (INTERIM / spec.map_id / "sync.json").write_text(json.dumps(sync, indent=1))
        inside.to_parquet(INTERIM / spec.map_id / "captions.parquet", index=False)

        e = sync["evidence"]
        bad = [k for k, v in sync["checks"].items() if v is False]
        if bad:
            failed.append(spec.map_id)
        print(f"{spec.map_id}: {sync['model']}, offset {sync['offset_at']:.1f} s, drift {sync['drift_per_1000s']:+.2f} s/1000s"
              f" | held-out {sync['fits'][sync['model']]['held_out']:.0f} vs chance {e['chance']:.0f}"
              f" | peak {e['peak']['kills']}/{e['peak']['runner_up']}"
              f" | counts {e['count_tags']['agree']:.0%} vs {e['count_tags']['chance']:.0%} (n={e['count_tags']['n']})"
              f" | bomb {e['bomb_words']['agree']:.0%} vs {e['bomb_words']['chance']:.0%} (n={e['bomb_words']['n']})"
              f" | {len(inside)} words" + (f" | FAILED: {', '.join(bad)}" if bad else ""))
        steps = [(r["round"], r["step"]) for r in sync["rounds"] if r["step"]]
        if steps:
            print(f"   steps in offset (round, seconds): {steps}")

    # Maps sharing a VOD must not overlap.
    for vod_id, s in spans.items():
        s.sort()
        for (a0, a1, ma), (b0, b1, mb) in zip(s, s[1:]):
            if b0 < a1:
                print(f"OVERLAP on {vod_id}: {ma} ends at {a1:.0f} s, {mb} starts at {b0:.0f} s")
                failed.append(mb)
    print(f"\nsynced {len(wanted)} maps" + (f", checks failed on {sorted(set(failed))}" if failed else ", all checks passed"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
