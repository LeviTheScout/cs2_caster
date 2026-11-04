"""Split each map's caption words into utterances: short units, about one remark each.

    python -m cs2c.utterances [map_id or vod_id ...]

A new utterance starts after a sentence end (. ? !), at a speaker change (>>),
after a silence of LONG_PAUSE seconds, or after a pause of PAUSE seconds when
the next word starts a sentence. Casters pause mid-sentence ("I took my
chance ... and I had none."), and captions write a continuation in lowercase;
a name tag or a number after a pause counts as a start. One longer than MAX_WORDS is split at the
comma nearest its middle, or else at its longest pause. Units are never
merged here: the matcher (Step 6) groups consecutive ones with the demo's help.

Sound tags ("[music]", "[ __ ]" for a censored word, "[screaming]") stay inside
the utterance they fall in, since they often sit mid-sentence. They count as no
words and are listed in `sounds`.

Only facts are recorded, nothing is judged: times on both clocks, round and
phase, the cleaned and the raw text, the tags inside, and delivery signals
(words per second, share of shouted words) for intensity later.

Needs the maps synced (cs2c.sync). Writes data/interim/<map_id>/utterances.parquet.
"""
import collections
import sys

import numpy as np
import pandas as pd

from . import captions
from .paths import INTERIM, pilot_maps

PAUSE = 1.5        # seconds between word starts that end an utterance, unless the next word is lowercase
LONG_PAUSE = 4.0   # seconds that always end one
MAX_WORDS = 30
MIN_PART = 8       # words each side of a split of a long utterance
LAST_WORD = 0.6    # seconds a final word is assumed to last before a pause
TAG_KINDS = ["player", "team", "weapon", "utility", "side", "count", "map"]


def boundaries(words: pd.DataFrame) -> list[tuple[int, int]]:
    """[start, end) row ranges of the utterances."""
    t, raw, speaker = words.t.to_numpy(), words.raw.tolist(), words.speaker.to_numpy()
    sound, kind = words.sound.to_numpy(), words.kind.tolist()

    def starts_new(i):
        if i == len(words) or speaker[i] or (not sound[i - 1] and raw[i - 1].endswith(captions.SENTENCE_END)):
            return True
        gap = t[i] - t[i - 1]
        lowercase = not kind[i] and not sound[i] and captions.split_token(raw[i])[1][:1].islower()
        return gap >= LONG_PAUSE or (gap >= PAUSE and not lowercase)

    out, start = [], 0
    for i in range(1, len(words) + 1):
        if starts_new(i):
            out += split_long(words, start, i)
            start = i
    return out


def split_long(words: pd.DataFrame, a: int, b: int) -> list[tuple[int, int]]:
    """Split [a, b) at the comma nearest its middle, else at its longest pause, until no part is over MAX_WORDS."""
    n = word_counts(words.iloc[a:b])
    if n.sum() <= MAX_WORDS:
        return [(a, b)]
    before = np.cumsum(n)  # words up to and including each row
    cuts = [i for i in range(a + 1, b) if before[i - a - 1] >= MIN_PART and n.sum() - before[i - a - 1] >= MIN_PART]
    if not cuts:
        return [(a, b)]
    mid = n.sum() / 2
    commas = [i for i in cuts if words.raw.iat[i - 1].endswith(",")]
    if commas:
        cut = min(commas, key=lambda i: abs(before[i - a - 1] - mid))
    else:
        cut = max(cuts, key=lambda i: words.t.iat[i] - words.t.iat[i - 1])
    return split_long(words, a, cut) + split_long(words, cut, b)


def word_counts(words: pd.DataFrame) -> np.ndarray:
    """Spoken words per row: an alias span ("heavy got") is several, a sound tag none."""
    return np.where(words.sound, 0, words.raw.str.split().str.len()).astype(int)


def shouted(raw: str, kind: str) -> bool:
    core = captions.split_token(raw)[1]
    return not kind and len(core) > 1 and core.isalpha() and core.isupper()


def build(words: pd.DataFrame) -> pd.DataFrame:
    words = words.reset_index(drop=True)
    n_words = word_counts(words)
    t = words.t.to_numpy()
    rows = []
    for a, b in boundaries(words):
        w = words.iloc[a:b]
        last = b - 1
        end = min(t[b], t[last] + LAST_WORD) if b < len(words) else t[last] + LAST_WORD
        n = int(n_words[a:b].sum())
        spoken = w[~w.sound]
        row = {
            "utt": len(rows), "round": int(w["round"].iat[0]), "phase": w.phase.iat[0], "phase_end": w.phase.iat[-1],
            "round_end": int(w["round"].iat[-1]),
            "t_start": float(t[a]), "t_end": float(end),
            "demo_start": float(w.demo_time.iat[0]), "demo_end": float(w.demo_time.iat[-1] + end - t[last]),
            "seconds_into_round": float(w.seconds_into_round.iat[0]),
            "text": " ".join(w.text), "raw": " ".join(w.raw),
            "n_words": n, "words_per_s": round(n / max(end - t[a], 0.5), 2),
            "shouted": round(sum(shouted(r, k) for r, k in zip(spoken.raw, spoken.kind)) / max(len(spoken), 1), 3),
            "speaker_change": bool(w.speaker.iat[0]),
            "sounds": [r for r in w.raw[w.sound]],
            "first_row": a, "rows": b - a,
        }
        sure = w[w.sure]
        for kind in TAG_KINDS:
            row[kind + "s"] = sure.entity[sure.kind == kind].tolist()
        row["player_ids"] = sure.steamid[sure.kind == "player"].tolist()
        row["unsure"] = w.entity[(w.kind != "") & ~w.sure].tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def check(words: pd.DataFrame, utt: pd.DataFrame) -> dict:
    """Every row in exactly one utterance; no speaker change or long silence inside one."""
    covered = np.zeros(len(words), int)
    inside_break = 0
    t, speaker = words.t.to_numpy(), words.speaker.to_numpy()
    for a, n in zip(utt.first_row, utt.rows):
        covered[a:a + n] += 1
        inside_break += int(speaker[a + 1:a + n].any() or (np.diff(t[a:a + n]) >= LONG_PAUSE).any())
    return {"every row once": bool((covered == 1).all()), "no break inside": inside_break == 0}


def main(argv: list[str]) -> int:
    maps = pilot_maps()
    by_vod = collections.defaultdict(list)
    for spec in maps.values():
        by_vod[spec.vod_id].append(spec.map_id)
    wanted = [m for a in argv for m in ([a] if a in maps else by_vod.get(a, []))] if argv else list(maps)
    unknown = [a for a in argv if a not in maps and a not in by_vod]
    if unknown:
        raise SystemExit(f"not a map_id or vod_id in pilot.tsv: {unknown}")
    failed = []
    for map_id in wanted:
        src = INTERIM / map_id / "captions.parquet"
        if not src.exists():
            raise SystemExit(f"{map_id}: not synced yet, run python -m cs2c.sync {map_id}")
        words = pd.read_parquet(src)
        utt = build(words)
        checks = check(words, utt)
        utt.to_parquet(INTERIM / map_id / "utterances.parquet", index=False)
        n = utt.n_words
        bad = [k for k, v in checks.items() if not v]
        failed += [map_id] if bad else []
        print(f"{map_id}: {len(utt)} utterances | words median {n.median():.0f}, 95th {n.quantile(.95):.0f}, "
              f"max {n.max()} | 1-2 words {(n <= 2).mean():.0%} | with a player tag {(utt.players.str.len() > 0).mean():.0%} | "
              f"phases {utt.phase.value_counts().to_dict()}" + (f" | FAILED: {', '.join(bad)}" if bad else ""))
    print(f"\nsplit {len(wanted)} maps" + (f", checks failed on {failed}" if failed else ", all checks passed"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
