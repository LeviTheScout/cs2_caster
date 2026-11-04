"""Learn misheard player names from the demos.

    python -m cs2c.learn_aliases [--check]

A misheard name behaves like a name: it is said just after that player's
kills and deaths far more often than chance. For every word or two-word phrase
the corrector left untagged, this counts how often it lands 3 s before to 11 s
after each player's events (using the clock from cs2c.sync), against the same
count with the words moved to wrong times. Evidence is pooled over every map
the player is on.

- Accepted: strong timing evidence, one clear player, and a spelling at least
  loosely like the name ("Casper sky" for xKacpersky). Written to
  data/meta/aliases_learned.tsv, which clean_captions reads with aliases.tsv.
- Review: strong timing but a different spelling, which may be a nickname or a
  word tied to a team ("Brazilian"). Written to
  data/interim/captions/aliases_review.tsv. Copy real ones into aliases.tsv;
  add others there with kind "english" so they are never names.

Captions are re-cleaned here with aliases.tsv only, so a word learned on an
earlier run is judged again from the evidence and not from its own tag.

--check hides the hand-written player aliases and reports how many of them
this finds, and whether any goes to the wrong player. Run it after changing a
threshold.

Needs clean_captions and sync run first. Rerun both after this.
"""
import collections
import csv
import json
import math
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import captions
from .clean_captions import LEARNED
from .paths import INTERIM, pilot_maps, players, team_names
from .sync import CHANCE_SHIFTS, DELAY_S

REVIEW = INTERIM / "captions" / "aliases_review.tsv"
MIN_SAID = 3          # times said on maps where the player plays
MIN_RATIO = 3         # hits next to the player's events over hits at chance
MAX_RIVAL = 0.5       # the next player may have at most this share of the best player's hits
# To accept: (spelling similarity 0-100 at least, chance of that many hits at most, hits at least).
# The closer the spelling, the less timing proof it needs.
ACCEPT = [(70, 0.05, 2), (50, 0.01, 3)]
MIN_SIMILAR_ENGLISH = 75  # an English word heard capitalised ("Fallen") must also be spelt this close
REVIEW_P = 1e-3           # strong timing alone sends a word to review
REVIEW_HITS = 3


def poisson_tail(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    term = total = math.exp(-lam)
    for i in range(1, k):
        term *= lam / i
        total += term
    return max(0.0, 1.0 - total)


def player_events(map_id: str, ids: dict[str, str]) -> dict[str, np.ndarray]:
    """Steam id -> sorted demo times of the player's kills, deaths, plants and defuses."""
    k = pd.read_parquet(INTERIM / map_id / "kills.parquet")
    b = pd.read_parquet(INTERIM / map_id / "bomb.parquet")
    b = b[b.event.isin(["planted", "defused"])]
    rows = [(ids.get(n), t) for n, t in zip(k.attacker[k.kind == "kill"], k.game_time[k.kind == "kill"])]
    rows += [(ids.get(n), t) for n, t in zip(k.victim, k.game_time)]
    rows += [(ids.get(n), t) for n, t in zip(b.player, b.game_time)]
    out = collections.defaultdict(list)
    for sid, t in rows:
        if sid:
            out[sid].append(t)
    return {sid: np.sort(ts) for sid, ts in out.items()}


def hits(said: np.ndarray, events: np.ndarray) -> int:
    """Times said from 3 s before to 11 s after one of the events."""
    i = np.searchsorted(events, said - DELAY_S[-1])
    return int(np.sum((i < len(events)) & (events[np.minimum(i, len(events) - 1)] <= said - DELAY_S[0])))


def candidates(table: pd.DataFrame, english: set[str]) -> list[tuple[tuple[str, ...], bool, int]]:
    """(keys, is an English word, row) for each untagged word or two-word phrase that could be a name.

    A word counts if it is not English, or is an English word capitalised mid-sentence.
    A phrase counts if one of its words is not English and nothing separates them.
    """
    raws, kinds, sound, speaker = table.raw.tolist(), table.kind.tolist(), table.sound.tolist(), table.speaker.tolist()
    parts = [captions.split_token(w) for w in raws]
    out = []
    for i, (_, core, post) in enumerate(parts):
        if kinds[i] or sound[i] or not core:
            continue
        k = captions.key(core)
        if len(k) < 3 or k.isdigit():
            continue
        is_english = core.lower() in english or k in english
        start = i == 0 or speaker[i] or sound[i - 1] or raws[i - 1].endswith(captions.SENTENCE_END)
        if not is_english:
            out.append(((k,), False, i))
        elif core[0].isupper() and not start:
            out.append(((k,), True, i))
        j = i + 1
        if j < len(raws) and not kinds[j] and not sound[j] and not speaker[j] and post in ("", "-") and parts[j][1]:
            k2 = captions.key(parts[j][1])
            if k2 and not k2.isdigit() and (not is_english or (parts[j][1].lower() not in english and k2 not in english)):
                out.append(((k, k2), False, i))
    return out


def similarity(keys: tuple[str, ...], name: str) -> float:
    said = "".join(keys).translate(captions.LEET)
    return max(fuzz.ratio(said, "".join(f)) for f in captions.name_forms(name))


def gather(aliases: list[dict], english: set[str]):
    """Re-clean each map's captions with `aliases`, then collect candidate sightings and player events."""
    known, teams = players(), team_names()
    seen = collections.defaultdict(list)  # keys -> [(map_id, demo time, is English, snippet)]
    events, on_map = {}, {}
    for spec in pilot_maps().values():
        synced_file = INTERIM / spec.map_id / "captions.parquet"
        if not synced_file.exists():
            raise SystemExit(f"{spec.map_id}: not synced yet, run python -m cs2c.sync {spec.map_id}")
        synced = pd.read_parquet(synced_file)
        info = json.loads((INTERIM / spec.map_id / "map_info.json").read_text())
        ids = {p["name"]: p["steamid"] for ps in info["rosters"].values() for p in ps}
        roster = {team: [p["steamid"] for p in ps] for team, ps in info["rosters"].items()}
        vocab = captions.Vocab.build(aliases, known, teams, roster, english)
        words = captions.read_words(spec.caption_file)
        words = words[(words.t >= synced.t.min()) & (words.t <= synced.t.max())].reset_index(drop=True)
        table = captions.clean(words, vocab)
        demo_time = np.interp(table.t, synced.t, synced.demo_time)
        texts = table.text.tolist()
        for keys, is_english, i in candidates(table, vocab.english or set()):
            seen[keys].append((spec.map_id, demo_time[i], is_english, " ".join(texts[max(0, i - 8):i + 9])))
        events[spec.map_id] = player_events(spec.map_id, ids)
        on_map[spec.map_id] = set(ids.values())
    return seen, events, on_map


def judge(seen, events, on_map, names: dict[str, str]) -> list[dict]:
    """Timing evidence for every candidate said MIN_SAID times, and a decision."""
    rows = []
    for keys, sightings in seen.items():
        # An English word only counts where it was heard capitalised mid-sentence; elsewhere it is the word.
        is_english = all(s[2] for s in sightings)
        if len(sightings) < MIN_SAID:
            continue
        by_map = collections.defaultdict(list)
        for map_id, t, _, _ in sightings:
            by_map[map_id].append(t)
        scores = []
        for sid in {s for m in by_map for s in on_map[m]}:
            said = h = 0
            lam = 0.0
            for map_id, ts in by_map.items():
                if sid not in on_map[map_id]:
                    continue
                ts = np.array(ts)
                ev = events[map_id].get(sid, np.array([]))
                said += len(ts)
                if len(ev):
                    h += hits(ts, ev)
                    lam += float(np.mean([hits(ts + s, ev) for s in CHANCE_SHIFTS]))
            scores.append((h, lam, said, sid))
        scores.sort(key=lambda x: (-x[0], x[1]))
        (h, lam, said, sid), rival = scores[0], (scores[1][0] if len(scores) > 1 else 0)
        p = poisson_tail(h, lam)
        ratio = h / lam if lam else float("inf")
        if said < MIN_SAID or ratio < MIN_RATIO:
            continue
        sim = similarity(keys, names[sid])
        accept = (rival <= MAX_RIVAL * h and (not is_english or sim >= MIN_SIMILAR_ENGLISH)
                  and any(sim >= s and p <= mp and h >= mh for s, mp, mh in ACCEPT))
        if accept:
            decision = "accept"
        elif p <= REVIEW_P and h >= REVIEW_HITS:
            decision = "review"
        else:
            continue
        example = next(s[3] for s in sightings if s[0] in by_map and sid in on_map[s[0]])
        rows.append({"decision": decision, "entity": names[sid], "alias": " ".join(keys), "said": said, "hits": h,
                     "chance": round(lam, 2), "ratio": round(ratio, 1), "p": f"{p:.1e}", "rival_hits": rival,
                     "similarity": round(sim), "english": is_english, "steamid": sid, "example": example})
    # A phrase must be spelt closer to the name than either word alone: "casper sky" is xKacpersky,
    # "and carrian" is "carrian" plus a filler word.
    rows = [r for r in rows if " " not in r["alias"]
            or all(r["similarity"] > similarity((w,), r["entity"]) for w in r["alias"].split())]
    return sorted(rows, key=lambda r: (r["decision"], r["entity"].lower(), -r["hits"]))


def check(aliases, english, names):
    """Hide the hand-written player aliases and see how many this finds, and for whom."""
    hidden = [a for a in aliases if a["kind"] != "player"]
    truth = {tuple(captions.key(w) for w in a["alias"].split()): a["entity"] for a in aliases if a["kind"] == "player"}
    seen, events, on_map = gather(hidden, english)
    rows = judge(seen, events, on_map, names)
    found = {tuple(r["alias"].split()): r for r in rows}
    # Only aliases that come back as candidates often enough can be found at all.
    testable = {k: e for k, e in truth.items() if len(seen.get(k, [])) >= MIN_SAID}
    right = [k for k, e in testable.items() if k in found and found[k]["entity"] == e]
    wrong = [(k, e, found[k]["entity"], found[k]["decision"]) for k, e in testable.items()
             if k in found and found[k]["entity"] != e]
    acc = [k for k in right if found[k]["decision"] == "accept"]
    print(f"hand aliases: {len(truth)}, said often enough to test: {len(testable)}; found with the right player: "
          f"{len(right)} ({len(acc)} accepted, {len(right) - len(acc)} to review); with a wrong player: {len(wrong)}")
    missed = [k for k in testable if k not in found]
    print(f"  missed: {', '.join(' '.join(k) for k in missed)}")
    for k, e, got, dec in wrong:
        print(f"  WRONG {' '.join(k)!r}: hand says {e}, found {got} ({dec})")
    unknown = [r for r in rows if tuple(r["alias"].split()) not in truth]
    print(f"other candidates found: {sum(r['decision'] == 'accept' for r in unknown)} accepted, "
          f"{sum(r['decision'] == 'review' for r in unknown)} to review")
    return rows, truth


def main(argv: list[str]) -> int:
    aliases, english = captions.read_aliases(), captions.english_words() or set()
    english = english | {captions.key(a["alias"]) for a in aliases if a["kind"] == "english"}
    names = players()
    if "--check" in argv:
        check(aliases, english, names)
        return 0
    rows = judge(*gather(aliases, english), names)
    accepted = [r for r in rows if r["decision"] == "accept"]
    with open(LEARNED, "w", newline="") as f:
        f.write("# Written by python -m cs2c.learn_aliases: misheard names proven by demo timing. Do not edit;\n"
                "# move a wrong one into aliases.tsv with kind english, or a right one there to keep it by hand.\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["kind", "entity", "alias", "write", "said", "hits", "chance", "similarity"])
        for r in accepted:
            w.writerow(["player", r["entity"], r["alias"], "", r["said"], r["hits"], r["chance"], r["similarity"]])
    review = [r for r in rows if r["decision"] == "review"]
    pd.DataFrame(review, columns=list(rows[0]) if rows else []).drop(columns=["decision"], errors="ignore").to_csv(
        REVIEW, sep="\t", index=False)
    print(f"{len(accepted)} aliases learned -> {LEARNED}\n{len(review)} to review -> {REVIEW}")
    for r in accepted:
        print(f"  {r['entity']:12s} {r['alias']!r:22s} said {r['said']:3d}, {r['hits']:3d} by events vs "
              f"{r['chance']:5.1f} by chance, spelling {r['similarity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
