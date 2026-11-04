"""Turn each VOD's auto captions into a cleaned word table.

    python -m cs2c.clean_captions [map_id or vod_id ...]

Needs the maps parsed first (rosters come from map_info.json). Writes, under
data/interim/captions/:
  <vod_id>.parquet  one row per word (alias spans merged), see captions.clean
  fixes.tsv         every distinct change made, with counts, to review
  suggest.tsv       frequent words that matched nothing and are not English,
                    with the closest name on that VOD: add real misses to
                    data/meta/aliases.tsv and run again
New players found in the demos are added to data/meta/players.tsv. Aliases
come from data/meta/aliases.tsv (by hand) and aliases_learned.tsv (written by
cs2c.learn_aliases once the maps are synced).
"""
import collections
import csv
import json
import sys

import pandas as pd
from rapidfuzz import fuzz, process

from . import captions
from .paths import INTERIM, META, pilot_maps, players, team_names

OUT = INTERIM / "captions"
LEARNED = META / "aliases_learned.tsv"
SUGGEST_MIN_COUNT = 3
SUGGEST_MIN_SCORE = 60


def vod_rosters(specs) -> dict[str, dict[str, str]]:
    """Team -> {steam id: in-game name} over every map on the VOD."""
    roster: dict[str, dict[str, str]] = {}
    for spec in specs:
        info_file = INTERIM / spec.map_id / "map_info.json"
        if not info_file.exists():
            raise SystemExit(f"{spec.map_id}: not parsed yet, run python -m cs2c.parse_demo {spec.map_id}")
        for team, ps in json.loads(info_file.read_text())["rosters"].items():
            for p in ps:
                roster.setdefault(team, {})[p["steamid"]] = p["name"].strip()
    return roster


def add_new_players(rosters: list[dict[str, dict[str, str]]]) -> dict[str, str]:
    """Append players not yet in players.tsv, named as most demos write them. Returns the full table."""
    known = players()
    seen = collections.defaultdict(collections.Counter)
    for roster in rosters:
        for ps in roster.values():
            for sid, name in ps.items():
                seen[sid][name] += 1
    new = {sid: c.most_common(1)[0][0] for sid, c in seen.items() if sid not in known}
    if new:
        path = META / "players.tsv"
        exists = path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            if not exists:
                w.writerow(["steamid", "name"])
            w.writerows(sorted(new.items(), key=lambda x: x[1].lower()))
        print(f"added {len(new)} players to players.tsv: {', '.join(sorted(new.values(), key=str.lower))}")
    return {**known, **new}


def suggest(vod_id: str, table: pd.DataFrame, vocab: captions.Vocab, names: list[str]) -> list[dict]:
    """Untagged non-English words said at least SUGGEST_MIN_COUNT times, with the closest name on the VOD."""
    english = vocab.english or set()
    cores = table.loc[(table.kind == "") & ~table.sound, "raw"].map(lambda w: captions.split_token(w)[1])
    counts = collections.Counter(c for c in cores if len(captions.key(c)) >= 3 and c.lower() not in english
                                 and captions.key(c) not in english and not captions.key(c).isdigit())
    keyed = {captions.key(n).translate(captions.LEET): n for n in names}
    rows = []
    for word, n in counts.most_common():
        if n < SUGGEST_MIN_COUNT:
            break
        best = process.extractOne(captions.key(word), list(keyed), scorer=fuzz.ratio)
        close = best and best[1] >= SUGGEST_MIN_SCORE
        rows.append({"vod_id": vod_id, "word": word, "count": n,
                     "closest": keyed[best[0]] if close else "", "score": round(best[1]) if close else ""})
    return rows


def main(argv: list[str]) -> int:
    maps = pilot_maps()
    by_vod = collections.defaultdict(list)
    for spec in maps.values():
        by_vod[spec.vod_id].append(spec)
    wanted = [maps[a].vod_id if a in maps else a for a in argv] or list(by_vod)
    unknown = [v for v in wanted if v not in by_vod]
    if unknown:
        raise SystemExit(f"not a map_id or vod_id in pilot.tsv: {unknown}")
    wanted = list(dict.fromkeys(wanted))

    rosters = {v: vod_rosters(by_vod[v]) for v in wanted}
    known = add_new_players(list(rosters.values()))
    teams, aliases, english = team_names(), captions.read_aliases(), captions.english_words()
    if LEARNED.exists():  # misheard names proven by demo timing, see learn_aliases
        aliases += captions.read_aliases(LEARNED)
    if english is None:
        print(f"no word list at {captions.DICT_FILE}: every name alias needs a mid-sentence capital")
    OUT.mkdir(parents=True, exist_ok=True)
    fixes, suggestions = collections.Counter(), []
    for vod_id in wanted:
        roster = {team: list(ps) for team, ps in rosters[vod_id].items()}
        vocab = captions.Vocab.build(aliases, known, teams, roster, english)
        words = captions.read_words(by_vod[vod_id][0].caption_file)
        table = captions.clean(words, vocab)
        table.to_parquet(OUT / f"{vod_id}.parquet", index=False)

        tagged = table[table.kind != ""]
        changed = table[table.text != table.raw]
        for r in changed.itertuples():
            fixes[(vod_id, r.kind, r.entity, captions.split_token(r.raw)[1], captions.split_token(r.text)[1])] += 1
        names = [known[sid] for ps in roster.values() for sid in ps] + list(roster)
        suggestions += suggest(vod_id, table, vocab, names)
        maps_on = ", ".join(s.map_id for s in by_vod[vod_id])
        print(f"{vod_id} ({maps_on}): {len(words)} words, {len(changed)} fixed, "
              f"{tagged.sure.sum()} tagged {tagged[tagged.sure].kind.value_counts().to_dict()}, "
              f"{(~tagged.sure).sum()} unsure")

    with open(OUT / "fixes.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["vod_id", "kind", "entity", "caption", "fixed", "count"])
        for (vod_id, kind, entity, raw, text), n in sorted(fixes.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], -x[1])):
            w.writerow([vod_id, kind, entity, raw, text, n])
    pd.DataFrame(suggestions, columns=["vod_id", "word", "count", "closest", "score"]).to_csv(
        OUT / "suggest.tsv", sep="\t", index=False)
    print(f"\nwrote {len(wanted)} tables, {OUT / 'fixes.tsv'} ({len(fixes)} distinct fixes), "
          f"{OUT / 'suggest.tsv'} ({len(suggestions)} words to review)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
