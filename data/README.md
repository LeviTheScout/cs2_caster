# Data

Raw downloads and generated tables live here and are not tracked by git, except `meta/`, which holds small files kept by hand or produced once.

## `raw/` (not tracked)

- **`raw/demos/`:** demo files from the Hugging Face dataset `cs2povarchive/cs2-demos`, at the archive's own paths.
- **`raw/captions/`:** YouTube auto captions (`en-orig`, json3, with a time for each word), named `<vod_id>.en-orig.json3`.
- **`raw/demos/.cache/`:** Hugging Face download metadata. Safe to delete.

## `interim/` (not tracked, rebuilt by `python -m cs2c.parse_demo`)

One folder per map, `interim/<map_id>/`:

| File | One row per |
|---|---|
| `map_info.json` | the map: teams, rosters, which events the demo has, dropped deaths, check results |
| `rounds.parquet` | round: timings, winner, reason, score, sides, timeout |
| `kills.parquet` | death: kind, players, teams, sides, positions, weapon, flags, killer HP, victim weapon, blind info, players alive |
| `bomb.parquet` | bomb event: plant or defuse start, plant, defuse, explode, drop, pickup; site, kit, players alive |
| `damage.parquet` | hit: real damage (HP lost), raw damage, HP left |
| `utility.parquet` | grenade: thrown from, landed at, went off or not, duration, damage, blinds |
| `blinds.parquet` | player blinded (only demos that record it) |
| `player_rounds.parquet` | player per round: money, buy, equipment, inventory, stats, survived, MVP |
| `states.parquet` | player per second of live play: HP, armour, money, weapon, bomb carrier, position (`filled` marks seconds carried over a recording gap) |
| `pickups.parquet` | weapon or kit picked up during a round |
| `timeline.parquet` | every event of the map in time order with a short summary; `in_hltv` marks events HLTV's live log sends |
| `sync.json` | demo-to-VOD clock (`python -m cs2c.sync`): offset model, per-round offsets and evidence, checks |
| `captions.parquet` | caption word inside the map: cleaned text and tags, round, phase (`freeze`, `live`, `over`), demo time |
| `utterances.parquet` | utterance (about one remark, `python -m cs2c.utterances`): times on both clocks, round and phase, cleaned and raw text, the tags inside, words per second, share of shouted words |

`interim/captions/` holds the whole-VOD caption tables from `python -m cs2c.clean_captions`: `<vod_id>.parquet` (one row per word), `fixes.tsv` (every change made) and `suggest.tsv` (frequent unmatched words to review).

## `meta/` (tracked)

- **`pilot.tsv`:** the 20 pilot maps (2 events, both caster crews, all 7 maps in the pool, every stage) with demo files, VOD and caption file.
- **`team_names.tsv`:** team name as each organiser writes it in demos, mapped to one standard name. The parser stops if a name is missing.
- **`cologne_vods.json`:** Cologne series matched to ESL Archives VODs, matched by script, with one known wrong match.
- **`cac2026_vods.tsv`:** all 30 CS Asia Championships 2026 series matched to PGL VOD's single-match VODs.
- **`players.tsv`:** steam id to the player's name as we write it (in-game names vary between demos). New players are added by `clean_captions`.
- **`aliases.tsv`:** hand-checked misheard forms of names and game terms, used by `clean_captions`.
- **`aliases_learned.tsv`:** misheard player names proven by demo timing, written by `python -m cs2c.learn_aliases`, with the evidence for each. Don't edit it; move a wrong one into `aliases.tsv` with kind `english`. Its review list is `interim/captions/aliases_review.tsv`.
- **`align_prototype/<map_id>.align.json`:** offsets from the old prototype aligner. **Superseded** by `interim/<map_id>/sync.json`; its per-round offsets were noise.
