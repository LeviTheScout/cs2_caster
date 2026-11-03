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

## `meta/` (tracked)

- **`pilot.tsv`:** the 20 pilot maps (2 events, both caster crews, all 7 maps in the pool, every stage) with demo files, VOD and caption file.
- **`team_names.tsv`:** team name as each organiser writes it in demos, mapped to one standard name. The parser stops if a name is missing.
- **`cologne_vods.json`:** Cologne series matched to ESL Archives VODs, matched by script, with one known wrong match.
- **`cac2026_vods.tsv`:** all 30 CS Asia Championships 2026 series matched to PGL VOD's single-match VODs.
- **`align_prototype/<map_id>.align.json`:** demo-to-VOD clock offsets from the prototype aligner, with VOD time = demo `game_time` + `off`. The `offs` field holds per-round offsets, keyed by round minus 1. These are temporary until Step 3 replaces them.
