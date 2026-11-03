# Data

Raw downloads go here. Nothing in this folder is edited by hand except `meta/`.

- `raw/demos/`: demo files from the Hugging Face dataset `cs2povarchive/cs2-demos`. Paths match the archive, and checksums were verified on 2026-11-03.
- `raw/captions/`: YouTube auto captions (`en-orig`, json3, with a time for each word).
- `meta/pilot.tsv`: the 5 pilot maps, covering 2 events, 2 caster crews, Bo1 and finals.
- `meta/cologne_vods.json`: Cologne series mapped to ESL Archives VODs (matched by script; one known false match).
- `meta/cac2026_vods.tsv`: all 30 CS Asia Championships 2026 series mapped to PGL VOD's single-match VODs.
- `meta/align_prototype/`: the prototype aligner's offsets for the 2 CAC pilot maps. Its convention is VOD time = demo `game_time` + `off`.
- `raw/demos/.cache/`: Hugging Face download metadata. Safe to delete.
