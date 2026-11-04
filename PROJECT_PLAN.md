# CS2 Live AI Commentary: Project Plan and Findings

Last updated: 4 November 2026

This file records the project goal, the decisions made so far, what was tested and found, what is still unknown, and what to do next. The starting point was a Gemini conversation saved as `blank.pdf`. Many of its claims were checked and some turned out to be wrong (see section 11).

**How to read the evidence labels**

| Label | Meaning |
|---|---|
| **Tested** | We ran it on real files or data in this project |
| **Reported** | It comes from community documentation or third-party sources, and we have not confirmed it ourselves |
| **Estimate** | Our planning judgement, not a measured result |
| **Unknown** | Still needs to be found out |

---

## 1. Goal

Generate live commentary for pro CS2 matches using a fine-tuned LLM:

- **During rounds:** play-by-play lines on kills, clutches and objectives.
- **During the buy phase:** analysis of the last round, standout stats, economy and unusual buys.
- **Intensity:** lines should sound calm or excited depending on how important or unlikely the moment is.
- **Live input:** the HLTV Scorebot feed.
- **Training data:** past pro match demos paired with real caster commentary from YouTube VODs.

**Preference that shapes the design:** no hand-written if/else rules that score how important or exciting a play is. Code may calculate plain facts, but the model decides what matters and how excited to sound.

**Product (decided 2026-11-03): a written live ticker, like Cricbuzz's ball-by-ball text or OneFootball's live ticker. It is not spoken-style casting.**
- **Driven by the timeline:**
  - Entries follow the match's event timeline (`timeline.parquet`).
  - Every kill is reported. Kills in a rush are reported together in one entry.
- **Entry types:**
  - kills, and groups of kills
  - bomb plant, defuse and explosion
  - round-end summary with the score
  - buy-phase preview: economy and stakes
  - occasional state updates, e.g. a clutch setup with HP and time left
- **Style is learned from the casters, never written by hand.** The training targets are the casters' own words for each moment, so the model learns from real casting:
  - phrasing ("swings in for one", "quick on the trade")
  - what matters
  - how much to stress it
- **No hand-written style examples or style rules.**
- **Cleanup only, no rewriting of style:**
  - fix misheard names using the rosters
  - remove caption marks (`>>`) and fillers
  - cut the parts of an utterance that are off-topic or rely on facts the input lacks
- **Emphasis on the page:** exclamation marks and capitals come from the intensity of the casters' reaction, which is learned, not added by rules.
- **Two layers, as in Cricbuzz:**
  - Every event gets a plain factual report.
  - Moments the casters reacted to carry the casters' words and emphasis.

---

## 2. Summary of where things stand

| Question | Answer | Evidence |
|---|---|---|
| Can past pro demos be parsed? | Yes | Tested on 6 pro demo files |
| Is there a good source of demos? | Yes: `cs2povarchive/cs2-demos` (Cologne Major 2026) or HLTV | Archive inventoried in full; 4 files parsed |
| Do demos contain what the live feed sends? | Yes, and more | Tested against the reported feed format |
| Can demos be converted to the feed's format? | Structure yes; some values can't be identical, which shouldn't affect training | Tested on 4 files; no real feed capture yet |
| Do VODs exist for the demos? | 105 of 106 Cologne series have a single-match VOD | Matched by title |
| Do kills line up with commentary? | Yes. Offset plus a small drift on ESL VODs; every pilot map synced and checked | Tested on all 20 pilot maps (Step 2) |
| Captions or Whisper for mapping? | Captions | Tested on one map |
| Is strict one-to-one event mapping possible? | No; use short event groups | Tested on one map |
| Can intensity be measured automatically? | **Not yet.** Simple signals are too weak | Tested |
| Is there a CS2 model worth building on? | No; start from a plain base model | Searched |
| Has the real live feed been captured? | **No. This is the biggest remaining unknown** | Not done |

---

## 3. Architecture

```
TRAINING
  Pro demo (.dem) ──> convert to shared format ──┐
                                                 ├─> align ──> training pairs ──> fine-tune one model
  VOD captions + audio ──> clean, correct names ─┘
                                                     (event group + intensity level + commentary)

LIVE
  HLTV feed ──> convert to shared format ──> state tracker ──> prompt ──> model
                                                                            │
          viewer <── stream delay <── fact checker <── speaking scheduler <─┘
```

**Decisions**

- **One model, trained once.** Two chained models would add delay and pass errors along. Gemini was right about this.
- **Shared input format.** Demos and the live feed go through the same code, so the model sees the same text in training and live use.
- **Code calculates facts, the model judges.** Man count, round kills, money changes and streaks are computed in code because LLMs are unreliable at arithmetic. Importance and tone are left to the model.
- **No `<analysis>` reasoning step during live use.** It delays the first word, and real transcripts contain no analysis to train on.
- **Two modes, one model:** play-by-play during rounds and analysis during the buy phase, using different prompt layouts.
- **Default output is text first.** Voice is still your decision (see section 10).

---

## 4. Live feed: HLTV Scorebot

**Status:** Reported only. Sources are the [`woq/hltv-bot` docs](https://github.com/woq/hltv-bot/blob/HEAD/docs/scorebot-data.md) (updated September 2026, most recent), the [`gigobyte/HLTV` client types](https://github.com/gigobyte/HLTV/blob/master/src/endpoints/connectToScorebot.ts) and the [`hltv-python` docs](https://github.com/bombsimon/hltv-python/blob/HEAD/DOCUMENTATION.md). HLTV publishes no schema.

### What it reportedly sends

The feed has two streams: `scoreboard` (full current state) and `log` (individual events).

**`scoreboard`**

| Level | Fields |
|---|---|
| Match and map | `mapName`, `currentRound`, `counterTerroristScore`, `terroristScore`, `ctTeamName`, `terroristTeamName`, `ctTeamId`, `tTeamId`, `startingCt`, `startingT`, `live`, `frozen`, `bombPlanted` |
| Round history | `ctMatchHistory` and `terroristMatchHistory`: winner, win type and survivors for each round |
| Player state | `alive`, `hp`, `money`, `primaryWeapon`, `kevlar`, `helmet`, `hasDefuseKit` |
| Player stats | `score` (kills), `deaths`, `assists`, `damagePrRound` (ADR), `advancedStats`: `kast`, `entryKills`, `entryDeaths`, `multiKillRounds`, `oneOnXWins`, `flashAssists` |
| Identity | `nick`, `name`, `steamId`, `dbId` |

**`log` events**

| Event | Fields |
|---|---|
| `Kill` | Killer, victim, both sides, `weapon`, `headShot`, `eventId`, `killerX/Y`, `victimX/Y`, `killerId`, `victimId`, `flasherNick`, `flasherSide` |
| `Assist` | Assister, victim, `killEventId` |
| `BombPlanted` | Planter, `ctPlayers`, `tPlayers`, `bombSite` (sometimes missing) |
| `BombDefused` | Defuser |
| `RoundStart`, `RoundEnd` | `RoundEnd` has `winner`, `winType` and the score |
| Other | `Suicide`, `PlayerJoin`, `PlayerQuit`, `MatchStarted`, `MatchOver`, `Restart`, `Reconnect`, `Disconnect` |

### What it reportedly does not send

- Smoke, wallbang, no-scope, blind and distance flags on kills
- Continuous player positions and aim direction
- Grenades, grenade paths and held utility
- Secondary weapon, exact armour value and equipment value
- Round clock and bomb timer
- Individual hits (HP only changes on a scoreboard update, reportedly every few seconds)
- A guaranteed series score (`ctTeamScore` is often just the map score)

### Derivable by your code

Man count, round kills (2K, 3K, ace), first kill, trades, win and loss streaks, a player's kills over recent rounds, money changes and what each team bought.

**To do in the facts step: economy facts per team.** Casters talk about buys in about 10–13% of freeze-time utterances and 8–12% of live-round ones (keyword count, 2026-11-03). Examples: "a little broken", "now they're at the full buy". Code gives plain facts and never labels a buy "eco", "force" or "broken"; the model learns that wording from the casters.

- **Team money and spend:** money before the buy, and the amount spent.
- **Team equipment value after the buy.** The rich tier uses the exact value. The HLTV tier uses an estimate from each player's main weapon and armour, since HLTV doesn't send equipment value, pistols or utility.
- **Weapon summary:** rifles, AWPs, SMGs, shotguns and pistols.
- **Armour, helmets and kits.**
- **Next-round affordability:** how many players can afford a rifle with armour next round, worked out from the game's fixed prices.
- **Loss streak, and the team's buys in the previous rounds.**
- **Pistol round:** round 1 or 13.
- **When to build frames:** during freeze time, when money is known, and at the live tick, when the buys are visible.
- **Sources:** the data is already in `player_rounds` (`money_start`, `spent`, `equip_value`, `inventory`, `armor`, `helmet`, `defuser`, `money_end`). Live, HLTV's scoreboard gives money, main weapon, armour, helmet and kit, and the round history gives the loss streak.

### Reported quirks to handle

- `RoundEnd` and `RoundStart` are sometimes skipped, especially at half-time, so also detect rounds from score changes.
- Reconnects and half-time resend old events, and `eventId` can reset, so duplicates must be removed.
- Events don't always arrive in order, and key names vary.
- During freeze time, `alive` can be false for all 10 players, so don't calculate man count then.

### Minimap

HLTV's minimap is probably driven by the `Kill` event's X/Y fields. `woq/hltv-bot` calls them minimap coordinates and ignores them; its only example shows zeros, which may be a placeholder. **Unknown** whether they carry real values.

If they do, you would get a position only at the moment of a kill, not continuous movement. The plan would be:

1. Keep raw coordinates out of the prompt.
2. Build a per-map lookup from position to callout using demos, which contain both position and `last_place_name`.
3. Work out the conversion from HLTV's scale to demo coordinates for each map.
4. Pass only the callout to the model, such as `victim_location: Connector`.

If they don't, the model has no location data, and transcript lines naming positions should be filtered out of training.

### Access

- **University wifi blocks HLTV.** From this machine, `www.hltv.org` and the scorebot hosts timed out while other sites worked. They all resolved to one Amazon Mumbai IP. You confirmed the wifi blocks HLTV. Use another network, such as a phone hotspot or home wifi.
- **Avoid using a VPN to get past the block on campus**, since that could break university network rules.
- **Scripted connections may fail on any network.** It's reported that Cloudflare often rejects them with a 403, and that only a real Chrome session stays stable. An [older client issue](https://github.com/gigobyte/HLTV/issues/904) reports the same error.
- **Don't build on copied cookies or browser impersonation.** That works around HLTV's bot protection and could break at any time.
- **The feed is unofficial and undocumented.** The hltv client's [README warns](https://cdn.jsdelivr.net/npm/hltv@3.5.0/README.md) that abuse will likely get your IP banned. HLTV's current terms have not been read. If this goes public, get HLTV's permission or a licensed feed.
- **No archive of past feed messages was found.** The feed can't be the main training source because you could only record matches as they happen.

---

## 5. Training data: demos

### Where demos come from

- **There is no Valve site for pro demos that we could find.** Valve's in-game download reportedly covers only your own recent matches, for about 30 days.
- `.dem` is Valve's file format. HLTV is where pro demos are hosted.
- "dem2" is not a file type. `demoparser2` is the Python library that reads `.dem` files.

### Main source: [`cs2povarchive/cs2-demos`](https://huggingface.co/datasets/cs2povarchive/cs2-demos)

**Tested:** full file inventory, plus 4 files downloaded with checksums matching Hugging Face's records and parsed.

| | Cologne Major 2026 | `cac2026` (another event) |
|---|---|---|
| Files | 219 | 64 |
| Series | 106 | 30 |
| Size | 92.9 GB | 27.7 GB |
| Status | 4 files parsed | Not opened |

- **Total:** 283 demo files, 120.6 GB. It contains only demos and a README with a licence line: no VODs, captions or metadata.
- **Cologne breakdown:** 33 Stage 1 series, 33 Stage 2, and 40 Stage 3 and playoffs.
- **Distinct Cologne maps:** about 198, after joining split files and removing the duplicate. Median file size is 444 MB.
- **Reachable on university wifi:** Hugging Face works from this network.
- **Origin is unverified.** It's an unofficial upload, and the Apache 2.0 tag is just what the uploader set. Folder numbers look like HLTV match IDs.

**Archive quirks**

| Issue | Detail | Handling |
|---|---|---|
| Split maps | 19 maps are split across `p1`/`p2`/`p3` files (39 files) | Join them. On Vitality vs MOUZ Dust2, part 2 continued exactly from part 1 (round 13, 6-6, kills continuing from 76) |
| Fake suicides | That part 2 opened with 10 fake suicides | Remove them |
| Duplicate | `g2-vs-spirit-m1-overpass(1).dem` is identical to the normal file | Skip it |
| False rounds | Some files have an empty round at the very start | Filter it out |
| Kill totals | In one demo the scoreboard total was 1 below the kill events | Probably a team kill; not checked |

### Other demo sources

- **HLTV match pages:** the safest origin, but blocked on your wifi and downloaded match by match. Series come as compressed archives.
- **[`awpy-fixtures`](https://github.com/pnxenopoulos/awpy-fixtures):** a few demos from various sources, including two labelled as HLTV pro matches, 9z vs PARIVISION and B8 vs Vitality. Good for testing, too few for training. Its manifest includes published stats for the 9z vs PARIVISION map.
- **[`cs2-pro-round-economy`](https://huggingface.co/datasets/sneakyzero/cs2-pro-round-economy):** economy data from 1,988 pro maps, with no commentary. Possibly useful for judging which buys or stats are unusual.

### What the parser does in practice

**Tested** with `demoparser2` 0.42.1.

- **Verified against published data:** on 9z vs PARIVISION, the final score was 13-10. dgt (20 kills, 7 deaths) and Jame (12 kills, 12 deaths) matched exactly.
- **ADR:** close but not identical. dgt 85.4 vs 84.7, and Jame 53.0 vs 52.7.
- **Speed:** each checked demo took about 3–6 seconds.
- **Halves:** half-time side swaps come through correctly.
- **Demo size:** about 350–550 MB for typical maps, with some files up to 921 MB.

**Traps found**

- `parse_event` has no `player_extra` argument, so Gemini's sample code fails with a `TypeError`.
- **Recordings differ.** The ESL demos lack events such as `begin_new_match` in some files, and some lack `bomb_exploded`. Don't assume any event exists.
- A missing event returns an empty list rather than a table.
- `round_end` values differ: text such as `t_killed` in pro demos, and numeric codes in the matchmaking test demo. `total_rounds_played` also appeared to count differently between recordings.
- `damage_total` only updates some time after a round ends, and `dmg_health` can exceed the victim's remaining HP. To calculate ADR during a round, cap damage at remaining HP yourself.
- Deaths with weapon `world` or no attacker are not kills, and neither are self-kills.
- Team names can be short forms, such as "PV" for PARIVISION.
- Inventory uses display names ("AK-47"), while kill events use internal names ("ak47"), so a mapping is needed.

### Converting demos to HLTV's format

**Tested:** HLTV-style `scoreboard` and `log` messages were rebuilt from 4 archive files. No expected field was missing from the structure, and 64 of 72 audited fields were always filled. "HLTV format" here means the reported format, since there's no capture yet.

| Status | Fields |
|---|---|
| Direct from the demo | Map, score, round, HP, money, K/D/A, alive, armour, helmet, kit, weapon, headshot, assists, flash assists, bomb plant, defuse, site, round winner |
| Calculated by code | ADR, round history, entry kills and deaths, multi-kill rounds, primary weapon, man count at plant |
| Not in the demo | HLTV IDs: `dbId`, `killerId`, `victimId`, `ctTeamId`, `tTeamId`, `startingCt`, `startingT`. The model never sees these, so they don't affect training |
| Not computed in this test | KAST and 1vX wins. Both can be calculated, but HLTV's definitions may differ |
| Uncertain | Kill X/Y: demos use map coordinates, and HLTV's scale is unknown |
| Will differ | Event IDs, which have to be generated, and possibly team and player names |
| Filled 50% of the time | `primaryWeapon`, which is empty for dead players and pistol-only buys. What HLTV sends in those cases is unknown |

**Remaining differences between demos and the live feed**

- **Extra data:** demos include smoke and wallbang flags and exact positions. These must stay out of the model's input. Keep them separately for filtering transcripts.
- **Fresher data:** demo HP is exact at every moment, while live HP may be a few seconds old. Once a capture shows the update rate, training data should copy that delay.
- **Names:** team and player names may differ, so a lookup table is needed.

### Match card: details outside the round data

Neither source has one table of match details, so build a small "match card" per map, in the same format for training and live.

**Tested:** demos hold team names as set on the server and player nicknames. These can be short forms: the Perfect World demo says `PV` where HLTV says PARIVISION. The demo header names only the server ("ESL Match Server #1", or the Perfect World event server), not the event.

| Card section | Live source | Training source |
|---|---|---|
| Teams, players | Feed (`ctTeamName`, `terroristTeamName`, player nicks) | Demo, plus the name lookup table |
| Event, stage, format, map number, map pick | HLTV match page, entered by hand before the match | Archive folder name (event, HLTV match ID, map number) plus the HLTV match page or Liquipedia |
| Series so far: earlier map scores, player stats | Your code keeps it from earlier maps | Calculated from the series' earlier demos |
| Background: ranking, recent results, head-to-head | Collected before the match | Liquipedia or HLTV results, **as of the match date** |

**Rules**
- **No future information:** background details in training must be what was known before that match, or the model learns from information it won't have live.
- **Only what you can fill live:** keep a field in the card only if it can be filled in live. Pages behind Cloudflare are collected by hand in a browser, not scraped.
- **Training sources found 2026-11-03:**
  - The archive folder gives the HLTV match ID, team slugs and map number (`m1`–`m5`).
  - The VOD title gives the stage (e.g. "Group A", "Quarter-final", "Stage 1 ROUND #1").
  - The demo gives team names, rosters and the map.
  - The other demos of the same series give the series score before each map.
  - Earlier archive matches in the same event give a team's record so far and any rematch, already as of the right date.
  - Still external: map picks (HLTV match page or Liquipedia) and rankings.
  - Valve's published regional standings may give dated rankings. **Unknown** until checked.
- **Background is small:** about 5 fields. Casters' storylines go beyond any card (roster news, a player's past form, jokes). Remove lines that depend on facts not in the card, or the model learns to make them up.

---

## 6. Commentary: VODs and captions

### Which VODs

| Source | Format | Use |
|---|---|---|
| **ESL Archives (Cologne 2026)** | Single series | **Main data** |
| PGL (Copenhagen 2024) | Some single-map videos, 36–45 min | Handy for simple tests |
| StarLadder (Budapest 2025) | At least some single series | Possible, after checking the rest of the event |
| BLAST (Austin 2025) | Whole-day or multi-match streams, 6–7 hours | Later; desk segments and ads need cutting |
| Co-streams (ohnepixel, fl0m, MN cast) | Streamer reactions | **Avoid** |

**Cologne coverage (Tested, by title):** ESL Archives has a single-match VOD for 105 of 106 series. Legacy vs MIBR was not found, though it may be in one of the 27 full-day videos from ESL Counter-Strike. A few title matches came from loose search and should be rechecked when each VOD is used.

**Captions:** every VOD checked had automatic English captions and no human-written subtitles (6 of 6 sampled, plus others downloaded).

### Captions compared with Whisper

**Tested** on B8 vs Vitality Mirage, 40 minutes of audio, using `large-v3-turbo`. This map comes from a different event, not from the archive.

| Measure | Captions | Whisper |
|---|---|---|
| Kills matched, automatic name matching | **36/123** | 27/123 |
| Kills matched, with a misheard-name table | **58/123** | 40/123 |
| Typical delay from kill to name | **1.4 s** | 2.9 s |
| Killer and victim both named | **16** | 8 |

**Decision: use YouTube captions for mapping and keep the audio for intensity analysis. Whisper is the fallback.**

**Limits of this result**

- It's one map and one Whisper model.
- The misheard-name table was built mostly from caption errors, which favours captions in that row. The automatic row doesn't have that bias.
- On the older PGL video, Whisper's text was clearly better: it had punctuation, `karrigan` and "9-1" where captions had "carrian" and "91".
- This ESL video's captions had punctuation and `>>` speaker-change marks, while the older PGL video's (Copenhagen 2024) had almost no punctuation. **Update 2026-11-04:** all 18 pilot VODs, ESL and PGL 2026, have punctuation (sentences average 6–12 words). `>>` marks are sparse, especially on PGL.

**Known caption problems**

- **Misheard names:** `Aleksib` to "Alexi", `karrigan` to "carrian", `b1t` to "bit", `ropz` to "Robs", `flameZ` to "Flamesey", `esenthial` to "Sentiel".
- **Garbled numbers:** "9-1" to "91", "7-0" to "70", and possibly "1v3" to "1 B3".
- **Censoring:** some words become `[ __ ]`, apparently including `s1zzi`'s name, which costs mapping hits.
- **Tags:** some text such as `[Music]` and `[applause]` needs removing.

**Whisper details:** about 8 minutes for 40 minutes of audio on this CPU, so no GPU is required for small batches. It sometimes invents text at the end of clips (repeated phrases, "Thank you"), and mishears names too ("Zaiwu", "Kent Cizor", "Messi").

### Mapping events to commentary

**Tested:** kill times from each demo were compared with name mentions in the captions, trying every time offset. A real alignment should spike at one offset.

| Demo and VOD | Kills matched at best offset | At a typical offset |
|---|---|---|
| B8 vs Vitality, Mirage | 36/123 | 5 |
| GamerLegion vs NRG, Inferno (Bo1, Stage 1) | 64/144 | 6 |
| Vitality vs MOUZ, Dust2 part 1 | 30/76 | 4 |
| Vitality vs MOUZ, Dust2 part 2 | 23/64 | 3 |
| FURIA vs Falcons, Mirage (grand final) | 41/134 | 5 |

**Findings**

- **The signal is real.** Spot checks agree too: "It's ZywOo in the clutch... he misses and PL closes it down" lines up with npl killing ZywOo in the demo.
- **A single offset held across a map** in these tests, and the two Dust2 parts landed within 4 seconds of each other. This was better than expected, since pauses could break it.
- **A few rounds fit better at very different offsets.** This may be noise from having few kills, or may be real pauses. **Unknown** which.
- **Casters often speak at or just before the kill.** The search window should start a little before the event, not at it as Gemini proposed.
- **Strict one-to-one mapping won't work.** Using the misheard-name table, a player was named within 6 seconds for only 58 of 123 kills, and both killer and victim for only 16. Some unnamed kills still get commentary ("He nails the head shots"), so true coverage is higher. Casters also merge kills ("Three kills from that...").

**Decision: pair a short group of events with the commentary span that covers it.**

**Decision (2026-11-03): every kill gets commentary.** No kill is skipped live, and the model never learns `<silent>` for a kill. Importance is learned from how the model comments (calm or excited, wording, and the version 2 intensity labels), not from whether it does.
- **Live:** kills in a burst, such as a trade 0.3 s later, share one line, so every kill is covered without falling behind.
- **Target lines when casters don't name a kill** (a player was named for only 58 of 123 kills in the test):
  1. Group the kill with nearby kills the casters did talk about.
  2. Use the casters' line wherever they talked about the kill.
  3. **Decided 2026-11-03:** kills no caster line covers get a **plain factual report**, like Cricbuzz's line for every ball even when TV commentators say nothing.
     - Example: "Zero kills Spinx with the AK-47. 4v4."
     - It covers only the facts: who, weapon, headshot, flash assist, man count, and position when the tier has it.
     - It carries no flavour or emphasis. Those come only from the casters.
     - This teaches importance: kills the casters ignored train toward plain reports, and kills they reacted to train toward their words and emphasis.
     - **Wording is not hard-coded or repetitive** (user, 2026-11-03).
       - Collect real low-key caster lines from the captions, e.g. "and Jee goes down".
       - A larger model writes each plain line from the facts, using that caster wording as its vocabulary.
       - Keep it only if the fact checker passes it.
       - Track what share of the training data these lines make up.

**Entries and triggers (decided 2026-11-03):**
- **Kill groups always get an entry.**
- **Grouping:**
  - Kills are grouped by a time gap, starting at 3 s and tuned on the pilot.
  - A caster's summary of kills from several groups (e.g. "three kills from Grim") goes with the last group it covers.
  - That group's input includes the earlier kills, as recent events and previous entries.
- **Moments where the model may post or stay `<silent>`:**
  - a clutch setup (the man count reaches 1vX)
  - a timeout
  - a weapon pickup
  - a grenade landing (rich tier only)
  - Whether to post is learned from whether the casters talked about it.
- **Always an entry:**
  - bomb plant, defuse and explosion
  - round end
  - buy phase
- **Context in every input**, from fixed to fast-changing (about 600–1,000 tokens):
  1. **Match card:** event, stage, format, map number, series score, teams, rosters.
  2. **Map so far:** score, round, half and sides, match point, recent round results, streaks, economy.
  3. **Player form on this map:** K/D, ADR and multi-kills, at least for the players involved.
  4. **The whole current round so far:** every event since the round went live, plus the current state (alive, HP, weapons, bomb status, time into round).
  5. **Recent entries:** this round's entries, plus the last one or two from the previous round.
  6. **What to report now.**
  - **Order:** layers 1–3 stay fixed through a round and come first, and layers 4–6 grow at the end. This keeps prompt caching effective on CPU.
  - **Training vs live:** training uses the previous targets as the recent entries. Live, each entry must pass the fact checker before it becomes context.
  - **Tiers:** each tier fills the layers only with the fields it has.
- **Context:** every kill also stays in the input for later lines.

### Caption pipeline, step by step (decided 2026-11-04)

Each step is a script that reruns on new maps without hand work, except where marked.

| Step | What | Status |
|---|---|---|
| 0 | Parse demos (`cs2c.parse_demo`) | Done, 20 maps |
| 1 | Clean captions: names and game terms from `aliases.tsv` plus each name's own spellings, scoped to the teams on the VOD (`cs2c.clean_captions`) | v1 done: 18 VODs, 8,606 words fixed |
| 2 | Sync demo clock to VOD, cut each map's words out, label round and phase (`cs2c.sync`) | **Done, all 20 maps pass** |
| 3 | Learn missing aliases from the demos: unknown words that keep landing next to one player's events (`cs2c.learn_aliases`) | Done: 5 learned automatically, 9 more added to `aliases.tsv` after review, 2 rejected ("alex", "magna") |
| 4 | Split words into utterances (`cs2c.utterances`) | Done: 18,153 utterances, median 5–10 words |
| 5 | Group events into reportable entries, with facts | Planned |
| 6 | Match utterances to entries | Planned |
| 7 | Hand-label a gold set (about 100 entries), measure and tune | Planned, **hand work** |
| 8 | Turn matched caster words into training targets | Planned |
| 9 | Build input frames per tier | Planned |
| 10 | Assemble the dataset, split by series | Planned |

**Step 2 findings (Tested 2026-11-04, 20 maps):**
- **Evidence:** caption words tagged with a player's steam id, against that player's kills and deaths. A mention counts most at 2–6 s after the kill, and the offset says when casters react, not the exact frame shown.
- **One clear peak per map**, 1.4–3× the next best offset.
- **The offset drifts on ESL VODs**, steadily, by 2–5.5 s per 1,000 s of demo, which is up to 17 s over the grand-final Inferno. It does not jump at pauses or half-time. PGL VODs are nearly flat.
  - **Model:** offset = start + drift × time, or a constant when drift doesn't help on held-out rounds (6 of 20 maps, all with little drift).
  - The demo clock is exactly tick ÷ 64, so the drift comes from the server's tick timing or the broadcast, not the parser.
- **No jumps** anywhere, including the Dust2 map split over two demo files. Steps are still allowed when they add at least 4 kills of evidence, so a cut VOD would show up.
- **Fitting each round separately, as the prototype did, was wrong:** its offsets jumped ±100 s from noise.
- **Checks on every map:**
  - Held-out rounds score 3.8–8× chance.
  - Count tags ("1v3") match real alive counts 44–100% of the time, against 3–14% at wrong offsets.
  - "plant" and "defuse" words land at bomb events 3–20× more often than at wrong offsets.
  - Neither check is used in the fit.
  - A read of sample kills against the captions confirmed it, including both ends of the most-drifted map.
- **Output:** 155,414 words inside maps, out of 346,451 in the VODs. The rest is pre-show, breaks and desk talk.

**Step 3 findings (Tested 2026-11-04):**
- **Method:** a word the corrector left untagged is tested against each player on its maps. It is counted when it lands 3 s before to 11 s after that player's kill, death, plant or defuse, and compared with the same words moved to wrong times.
  - **Accepted** if the timing is strong, one player clearly wins, and the spelling is loosely like the name. Closer spelling needs less timing proof.
  - **Otherwise review**, if the timing alone is strong.
- **Answer key:** the 296 hand-written player aliases were hidden, and 134 of them come up often enough to test.
  - 43 were found with the right player: 23 accepted and 20 sent to review.
  - **None was accepted for a wrong player.** The 4 wrong-player cases all went to review: teammates in the same fights, and "alex", which is alex666 on B8's VOD and xelex on MOUZ's.
- **Conservative by design:** a form said only a few times can't be proven. The rest stays with the hand table and the review list.
- **Learned on the real data:** kuk (kyousuke), sias (SunPayus), towson (Tauson), tases (TeSeS), zentarz and zentar (XANTARES). Every sighting was read in context.

**Step 4 findings (Tested 2026-11-04):**
- **Punctuation is good on both crews.** Sentences average 6–12 words, so a sentence end is the main break.
- **Speaker marks (`>>`) are sparse**, 17–313 per map, so they are used only as hard breaks.
- **Casters pause mid-sentence.** A pause of 1.5 s or more ends an utterance only when the next word starts a sentence: a capital, a name tag or a number. A silence of 4 s or more always ends one.
- **Units are never merged here.** The matcher groups consecutive ones.
- **Noise units remain** ("[cheering]", and "heat", a YouTube caption artefact during music). They hold no game terms, and Step 8 drops them.

**Proposed method, in two levels (2026-11-03).** Level 1 is now built as Step 2; see its findings above. The per-round refinement and per-pause offsets proposed here proved unnecessary. The offset is a line per map, and bomb words and count tags are used as checks rather than fit inputs. Level 2 is Steps 5–6.

1. **Clock alignment.**
   - Find one offset per map from name hits (tested).
   - Refine it per round using anchors: kills, "plant is down", and score call-outs at round end.
   - Allow a new offset after pauses.
   - Fallback: read the score and round timer from the broadcast video with OCR.
2. **Local matching.**
   - Split captions into utterances at speech pauses.
   - Group events that come close together.
   - Score each candidate span for each group in a window from about 1.5 s before to about 6 s after:
     - names
     - weapons
     - kill-count words
     - man counts against computed alive counts
     - bomb words
     - score numbers
     - closeness in time
   - Assign spans in time order with a monotonic alignment (dynamic programming), so a span goes to only one group.
   - Keep high-confidence pairs only.
   - Use a model as judge for unclear cases, if needed.
3. **Phase rules.**
   - Freeze-time speech pairs with the round summary.
   - Names of players who died earlier, mentioned after the round ends, are replay talk: drop them.
   - Speech outside map time is dropped.
4. **Gold set.** Hand-label about 100 event groups from 2 maps, then measure the aligner's precision and recall and tune its thresholds.

### Other cleaning needed

- Remove sponsor reads, desk chat and off-topic talk (the test map included a chat about firearms).
- Set aside lines about things the input can't support, such as positions, smoke and player history. Otherwise the model learns to invent them. This is the main source of made-up details. A real example: "Frozen in the smoke but spammed out by Alexi... up in the top mid."
- Watch for replays between rounds, when casters discuss a kill from a minute ago.
- Skip speaker separation at first. It's weak with overlapping speech, and published error rates of 11.7–17% come from easier audio. Use game phase as a stand-in for style: live rounds for play-by-play and freeze time for analysis.

### Talk away from events

**Tested 2026-11-03 on 3 pilot maps:** during live rounds, 56–62% of caster utterances are not within about 6 s of a kill, plant, defuse or round end. There are also 90–117 utterances per map in freeze time.

Quiet talk falls into 4 kinds:
- **Grounded in facts we have:** "5-1, NRG have had a couple of chances, but GamerLegion dictating the tone". Keep it.
- **Reads of play needing data we lack:** setups, rotations, utility, positions. Example: "they're scaling this execute and MOUZ not buying it". Drop it, or keep it for the rich tier.
- **Story beyond the match card:** drop it, unless the card has the fact.
- **Banter and production:** "check in on the B stream". Drop it.

**Proposed handling:**
- Add "quiet tick" frames every ~10–15 s with no events.
- For each quiet tick, the target is a grounded line if casters said one, otherwise `<silent>`.
- This lets the model learn from data when there's nothing worth saying.
- Keep the `<silent>` share moderate, and tune it, so the model doesn't learn to stay quiet everywhere.
- Text readers don't need every second filled.
- **Unknown:** what share of quiet talk is grounded. The Phase 2 tagging measures it.

### Training on full data while staying honest on less

Proposed 2026-11-03. This replaces the plain "set aside unsupported lines" rule above: one model learns from all demo data but must never mention what its input doesn't contain.

- **Tiers of input matching real sources:**
  - **HLTV tier:** the reported feed fields only.
  - **Rich tier:** adds utility, inventories and positions, the fields CS2's Game State Integration (GSI) may provide. **Unknown** until tested.
  - **Full tier:** everything in the demo.
- **Availability header:** each frame starts with a line listing which data kinds are present and which are absent. Every section also appears in every tier, marked `NOT REPORTED` when absent rather than left out.
- **The HLTV tier must mirror exactly what HLTV sends.** For example, it keeps flash assists (`flasherNick` on kills) while removing smokes, molotovs and HE grenades.
- **Tag every commentary line with the data it depends on.** Examples: kill, bomb, economy, utility, position, history. Use keyword lists plus a small model as the classifier, checked against a hand-rated sample. The demo shows whether the utility or event really happened, which also catches mapping errors.
- **Pair each line with tiers that support it.** A smoke line trains only with tiers that include utility. In the HLTV tier, the target is a line that uses only HLTV facts. That comes from the same moment's other lines or a rewrite by a larger model, and is kept only if it passes the validator.
- **Never** train a line whose facts are missing from its input. That one mistake would teach the model to invent.
- **Also drop individual fields at random within a tier,** such as `bombSite` (sometimes missing live) and stale HP. This teaches the model to cope with gaps.
- **Tier mix is a setting to test, not a fixed rule.**
  - Start with 70% HLTV tier and 30% richer, because HLTV is the live input and not inventing on it matters most.
  - Switch to 70% richer if GSI becomes the live source.
  - Compare against 70% richer, whose examples are real lines with full grounding that need no rewrites.
  - Pick using two measures: the HLTV-tier invention rate and blind preference on HLTV-tier input, since that is the live source unless GSI works.
  - A line that uses only HLTV facts is valid in every tier, so the two tiers can share the same moments.
- **Test with ground truth:**
  - In the HLTV tier, any mention of utility or positions counts as an invention.
  - In richer tiers, check each utility mention against the demo.
  - The validator stays as a backstop at run time.
- **Expected costs:**
  - About 1.5–2× the examples and training time.
  - HLTV-tier lines will be plainer, because many lively lines are about utility.
  - The richer tiers pay off live only if GSI or another source supplies those fields.

---

## 7. Intensity

**Your idea:** a per-event intensity level, learned from data, reflecting how important or unlikely a moment is, and shown in the written commentary through word choice, capitals and so on.

**View: it's a good idea and worth building, but it can't be measured reliably yet.**

### Test results

**Tested:** audio on one map; captions on 4 matches (5 demo files, 530 kills).

| Signal | Result | Verdict |
|---|---|---|
| Raw loudness | 1vX kills -0.12 vs -0.08 for others (z-scores). The range is narrow (median -28 dB, 95th percentile -24.5 dB), and the "loudest" kills were mostly normal chat | Too weak alone |
| Speech speed | No clear difference in the same comparisons | Too weak alone |
| Capitals in captions | Appear on shouted lines, and those spans were 5.6 dB louder. But only 13 of 530 kills (2.5%) had them, with none on the grand final map or Vitality vs MOUZ | Useful hint, unusable as a label |

- Capitals were more common on clutch kills: 14% of 1vX kills vs 1.4% of others, but that's only 6 kills. At least one run was a false alarm ("TERRORISTS WIN").
- Kills at match point and in close games had no capitals, but with so few capitals overall that says little.
- **Limits:** the audio still has game sound and crowd mixed in, and the audio-tested map was a one-sided 13-5. This shows that easy signals are weak, not that excitement can't be measured.

### Two separate things

- **Importance and unlikeliness** come from the game. A small separate model could learn how much each event changes a team's chance of winning the round, using man count, economy and bomb state from demos. This is learned rather than if/else, and it works live. **Not yet tested.**
- **Caster excitement** comes from the audio, and is what the text should copy. It usually follows importance, but not always.

Importance becomes an input fact, and excitement is what the model learns to output.

**Win-probability model (parallel track, planned 2026-11-04):**
- **What:** a small separate model gives each moment's chance of winning the round, and each entry's swing in that chance, as input facts.
- **Data:** `states.parquet` per second, labelled with the round winner from `rounds.parquet`.
- **Inputs:** HLTV-tier only, so it runs live: alive per side, HP, armour, main weapons, equipment value, kits, bomb status and site, time since plant, round time.
- **Size:** 20 maps is only about 450 rounds. It needs a few hundred archive maps, about 10,000 rounds. These only need parsing, no VODs.
- **Training and checks:** LightGBM or logistic regression, split by match, judged on log loss, Brier score and calibration.

### Proposed design

1. **The model predicts the level first, then writes the line.** For example: `[INTENSITY: PEAK] HE DOES IT!` It stays one model, adds almost no delay, and the level is available for text styling or voice later.
2. **Use 4 levels:** calm, engaged, high and peak. The labels are too noisy for 1–10.
3. **Style the text from the level.** Word choice comes from what casters really said at that level. Since captions rarely show capitals, code adds capitals to training text for peak lines only. That's formatting, not a judgement about importance, so it fits the no-rules approach.
4. **Give the model the stakes as facts.** The feed doesn't reliably give series score or tournament stage, so "tournament-defining" needs extra details from the match page, such as stage and map number.

### Expected limits

- **Hidden causes:** if casters shout about a blind shot through smoke or a flick, the feed can't show why, so the model will miss some peaks.
- **Rare peaks:** the model could play it safe, so peak examples need extra weight in training.

### How to get reliable labels

1. A person, ideally you, listens to about 200 moments across 3–4 maps, including a final, and rates each from calm to peak. This is the ground truth.
2. Test better signals against those ratings: isolated voice (for example with Demucs), pitch, and a mix of measures.
3. If one agrees with the ratings, use it to label all training data automatically.
4. If none does, base the level on importance alone, or on a model trained from your ratings.

---

## 8. Model and data size

### Base model

**Update 2026-11-03: no 24/7 serving is planned.** The user will run the model on a GPU server only when testing or analysing, so CPU speed no longer limits model choice.
- Qwen3-4B-Instruct-2507, Qwen3-8B and Gemma 4 E4B compete on quality alone. Fine-tune them and compare on held-out series.
- The CPU speed test is optional.
- The notes below about speed on CPU apply only if the model is ever run on this PC.

Updated 2026-11-02 after the model survey in [reports/Open models for CS2 commentary.md](reports/Open%20models%20for%20CS2%20commentary.md). The CPU speeds there are **Estimates** from memory bandwidth, not measurements on this PC.

- **Primary: Qwen3-4B-Instruct-2507** (Apache 2.0).
  - It has no thinking mode, so no `<think>` tags can leak into the output.
  - Qwen reports Creative Writing v3 at 83.5, against 64.5 for Qwen3-8B with thinking off. These come from different eval versions, so the gap is indicative.
  - It's a plain transformer, the easiest case for prompt-cache reuse in llama.cpp.
  - It trains easily on a free T4.
- **Comparison: Gemma 4 E4B** (now Apache 2.0). It's in the same speed class, and Gemma has the strongest independent prose scores among small models.
  - **Main risk:** its sliding-window attention may stop the prompt cache being reused. Check `tokens_cached`.
- **Qwen3-8B is now only a speed reference and the GPU fallback.** A dense 8B is estimated at about 4.5–8.5 s per line on this CPU even with caching, about twice the budget.
- **Stretch option: Qwen3-30B-A3B-Instruct-2507.** It has the best writing scores and generates at roughly 4B speed, but it can't train on a T4. Test it untrained first, and fine-tune it on a Modal L4 only if it clearly wins.
- **Rejected:**
  - Qwen3.5: Unsloth advises against QLoRA, and it has Ollama problems.
  - gpt-oss-20b: its reasoning can't be turned off.
  - Phi and Llama: weaker formatting and prose.
  - In reserve: Nemotron 3 Nano 4B and LFM2.5-1.2B.
- **Prompt layout matters as much as the model.** Put the match card and round state first and keep them identical between calls. Put recent events and previous lines at the end.
- **Train with QLoRA** on a free T4 (Colab or Kaggle) with Unsloth.
- **Before training, re-check Hugging Face for Qwen4** (reported as imminent in late October 2026). Check its licence and cache behaviour.
- **Always test the same model with no fine-tuning**, given a few real caster lines as examples. If fine-tuning doesn't beat that, the extra work isn't paying off.

**Hardware (Tested):** `nvidia-smi` isn't installed, so this machine probably has no usable NVIDIA GPU. It has 31 GB RAM, 24 threads and about 302 GB of free disk. That's fine for parsing, captions and small Whisper jobs. Training needs a cloud GPU, a free T4 is not dependable for hosting live, and whether this CPU is fast enough for live inference is untested.

### Existing models: nothing to build on

Searched Hugging Face, GitHub and the wider web. It's hard to prove nothing else exists.

| Found | What it is | Useful? |
|---|---|---|
| [`Qwen3-1.7B-cs2-commentators`](https://huggingface.co/Dannys0n/Qwen3-1.7B-cs2-commentators) | A class project with casters inspired by the Portal 2 announcer and turrets. Only knows locations on Dust2, expects inputs the feed lacks, and lists no licence | No |
| [Fine-tuned Qwen3-VL on CS2 questions](https://arxiv.org/html/2604.12320v1) | Video Q&A benchmark, not commentary (56.37% to 58.68% accuracy) | No |
| [League of Legends event-to-commentary project](https://github.com/XXoussam/Esports-Event-to-Commentary-Generation-LoL) | Closest approach, but a different game | Read its "event faithfulness" evaluation |
| [`cs2-ai-commentator`](https://github.com/HullyMully/cs2-ai-commentator) | Prototype working from video, with templates or optional LLM generation | Architecture ideas at most |
| Valorant, cricket and chess commentary models | Other games and sports | No |
| [F1 live commentary adapter](https://huggingface.co/jsantillana/qwen2.5-3b-f1-commentary-lora-v2) | Short, grounded commentary from a structured live snapshot, on Qwen2.5-3B. Closest match to this project's design | Study the design, don't build on it. Research-licensed base |
| [`matchcommentary`](https://huggingface.co/abocide/matchcommentary) and [DeepSport](https://arxiv.org/pdf/2511.12908) | Commentary from video | No, the feed has no video |

**General-purpose commentary models:** none found. Everything is a small fine-tune for one sport, and many have blank cards. Building on one would mean untraining that sport's vocabulary, and adapters tie you to their older, smaller base models. Real caster transcripts give you the commentary style directly.

Building on a fine-tuned model wouldn't gain much anyway. There's no general model that has "learned CS2", these small fine-tunes mainly teach a style and output format, and your input spells out the facts already.

**Optional later experiment:** if a Qwen3-8B fine-tune falls short, try training first on public commentary from several sports, then on CS2, and keep it only if it beats direct training on held-out series.

### How much data

**Estimate**, based on measured maps with 18–23 rounds and about 123–143 kills.

- **Target: 40–60 maps, about 15–25 Bo3 series.** Count maps, not matches.
- **Examples per map:** roughly 60–100 after cleaning, including round endings and buy phases.
- **Overall target:** about 3,000–5,000 examples.
- **Build up in stages:** 5 maps to prove the pipeline, 20 maps for the first model, and 40–60 maps only if results are still improving.
- **Hold out 4–5 series** the model never trains on, ideally with some unseen teams and casters. Keep every map of a series on the same side of the split.
- **Use more than one tournament.** With Majors only, the tournament and stage inputs never change, so the model can't learn that stakes differ. It would also copy one casting crew. The archive's `cac2026` folder is a second, non-Major event: 30 series, 64 maps, Bo1 to Bo5. Its folder names begin with HLTV match IDs (2394199–2394228). **Tested 2026-11-03:**
- It is the CS Asia Championships 2026, held in Shanghai, 20–24 May.
- All 30 series have an English single-match VOD on the "PGL VOD's" YouTube channel, with auto captions. The casters are PGL's, a different crew from Cologne's ESL casters.
- The mapping is in `data/meta/cac2026_vods.tsv`.
- Alignment works on 2 test maps, with 29 and 45 kills matched at the best offset. Rough mix: about 30 Cologne maps and about 20 `cac2026` maps. Keep a few `cac2026` series held out to test on an unseen event. A third organiser's event, downloaded from HLTV in a browser off campus, is optional.

### Which formats

| Format | At Cologne | Use |
|---|---|---|
| Bo3 | Swiss deciders, Stage 3 | **Most of the data** |
| Bo1 | Other Swiss matches in Stages 1–2 | **Some**, for more teams and calmer games |
| Bo5 | Probably the grand final, not confirmed by sources | **Just a few**, to avoid everything sounding like peak hype |
| Bo2 | Not in the format checked | Skip |

Mix in early Swiss games so the model learns calm commentary and more player names, use different casters, and don't let one team dominate. Only Cologne has been checked for demos and VODs.

---

## 9. Parts of the live system still to design

The Gemini chat focused on what to say and didn't cover these:

- **When to speak:** deciding when to talk or stay silent, especially when kills come in bursts. Events must be merged or dropped because generation takes time.
- **Avoiding repetition:** the model needs its previous lines as context, and training pairs should include them.
- **Stream sync:** a delay setting, so commentary doesn't spoil what's on the viewer's stream. The feed and any given stream will run at different delays.
- **Fact checking:** a validator that drops or regenerates lines containing names or numbers not in the input.
- **Replay tool:** record feed messages to disk and replay them at real speed, so development doesn't depend on a live match.

### Where the model runs

**Decided 2026-11-03: on demand, not 24/7.**
- **Offline testing:** replay recorded matches in Colab or Kaggle notebooks.
- **Live test runs:** a pay-per-second GPU such as Modal, started only when needed.
- **This PC:** optional. The CPU analysis below matters only if local serving is ever wanted.

Training and serving are separate jobs. QLoRA training needs a GPU for a few hours, and free notebook GPUs (Colab, Kaggle) can do it. Serving has to stay up for a 2–3 hour series and answer within a few seconds.

**Tested:** this PC has an i7-13700K (24 threads) and 32 GB RAM. Its NVIDIA card is a GT 730, too old for LLMs, so local means CPU only.

**Tested 2026-11-03:** the RAM is DDR5 at 4800 MT/s, and `dmidecode` lists only one module, so it is probably single-channel. That gives about 38 GB/s peak, roughly half of dual-channel.

- **Estimate:** a 4B model at Q4 generates about 10–13 tokens/s, so a 40-token line takes 3–4 s before reading the prompt. An 8B is about 5–6 tokens/s.
- **Cheapest fix:** add a second, matching stick for dual channel, which roughly doubles generation speed. Also check whether XMP or EXPO allows a higher speed.
- **Otherwise:** use shorter lines, a smaller or mixture-of-experts model, or the GPU fallback.

| Option | Verdict |
|---|---|
| **This PC, CPU only**: merge the adapter, quantise to GGUF at 4–5 bit, serve with llama.cpp or Ollama | **First choice for a 4B-class model.** Free, no quotas, and next to the browser-based feed reader. **Estimate (revised):** a 4B-class model takes about 2.2–4.5 s per line with the prompt cached. An 8B takes about 4.5–8.5 s, too slow. RAM type (DDR4 or DDR5) changes speed by about 1.7×. Measure it with the bake-off in the model report |
| **Modal**: pay-per-second GPU, scales to zero | **Fallback if the CPU is too slow.** **Reported:** the Starter plan includes $30/month credit and an L4 costs about $0.80/hour, which is roughly 37 hours, or about 12–15 series a month. Start it before the match to avoid a cold start |
| **Alibaba Model Studio**: token-billed deployment of Qwen3-8B LoRA fine-tunes | Possible pay-per-token option. **Reported:** meant for testing and low traffic, and released after a month unused |
| **Hugging Face ZeroGPU (free)** | No. **Reported:** about 5 GPU-minutes a day |
| **Hugging Face free CPU Space** | No. Slower than this PC |
| **Colab or Kaggle** | No for live serving. Sessions end, and Colab's free tier restricts running a web UI or remote proxy instead of the notebook, which a tunnel to your PC would amount to. Fine for training and for offline tests inside the notebook: the few-shot baseline, held-out evaluation, and replaying recorded frames at real speed. Colab's T4 speed says nothing about CPU speed on this PC |
| **Hosted APIs** (Groq and similar) | No. They can't load your fine-tune |

The live system must run on a network that reaches HLTV (section 4). If this PC stays on the campus network, either run the whole system from another network or use a remote GPU.

### How to evaluate

Judge every model on the same three things:

1. How often it invents facts not in the input. Check that every name, weapon and number in a line appears there.
2. How quickly the first word appears, and the delay from event to output.
3. Whether people prefer its lines in a blind test against other versions.

Also track how often it repeats itself. Your recorded feed captures are the final test, because they're exactly what the model will see in production.

---

## 10. Plan

### Your decisions

- **Text or voice output?** Text is recommended first. Voice adds speech-generation delay and needs the excitement carried into the audio.
- **Private project or public release?** This decides how much the HLTV, YouTube and caster-rights questions matter.

### Phase 0: Confirm the inputs

- [ ] **Capture the live feed.** On a non-university network, open a live match on HLTV. In DevTools, go to Network, filter by WS, reload, let it run for several rounds (including a half-time if possible), and save a HAR file. Capture 2–3 matches over time.
- [ ] **From the capture, confirm:**
  - Real field names and how often the scoreboard updates
  - Whether `victimX/Y` hold real values that match the minimap
  - Whether `primaryWeapon` shows pistols, and what dead players show
  - Whether `alive` is false during freeze time
  - Team and player name formats
- [ ] **Verify the archive.** Download 2–3 of the same maps from HLTV, extract them and compare with the archive copies. If checksums differ, compare scores and kills, or use HLTV instead.

**Done when:** the real field list is known and the archive is trusted or replaced.

### Phase 1: Shared format and converter

- [ ] Define the shared format from the capture, not from the docs.
- [ ] Build the demo converter: join split files, skip the duplicate, filter false rounds and fake suicides, and handle missing events.
- [ ] Keep demo-only facts (smoke, wallbang, places) separate from the prompt, with a test proving they never leak in.
- [ ] Add the name lookup table and weapon name mapping.
- [ ] Copy the feed's update delay into demo state.
- [ ] Decide on KAST and 1vX wins once HLTV's values can be compared.

**Done when:** a recorded feed message and the same moment from a demo produce the same style of prompt.

### Phase 2: Commentary pipeline on 5 maps

- [x] Download captions for 20 maps with mixed stages and casters. Audio not yet downloaded; it's only needed for intensity.
- [x] Clean captions: misheard names and game terms using the rosters, plus names learned from demo timing (Steps 1 and 3). Tags and censoring are kept as rows for now and removed in Step 8.
- [x] Align each map (Step 2). No round sits at an unusual offset once drift is modelled.
- [x] Split captions into utterances (Step 4).
- [ ] Group events and pair them with commentary spans, and drop kills with no clear commentary.
- [ ] Filter desk chat.
- [ ] Tag each line with the data it depends on: keyword and per-map callout lists first, a small model for unclear lines, and about 200 lines checked by hand.
- [ ] Measure what share of lines need utility, positions or history. This decides how much HLTV-tier rewriting is worth.
- [ ] Build the HLTV-tier targets: another supported line from the same moment first, a validated rewrite second, and otherwise use the example in richer tiers only.
- [ ] Read a sample by hand to check quality, and compare with Whisper if captions fall short.

**Done when:** a person reading a random sample agrees that the pairs are correct.

### Phase 3: Intensity labels

- [ ] Hand-rate about 200 moments across 3–4 maps, including a final.
- [ ] Test isolated voice, pitch and combined signals against those ratings.
- [ ] Choose how labels are produced.
- [ ] Try the win-chance model for importance, using demos.
- [ ] Add stakes from match pages: stage, map number and series score.

**Done when:** an automatic label agrees well enough with your ratings, or a fallback is chosen.

### Phase 4: First model

- [ ] Grow the dataset to 20 maps and set aside the held-out series.
- [ ] Run the speed bake-off on stock GGUFs: Qwen3-4B-2507, Gemma 4 E4B, Qwen3-8B and Qwen3-30B-A3B-2507. Check RAM type, `llama-bench`, and `tokens_cached` under replayed frames.
- [ ] Test the untuned base models with a few example lines as the baseline.
- [ ] Fine-tune Qwen3-4B-Instruct-2507 and Gemma 4 E4B on a free T4 with identical data, and compare them.
- [ ] Evaluate on the three measures in section 9.

**Done when:** fine-tuning clearly beats the baseline, or you know why it doesn't.

### Phase 5: Scale up

- [ ] Grow to 40–60 maps only while held-out results keep improving.
- [ ] Add the buy-phase analysis pairs if they weren't already included.

### Phase 6: Live system

- [ ] Measure local CPU speed with the quantised model (time to first word and total time per line). Switch to a remote GPU only if it's too slow.
- [ ] Build the feed reader using a normal browser session, within the access limits in section 4.
- [ ] Build the state tracker: duplicates, missing round events, out-of-order messages and round-over-round stats.
- [ ] Add the speaking scheduler, previous-line context, fact checker and stream delay setting.
- [ ] Test with the replay tool first, then on a live match.
- [ ] Run the final evaluation on recorded feed captures.

---

## 11. Gemini's claims, checked

| Claim | Verdict |
|---|---|
| One end-to-end model rather than two | Agree |
| Mask demos down to live-feed fields | Agree, though it's incomplete (HP freshness, commentary mentioning unseen facts) |
| "Zero domain shift" | **Overstated.** Inputs can match, but commentary still mentions things not in the input |
| The feed has `throughSmoke` and `penetrated` flags | **Reportedly false** |
| The feed has no coordinates at all | **Reportedly false** for kills |
| ADR isn't directly available | **Reportedly false** (`damagePrRound`) |
| Bomb site is available, and "32s remaining" | Site is only sometimes present, and no timer field is reported |
| `throughSmoke` drama examples ("blind through the smoke") | Can't be grounded in the live feed |
| HLTV archives pro demos | Consistent with what we found; not checked directly because of the wifi block |
| Demo code using `player_extra` | **Fails** on `demoparser2` 0.42.1 |
| Parsing takes about 3–5 seconds | Roughly matches the checks run |
| A single global offset for alignment is enough | Mostly held in our tests, but pauses are unresolved |
| Using "Round 1" or "Pistol round" phrases to find the offset | Not needed; name mentions gave a clear offset |
| YouTube captions are unusable | **Too harsh.** They beat Whisper on mapping in our one test, though they do mangle names and censor words |
| Whisper fixes player names and gives all-caps shouting | **False.** It also mangles names, and it doesn't capitalise shouting |
| Synthetic data gives an "identical result" | **False.** The model would learn an LLM's imitation of casters. Some providers' terms also restrict training models on their outputs |
| "5–10 matches" sweet spot, and 50+ overfits | **No evidence.** Overfitting to one caster comes from too few casters, not too much data |
| Commentary at 7-7 "locks up the half" | **Wrong for CS2.** The half comes after round 12 |
| `<analysis>` reasoning step | Rejected for live use: it adds delay and has no real training targets |
| Free T4 hosting | Not dependable for a live service |
| Llama 3 8B as the model | Newer small models are recommended more often now |

---

## 12. Risks

| Risk | Impact | Response |
|---|---|---|
| The real feed differs from the docs | Schema and training mismatch | Capture before building anything |
| HLTV access breaks, or an IP is banned | Live product stops | Keep usage gentle, keep a replay tool, and consider a licensed feed |
| Intensity can't be measured well | Excitement control is weak | Fall back on importance or hand-trained labels |
| Model invents details | Wrong commentary on air | Filter training data and add a live fact checker |
| Unofficial archive is altered or removed | Bad or lost training data | Verify against HLTV, and keep your own copy |
| YouTube's terms and caster rights | Limits on public release | Lower risk as a private project; check before publishing the model or dataset |
| Captions vary between VODs | Uneven dataset quality | Check per VOD, with Whisper as a fallback |
| Only Cologne is well checked | Style tied to one event's casters | Add other events if casters become a limit |

---

## 13. Test files

Test scripts and downloads from this work are in a temporary folder and will not last:

`/tmp/claude-1001/-home-dharmikpatel-Desktop-cs2-commentry/eb3770ca-4272-42bd-b245-e572cc535f61/scratchpad`

Its `extra_code_not_requested` folder holds an unfinished early extractor, with 2 of its 16 tests failing. It was written against a matchmaking demo before the pro demos were tested, so it shouldn't be treated as a working base. This file is the lasting record of the results.

## 14. Sources

**Feed:** [woq/hltv-bot data](https://github.com/woq/hltv-bot/blob/HEAD/docs/scorebot-data.md), [woq/hltv-bot transport](https://github.com/woq/hltv-bot/blob/HEAD/docs/scorebot-transport.md), [woq/hltv-bot Cloudflare notes](https://github.com/woq/hltv-bot/blob/HEAD/docs/cloudflare.md), [gigobyte/HLTV types](https://github.com/gigobyte/HLTV/blob/master/src/endpoints/connectToScorebot.ts), [gigobyte/HLTV issue #904](https://github.com/gigobyte/HLTV/issues/904), [hltv README](https://cdn.jsdelivr.net/npm/hltv@3.5.0/README.md), [hltv-python docs](https://github.com/bombsimon/hltv-python/blob/HEAD/DOCUMENTATION.md), [hltv.org scan](https://scanmalware.com/result/25aa20fc-ef74-41f8-948d-8d0f51526629)

**Demos:** [cs2povarchive/cs2-demos](https://huggingface.co/datasets/cs2povarchive/cs2-demos), [awpy-fixtures](https://github.com/pnxenopoulos/awpy-fixtures), [demoparser](https://github.com/LaihoE/demoparser), [cs2-pro-round-economy](https://huggingface.co/datasets/sneakyzero/cs2-pro-round-economy), [CS2 demo guide](https://csdb.gg/guides/demo-guide/), [How to download a CS2 demo](https://csdb.gg/blog/guides/how-to-download-and-watch-a-cs2-demo)

**VODs used in tests:** [B8 vs Vitality](https://www.youtube.com/watch?v=1bUFCQf0l28), [GamerLegion vs NRG](https://www.youtube.com/watch?v=2OZOsByVSfY), [Vitality vs MOUZ](https://www.youtube.com/watch?v=2ObF6lvJuOI), [FURIA vs Falcons](https://www.youtube.com/watch?v=p6U28DEBn8A), [FaZe vs NAVI Inferno](https://www.youtube.com/watch?v=vdXveK6G70g)

**Transcription:** [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization/), [WhisperX diarization guide](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/whisperx-diarization-word-level-timestamps), [WhisperX alignment guide](https://bestllmfor.com/guides/whisperx-alignment-diarization/)

**Models:** [Awesome open-weight models](https://github.com/phlx0/awesome-open-weight-models), [Fine-tuning guide](https://www.kunalganglani.com/blog/fine-tune-open-source-llm-lora-qlora), [Small LLMs that fit in 8 GB](https://pinggy.io/blog/small_llms_that_fit_in_8gb_memory/), [Best local LLM by VRAM](https://vdf.ai/blog/best-local-llm-by-vram/), [Qwen3-1.7B-cs2-commentators](https://huggingface.co/Dannys0n/Qwen3-1.7B-cs2-commentators), [cs2-commentators dataset](https://huggingface.co/datasets/Dannys0n/cs2-commentators), [EgoEsportsQA](https://arxiv.org/html/2604.12320v1), [League of Legends commentary](https://github.com/XXoussam/Esports-Event-to-Commentary-Generation-LoL), [cs2-ai-commentator](https://github.com/HullyMully/cs2-ai-commentator)

**Format:** [IEM Cologne Major 2026 guide](https://www.exitlag.com/news/iem-cologne-major-2026-full-guide/), [Cologne Major format preview](https://esportnow.gg/cs2/news/iem-cologne-major-2026-preview)
