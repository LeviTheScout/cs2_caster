# Demo fields: what the parser gives for each event

Checked on 2026-11-03 with `demoparser2` 0.42.1 on two pilot demos:
- an ESL Cologne demo (FURIA vs Falcons, Mirage)
- a Perfect World CAC demo (MOUZ vs TYLOO, Inferno)

✅ means present in that demo, ❌ means missing.

**Extra columns:** any event can have extra columns added at parse time:
- `player=[...]`: state of the event's players at that tick, e.g. `X`, `Y`, `Z`, `last_place_name`, `health`, `team_name`, `team_clan_name`. Each gets the prefix `attacker_`, `user_` and so on.
- `other=[...]`: game-wide values, e.g. `game_time`, `total_rounds_played`, `is_warmup_period`.

## Kills and damage

| Event | Fields | ESL | PW |
|---|---|---|---|
| `player_death` | `tick`, `attacker_name`/`_steamid`, `user_name`/`_steamid` (the victim), `assister_name`/`_steamid`, `assistedflash`, `weapon`, `headshot`, `hitgroup`, `distance`, `dmg_health`, `dmg_armor`, `penetrated` (wallbang), `thrusmoke`, `attackerblind`, `attackerinair`, `noscope`, `dominated`, `revenge`, `wipe`, `noreplay`, weapon item ids | ✅ | ✅ |
| `player_hurt` | `tick`, `attacker_name`, `user_name`, `weapon`, `dmg_health`, `dmg_armor`, `health` and `armor` left, `hitgroup` | ✅ | ✅ |
| `weapon_fire` | `tick`, `user_name`, `weapon`, `silenced` | ✅ | ✅ |

## Bomb

| Event | Fields | ESL | PW |
|---|---|---|---|
| `bomb_beginplant` | `tick`, `user_name`, `site` | ❌ | ✅ |
| `bomb_planted` | `tick`, `user_name`, `site` | ✅ | ✅ |
| `bomb_begindefuse` | `tick`, `user_name`, `haskit` | ❌ | ✅ |
| `bomb_defused` / `bomb_exploded` | `tick`, `user_name`, `site` | ✅ | ✅ |
| `bomb_dropped` / `bomb_pickup` | `tick`, `user_name` | ✅ | ✅ |

`site` is a map entity number (e.g. 81, 429), not A or B. To get A or B, use the planter's `last_place_name` (`BombsiteA` or `BombsiteB`).

## Rounds

| Event | Fields | ESL | PW |
|---|---|---|---|
| `round_start` (added by the parser) | `tick`, `round` | ✅ | ✅ |
| `round_freeze_end` (buy time over, round live) | `tick` | ✅ | ✅ |
| `buytime_ended` | `tick` | ❌ | ✅ |
| `round_end` (added by the parser) | `tick`, `round`, `winner` (CT/T), `reason` (e.g. `ct_killed`, `bomb_exploded`, `bomb_defused`, `t_killed`, `time_ran_out`) | ✅ | ✅ |
| `round_announce_match_point`, `round_announce_last_round_half` | `tick` | ✅ | ✅ |
| `cs_win_panel_match` (map over) | `tick` | ✅ | ✅ |

## Utility

| Event | Fields | ESL | PW |
|---|---|---|---|
| `smokegrenade_detonate` / `_expired` | `tick`, `user_name` (thrower), `x`, `y`, `z` (landing point), `entityid` | ✅ | ✅ |
| `flashbang_detonate` | `tick`, `user_name`, `x`, `y`, `z`, `entityid` | ✅ | ✅ |
| `hegrenade_detonate` | `tick`, `user_name`, `x`, `y`, `z`, `entityid` | ✅ | ✅ |
| `inferno_startburn` / `_expire` (molotov or incendiary) | `tick`, `user_name` (thrower), `x`, `y`, `z`, `entityid` | ✅ | ✅ |
| `player_blind` | `tick`, `attacker_name` (who flashed), `user_name` (who was blinded), `blind_duration` (s), `entityid` | ❌ | ✅ |
| `grenade_thrown` | `tick`, `user_name`, `weapon` | ✅ | ❌ |

**Utility damage:** `player_hurt` with weapon `hegrenade`, `inferno` or `molotov`.

## Equipment

| Event | Fields | ESL | PW |
|---|---|---|---|
| `item_pickup` | `tick`, `user_name`, `item`, `defindex`, `silent` | ✅ | ✅ |
| `item_equip` | weapon switches | ❌ | ✅ |
| Purchases | **no event in either demo.** Work them out from inventory changes during buy time | ❌ | ❌ |

## Player state at any tick (`parse_ticks`)

Per player:
- `name`, `steamid`
- `team_name` (the side, CT/T), `team_clan_name` (the team)
- `health`, `armor_value`, `has_helmet`, `has_defuser`
- `balance` (money)
- `is_alive`
- `X`, `Y`, `Z`
- `last_place_name` (the map's callout, e.g. `Banana`, `Jungle`, `SnipersNest`)
- `inventory` (all items, including grenades)

## Round end and running stats

**Round-end events:**
- `round_end`: winner and reason, from the parser.
- `round_officially_ended`: PW only.
- `cs_win_panel_match`: map over.

There is **no round MVP event**. The MVP is the player whose `mvps` counter went up.

**Per-player running totals**, in both demos. Read them with `parse_ticks` at each `round_end` tick:
- `kills_total`, `deaths_total`, `assists_total`, `headshot_kills_total`
- `damage_total`, `utility_damage_total`, `enemies_flashed_total`
- `mvps`, `score`, `objective_total`, `alive_time_total`
- `balance`, `cash_spent_this_round`, `total_cash_spent`, `cash_earned_total`, `kill_reward_total`, `money_saved_total`
- `round_start_equip_value`, `current_equip_value`, `equipment_value_total`

**Per-team values:** `team_rounds_total` (the score), `team_score_first_half`, `team_score_second_half`.

A single round's stats are the difference between two consecutive round ends.

**What code works out from these:** multi-kills, entry kills, trades, clutches (1vX), survivors and saved weapons, KAST, ADR, loss streak and loss bonus, and team money for the next buy.

ADR worked out from `player_hurt` matched HLTV's published figure to within 1 (tested 2026-11-02).

## Noise to ignore

`player_footstep`, `player_jump`, `player_sound`, `weapon_reload`, `weapon_zoom`, `fire_bullets`, `hltv_chase`, `hltv_fixed`, `server_cvar`, `chat_message`, `player_ping`.

## Traps

- **Weapon names:**
  - `m4a1` is the **M4A4**, and `m4a1_silencer` is the M4A1-S.
  - `weapon_fire` names include the `weapon_` prefix, e.g. `weapon_knife_butterfly`.
- **Side vs team:** `team_name` is the side, which swaps at half. Use `team_clan_name` for the team.
- **Missing events:** an event a demo doesn't record parses as an empty list, not an empty table. Check before using it.
- **Organiser differences:** event lists differ by organiser, as the ✅/❌ columns show. The converter must work when an event is missing.
