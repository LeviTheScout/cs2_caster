# Dataset and Fine-Tuning Plan

There are two projects. The dataset is the larger and riskier one. Fine-tuning depends on it, but its setup can start early.

The details behind each step are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

---

## Project 1: Caption-aligned dataset

**Goal:** about 3,000–5,000 training pairs from 40–60 maps. Each pair is one input frame (game state and recent events) and the caster line that goes with it.

**Version 1 leaves out intensity labels.** They need their own labelling work, so they come in version 2.

### Steps

1. **Pick the maps**
   - Choose 5 pilot maps with mixed stages and casters, mostly from Cologne.
   - Check whether `cac2026` has English VODs with captions. If not, choose another non-Major event.
2. **Collect the data**
   - Demo files from the Hugging Face archive.
   - YouTube captions in json3 format.
   - A table that maps team and player names to their HLTV versions.
   - A match card for each map.
3. **Parse each demo** into a timeline of events and game state, then cut that down to an HLTV-only version.
4. **Clean the captions**
   - Remove tags and censoring.
   - Fix misheard player names using the rosters.
5. **Align each map**
   - Find the time offset between the demo and the VOD from name matches.
   - Check rounds that fit a different offset.
   - Join split p1/p2 demo files.
6. **Pair events with lines**
   - Pair a short group of events with the caption span that covers it. Start the window slightly before the event.
   - Pair freeze-time talk with a round summary.
   - Drop unmatched kills, desk chat, sponsor reads and replay talk.
7. **Tag each line** with the data it relies on: kill, bomb, economy, utility, position or history.
8. **Build the tiers**
   - Mix 70% HLTV tier and 30% richer tier.
   - A target line may only mention facts present in its input.
   - Mark missing sections `NOT REPORTED`.
9. **Split the data** by series. Hold out 4–5 series, including some from the non-Major event.
10. **Check by hand**
    - Read 100 random pairs and record what share are correct.
    - Fix whatever is causing the errors.

**Output:** JSONL files (train, validation, test) and a short data card describing sources, counts and known issues.

**Stages:** 5 maps → 20 maps → 40–60 maps. Grow only once the hand check passes at the current size.

**Done when:** a random sample of pairs reads as correct, and the share of lines that rely on missing data is known.

---

## Project 2: Fine-tuning

**Goal:** a fine-tuned model that beats the untuned baseline on held-out series and runs fast enough on your PC.

### Steps

1. **Before any training** (can start during Project 1)
   - Run the speed test on your PC with stock models: Qwen3-4B-Instruct-2507, Gemma 4 E4B, Qwen3-8B and Qwen3-30B-A3B. Check the RAM type, measure speed with `llama-bench`, and confirm prompt-cache reuse.
   - Run the untuned few-shot baseline on held-out frames.
2. **Set up training**
   - Unsloth QLoRA on a free Kaggle or Colab T4.
   - Use one fixed chat template.
   - Save checkpoints to Drive or Kaggle output.
3. **First run** on the 20-map data:
   - Train Qwen3-4B-Instruct-2507 and Gemma 4 E4B.
   - Use the same data and the same settings for both.
4. **Evaluate** on the held-out series:
   - Format: the output parses every time.
   - Inventions: the fact checker, plus the demo ground truth for utility.
   - Blind preference against the baseline and against real caster lines.
   - Repetition.
5. **Experiments, one at a time**
   - Tier mix: 70/30 vs 30/70.
   - 4B vs 8B.
   - Then version 2 of the data with intensity labels.
6. **Export and test live-like**
   - Merge the fine-tune into the model and convert it to GGUF (Q5_K_M).
   - Replay recorded matches on your PC and measure time per line.

**Done when:** the best model beats the baseline on held-out series and meets the speed target. If it can't, you know why.

---

## Order

```
Project 1:  pilot 5 maps ──► 20 maps ──────────► 40–60 maps
Project 2:  speed test + baseline ──► first runs on 20 maps ──► experiments
```

Neither has to finish before the other starts: Project 2's setup runs during the pilot, and training starts at 20 maps.
