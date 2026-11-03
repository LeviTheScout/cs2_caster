# Open models for CS2 commentary: CPU inference speed, fine-tuning practicality, and evidence on small models learning a short style task (researched 2026-11-02)

Scope note: these are research notes for a report writer. "Measured" means a third party ran the benchmark. "Estimate" means I derived it from bandwidth or compute arithmetic. I found no published llama.cpp benchmark for the exact target CPU (i7-13700K), so every 13700K number below is an estimate. arxiv.org could not be fetched in this session, so the arXiv items rely on search-result abstracts and summaries. The official EQ-Bench and Kaggle docs pages render with JavaScript and could not be read.

---

## Q1. Measured llama.cpp CPU speeds (pp512 / tg128) on desktop x86, the bandwidth rule, DDR4 vs DDR5, and P-core vs E-core threads

### Takeaway
On a dual-channel DDR5 desktop, MoE models with about 3B active parameters (Qwen3-30B-A3B, gpt-oss-20b) generate about 16–30 tok/s on CPU, roughly twice the speed of a dense 8B model. LFM2-8B-A1B, with about 1.5B active, decodes at about 70–75 tok/s in the vendor's own CPU benchmark. Token generation is limited by memory bandwidth, so DDR4-3200 costs about 40% of tg versus DDR5-5600. Prompt processing (pp) is limited by compute, and that is the real latency risk for 500–1,500-token prompts. On Intel hybrid CPUs, pin generation threads to the P-cores.

### Cited Findings

**MoE, about 3–4B active**
- **Qwen3-30B-A3B, AMD Ryzen 9 9950X (measured, community, about mid-2025, older data):** CPU-only on 2×48 GB DDR5-6400 (about 88 GB/s measured RAM bandwidth), the author reports ">500 tok/s prompt processing and >30 tok/s token generation". This was a tuned rig using the ik_llama.cpp fork and ubergarm's custom quants, so treat it as a best case. — [ubergarm/Qwen3-30B-A3B-GGUF discussion #2](https://huggingface.co/ubergarm/Qwen3-30B-A3B-GGUF/discussions/2)
- **Qwen3 30B-A3B, unnamed CPU, dual-channel DDR5 (reported, about May 2025, older data):** DDR5-5600 with a Q4 quant gives 18–20 tok/s. DDR5-6000 with Q8 gives 25 tok/s. No CPU model or pp numbers are given. — [Hardware Corner Qwen3 hardware guide](https://www.hardware-corner.net/guides/qwen3-hardware-requirements/)
- **Qwen3-Coder 30B, consumer CPUs (anecdotal, 2025):** users with a "Ryzen 9 7950X3D or even a Ryzen 5 5600G & 32GB RAM" report 12–15 tok/s. The quant is not stated. — [Arsturn guide](https://www.arsturn.com/blog/running-qwen3-coder-30b-at-full-context-memory-requirements-performance-tips)
- **Qwen3-30B-A3B Q4_K_M, Radxa Orion O6 (12-core ARM board; measured, community, 2025):** tg128 about 16 tok/s and pp512 about 23 tok/s at -t 7. The Instruct-2507 variant gets tg about 15. Another quant reaches about 20 tg. This is not x86, but it shows pp can be very low on weak SIMD hardware. — [Radxa forum llama.cpp benchmarks](https://forum.radxa.com/t/llama-cpp-benchmarks/27813)
- **gpt-oss-20b Q4_K_M, AMD Threadripper 1950X (16-core Zen 1, capped at 3.6 GHz; measured, 2025-10-17):**
  - Mainline llama.cpp at 16 threads: pp512 about 56.8 tok/s, tg128 14.3–15.6 tok/s.
  - ik_llama.cpp fork: pp512 about 88–92 tok/s, tg128 16–18.5 tok/s.
  - The best everyday preset (`-fa -fmoe`, f16 KV) gives about 91.7 pp512 and 17.0 tg128.
  - "Scaling beyond 16 cores didn't help decode, 14–16 was best." Flash attention gave about 3–5% prefill gain and no decode gain. q8_0 K-cache costs "≤1%" quality.
  - RAM type was not stated, and the memory was not fully quad-channel.
  — [Nikro, "CPU-only LLM Inference"](https://nikro.me/articles/professional/cpu-only-llm-inference/)
- **Qwen3-Coder-Next (80B, 3B active), CPU path (GitHub issue, 2026):** the issue says CPU inference is "~5x slower than expected". It estimates that a 3B-active model on DDR5-5600 "should" reach about 20–30+ tok/s in theory. Mainline CPU MoE paths can fall well short of the bandwidth ceiling for newer architectures. — [llama.cpp issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480)

**MoE, about 1B active (LFM2)**
- **LFM2-8B-A1B, Q4_0, llama.cpp CPU backend, AMD Ryzen AI 9 HX 370 laptop (vendor-measured, LFM2 Technical Report, Nov 2025):**
  - The table row reads "1,320 · 1,185 · 74.9 · 69.1". I interpret it as prefill about 1,320 / 1,185 tok/s and decode about 74.9 / 69.1 tok/s at 1K / 4K context. The column split is my reading of the snippet.
  - Per the report, LFM2-8B-A1B matches Granite-4.0-H-Tiny on prefill, with 1.1–1.3× higher decode. It has about 2.8–3.7× higher decode than dense 4B-class baselines such as Qwen3-4B.
  — [LFM2 Technical Report (arXiv 2511.23404)](https://arxiv.org/html/2511.23404v1)
- **Contradiction:** a third-party guide reports only about 19.4 tok/s for LFM2-8B-A1B on the same HX 370, with an unclear setup. The search summary did not say which of these two pages it came from. — [tinyweights.dev guide](https://tinyweights.dev/posts/run-lfm2-5-8b-a1b-locally/) / [xugj520.cn article](https://www.xugj520.cn/en/archives/lfm2-8b-a1b-sparse-moe-mobile.html)
- **LFM2.5-8B-A1B (successor, same architecture; vendor, 2026):** decodes 253 tok/s on an M5 Max and 146 tok/s on a Ryzen AI Max+ 395, using under 6 GB. These are not desktop x86 DDR5 figures. — [Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b)

**Dense 7–9B**
- **Qwen3-8B, Ryzen 9 7950X, 64 GB DDR5-6000 dual-channel, CL36, llama.cpp b9838 (page dated 2026-07-09):**
  - The page claims pp512 210.3 ± 3.1 and tg128 26.4 ± 0.3 for Q4_K_M (4.90 GiB) at 16 threads.
  - It also claims tg 22.8 for Q5_K_M and 18.4 for Q8_0.
  - **Treat as unreliable.** 26.4 tok/s × 5.26 GB = about 139 GB/s of weight reads per second, which is more than the 96 GB/s theoretical maximum of dual-channel DDR5-6000.
  - The page also has internal inconsistencies: the TTFT does not match pp512, the end-to-end time ignores TTFT, and VRAM is mislabeled.
  - The Q4→Q5→Q8 tg ratios are directionally plausible.
  — [markaicode benchmark](https://markaicode.com/benchmarks/llamacpp-tokens-per-second-benchmark/)
- **Older AVX2 comparisons, LLaMA-3.1-8B Q4_K_S, 7950X at 16 threads (about early 2025, older data):** a llama.cpp-vs-ik_llama.cpp table exists in this discussion. A commenter notes that the 7950X is "much faster" than a laptop 8945HS. I did not extract the exact numbers. — [ik_llama.cpp discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)

**Intel hybrid (P-core / E-core) threading**
- **12th-gen Intel:** a user reports about 3× better performance using P-cores only. The E-cores bottleneck the P-cores, which "finish their work earlier and have to wait on the E-cores". A 13600KF user saw similar losses. (2023, older data, pre-dates many llama.cpp threading changes.) — [llama.cpp discussion #572](https://github.com/ggml-org/llama.cpp/discussions/572)
- **i9-13900K on Win11:** about 50% CPU utilisation with all threads. One user got the best result at 10 threads with work forced onto the 8 P-cores. (2023, older data.) — [llama.cpp issue #842](https://github.com/ggerganov/llama.cpp/issues/842)
- **Thread scaling:** token generation is memory-bandwidth-bound. 12 threads (all logical) was 2.4% slower than 8. — [InventiveHQ thread-scaling test](https://inventivehq.com/blog/cpu-thread-scaling-llm-inference)
- **Default behaviour:** on a 13th-gen hybrid laptop, llama.cpp used all 14 threads and left P/E placement to the OS scheduler. — [CEUR-WS Vol-4164 paper 11](https://ceur-ws.org/Vol-4164/paper11.pdf)
- **llama-server thread flags:** `-t/--threads` sets "threads to use during generation" (default -1). `-tb/--threads-batch` sets threads "during batch and prompt processing (default: same as --threads)". Prefill and decode can therefore use different thread counts. (Fetched 2026-11-02.) — [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

### Inferences

**Bandwidth rule (calculation):**
- Theoretical bandwidth = MT/s × 8 bytes × channels:
  - DDR4-3200 dual-channel = 51.2 GB/s
  - DDR5-4800 dual-channel = 76.8 GB/s
  - DDR5-5600 dual-channel = 89.6 GB/s
  - DDR5-6000 dual-channel = 96 GB/s
- The i7-13700K officially supports up to DDR5-5600 or DDR4-3200, per Intel's published spec. I did not re-verify that spec this session.
- Achievable bandwidth is about 70–90% of theoretical. The 9950X example measured about 88 of 102.4 GB/s.
- tg ceiling ≈ effective bandwidth ÷ bytes of weights read per token. For MoE, use active-expert bytes plus shared layers.
- Measured tg usually lands at about 50–80% of that ceiling.

**Estimated tg128 for an i7-13700K at -t 8.** Assumes about 60–70 GB/s effective on DDR5-5600 and about 38–45 GB/s on DDR4-3200. These are estimates, not measurements.

| Model (quant) | Bytes/token (approx.) | tg on DDR5-5600 | tg on DDR4-3200 |
|---|---|---|---|
| Llama-3.2-3B / SmolLM3-3B Q4_K_M | ~2.0 GB | ~20–30 | ~13–20 |
| Qwen3-4B / Phi-4-mini Q4_K_M | ~2.5 GB | ~17–25 | ~11–16 |
| Dense 8B Q4_K_M (Qwen3-8B, Llama-3.1-8B, Ministral-8B) | ~5 GB | ~9–13 | ~6–8 |
| Dense 8B Q5_K_M | ~5.8 GB | ~8–11 | — |
| Qwen3-30B-A3B Q4_K_M (~3.3B active) | ~2 GB | ~18–28 | ~12–18 |
| gpt-oss-20b MXFP4 (~3.6B active) | ~2 GB | ~16–28 | — |
| LFM2-8B-A1B Q4 (~1.5B active) | ~0.9 GB | ~40–70 | — |

- The 13700K has no AVX-512, while Zen 4/5 and the HX 370 do. Vendor and 9950X numbers will therefore overstate 13700K prefill, and probably have less effect on decode.
- **RAM footprint:** a dense 8B and all 3–4B models fit easily in 32 GB. Qwen3-30B-A3B Q4_K_M (about 17–19 GB) and gpt-oss-20b (about 12–13 GB) fit, but leave less headroom if CS2, a demo parser or OBS share the box. These file sizes are approximate and were not re-verified.
- **Thread recommendation:** start with `-t 8` (P-cores only) for generation. Sweep `-tb` over 8, 16 and 24 for prompt processing. Prefill is compute-bound, so E-cores may help there even though they hurt decode. Settle it with one `llama-bench -t 8,12,16,24` run on the real machine.

### Gaps
- No measured llama.cpp pp512/tg128 for an i7-13700K, i9-13900K, i9-14900K or Ryzen 7000 desktop on any candidate model was found with trustworthy numbers. The one 7950X/Qwen3-8B source is physically implausible on tg.
- No desktop-x86 CPU numbers were found for:
  - Gemma 4 E2B/E4B
  - Granite 4.0-H (only the vendor's relative comparison above)
  - Nemotron Nano 9B v2 (hybrid Mamba)
  - Ministral 8B
  - SmolLM3-3B
- Hybrid Mamba CPU speed on x86 is therefore unmeasured in these notes.
- I did not extract the exact numbers from ik_llama.cpp discussion #164, which is the best older 7950X source.
- Whether the user's machine has DDR4 or DDR5 is the single biggest unknown for tg. It is a roughly 1.6–1.75× swing.

---

## Q2. Prompt caching and KV reuse in llama.cpp server, and a per-line latency estimate

### Takeaway
llama-server keeps the KV cache for the last prompt in each slot by default (`cache_prompt: true`). On the next request it evaluates only the suffix that differs. If the prompt is laid out so that only the newest events change, per-line prefill drops from about 1,000+ tokens to about 100–250. That is the difference between missing and meeting the 2–3 s target for 3–4B dense and A3B MoE models on this CPU.

### Cited Findings
All items below come from the [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), fetched 2026-11-02.
- **`cache_prompt`:** "Re-use KV cache from a previous request if possible." Default `true`. "The common prefix does not have to be re-processed, only the suffix that differs between the requests." The README warns that it "can cause nondeterministic results" because logits may differ across batch sizes.
- **Server flag `--cache-prompt` / `--no-cache-prompt`:** prompt caching is enabled by default.
- **Response fields:** `tokens_cached` is the number of prompt tokens reused. `tokens_evaluated` is the total evaluated. These can be used to verify cache hits in production.
- **`--cache-reuse N` / `n_cache_reuse`:** "min chunk size to attempt reusing from the cache via KV shifting". Default 0 (disabled). It lets the server reuse matching chunks that are not a strict prefix.
- **`--cache-ram N`:** maximum size in MiB of a host-RAM prompt cache. Listed default 8192; -1 means unlimited, 0 disables it. **`--cache-idle-slots`** saves idle slots into that cache (default on).
- **`--slot-save-path PATH`:** enables saving slot KV caches to disk.
- **`-np/--parallel`:** number of slots (default auto). **`id_slot`** pins a request to a slot. **`-sps/--slot-prompt-similarity`** (default 0.10) controls which slot a request is routed to by prompt match.

### Inferences
**Formula (estimate):**
latency ≈ HTTP/tokenise/sample overhead (~0.1–0.3 s) + N_new_prompt_tokens ÷ pp_speed + N_output_tokens ÷ tg_speed.
- N_output ≈ 1 intensity token + 20–40 words-as-tokens + EOS, so about 30–45 tokens.

**Prompt layout to maximise reuse:**
- Put fixed content first: system/style instructions (which a fine-tune can largely drop), then the match card. Put the round state next, and the newest events and previous lines last.
- Any change early in the prompt forces re-evaluation of everything after it.
- A sliding window of previous lines placed before the events will invalidate the cache every call. Either append-only within a round, or put the rolling history at the very end.

**Estimated per-line latency on an i7-13700K with DDR5.** pp ranges are my compute-scaling estimates from the 1950X, 9950X and HX 370 data above, not measurements. tg ranges come from the Q1 table. Output is 42 tokens.

| Model | pp est. (tok/s) | tg est. (tok/s) | No cache, 1,000-token prompt | Cached, ~150 new tokens |
|---|---|---|---|---|
| 3B dense Q4_K_M | 100–400 | 20–30 | ~4–12 s | **~1.6–3.6 s** |
| 4B dense Q4_K_M | 80–300 | 17–25 | ~5–15 s | **~2.2–4.5 s** |
| 8B dense Q4_K_M | 40–150 | 9–13 | ~10–30 s | ~4.5–8.5 s (misses target) |
| Qwen3-30B-A3B Q4_K_M | 60–250 | 18–28 | ~5.5–19 s | **~2.1–4.8 s** |
| gpt-oss-20b | 60–200 | 16–28 | ~6–19 s | ~2.3–5.2 s* |
| LFM2-8B-A1B Q4 | 300–1,000 | 40–70 | ~1.6–4 s | **~0.8–1.5 s** |

\* gpt-oss uses the harmony format and an analysis/reasoning channel. Any reasoning tokens add directly to the tg term, so the fine-tune or chat template must suppress them. The same applies to Qwen3 hybrid-thinking models: thinking must be disabled.

- **Conclusion:** only the ≤4B dense models, the A3B MoEs and the A1B MoE hit 2–3 s on CPU, and only with cache reuse working. LFM2-8B-A1B is the only candidate with comfortable margin even without caching.
- **8B dense on CPU** is about 2–3× over budget. It is the natural Modal L4 fallback. An L4 would run tg about 40–60+ tok/s for 8B Q4. That L4 figure is my estimate and was not researched here.
- **Pipelining:** generate line N+1 while TTS speaks line N, the pattern Kawamatsu et al. use (see Q5). This hides most of the latency regardless of model.

### Gaps
- **Possible risk (not verified this session):** prefix-reuse behaviour may differ for models whose state is not a plain transformer KV cache. This would affect:
  - gpt-oss-20b and Gemma (sliding-window attention)
  - LFM2 (conv/attention hybrid)
  - Granite-4.0-H and Nemotron-H (Mamba hybrids)
  For recurrent/SWA layers, llama.cpp cannot simply truncate the cache to an arbitrary common prefix. It may fall back to full re-processing unless checkpoint/full-SWA options are used. Check `tokens_cached` on the actual model before relying on cache hits.
- Ollama's exact prefix-reuse behaviour and its exposure of `cache_prompt` / `--cache-reuse` were not researched. Using llama-server directly gives explicit control.
- No measured pp512 for the 13700K exists in these notes. The pp term dominates uncached latency and is the least certain number.

---

## Q3. QLoRA on a free T4 (16 GB): what fits, time per epoch, Kaggle vs Colab limits, and no bf16

### Takeaway
Every dense candidate up to 14B fits QLoRA on a 16 GB T4 with Unsloth. 27B+ dense and the 30B-A3B MoE do not fit on one T4. The T4 has no bf16 (and no FP8), so training runs in fp16. My estimate for one epoch of about 4,000 × 1,500-token examples is about 2–4 h for a 4B model and about 4–8 h for an 8B model. That fits Kaggle's roughly 30 GPU-h/week only with checkpointing across sessions for 8B.

### Cited Findings
- **Unsloth minimum VRAM, QLoRA 4-bit (no date stated on page; fetched 2026-11-02):** 3B 3.5 GB, 7B 5 GB, 8B 6 GB, 9B 6.5 GB, 11B 7.5 GB, 14B 8.5 GB, 27B 22 GB, 32B 26 GB. The page notes "some models need more VRAM than shown" and suggests batch size 1–3 on OOM. Minimum GPU is CUDA capability 7.0 (T4 included). — [Unsloth requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
- **Qwen3-14B on a free T4:** Unsloth says it fits on a free Colab Tesla T4 (16 GB) and provides free notebooks. — [Unsloth Qwen3 run & fine-tune docs](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)
- **Community example:** a notebook fine-tunes Qwen3-14B on a free Colab T4 and exports GGUF, merged 16-bit/4-bit and LoRA formats. — [asshejan GitHub repo](https://github.com/asshejan/Fine-Tuning-Qwen3-14B-and-Chat-with-Unsloth-)
- **T4 lacks FP8:** an Unsloth maintainer said a Qwen3-8B FP8 GRPO notebook was wrongly labelled for T4, because "free Colab NVIDIA Tesla T4 GPUs don't support FP8". It needs at least an L4. The maintainer also said about 99% of their other notebooks target free T4s. (2025/2026.) — [Unsloth issue #3669](https://github.com/unslothai/unsloth/issues/3669)
- **Kaggle quota (secondary sources, conflicting):**
  - Weekly GPU quota is about 30 h, "sometimes higher", reset weekly. — [paradox.engineer](https://www.paradox.engineer/resources/kaggle-free-gpu-tpu-compute)
  - Another page is titled "50 Hours Weekly Quota", probably GPU + TPU combined. — [resourify](https://resourify.com/resources/kaggle)
  - Sessions are cited as 12 h by some sources and 9 h by others. One blog headline says "Kaggle gave you 12 hours" and discusses long jobs dying. — [LuminoAI blog](https://www.luminoai.in/blog/kaggle-gave-you-12-hours-your-training-job-needed-more)
  - Accelerators are a P100 or T4 / 2×T4.
  - One source claims 2×T4 consumes 2 quota-hours per wall-clock hour. This is unverified and contradicts common understanding, so check it.
  - A 2026 comparison says Kaggle lets users link a Colab Pro subscription to raise the GPU quota. — [Kaggle vs Colab 2026 (Medium)](https://lalatenduswain.medium.com/kaggle-vs-google-colab-which-cloud-notebook-platform-should-you-choose-in-2026-da053a02fcb7); [AIMultiple free cloud GPU list](https://aimultiple.com/free-cloud-gpu)
- **Colab free:** no published guarantee. Third-party claims put it at "15–30 hours weekly on T4/P100 with 12-hour session limits", subject to Google's discretion. — [Kaggle vs Colab 2026 (Medium)](https://lalatenduswain.medium.com/kaggle-vs-google-colab-which-cloud-notebook-platform-should-you-choose-in-2026-da053a02fcb7); [Alibaba free-GPU explainer](https://electronics.alibaba.com/question/free-gpu-access-in-2026-real-options-limits)

### Inferences
**Fit at sequence length 2048 (estimate):**
- 3–4B: very comfortable; batch 4–8 with gradient checkpointing.
- 7–9B: fits; batch 1–4 with Unsloth's gradient checkpointing.
- 14B: fits per Unsloth, but tight at 2048.
- Does not fit on one T4:
  - Qwen3-30B-A3B. Total parameters drive memory at about 4.5 bits per weight, so roughly 17–18 GB or more, which exceeds 16 GB.
  - 27B+ dense.
- gpt-oss-20b QLoRA on a T4 is borderline. See Gaps.

**Time per epoch (estimate):**
- 4,000 × 1,500 = 6M tokens per epoch.
- With frozen 4-bit base weights, LoRA backward still needs activation gradients, and gradient checkpointing re-runs the forward pass. That makes about 6 × N_params FLOPs per token.
- T4 fp16 tensor peak is about 65 TFLOPs. Assume about 15–25% effective utilisation, because 4-bit dequant overhead and short batches hurt throughput.
- **4B:** 6 × 4e9 × 6e6 = 1.4e17 FLOPs ÷ 10–16 TFLOPs ≈ **2.5–4 h/epoch**.
- **8B:** about **5–8 h/epoch**.
- **3B:** about 2–3 h.
- **1–1.5B:** under 1.5 h.
- Masking the loss to responses only (`train_on_responses_only`) does not cut compute much, because the prompt is still forwarded and back-propagated through. Shortening the prompt, for example to about 800 tokens, is the main lever and roughly halves time.
- At 3–5k examples, 2–3 epochs is typical. A 4B model therefore needs about 6–12 GPU-h, which is one Kaggle week. An 8B needs about 12–24 GPU-h, so multiple sessions with checkpoints written to /kaggle/working or Drive.

**No bf16 on T4:**
- Training uses fp16 with loss scaling. Unsloth's `is_bfloat16_supported()` returns False on T4, and the notebooks then set `fp16=True`.
- Risks are overflow in models whose activations exceed the fp16 range, and slightly noisier optimisation.
- Models released in bf16 are upcast or downcast on load.
- These points come from general knowledge of Turing GPUs and the Unsloth notebook pattern; I did not re-verify them in a fetched source this session.

### Gaps
- No official Kaggle or Colab doc page could be read; the Kaggle docs page rendered empty. The exact 2026 weekly quota, session cap, and 2×T4 accounting are unverified.
- No measured Unsloth T4 wall-clock times for 4B/8B at sequence length 1,500–2,048 were found.
- Unsloth gpt-oss-20b QLoRA on a free T4 was not verified. I recall Unsloth advertising a T4-fitting gpt-oss-20b notebook in Aug 2025, but I found no source this session.
- Unsloth support status was not checked for:
  - LFM2 / LFM2-8B-A1B
  - Granite 4.0-H
  - Nemotron Nano 9B v2
  - Gemma 4 E2B/E4B
  - SmolLM3
  4-bit QLoRA of MoE or Mamba-hybrid architectures is generally less mature than for dense transformers. I flag this as a risk; it was not verified.
- Gemma-family fp16 overflow on T4 (reported historically for Gemma 3) was not re-verified for Gemma 4.

---

## Q4. Exporting the QLoRA fine-tune to GGUF, and Q4_K_M vs Q5_K_M vs Q8_0 quality loss

### Takeaway
The standard path is to merge the LoRA into the 16-bit base, convert to GGUF, then quantise. Unsloth does all three in one call, and the free-T4 notebooks already include it. Measured KL divergence on 7–8B models shows Q4_K_M noticeably worse than Q5_K_M, and Q8_0 near-lossless. Small models lose more per bit. For a 3–4B fine-tune, Q5_K_M, Q6_K or even Q8_0 is affordable on CPU. For 8B, Q4_K_M or Q5_K_M is the speed and quality tradeoff.

### Cited Findings
- **Unsloth export formats:** Unsloth's fine-tuning flow exports to GGUF, merged 16-bit and 4-bit, and LoRA adapters for llama.cpp/Ollama. The model cards describe GGUF, 4-bit and 16-bit safetensors exports. — [asshejan repo](https://github.com/asshejan/Fine-Tuning-Qwen3-14B-and-Chat-with-Unsloth-); [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)
- **Llama-3.1-8B KL divergence vs full precision (blog, measured):**
  - Mean KL: Q8_0 0.0014, Q5_K_M 0.0108, Q4_K_M 0.0282.
  - Top-token agreement: 97.7%, 96.0% and 91.9% respectively.
  - Q4_K_M "disagrees about the top token roughly once in twelve tokens".
  — [bshadmehr.me](https://bshadmehr.me/writing/quantization-q4-q8-explained/)
- **Mistral-7B on wiki.test (gist, about 2024, older data):** median KL is 0.0075 for Q4_K_M (4.83 bpw) and 0.0043 for Q5_K_M (5.67 bpw). The Q8_0 row in that table is corrupt (881.95). — [Artefact2 gist](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)
- **Imatrix and source precision:** Q4_K_M quantised from fp16 vs fp32 sources "trade blows" in llama.cpp KLD runs. — [bartowski HF post](https://huggingface.co/posts/bartowski/608656345183499)
- **Small-model penalty (blog, 2026-10):** on a 135M model, perplexity rose 5.6% for Q4_K_M, 4.1% for Q5_K_M and 1.0% for Q6_K. The post says small models "break the rules". — [logeshwaran.org](https://www.logeshwaran.org/2026/10/gguf-quantization-types-explained.html)
- **Qwen3-4B (unverified secondary claim):** Q4_K_M loses 6.4% on HumanEval vs Q8. The methodology is unclear. — [bestllmfor.com](https://bestllmfor.com/guides/quantization-q4-vs-q5-vs-q8-quality/)
- **Speed cost of higher quants:** Qwen3-8B CPU tg falls from 26.4 (Q4_K_M) to 22.8 (Q5_K_M) to 18.4 (Q8_0). The absolute numbers are suspect (see Q1), but the ratios track file size. — [markaicode](https://markaicode.com/benchmarks/llamacpp-tokens-per-second-benchmark/)

### Inferences
- **Effect on short lines:** at Q4_K_M's roughly 92% top-token agreement, a 30-token line would see about 2–3 argmax differences on average.
  - With sampling at temperature above 0, this mostly shows up as subtle stylistic drift rather than errors.
  - It can, however, flip the leading intensity token or a player name. Constrain these with a grammar, or validate them against the frame.
- **Recommendation:**
  - 3–4B: use Q5_K_M or Q6_K. The tg cost versus Q4_K_M is about 15–25%, still within budget.
  - 8B: Q4_K_M is forced by speed.
  - MoE A3B: Q4_K_M is the usual choice.
- Run `llama-perplexity --kl-divergence` on the fine-tuned model against held-out commentary frames. This is cheap, and better than relying on generic numbers.
- Generating an importance matrix (imatrix) from the project's own training prompts should reduce Q4 loss for this domain. This is an inference, not measured here.

### Gaps
- No study was found that measures quantisation loss specifically on fine-tuned small models for short stylistic outputs.
- The `save_pretrained_gguf` / `push_to_hub_gguf` API signatures and the supported `quantization_method` list were not re-read from Unsloth docs this session.
- Merge pitfalls were not researched: merging into a 4-bit vs 16-bit base, and chat-template or EOS mismatches after GGUF conversion.

---

## Q5. Does a fine-tuned 3–4B model match 8B on short, style-heavy generation? Esports/sports commentary papers, data-to-text, hallucination

### Takeaway
There is no CS2 or CS:GO commentary-generation paper. The closest work is on League of Legends data-to-commentary, which used T5 and Llama-2 with the 13B model best, and on real-time game commentary timing (2026). Direct evidence that a fine-tuned 3–4B model equals an 8B on a style task is thin. The few LoRA size studies found show the 7–8B model ahead on most settings, by modest margins. Faithfulness to the input frame is the main failure mode reported for esports data-to-text.

### Cited Findings

**Esports and game commentary**
- **Wang & Yoshinaga, "Commentary Generation from Data Records of Multiplayer Strategy Esports Game" (about 2024, older data):**
  - LoL datasets pair structured match records with commentary (the "LoL19" dataset).
  - T5 and Llama-2 variants were tested, and Llama-2-13B scored best overall.
  - The authors report models "struggle with long-term coherence and strategic narrative".
  - Source is an aggregator review; the venue is unconfirmed.
  — [Liner review](https://liner.com/review/commentary-generation-from-data-records-multiplayer-strategy-esports-game)
- **Student project, LoL event-log-to-commentary:**
  - Fine-tunes T5-base and Qwen2.5-7B. The search snippet also says "13B", which Qwen2.5 does not have, so treat that as an error.
  - Proposes an "Event Faithfulness Score" penalising hallucinated mentions.
  - Coursework, not peer-reviewed, but a useful reference implementation.
  — [XXoussam GitHub](https://github.com/XXoussam/Esports-Event-to-Commentary-Generation-LoL)
- **Kawamatsu et al., "Low-Latency Real-Time Audio Game Commentary System via LLM-Based Parallel Text Generation" (arXiv 2606.13322, Jun 2026; IJCAI-26 demo):**
  - Generates the next utterance while the current one is being spoken, and keeps a buffer of candidate utterances so TTS starts immediately.
  - Mean silence between utterances fell from 9.6 s to 0.3 s versus sequential baselines.
  - Timing similarity to professional commentators improved by over 40%.
  - A 120-person user study preferred the rhythm.
  - Base model not retrieved (arXiv blocked).
  — [arXiv PDF](https://arxiv.org/pdf/2606.13322); [IJCAI proceedings](https://www.ijcai.org/proceedings/2026/972); [papers.cool](https://papers.cool/arxiv/2606.13322)
- **"Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches" (arXiv 2603.02655, LREC 2026):** covers racing and fighting games in Japanese and English. Dynamic-interval decoding aligns better with human timing, using prompting only (no fine-tuning). — [arXiv](https://arxiv.org/pdf/2603.02655)
- **Game-MUG (arXiv 2404.19175, Apr 2024, older data):** a multimodal LoL game-situation understanding and commentary dataset. — [arXiv](https://arxiv.org/pdf/2404.19175)
- **MDPI Mathematics (2025):** combines temporal event prediction with an LLM for automatic commentary in a video game (a ball game, not esports). — [doi:10.3390/math13172738](https://doi.org/10.3390/math13172738)
- **Sports survey (arXiv 2608.14377, Aug 2026):** "A Survey of Large Models in Sports" catalogues fine-tuned (M)LLM commentary systems, citing Wang & Yoshinaga 2024. — [arXiv](https://arxiv.org/pdf/2608.14377)
- **CS2 mention:** CS2 appears only in an egocentric esports video QA benchmark (EgoEsportsQA, arXiv 2604.12320, Apr 2026), not in commentary generation. — [arXiv](https://arxiv.org/pdf/2604.12320)
- **Data resource:** LoL Worlds 2025 broadcast transcripts on Kaggle, useful as an example of caster-style text. — [Kaggle dataset](https://www.kaggle.com/datasets/troykueh/lol-worlds-2025-broadcast-transcripts)

**Model size after LoRA fine-tuning**
- **Course Q&A (arXiv 2609.13918, Sep 2026):** LoRA-tuned 7B models outperformed 3B models at most LoRA ranks. ROUGE and BERTScore measure proximity, not correctness. — [arXiv](https://arxiv.org/pdf/2609.13918)
- **"How Small Can You Go? LoRA Fine-Tuning 270M–8B Models for Merchant Information Extraction" (arXiv 2606.08051, Jun 2026):** spans 0.27B–8B. The best sub-8B model reached F1 96.60%. It is an extraction task, not stylistic generation. — [arXiv](https://arxiv.org/html/2606.08051v1)
- **Healthcare summarisation (arXiv 2512.16189, Dec 2025):** LoRA "significantly improves the performance of all open-source models", with Ministral-8B best among open models. Residual factual errors were handled by a verification pipeline. — [arXiv](https://arxiv.org/pdf/2512.16189)
- **Production case study (single, not peer-reviewed):** LoRA on 12k domain documents reduced hallucination 22% relative to base Llama-3.1-8B. — [Academia.edu](https://www.academia.edu/168308350/Reducing_Hallucinations_in_Domain_Specific_LLMs_via_LoRA_Fine_Tuning_A_Production_Case_Study)

### Inferences
- **The task is narrow.** One line, 20–40 tokens, a fixed vocabulary of players, weapons, sites and economy terms. Fine-tuning on 3–5k examples will teach the format and caster register to any 3B+ model.
- **Where 8B or A3B models likely still win:**
  - Picking the right salient fact from a 1,000-token frame.
  - Not inventing kills, players or round states (faithfulness).
  - Varying phrasing over hundreds of lines in a match.
- This matches the LoL finding that the largest model was best, and the LoRA size studies that show 7–8B ahead at most settings.
- **Mitigations that let a 3–4B model compete** (design inference):
  - Pre-select the salient facts in code, which keeps "facts only" in code and leaves tone and importance to the model.
  - Shorten and normalise the frame.
  - Use a faithfulness check against the frame, such as the Event Faithfulness Score idea.
- A3B MoE models give roughly 8B-plus quality at roughly 3B decode cost on CPU. This makes Qwen3-30B-A3B the best "quality within CPU budget" candidate if it can be fine-tuned, but it cannot be QLoRA'd on a free T4 (Q3).
- LFM2-8B-A1B is the best latency candidate, with unverified fine-tuning tooling.

### Gaps
- No paper directly compares fine-tuned 3–4B vs 8B on data-to-text, sports commentary, or persona/style generation with hallucination metrics.
- The base models used by Kawamatsu et al. (2026) and the pause-aware paper were not retrieved.
- No CS:GO or CS2 commentary dataset or paper was found.

---

## Q6. Creative-writing and instruction-following leaderboards (EQ-Bench Creative Writing v3, IFEval) for 1–14B open models

### Takeaway
I could not obtain trustworthy, current EQ-Bench Creative Writing v3 rankings for 1–14B open models. The official page is JS-rendered, and the aggregators show filtered or inconsistent subsets. The available normalised scores put Qwen3-VL-30B-A3B at about 0.85 and Qwen3-VL-8B at about 0.82, against Qwen3-VL-4B at about 0.76. For IFEval, Qwen3-4B-Instruct-2507 (about 83) is well ahead of Phi-4-mini (70.1), Llama-3.2-3B (68.0) and Ministral-3B (47.5). These are base-model signals only; a 3–5k-example fine-tune will dominate style.

### Cited Findings
- **EQ-Bench Creative Writing v3, llm-stats.com (14-model filtered board, updated 2026-11-02; normalised 0–1 scores, likely vendor-reported):**
  - Qwen3-235B-A22B-Instruct-2507: 0.875 (#1)
  - Qwen3-VL-32B-Instruct: 0.856 (#6)
  - Qwen3-Next-80B-A3B-Instruct: 0.853 (#8)
  - Qwen3-VL-30B-A3B-Instruct: 0.846 (#9)
  - Qwen3-VL-30B-A3B-Thinking: 0.825 (#11)
  - Qwen3-VL-8B-Thinking: 0.824 (#12)
  - Qwen3-14B: 0.803 (#13)
  - Qwen3-VL-4B-Thinking: 0.761 (#14)
  - No Gemma, Llama, Ministral, Phi, gpt-oss, LFM, Granite or SmolLM entries.
  — [llm-stats Creative Writing v3](https://llm-stats.com/benchmarks/creative-writing-v3)
- **Aggregator listing:** gemma-3-27b-it Elo 1259 / rubric 15.33, Mistral-Small-3.2-24B-Instruct-2506 Elo 1242.2 / rubric 14.31, Mistral-Small-24B-Instruct-2501 Elo 833.2. The search summary did not say which of these two pages gave these numbers; they are not verified against eqbench.com. — [benchmarklist.com](https://benchmarklist.com/benchmarks/creative_writing_v3/) / [evals.report](https://evals.report/benchmarks/eq-bench?tab=scores)
- **Gemma 4 31B:** reportedly scores a higher CW v3 rubric than Mistral Large 3. No figure given. — [Featherless blog](https://featherless.ai/blog/best-ai-for-creative-writing-2026)
- **Sophon EQ-Bench page (Aug 2026):** shows only the top 21, with no small models visible. The page is internally inconsistent: GPT-6 Astra is labelled both 1741.5 and about 2177. — [sophon.at](https://sophon.at/evals/eqbench-creative-writing)
- **Official EQ-Bench leaderboard:** the table did not load (JS-rendered). — [eqbench.com](https://eqbench.com/creative_writing.html)
- **IFEval, Phi-4-Mini Technical Report, Table 7 (0-shot; Mar 2025, older data):** Phi-4-mini 70.1, Llama-3.2-3B-Instruct 68.0, Ministral-3B 47.5. — [arXiv 2503.01743](https://arxiv.org/pdf/2503.01743)
- **IFEval, Qwen3-4B-Instruct-2507:** 83.4. Secondhand; I believe it matches Qwen's model card but did not verify. — [DEV Community post](https://dev.to/nadiateta/llm-research-blog-the-model-that-fits-our-project-201b)

### Inferences
- Among the CPU-feasible sizes, the Qwen3 family is the only one with consistent CW v3 coverage. Within it, A3B MoE scores higher than 8B, and 8B higher than 4B, by about 0.02–0.06 on the normalised scale.
- For a model fine-tuned on 3–5k caster lines, IFEval matters little: the format is learned. CW v3 measures long-form fiction, not 30-token play-by-play.
- Treat both leaderboards as weak tie-breakers. Base-model fluency and vocabulary matter, but an in-house blind A/B on 100 held-out frames is the decisive test.

### Gaps
- The following have no verified EQ-Bench CW v3 or IFEval numbers in these notes:
  - Gemma 4 E2B/E4B
  - Qwen3.5 small models
  - SmolLM3-3B
  - LFM2 / LFM2-8B-A1B
  - Granite 4.0-H
  - Nemotron Nano 9B v2
  - gpt-oss-20b
  - Ministral 8B
  - Llama-3.1-8B
- The official eqbench.com table and the model cards should be checked directly; WebFetch could not render them.
- The llm-stats board is a 14-model subset and is probably vendor-reported, not EQ-Bench-run. Ranks there are not global ranks.
