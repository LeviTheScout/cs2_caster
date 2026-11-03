# Gemma and Phi open-weight families (1B–14B) as QLoRA base models for a live CS2 caster-line generator — status as of 2 Nov 2026

Scope note: researched 2026-11-02. Every fact carries the date of its source where known. "Official" = Google/Microsoft/model card/tech report; "independent" = third-party leaderboards; "community" = blogs, HN, Reddit, HF community cards. Facts that came only through search-result summaries (pages not opened) are marked "(search snippet)".

---

## 1. Model inventory: release dates, sizes, architecture (dense / MoE / "effective" params), context, licence

### Takeaway
The current Gemma generation is **Gemma 4** (E2B, E4B, 26B-A4B MoE, 31B on 31 Mar / 2 Apr 2026; **12B "Unified" dense on 3 Jun 2026**), and it **switched from the restrictive Gemma Terms of Use to Apache 2.0**. Microsoft has released **no Phi-5** as of the latest sources found (Aug 2026); the newest Phi is **Phi-4-reasoning-vision-15B (4 Mar 2026, MIT)**, so the relevant text Phi models are still Phi-4 (14B, Dec 2024) and Phi-4-mini (3.8B, Feb 2025), all MIT.

### Cited Findings

**Gemma timeline (official release log)**
- Google's official release log lists: Gemma 3 1B/4B/12B/27B on "March 10, 2025"; Gemma 3n E2B/E4B on "June 26, 2025"; Gemma 3 270M on "August 14, 2025"; Gemma 4 E2B/E4B/31B/26B-A4B on "March 31, 2026"; "Gemma 4 MTP for E2B, E4B, 31B, and 26B A4B" on "April 16, 2026"; "Gemma 4 12B Unified" on "June 3, 2026"; EmbeddingGemma 2 (740M) on "October 6, 2026". No Gemma 5 / Gemma 4n entry. — [Gemma releases, ai.google.dev](https://ai.google.dev/gemma/docs/releases)
- Date conflict: Google's launch blog is dated **Apr 02, 2026** and Wikipedia gives 2 April 2026 for Gemma 4, versus 31 March 2026 on the release log. Similarly, Wikipedia gives Gemma 3 as 2025-03-12 vs 10 March on the release log. — [Google blog, Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/); [Wikipedia: Gemma](https://en.wikipedia.org/wiki/Gemma_(language_model)); contradicted by [Gemma releases](https://ai.google.dev/gemma/docs/releases)
- Nothing new in the LLM line in Oct 2026: the October Gemma releases are EmbeddingGemma 2 (embedding model) and a "DiffusionGemma" entry described as built on Gemma 4 (no details found; the HF Gemma 4 blog says DiffusionGemma trades quality for speed, e.g. MMLU Pro 77.6% vs 82.6% for 26B-A4B). (search snippet for the DeepMind page) — [Gemma releases](https://ai.google.dev/gemma/docs/releases); [HF blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)

**Gemma 4 (official model card / tech report / HF blog, Apr–Jul 2026)**

| Model | Params | Layers | Context | Modalities | Source |
|---|---|---|---|---|---|
| Gemma 4 E2B | "2.3B effective (5.1B with embeddings)" | 35 | 128K | text, image, audio | [HF card E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| Gemma 4 E4B | "4.5B effective (8B with embeddings)" | 42 | 128K | text, image, audio | [HF card E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| Gemma 4 12B Unified | "11.95B dense, encoder-free" | n/a | 256K | text, image, audio (video) | [HF blog](https://huggingface.co/blog/gemma4); [Google Dev blog 12B](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/) |
| Gemma 4 26B A4B (MoE) | 25.2B total, 3.8B active | 30 | 256K | text, image | [HF card E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| Gemma 4 31B | 30.7B dense | 60 | 256K | text, image | [HF card E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |

- All Gemma 4 sizes: 262K vocabulary; sliding window 512 for small models, 1024 for large; vision encoder ~150M and audio encoder ~300M on E2B/E4B; ~550M vision encoder on 26B/31B. — [HF card gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- **Licence: Apache 2.0** on all Gemma 4 HF cards; Google's blog calls it "a commercially permissive Apache 2.0 license". Wikipedia: Gemma 1–3 were "source-available" under the Gemma Terms of Use and Google switched licences with Gemma 4. — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/); [HF gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B); [Wikipedia](https://en.wikipedia.org/wiki/Gemma_(language_model))
- Gemma 4 12B: Google calls it "a dense multimodal model" using "a single decoder-only transformer" without separate encoders: raw 48x48 image patches through "a single matmul", raw 16 kHz audio in 40 ms frames projected linearly; "small enough to run locally on dedicated GPU laptops with 16GB VRAM"; a dedicated MTP model is released "to maximize local inference speeds". The tech report says it replaces the 550M vision encoder with a 35M matmul and drops the 305M audio encoder (search snippet for the tech report part). Apache 2.0. — [Google Developers Blog, 3 Jun 2026](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/); [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1)
- Gemma 4 pre-training data cutoff: January 2025; data = "web documents, code, images, and audio (for E2B, E4B and 12B)". Post-training "similar ... as in Gemma 3" plus a thinking mode. — [Gemma 4 Technical Report, arXiv 2607.02770, 2 Jul 2026](https://arxiv.org/html/2607.02770v1)

**What "effective" params (E2B/E4B) means**
- Tech report: "E2B and E4B use per-layer embeddings as in Gemma 3n, making them 2.3B and 4.5B effective out of 5B and 8B total parameters". Table 1 lists embedder params E2B "400M + 2,340M", E4B "670M + 2,820M" — the second number is the per-layer-embedding (PLE) table. — [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1)
- HF blog: PLE is "a parallel, lower-dimensional conditioning pathway alongside the main residual stream"; each token gets a small vector per layer from a token-identity lookup plus a learned projection; PLE dim is much smaller than hidden size, giving "meaningful per-layer specialization at modest parameter cost". The blog does not quantify RAM savings or say how PLE is offloaded. — [HF blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- HF card: PLE tables "are large but used only for quick lookups", hence effective << total. — [HF card gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- KV-cache sharing: "we share the KV cache with ratios of 20/35 and 18/42 for the E2B and E4B model"; together with other tricks this reduces "the global KV cache footprint by up to 37.5%". — [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1)
- Gemma 3n precedent (official card): "the raw parameter count of this model is 8B"; runs "with a memory footprint comparable to a traditional 4B model by offloading low-utilization matrices from the accelerator"; MatFormer nesting (E2B is a sub-model inside E4B, "Mix-and-Match" custom sizes). — [HF card gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)
- Gemma 4 docs/blog do **not** mention MatFormer for Gemma 4 (the HF blog doesn't mention it). — [HF blog](https://huggingface.co/blog/gemma4)

**Gemma 3 / 3n (older generation, superseded by Gemma 4)**
- Gemma 3 (Mar 2025): 1B, 4B, 12B, 27B; "128K tokens, with the exception of the 1B model that has 32K"; 417M SigLIP vision encoder on 4B/12B/27B (none on 1B); 4T training tokens for 4B, 12T for 12B. Licence: Gemma Terms of Use. — [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786); [Wikipedia](https://en.wikipedia.org/wiki/Gemma_(language_model))
- Gemma 3n (26 Jun 2025): E2B/E4B; "Total input context of 32K tokens"; licence field "gemma" linking to ai.google.dev/gemma/terms. — [HF card gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it); [Gemma releases](https://ai.google.dev/gemma/docs/releases)

**Phi family**
- Phi-4 (14B): technical report Dec 2024; "14-billion-parameter decoder-only transformer", 4K default context extended to 16K in midtraining; ~40% synthetic data in pretraining mix (~400B unweighted synthetic tokens across 50 dataset types); SFT + 2 rounds DPO (Pivotal Token Search). Wikipedia cites an HF licence page for Phi-4 dated December 11, 2024; MIT series licence. — [Phi-4 Technical Report, arXiv 2412.08905](https://arxiv.org/html/2412.08905); [Wikipedia: Phi](https://en.wikipedia.org/wiki/Phi_(LLM))
- Phi-4-mini-instruct: "February 2025", 3.8B dense decoder-only with GQA and shared in/out embeddings, 128K context, 200,064-token vocab, MIT, 5T training tokens incl. synthetic "textbook-like" data (no % given), data cutoff June 2024. — [HF card Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- Phi-4-mini-flash-reasoning: "June 2025" card (paper 9 Jul 2025), 3.8B, "hybrid SambaY architecture with Differential Attention" (SSM + GQA, Gated Memory Unit), 64K context, MIT, "designed and tested for math reasoning only". — [HF card Phi-4-mini-flash-reasoning](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)
- Phi-4-reasoning-vision-15B: "March 4, 2026", 15B, Phi-4-Reasoning backbone + SigLIP-2, "16,384 tokens" context, MIT, `<think>`/`<nothink>` modes, aimed at visual science/maths reasoning and computer-use agents. — [HF card Phi-4-reasoning-vision-15B](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)
- Other Phi-4 members (secondary sources): Phi-4-multimodal (5.6B, text+vision+speech), Phi-4-reasoning and Phi-4-reasoning-plus (14B); original Phi-4 announced 12 Dec 2024. (search snippet) — [frankx.ai Phi analysis 2026](https://www.frankx.ai/blog/phi-analysis-2026); [Turing Post SLM list](https://www.turingpost.com/p/slmslist)
- **Phi-5:** an Aug 2026 review says Microsoft has announced no Phi-5 of any size and no Phi-5 weights exist on HF/Ollama; as of 11 Jul 2026 the newest broadly released Phi is Phi-4-reasoning-vision-15B. A Spheron "Deploy Phi-5" guide describes a 14B Phi-5 based on "pre-release announcements" — treat as speculative. (search snippets) — [llmcheck, Aug 2026](https://llmcheck.net/blog/phi-5-large-28b-review/); [ArtificialWatch Phi-5 tracker](https://artificialwatch.com/model/phi-5); contradicted (unverified) by [Spheron blog](https://www.spheron.network/blog/deploy-phi-5-gpu-cloud/)

### Inferences
- For a public student project, Gemma 4 (Apache 2.0) and Phi (MIT) are both licence-clean; Gemma 3/3n carry the older Gemma Terms (usage restrictions that apply to derivatives), so Gemma 4 dominates Gemma 3/3n on licence as well as quality.
- E4B on disk/RAM behaves like an ~8B model (all weights must be loaded; Ollama tag is 6.6–9.5 GB), but per generated token it only does ~4.5B params of dense matmul, because PLE rows are looked up (one row per token per layer), not multiplied. So RAM cost ≈ 8B-class, decode speed ≈ 4B-class. With 32 GB RAM, RAM is not a constraint for any candidate ≤14B.
- The audio (~300M) and vision (~150M) towers on E2B/E4B add load size but no per-token cost for text-only prompts.
- Candidates in range for us: Gemma 4 E2B, E4B, 12B; Gemma 3 4B/12B and Gemma 3n E2B/E4B (superseded); Phi-4 14B, Phi-4-mini 3.8B. Phi-4-mini-flash-reasoning and Phi-4-reasoning(-vision) are math/vision reasoning models and do not fit a short-caster-line task. Gemma 4 26B-A4B is outside the size range and (see §2) not T4-trainable.

### Gaps
- No official Phi model release found between 4 Mar 2026 and 2 Nov 2026; the most recent "no Phi-5" confirmation found is Aug 2026, so a release in Sep–Oct 2026 can't be fully ruled out.
- Exact PLE RAM figures per quant level and whether llama.cpp memory-maps PLE separately were not found in official docs.
- Exact HF weight-release date for Phi-4 (vs Dec 2024 announcement) not verified.
- Gemma 4 12B context length: the HF blog says 256K, but the Google Dev blog does not state it.

---

## 2. Text-only QLoRA fine-tuning (Unsloth / PEFT / TRL) on a free T4 (16 GB) at ~2048 seq len; known gotchas

### Takeaway
Gemma 4 E2B/E4B are **officially supported by Unsloth with free Colab/Kaggle notebooks** (text, vision, audio) and E4B QLoRA is Unsloth's recommended small-model recipe. VRAM figures conflict (E4B "10GB" vs "LoRA requires 17GB"), so **E4B QLoRA at 2048 tokens probably fits a T4 but expect to verify**; E2B fits easily (8 GB). Several Gemma-specific bugs exist (use_cache garbage logits, fp16 overflow on T4, template mismatch), mostly fixed inside Unsloth, which is a strong argument for Unsloth over raw PEFT/TRL on a T4. Phi-4 14B also has a free-T4 Unsloth notebook but needed tokenizer/template fixes.

### Cited Findings

**Gemma 4 (Unsloth Gemma 4 training guide; screenshots dated 2026-04-07, page undated)**
- VRAM: "E2B: trains on 8GB VRAM", LoRA on 8–10GB; E4B intro says "requires 10GB VRAM" but a later bullet says "E4B LoRA requires 17GB VRAM" (unreconciled on the page); 31B QLoRA ~22GB; 26B-A4B LoRA >40GB and MoE QLoRA "not recommended"; GRPO RL on E2B in 9GB. Claimed ~1.5x faster, ~60% less VRAM than FA2 setups. — [Unsloth: Gemma 4 Fine-tuning Guide](https://unsloth.ai/docs/models/gemma-4/train)
- Unsloth recommends "train E4B QLoRA rather than E2B LoRA as the E4B is bigger and the quantization accuracy difference is miniscule". (search snippet of same page) — [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train)
- Free notebooks: E4B Inference+Text, Vision+Text, Audio; E2B Vision+Text, Audio, GRPO; 31B on Kaggle; 26B-A4B/31B vision on A100; plus Unsloth Studio (no-code). — [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train)
- Text-only recipe on the page: `FastModel.from_pretrained(..., load_in_4bit=True, max_seq_length=8192)`, LoRA r=8, alpha=8, dropout 0, `finetune_vision_layers` off, 3000 rows of FineTome-100k, batch 1 × grad-accum 4, `adamw_8bit`, LR 2e-4 (2e-5 for long runs), 1 epoch. — [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train)
- Gotchas listed by Unsloth:
  - **use_cache bug:** E2B/E4B produce garbage logits with `use_cache=False`, which QLoRA and `gradient_checkpointing=True` force; Unsloth's fix gives bit-exact parity ("max logit diff 0.000000, versus 48.937500 before"). — [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train)
  - **Loss looks wrong but isn't:** E2B/E4B training loss of 13–15 is "perfectly normal" for these multimodal models (26B/31B ~1–3). Losses of 100–300 indicate broken gradient accumulation (fixed in Unsloth). — same source
  - **fp16 overflow on T4 (audio path):** `-1e9` in `masked_fill` overflows fp16 on T4, raising a `c10::Half` RuntimeError; page doesn't say fixed. — same source
  - **Template:** use `gemma-4` template for E2B/E4B (and `gemma-4-thinking` for 26B/31B); use the same template at inference as training; `train_on_responses_only` with instruction part `<|turn>user\n` and response part `<|turn>model\n`. — same source
  - IndexError for 31B/26B inference (`num_kv_shared_layers = 0` → `layer_types[:-0]` empty). Not relevant to E-models. — same source
- Third-party guides: Lushbinary says "Start with E4B – it's fast, fits on free Colab T4 GPUs with QLoRA"; Wavect suggests starting with E2B QLoRA on free Colab. (community, search snippets) — [Lushbinary Gemma 4 LoRA guide](https://lushbinary.com/blog/fine-tune-gemma-4-lora-qlora-complete-guide/); [Wavect blog](https://wavect.io/blog/fine-tune-gemma-4-free-unsloth-colab/)
- HF/TRL: TRL has a Gemma 4 E2B GRPO example; Google's Vertex SFT example freezes "both the vision and audio towers" (on an H100, not a T4). No PEFT memory numbers given. — [HF blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- Gemma 4 12B: Google says it supports LoRA or full tuning via Hugging Face and Unsloth; because modalities share weights in the encoder-free design, "fine-tuning updates the whole multimodal path in one pass". Unsloth lists 12B as supported with no VRAM figure. — [Google Developers Blog 12B](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/); [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train)

**Gemma 4 chat template facts relevant to SFT**
- Gemma 4 natively supports the `system` role (Gemma 3 did not). Thinking is enabled by `<|think|>` at the start of the system prompt; for E2B/E4B, disabled thinking produces **no** empty thought block (larger models emit `<|channel>thought\n<channel|>`). Multi-turn history should keep only final answers. — [HF card gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)

**Gemma 3 / 3n (older)**
- Gemma 3 has activation overflow in fp16; Unsloth keeps intermediate activations in bfloat16 and claims to be "the only framework which works in float16 machines for Gemma 3 inference and training", i.e. free T4 notebooks work. — [Unsloth Gemma 3 blog](https://unsloth.ai/blog/gemma3); [Unsloth Gemma 3 docs](https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune)
- Gemma 3n: no activation overflow like Gemma 3, but still produces inf/NaN on T4 because Conv2D weights exceed fp16 max 65504; Unsloth autocasts those matmuls to fp32. Free Colab "Gemma3N (4B) Conversational" notebook; text layers fine-tuned by default (`finetune_vision_layers=False` etc.); fine-tuning vision/audio "will require much more VRAM" than free 15GB; Unsloth gradient checkpointing can't be applied to the vision encoder. Don't add a second `<bos>`. — [Unsloth Gemma 3n guide](https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune)

**Phi**
- Unsloth found Phi-4 bugs (Jan 2025): tokenizer used `<|endoftext|>` as BOS/EOS/PAD (fix EOS → `<|im_end|>`); PAD = EOS causes infinite generations after fine-tuning because pad is masked from loss (fix: unused token such as `<|dummy_87|>`); chat template always appended the assistant prompt. — [Simon Willison, 11 Jan 2025](https://simonwillison.net/2025/Jan/11/phi-4-bug-fixes); [Unsloth Phi-4 blog](https://unsloth.ai/blog/phi4) (403 when fetched; content via search snippet)
- Unsloth's free Phi-4 (14B) Colab notebook runs on a T4 16GB. (search snippet) — [Gwern-archived Reddit post](https://gwern.net/doc/www/old.reddit.com/ad2167533b5a148ade733871cf943c93632b15b9.html); [plushcap summary](https://www.plushcap.com/content/unsloth/blog/unsloth-phi-4-finetuning-bug-fixes-by-unsloth)
- Phi-4-mini chat format: `<|system|>…<|end|><|user|>…<|end|><|assistant|>`; vocab includes placeholder tokens "for fine-tuning". — [HF card Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- Phi-4-mini-flash-reasoning needs `trust_remote_code=True` and flash attention; tested on A100/H100 only. — [HF card Phi-4-mini-flash-reasoning](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)

### Inferences
- Our data (3–5k examples, ~500–1500 prompt tokens + ~30 output tokens) is tiny: one epoch at seq len 2048 is a few hundred to ~1,250 optimizer steps at batch 4 — well within a free Colab/Kaggle session for E2B/E4B; Phi-4 14B and Gemma 4 12B would be slower (more params) but still feasible on T4 in 4-bit (~7–9 GB weights), probably with less headroom for seq 2048.
- Our run is text-only, so the documented audio fp16 overflow should not trigger; keep `finetune_vision_layers=False` and don't feed audio. Use Unsloth rather than raw PEFT on a T4 to inherit the use_cache and fp16 fixes (T4 has no bf16).
- Use `train_on_responses_only` so the loss is only on the ~30-token caster line; otherwise the 1,000+ prompt tokens dominate the loss. Expect E4B raw loss in the 13–15 range per Unsloth, which is not a bug.
- Train with the non-thinking `gemma-4` template and keep thinking off at serving time: the E-models emit no empty thought block, which keeps output as a bare `<intensity> line`.
- Phi-4's pad/EOS bug class matters a lot for our short outputs: a wrong EOS means the model never stops after one line. Use Unsloth's fixed checkpoints if testing Phi.

### Gaps
- No primary source gives a measured peak VRAM for E4B QLoRA at seq len 2048 on a T4; the Unsloth page contradicts itself (10 GB vs 17 GB, possibly 16-bit LoRA vs QLoRA).
- No measured T4 fit/throughput for Gemma 4 12B QLoRA or Phi-4-mini QLoRA found.
- Whether the fp16 audio-mask overflow is fixed in current Unsloth releases is not stated.

---

## 3. llama.cpp / GGUF / Ollama support and CPU-only serving (i7-13700K, 32 GB, 2–3 s per line)

### Takeaway
Gemma 4 (incl. E2B/E4B) has **day-one llama.cpp and Ollama support** (llama.cpp PR #21309, build b8637; Ollama `gemma4:e2b/e4b/12b` tags), plus official GGUFs from ggml-org; Unsloth exports `q4_k_m` GGUF directly. Remaining rough edges concern Gemma 4's MTP/speculative drafters, not basic inference. **I found no trustworthy CPU-only speed benchmark** for any candidate; the main risk to the 2–3 s target is **prompt prefill of 500–1500 tokens on CPU**, not generating 30 tokens.

### Cited Findings
- Google blog: day-one support from Hugging Face (Transformers, TRL), llama.cpp, Ollama, Unsloth, vLLM, MLX, LM Studio, NVIDIA NIM; weights on HF, Kaggle, Ollama. — [Google blog, 2 Apr 2026](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- HF blog: "image+text support in llama.cpp from the get-go"; example `llama serve -hf ggml-org/gemma-4-E2B-it-GGUF`. — [HF blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- Gemma 4 architecture support landed via llama.cpp PR #21309 (build b8637); older builds fail with "unknown model architecture". Later PR #24282 adds Gemma-4 E2B/E4B MTP "assistant" (speculative decoding) support; a secondary report says llama-speculative and llama-bench fail with E2B/E4B MTP. (search snippets) — [PR #21309 mirror](https://gh.mise.run.place/ggml-org/llama.cpp/pull/21309); [aiweekly on PR #24282](https://aiweekly.co/alerts/llamacpp-adds-gemma-4-e2be4b-mtp-assistant-support)
- Official GGUFs: `ggml-org/gemma-4-E4B-it-GGUF`, run via `llama-server -hf ggml-org/gemma-4-E4B-it-GGUF`. A bartowski discussion notes a llama.cpp workaround for older GGUFs means regeneration only matters if imatrix was used. (search snippet) — [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF); [bartowski discussion](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/discussions/1)
- Ollama `gemma4` library: e2b 4.6–7.5 GB, e4b 6.6–9.5 GB (128K), 12b 7.7–8.0 GB (256K), 26b 16–19 GB, 31b 19–20 GB; tags list Text+Image input although the model table lists audio on E2B/E4B. — [Ollama library: gemma4](https://ollama.com/library/gemma4)
- Unsloth exports `save_pretrained_gguf` with `q4_k_m`, `q8_0`, `f16`, and has deployment guides for llama.cpp, llama-server and Ollama. — [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train)
- PLE placement: secondary guides say the llama.cpp loader keeps input-layer tensors including `per_layer_token_embd` in host (CPU) memory by default. Not confirmed in llama.cpp docs. (search snippet) — [dev.to: Gemma 4 E2B/E4B on L4 and small GPUs](https://dev.to/gde/size-your-model-to-where-you-run-it-gemma-4-e2b-and-e4b-on-an-aws-l4-and-two-small-gpus-1d47); [rohitghumare.com Gemma 4 guide](https://rohitghumare.com/guides/gemma-4/)
- Gemma 3n GGUF history: initially "only supported in text format for inference"; GGUF bugs fixed (`add_shared_kv_layers` float32→uint32; `per_layer_token_embd` had to be Q8_0, later Q4_0/Q4_1/Q5_0/Q5_1 confirmed working). — [Unsloth Gemma 3n guide](https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune)
- Speed data found (none is a clean desktop-CPU number):
  - GPU: E4B QAT Q4_0 at 38.0 tok/s on a GTX 1650 Ti 4 GB laptop and 74.0 tok/s on an NVIDIA L4 (no drafter), llama.cpp. (search snippet) — [dev.to: E2B/E4B on AWS L4](https://dev.to/gde/size-your-model-to-where-you-run-it-gemma-4-e2b-and-e4b-on-an-aws-l4-and-two-small-gpus-1d47)
  - GPU: E4B generation 60.41 tok/s on an RTX 3050 8 GB. (search snippet) — [dev.to: Gemma 4 on RTX 3050](https://dev.to/gde/gemma-4-on-a-used-8-gb-rtx-3050-116-tokenss-for-one-user-618-for-64-k6h)
  - llm-benchmark.de's "CPU" chart for E4B Q4_K_M actually groups GPU runs by host CPU (e.g. an "i5-7400" row matches an RTX 2060 run), so it is not CPU-only data. — [llm-benchmark.de run](https://llm-benchmark.de/benchmark/run-20260826-082948-e64f1d)
  - A third-party HF card reports Q4_K_M at 4.18 tok/s generation / 9.83 tok/s prompt on unspecified hardware, and a guide estimates "~8–12 tokens per second on CPU" (search snippets; exact URLs uncertain, hardware unspecified — low trust). — [sunil-pathak Q4_K_M card](https://huggingface.co/sunil-pathak/gemma-4-E4B-it-Q4_K_M); [llamabuilds E4B page](https://www.llamabuilds.ai/models/gemma-4-e4b)
- Gemma 4 MTP drafters (16 Apr 2026) and the 12B's dedicated MTP model are meant to speed local inference. — [Gemma releases](https://ai.google.dev/gemma/docs/releases); [Google Developers Blog 12B](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)
- Phi-4-mini-flash-reasoning (SambaY hybrid) card lists vLLM/SGLang/Docker Model Runner and does not mention llama.cpp/GGUF. — [HF card](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)

### Inferences
- **Decode speed (memory-bandwidth estimate, not measured):** CPU decode is bound by bytes of weights read per token. At Q4_K_M (~4.8 bits/weight), E4B reads ~4.5B dense params ≈ 2.7 GB/token (PLE rows are lookups); Gemma 4 12B ≈ 7 GB; Phi-4 14B ≈ 8.5 GB; Phi-4-mini ≈ 2.3 GB. On dual-channel DDR5 (13700K, ~60–80 GB/s practical; spec not sourced here), rough decode ceilings are ~20–25 tok/s (E4B, Phi-4-mini), ~8–10 tok/s (12B), ~7–9 tok/s (Phi-4 14B). A 35-token line (incl. intensity token) then takes ~1.5–2 s for E4B but ~4–5 s for 12B/14B — so **only ≤4B-class models plausibly meet 2–3 s on CPU**.
- **Prefill is the real bottleneck:** re-processing 1,000+ prompt tokens each line on a CPU could take several seconds even for E4B. Mitigation: keep the static part (system prompt, match card) as a stable prefix so llama.cpp/Ollama's prompt cache reuses the KV cache and only new events are prefilled; keep the dynamic tail short. This should be benchmarked with `llama-bench -ngl 0` on the actual i7-13700K before committing.
- MTP/speculative decoding is attractive in theory but E2B/E4B MTP had reported tool failures; don't depend on it for the first version.
- Phi-4 and Phi-4-mini use standard dense transformer architectures (Phi-3 lineage), so llama.cpp support is mature (inference; no 2026 issue found). Phi-4-mini-flash's hybrid SSM architecture is a GGUF risk.

### Gaps
- **No reliable CPU-only tok/s (prompt or generation) for Gemma 4 E2B/E4B/12B, Phi-4 or Phi-4-mini on a 13th-gen Intel desktop was found.** This must be measured locally.
- No primary llama.cpp issue found documenting PLE-specific CPU slowdowns or memory behaviour for Gemma 4 E-models.
- Not verified whether Ollama `gemma4:e4b` default tag is Q4_K_M or larger (6.6–9.5 GB range suggests several quants).

---

## 4. Benchmarks relevant to us: instruction following, creative writing, grounding/hallucination, English chat quality — and evidence for "Gemma prose" vs "dry Phi"

### Takeaway
Officially, Gemma 4 E4B scores **IFEval 96.7** (thinking mode) but only **IFBench 44.0** vs 74.0 for Gemma 4 12B, so harder format constraints favour 12B. Phi-4's own report admits IFEval 63.0 and weak strict-format adherence. Independently, **EQ-Bench Creative Writing v3** ranks Gemma models far above same-size peers (Gemma 3 4B 13.91 vs Llama 3.1 8B 8.89) and lists **no Phi model at all**; Gemma 4 E2B/E4B are also not on it. Grounding numbers exist only for Gemma 3 (FACTS Grounding 70.1/75.8 for 4B/12B). Evidence that Phi prose is dry is anecdotal.

### Cited Findings

**Instruction following**
- Gemma 4 IFEval (Table 5, **thinking mode**): 31B 98.9; 26B-A4B 98.5; 12B 97.2; E4B 96.7; E2B 94.6; Gemma 3 27B (non-thinking) 90.4. IFBench: 31B 76.0; 26B-A4B 72.0; 12B 74.0; **E4B 44.0; E2B 38.0**; Gemma 3 27B 32.0. — [Gemma 4 Technical Report, 2 Jul 2026](https://arxiv.org/html/2607.02770v1)
- Google: Gemma 4 "demonstrates significant improvements in math and instruction-following benchmarks" (no numbers in blog). — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- Gemma 3 IFEval (IT): 1B 80.2, 4B 90.2, 12B 88.9, 27B 90.4. — [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786)
- Phi-4 IFEval 63.0 vs Qwen 2.5 14B 78.7, GPT-4o-mini 80.0, Llama-3.3 70B 89.3; the authors call this a weakness and list "weak adherence to strict formatting requirements, such as tables or fixed bullet structures". Phi-4's weakest benchmarks are SimpleQA, DROP and IFEval. — [Phi-4 Technical Report](https://arxiv.org/html/2412.08905); [MS Research PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/P4TechReport.pdf)
- Phi-4-mini-instruct card: no IFEval reported; Arena Hard 32.8 (below Phi-3.5-mini's 34.4, Qwen2.5-7B's 55.5); overall 63.5 vs Qwen2.5-7B 67.9, Llama-3.1-8B 62.3. — [HF card Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

**General capability (official, Gemma 4 IT)**
- MMLU Pro: 31B 85.2, 26B-A4B 82.6, E4B 69.4, E2B 60.0, Gemma 3 27B 67.6. GPQA Diamond: E4B 58.6, E2B 43.4. LiveCodeBench v6: E4B 52.0, E2B 44.0. — [HF card gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- Gemma 3n E4B IT: MMLU 64.9, MMLU-Pro 50.6 (E2B 60.1 / 40.5). — [HF card gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)

**Creative writing (independent, LLM-judged)**
- EQ-Bench Creative Writing v3 (benchmarklist mirror, updated 15 Jul 2026; rubric 0–20 / Elo): Gemma 4 31B 16.01 / 1407.6; Gemma 4 26B-A4B 16.03 / 1286.9; **Gemma 4 12B 15.71 / 1270.4**; Gemma 3 27B 15.33 / 1259; Gemma 3 12B 14.63 / 1181.6; **Gemma 3 4B 13.91 / 1113.6**; Gemma 2 9B 11.16 / 957.9. Reference small models: Llama 3.2 3B 8.18 / 709.4; Llama 3.1 8B 8.89 / 800.8; Mistral Small 3.2 24B 14.31 / 1242.2. **No Phi models, no Gemma 4 E2B/E4B, no Gemma 3n** on the board. — [benchmarklist: Creative Writing v3](https://benchmarklist.com/benchmarks/creative_writing_v3/)
- Featherless (2026) also cites Gemma 4 31B at 16.01 rubric / Elo 1407.6. EQ-Bench uses an LLM grader (search snippet names "GLM 5.3 via OpenRouter"); rubric and Elo can rank models differently. — [Featherless: best open models for creative writing](https://featherless.ai/blog/best-ai-for-creative-writing-2026); [EQ-Bench creative-writing-bench GitHub](https://github.com/EQ-bench/creative-writing-bench)
- Community: an HN commenter says r/LocalLLaMA has lately favoured E4B for creative writing (especially uncensored GGUFs), and the 31B and its finetunes for serious creative work — explicitly secondhand. — [Hacker News thread on Gemma 4 12B](https://news.ycombinator.com/item?id=48385906)

**Chat quality (LMArena / Arena text)**
- Arena Text Elo (as of 19 Jun 2026, in tech report): Gemma 4 31B 1451 ±8; 26B-A4B 1438 ±8; Gemma 3 27B 1366 ±4. No Arena figures for E2B/E4B/12B. — [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1)
- Google (as of 1 Apr 2026): 31B was "#3 open model" on Arena AI text leaderboard, 26B #6. HF blog: estimated text-only LMArena 1452 (31B), 1441 (26B). — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/); [HF blog](https://huggingface.co/blog/gemma4)
- Gemma 3 27B IT preliminary Chatbot Arena Elo 1338 (+8/−9) on 8 Mar 2025. — [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786)
- Gemma 3n E4B IT: LMArena Text 1317.2 per an aggregator (not confirmed on LMArena itself). — [llmrun.dev Gemma 3n E4B benchmarks](https://llmrun.dev/model/google-gemma-3n-e4b-it/benchmarks)
- Phi-4 Arena Hard 75.4 vs Qwen 2.5 14B 70.2, GPT-4o-mini 76.2 (official). — [Phi-4 Technical Report](https://arxiv.org/html/2412.08905)

**Grounding / hallucination**
- Gemma 3 IT FACTS Grounding: 1B 36.4, 4B 70.1, 12B 75.8, 27B 74.9; SimpleQA 2.2 / 4.0 / 6.3 / 10.0. — [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786)
- Gemma 4 tech report gives no FACTS Grounding or SimpleQA numbers. — [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1)
- Phi-4 SimpleQA 3.0 (authors: partly deliberate refusal training); Phi-4-mini card: "does not have the capacity to store too much factual knowledge", recommends RAG; may "hallucinate function names or URL's". — [Phi-4 Technical Report](https://arxiv.org/html/2412.08905); [HF Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- HF blog: in a video demo the E2B model's audio-based answer included hallucinated content. — [HF blog](https://huggingface.co/blog/gemma4)

**Phi style evidence**
- Phi-4 report: synthetic data is ~40% of pretraining; long, elaborate answers even for simple questions as a side effect of chain-of-thought data; tuned mainly for single-turn. — [Phi-4 Technical Report](https://arxiv.org/html/2412.08905)
- Community guide: Phi models produce "technically correct but lifeless prose" because "synthetic training data optimizes for accuracy, not voice"; community says Microsoft may have over-optimised for benchmarks at the expense of general knowledge (informal, not a controlled study). — [InsiderLLM Phi models guide](https://insiderllm.com/guides/phi-models-guide/)
- Microsoft defends against overfitting claims: strong results on Nov 2024 AMC-10/12 contests that postdate training data. — [Phi-4 Technical Report](https://arxiv.org/html/2412.08905)

### Inferences
- The independent creative-writing data supports "Gemma writes well for its size": Gemma 3 4B beats Llama 3.1 8B by ~5 rubric points and Mistral Small 24B (Jan/Mar 2025 versions). Gemma 4 E4B is likely at or above Gemma 3 4B, but this is not measured.
- Gemma 4 IFEval numbers are thinking-mode; we'd run non-thinking, so true out-of-the-box format compliance for E4B is likely lower. IFBench 44 vs 74 suggests E4B is much weaker than 12B on unusual constraints. After SFT on 3–5k examples with one fixed output format, format compliance is mainly learned, so this gap matters less than it looks — but grounding (not inventing kills/players) is a capability the base must bring.
- FACTS Grounding favours 12B over 4B in Gemma 3 (75.8 vs 70.1); expect a similar ordering for Gemma 4 12B vs E4B.
- The weakest Phi-4 points (IFEval, strict formatting, verbosity) are exactly the ones a 20–40-token formatted line stresses; SFT can fix length/format but Phi's register is the larger unknown since no creative-writing benchmark data exists.

### Gaps
- No EQ-Bench Creative Writing, LMArena or IFBench data for Phi-4 / Phi-4-mini; no EQ-Bench data for Gemma 4 E2B/E4B or Gemma 3n.
- No non-thinking IFEval figures for Gemma 4.
- No grounding/hallucination benchmark (FACTS, HHEM/Vectara) for Gemma 4 or Phi-4-mini found.
- No benchmark specifically on casual/hype English register or sports commentary for any of these models.

---

## 5. USP and known weaknesses of each model for this use (repetition, refusals on "kill"/"headshot", verbosity)

### Takeaway
**Gemma 4 E4B** is the best-balanced candidate: Apache 2.0, first-class Unsloth/llama.cpp/Ollama support, 4.5B-class decode cost, and the best prose reputation in its size class; its weaknesses are an 8B-class load size, multimodal-specific training bugs, low IFBench, and weak evidence about its CPU prefill speed. **Gemma 4 12B** is the quality step-up (IFBench 74, EQ-Bench 15.71) but is likely too slow for 2–3 s on CPU. **Phi-4-mini** is the main Phi option (MIT, 3.8B, plain dense) but its own benchmarks, Arena Hard 32.8 and the family's IFEval/format/verbosity weaknesses, plus the lack of creative-writing evidence, make it a weak fit; **Phi-4 14B** is too slow for CPU. No source tests refusals on esports vocabulary like "kill"/"headshot".

### Cited Findings

**Refusals / safety**
- Gemma 4 tech report: models "significantly outperform Gemma 3 and 3n models in improving safety, while keeping unjustified refusals low"; no refusal-rate numbers given. — [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1)
- Community "abliterated" E4B card reports stock E4B refusing ~98.8% of a "hard" (harmful-request) prompt set; the abliterated version drifts off-topic, loops, and sometimes answers in the wrong language (self-reported). — [OBLITERATUS/gemma-4-E4B-it-OBLITERATED](https://huggingface.co/OBLITERATUS/gemma-4-E4B-it-OBLITERATED)
- A tutorial reports stock E4B politely declined a "developer mode" jailbreak prompt. (community) — [qwe.edu.pl Gemma 4 jailbreak tutorial](https://www.qwe.edu.pl/tutorial/gemma-4-jailbreak-system-prompt/)
- Phi-4-mini: resists most jailbreaks; culturally framed or long multi-turn attacks can work. — [HF Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

**Repetition / verbosity**
- Phi-4-mini and Phi-4-mini-flash-reasoning cards: responses "can become repetitive or inconsistent" over very long chats. — [HF Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct); [HF Phi-4-mini-flash-reasoning](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)
- Phi-4: long, elaborate answers even for simple questions. — [Phi-4 Technical Report](https://arxiv.org/html/2412.08905)
- Gemma recommended sampling: temperature 1.0, top_p 0.95, top_k 64 (Gemma 3n also min_p 0.0–0.01, repetition penalty 1.0). — [HF card gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it); [Unsloth Gemma 3n guide](https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune)

**Per-model USP / weakness summary (facts cited above, sources repeated)**
- Gemma 4 E4B — USP: Apache 2.0; 4.5B effective / 8B total; 128K ctx; native system role; no empty thought block when thinking is off; Unsloth free notebooks; day-one llama.cpp/Ollama; MMLU-Pro 69.4 (above Gemma 3 27B's 67.6). Weakness: IFBench 44.0; Unsloth's E4B VRAM figures conflict; multimodal training quirks (use_cache bug, loss 13–15, fp16 audio overflow); load size 6.6–9.5 GB in Ollama. — [HF card](https://huggingface.co/google/gemma-4-E4B-it); [tech report](https://arxiv.org/html/2607.02770v1); [Unsloth](https://unsloth.ai/docs/models/gemma-4/train); [Ollama](https://ollama.com/library/gemma4)
- Gemma 4 E2B — USP: smallest/fastest (2.3B effective, 5.1B total), trains in 8 GB. Weakness: IFEval 94.6 / IFBench 38.0, MMLU-Pro 60.0; HF demo showed hallucinated content. — [HF card](https://huggingface.co/google/gemma-4-E4B-it); [tech report](https://arxiv.org/html/2607.02770v1); [HF blog](https://huggingface.co/blog/gemma4)
- Gemma 4 12B Unified — USP: dense 11.95B, IFEval 97.2 / IFBench 74.0, EQ-Bench CW 15.71 (best measured ≤14B Gemma), Apache 2.0, own MTP drafter. Weakness: ~2.5–3× E4B's per-token compute (inference); fewer T4 fine-tune reports; encoder-free design means fine-tuning updates the whole multimodal path. — [tech report](https://arxiv.org/html/2607.02770v1); [benchmarklist](https://benchmarklist.com/benchmarks/creative_writing_v3/); [Google Dev blog](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)
- Gemma 3 4B/12B, Gemma 3n E2B/E4B — superseded by Gemma 4 (older licence; Gemma 3n limited to 32K ctx; fp16 T4 issues). Useful only as fallbacks with known EQ-Bench (3 4B 13.91, 3 12B 14.63) and FACTS (70.1 / 75.8) scores. — [Gemma 3 tech report](https://arxiv.org/html/2503.19786); [benchmarklist](https://benchmarklist.com/benchmarks/creative_writing_v3/)
- Phi-4-mini (3.8B) — USP: MIT, plain dense architecture, 128K ctx, strong maths for size (GSM8K 88.6). Weakness: Arena Hard 32.8, overall 63.5 vs Qwen2.5-7B 67.9; little factual knowledge; repetition in long chats; no IFEval or creative-writing data published. — [HF Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- Phi-4 (14B) — USP: MIT, Arena Hard 75.4. Weakness: IFEval 63.0, weak strict formatting, verbose, 16K context, tokenizer/EOS bugs needing Unsloth fixes. — [Phi-4 report](https://arxiv.org/html/2412.08905); [Simon Willison](https://simonwillison.net/2025/Jan/11/phi-4-bug-fixes)
- Phi-4-mini-flash-reasoning / Phi-4-reasoning(-plus) / Phi-4-reasoning-vision-15B — maths/visual-reasoning specialists ("designed and tested for math reasoning only"; vision/computer-use focus); not suitable. — [HF flash card](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning); [HF reasoning-vision card](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)

### Inferences
- "Kill"/"headshot" refusals are unlikely to matter after SFT: these are game words in a neutral reporting context, and 3–5k examples that use them will dominate the base model's behaviour. The 98.8% refusal figure is on genuinely harmful requests, not game commentary. Still worth a quick zero-shot smoke test on the base model.
- Repetition risk for live commentary comes mainly from the "previous commentary lines" in the prompt (the model copies them). Gemma's default repetition penalty 1.0 means we may need a mild penalty or n-gram blocking at serving time, tuned on held-out matches.
- Recommended shortlist for the comparison: Gemma 4 E4B (primary), Gemma 4 E2B (speed fallback), Gemma 4 12B (quality ceiling / Modal L4 fallback), Phi-4-mini (cross-family control with MIT licence). Phi-4 14B only makes sense on the GPU fallback.

### Gaps
- No source tests any Gemma or Phi model on violent-sounding esports vocabulary, or on casual/hype sports register.
- No measured repetition/verbosity comparison between Gemma 4 and Phi-4-mini on short-form generation.
- No community fine-tune reports (r/LocalLLaMA) comparing Gemma 4 E4B vs Phi-4-mini for style transfer with small datasets were found.
