# Other open-weight small LLMs (not Qwen, Gemma, Phi, Llama, Mistral, Granite or Liquid LFM) as QLoRA bases for a live CS2 caster-line generator, as of 2026-11-02

Scope note: research done 2026-11-02 via web search and fetches of model cards and docs. "Official" means a vendor model card, blog or docs page. "Independent" means a third-party measurement. "Community/secondary" means aggregators, blogs or Reddit snapshots. r/LocalLLaMA, LMArena and the live EQ-Bench table could not be read directly (the EQ-Bench page renders client-side and returned no rows), so community sentiment is thin. The use case is used as the yardstick throughout: 500–1500 token structured prompt in, `intensity + 20–40 token line` out, QLoRA on a free T4, CPU serving on an i7-13700K with 32 GB RAM via llama.cpp/Ollama at Q4–Q5, target 2–3 s per line.

## Q1. What exists: release date, sizes, architecture, total/active params, context, licence

### Takeaway
Few non-excluded families are serious for this job. They are OpenAI gpt-oss-20b (Aug 2025, 21B/3.6B-active MoE, Apache 2.0), NVIDIA's small Nemotron line (Nano 9B v2 from Aug 2025 and **Nemotron 3 Nano 4B from Mar 2026**, both hybrid Mamba-2, NVIDIA open model licence), and to a lesser degree Ai2 Olmo 3 7B (Nov 2025, Apache 2.0, fully open data), HF SmolLM3-3B (Jul 2025, Apache 2.0) and OpenBMB MiniCPM5-2B (Sep 2026, Apache 2.0). Most other 2026 releases from non-excluded labs are 30B+ total-parameter MoEs (GLM-4.7-Flash, Nemotron 3 / 3.5 30B-A3B, Kimi Linear 48B-A3B, Kolibri-1 78B-A3.5B), are reasoning- or code-specialised, or carry licence problems (Cohere CC-BY-NC; Tencent Hunyuan excludes the EU, UK and South Korea).

### Cited Findings

**OpenAI gpt-oss-20b (and newer OpenAI releases)**
- Released 2025-08-05 under Apache 2.0, alongside gpt-oss-120b — [TechCrunch, 2025-08-05](https://www.techcrunch.com/2025/08/05/openai-launches-two-open-ai-reasoning-models)
- 21B total and 3.6B active parameters. It is a MoE whose MoE weights were post-trained in MXFP4. It runs within 16 GB of memory. The licence is "Permissive Apache 2.0". The card says it "should only be used with the harmony format as it will not work correctly otherwise". Reasoning level (low/medium/high) is set in the system prompt, e.g. "Reasoning: high" (official, HF card, accessed 2026-11-02) — [HF openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- Model card paper published 2025-08-08 — [arXiv 2508.10925](https://arxiv.org/pdf/2508.10925)
- No newer OpenAI open-weight LLM was found. One blog, checked as of October 2026, says OpenAI had released no new open-weight model and that "gpt-oss-2" rumours are speculation with no model card or weights. That is a low-reliability secondary source — [aiera.blog](https://aiera.blog/?p=1948)
- A curated licence list (updated Oct 2026) still lists only GPT-OSS-20B (21B/3.6B, Apache 2.0) and gpt-oss-120b (5.1B active) for OpenAI — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)

**Hugging Face SmolLM3**
- SmolLM3-3B was released 2025-07-08: 3B parameters, about 11T training tokens, a Llama-style decoder with GQA and NoPE in every 4th layer. Context is 64k native and 128k with YaRN. It covers six languages (EN/FR/ES/DE/IT/PT). Dual mode uses `/think` and `/no_think`, with **thinking on by default**. The full recipe is public (official) — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- Licence Apache 2.0 — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)
- No SmolLM model newer than SmolLM3 was found in 2026 searches. Only older SmolLM2 sizes (135M/360M/1.7B) appear in Oct-2026 lists — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)

**AllenAI OLMo 2 / Olmo 3 / Olmo Hybrid (fully open data)**
- Olmo 3 launched 2025-11-20 as Base (7B, 32B), Think (7B, 32B), Instruct (7B) and RL-Zero (7B). Olmo 3.1 followed on 2025-12-12 with Think 32B and Instruct 32B plus refreshed RL-Zero 7B. Pretraining data (Dolma 3), post-training data (Dolci), code and intermediate checkpoints are all released (official) — [Ai2 blog: Olmo 3](https://allenai.org/blog/olmo3)
- Olmo 3 7B Instruct: Apache 2.0, 65K context, listed as released 2025-11-21 — [OpenRouter](https://openrouter.ai/allenai/olmo-3-7b-instruct-20251121)
- **Olmo Hybrid** (2026-03-05) is a 7B model that interleaves transformer layers with Gated DeltaNet linear-RNN layers. Ai2 reports RULER at 64k of 85.0 vs 70.9 for Olmo 3 7B (official) — [Ai2 blog: Olmo Hybrid](https://allenai-web.allen.ai/blog/olmohybrid); [Ai2 March 2026 newsletter](https://allenai.org/newsletters/2026-03-newsletter)
- Olmo Hybrid has a 3:1 Gated DeltaNet to attention ratio, a 65,536 context and Apache 2.0, with an `Olmo-Hybrid-Instruct-DPO-7B` checkpoint (secondary: inference host) — [Lambda](https://lambda.ai/inference-models/allenai/olmo-hybrid-instruct-dpo-7b)
- On 2026-10-02 Ai2 released OlmoCore 3, a MoE training stack. It is described as the base for its next-generation OLMo, but no new model had been announced (secondary) — [TechTimes](https://www.techtimes.com/articles/328470/20261002/ai2-releases-olmocore-3-open-moe-training-stack-delivers-27x-throughput-gain.htm)
- OLMo 2 (7B/13B) is superseded by Olmo 3 for this purpose. No OLMo 2 data was fetched this session.

**NVIDIA Nemotron Nano (hybrid Mamba)**
- **Nemotron Nano 9B v2**, released 2025-08-18: 9B, a Mamba2-Transformer hybrid (mostly Mamba-2 and MLP layers, only 4 attention layers), 128K context. It ships under the **NVIDIA Open Model License** and is "ready for commercial use". `/think` and `/no_think` control reasoning, and the default is reasoning **on** (official) — [HF nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- **Nemotron 3 Nano 4B**, on HF from 2026-03-16: 3.97B, a Mamba-2/Transformer hybrid with only 4 attention layers. The page does not call it a MoE. It is compressed from Nano 9B v2 with "Nemotron Elastic". The card gives 262K context, the **NVIDIA Nemotron Open Model License** and "ready for commercial use", with a clause against circumventing safety guardrails. English is the listed supported language. Reasoning is on by default and can be turned off with `enable_thinking=False` (official) — [HF nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
  - Conflict: Unsloth calls the 4B a "hybrid MoE" with 1M context — [Unsloth Nemotron 3 guide](https://unsloth.ai/docs/models/nemotron-3). NVIDIA's own card says the 4B was compressed from the dense 9B v2 and lists 262K, so the card should be preferred.
- **Nemotron 3 Nano 30B-A3B** (Dec 2025): about 30B total and about 3B active, a hybrid Mamba-2 + sparse MoE. An NVFP4 version followed in Jan 2026 — [vLLM blog, 2025-12-15](https://blog.vllm.ai/2025/12/15/run-nvidia-nemotron-3-nano.html); [NVIDIA research page](https://research.nvidia.com/labs/nemotron/Nemotron-3)
- Nemotron 3 Nano Omni 30B-A3B (Apr 2026) is multimodal and English only — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)
- **Nemotron 3.5 Lightning 30B-A3B** (reported release 2026-08-11) is a Mamba-2 + MoE + attention hybrid under **OpenMDW-1.1**. The card claims MMLU-Pro ≈81.9 and GPQA-Diamond ≈75.4, and the model targets agent and tool workloads — [HF NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4); [DataLearner (secondary)](https://www.datalearner.com/ai-models/pretrained-models/nemotron-3-5-lightning); [Unsloth Nemotron 3.5 guide](https://unsloth.ai/docs/models/nemotron-3.5)
- A curated list (Oct 2026) also names **"Nemotron 3.5 Bolt-E4B", a fine-tune of Gemma 4 E4B**. That is relevant to the project shortlist but belongs to the Gemma family, so it is flagged for the Gemma researcher — [xigh/open-weight-models](https://github.com/xigh/open-weight-models). October-2026 search roundups (taskade, tooldirectory.ai) also mention a "Nemotron 3.5 Bolt" VLM of about 74B total / 9B active under OpenMDW-1.1. I did not open those pages, so the exact attribution is unverified.

**DeepSeek R1 distills / small DeepSeek models**
- DeepSeek-R1-0528-Qwen3-8B is Qwen3-8B-Base with chain-of-thought distilled from R1-0528, under MIT, with commercial use and distillation allowed. Recommended temperature is 0.6. The card claims AIME24 86.0, AIME25 76.3 and GPQA-D 61.1 (official) — [HF deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- DeepSeek's 2026 releases are large. V4-Flash is 284B total / 13B active, MoE, 1M context, MIT, from 2026-04-24 (source conflict: another source lists the repo at 304B, released 2026-07-31) — [Simon Willison, 2026-04-24](https://simonwillison.net/2026/apr/24/deepseek-v4); [Yotta Labs, 2026-08-07](https://www.yottalabs.ai/post/deepseek-v4-release-date-specs-how-to-access-2026)
- The only other small DeepSeek models are specialists: DeepSeek-OCR at 3.3B (MIT), DeepSeek-OCR-2 at 3.4B and DeepSeek-Prover-V2-7B (Lean 4) — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)

**Zhipu / Z.ai GLM small models**
- GLM-4-9B-0414: 9B, MIT, Chinese and English focus. GLM-Z1-9B-0414 is the reasoning-trained sibling. Transformers, vLLM and SGLang are supported, and quantized versions exist for llama.cpp, Ollama and LM Studio (official) — [HF zai-org/GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414)
- **GLM-4.7-Flash** is a "30B-A3B MoE" (metadata shows 31B), model type `glm4_moe_lite`, MIT, English and Chinese. The card compares it with Qwen3-30B-A3B-Thinking-2507 and GPT-OSS-20B and reports AIME25 91.6, GPQA 75.2, SWE-bench Verified 59.2 and τ²-Bench 79.5. It mentions a "Preserved Thinking" mode (official) — [HF zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- GLM-4.7-Flash was released 2026-01-20 with 200K context (secondary) — [Verdent guide](https://www.verdent.ai/guides/model/glm-4-7-flash)

**Baidu ERNIE 4.5 small**
- ERNIE-4.5-21B-A3B-PT: 21B total / 3B active, 28 layers, 64 text experts (6 active) plus 2 shared, 131,072 context, Apache 2.0. Needs Transformers ≥4.54 and vLLM ≥0.10.2. Quantized versions exist for llama.cpp, Ollama and LM Studio. The card has no benchmarks and no release date (collection updated 2025-11-11) (official) — [HF baidu/ERNIE-4.5-21B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT)
- ERNIE-4.5-0.3B-PT: about 0.36B dense, 131,072 context, Apache 2.0, text-only post-trained model (official) — [HF baidu/ERNIE-4.5-0.3B-PT](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT)
- A reasoning variant, ERNIE-4.5-21B-A3B-Thinking, exists with GGUFs (Q8_0 is 23.21 GB) — [bartowski GGUF](https://huggingface.co/bartowski/baidu_ERNIE-4.5-21B-A3B-Thinking-GGUF)

**Tencent Hunyuan small**
- Hunyuan dense family (0.5B, 1.8B, 4B, 7B; Pretrain and Instruct), on HF from 2025-07-30. GQA, 256K native context. **Slow thinking is on by default** and is disabled with `enable_thinking=False` or a `/no_think` prefix (official) — [HF tencent/Hunyuan-7B-Instruct](https://huggingface.co/tencent/Hunyuan-7B-Instruct)
- **Licence: the Tencent Hunyuan Community License, whose header states it does not apply in the EU, UK or South Korea** — [HF Hunyuan-1.8B-Instruct LICENSE commit](https://huggingface.co/tencent/Hunyuan-1.8B-Instruct/commit/21ab9fd367ee99ba8001d34a182252ddb2ed255c); [canirun.ai licence summary (secondary)](https://canirun.ai/license/tencent-hunyuan-community/)

**ByteDance Seed-OSS**
- Seed-OSS-36B (Base with and without synthetic data, plus Instruct), released 2025-08-20 under Apache 2.0. It has a user-set "thinking budget": the default is unlimited thinking, and a budget of 0 gives direct answers. Recommended sampling is temperature 1.1, top_p 0.95 — [ByteDance Seed blog](https://seed.bytedance.com/en/blog/seed-oss-open-source-models-release); [GitHub seed-oss](https://github.com/ByteDance-Seed/seed-oss)
- 512K token context — [TMCnet](https://blog.tmcnet.com/blog/rich-tehrani/ai/bytedance-releases-seed-oss-36b-with-512k-token-context-window.html)
- 36B dense is outside the 1–14B range. No smaller general Seed-OSS model was found. ByteDance's small open models are specialists, e.g. UI-TARS-1.5-7B for GUI grounding — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)

**Moonshot / Kimi**
- Kimi-Linear-48B-A3B (Oct 2025): 48B total / 3B active, a linear-attention hybrid, 1M context, MIT — [xigh/open-weight-models](https://github.com/xigh/open-weight-models). No Kimi model under 48B total was found for 2026.

**Cohere**
- Command R7B (c4ai-command-r7b-12-2024) is described as 7B (8B in safetensors), with 128K context and 23 languages. It is under **CC-BY-NC-4.0 plus Cohere's Acceptable Use Policy**, the repo is gated, and the card calls it an "open weights research release" (official) — [HF CohereLabs/c4ai-command-r7b-12-2024](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024). Released 2024-12-13 — [DevelopersDigest](https://www.developersdigest.tech/models/command-r7b)
- Tiny Aya (Feb 2026) is an open-weight multilingual family covering 70+ languages. The licence was not confirmed, and one article says "commercial license terms apply" — [ai2.work (secondary)](https://ai2.work/technology/cohere-open-sources-tiny-aya-70-language-ai-models-in-2026/)
- North-Mini-Code-1.0 (CohereLabs) is a 30B/3B MoE agentic coder under Apache 2.0, listed in Oct 2026 — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)

**Other notable small or small-active models found (2025–2026)**
- **OpenBMB MiniCPM5-2B**, released 2026-09-07 under Apache 2.0: 2.52B dense (1.98B non-embedding), 131,072 context, hybrid-thinking mode. Base, SFT, GGUF and MLX builds plus a speculative-decoding draft model shipped on day one. It scores 15 on the Artificial Analysis Intelligence Index v4.2 (secondary news) — [AI Weekly](https://aiweekly.co/alerts/openbmb-ships-minicpm5-2b-a-25b-apache-20-on-device-model-averaging-539-across); [OrcaRouter](https://www.orcarouter.ai/blog/minicpm5-2b-open-weights-release)
- **Ant Group Ling-3.0-tiny**: 7.9B total / 1.3B active MoE, MIT, about 4.5 GB, "AA Intelligence Index 25" — [xigh/open-weight-models](https://github.com/xigh/open-weight-models). Another source says it scored 16 on AA Index v4.2 and puts it at about 7.8B — [AI Weekly](https://aiweekly.co/alerts/openbmb-ships-minicpm5-2b-a-25b-apache-20-on-device-model-averaging-539-across). The two figures probably come from different index versions and are unreconciled.
- **Zyphra ZAYA1-8B** (May 2026): 8.4B total / 0.76B active MoE, hybrid Mamba/attention, Apache 2.0, aimed at on-device use — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)
- **Arcee Trinity Nano Preview** (6B total / about 1B active, an "experimental chat model") and **Trinity Mini** (26B/3B, reasoning), released 2025-12-01 under Apache 2.0 — [OpenSourceForU](https://www.opensourceforu.com/2025/12/arcee-launches-apache-licensed-trinity-suite-to-challenge-chinas-open-weight-lead/). Unverified claim: on 2026-05-29 the whole Trinity family was moved, retroactively, to the OpenMDW-1.1 licence — [aiwiki](https://aiwiki.ai/wiki/arcee_ai); [Layer3 Labs](https://www.layer3labs.io/guides/arcee-trinity-large-explained)
- **TII Falcon-H1 / Falcon-H1R-7B**: Falcon-H1 is a hybrid family that runs "out of the box" in vLLM, SGLang, Transformers and llama.cpp — [GitHub tiiuae/Falcon-H1](https://github.com/tiiuae/Falcon-H1). Falcon-H1R-7B is a **reasoning** model built on Falcon-H1-7B-Base under the "Falcon-LLM License" (custom) — [HF tiiuae/Falcon-H1R-7B](https://huggingface.co/tiiuae/Falcon-H1R-7B)
- Shanghai AI Lab InternLM3-8B: 8B dense, Apache 2.0, has a thinking mode — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)
- Large-total MoEs with 3–4.3B active, all Oct-2026 listings: Aleph Alpha Kolibri-1 (78B/3.46B, German and English, Apache 2.0), Yandex AliceAI-Foundation-80B-A3B (base only), Krafton Raon-1-mini-preview (28.8B/4.3B, Apache 2.0, gated), Poolside Laguna XS 2.1 and JetBrains Mellum 2 (code) — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)
- Non-commercial small models to avoid: Salesforce xLAM-8B and MadeAgents Hammer2.1-7B, both CC-BY-NC 4.0 — [xigh/open-weight-models](https://github.com/xigh/open-weight-models)

### Inferences
- In 2026 the non-excluded labs mostly shipped 30B-A3B-class MoEs rather than dense 4–8B models. Only NVIDIA (Nemotron 3 Nano 4B), OpenBMB (MiniCPM5-2B), Ai2 (Olmo Hybrid 7B) and Zyphra (ZAYA1-8B) shipped new sub-10B-total models this year.
- Licence flags for a public student project:
  - **Command R7B** is NC, which is acceptable for a non-commercial student release but blocks any later commercial use.
  - **Hunyuan** is unusable if the student is in the EU, UK or South Korea.
  - **Falcon-LLM** and **NVIDIA open model** licences are custom but permit commercial use.
  - **OpenMDW-1.1** (Nemotron 3.5, possibly Trinity) is a permissive Linux Foundation model licence, but its text was not reviewed here.
  - Apache 2.0 or MIT covers gpt-oss, SmolLM3, Olmo 3, ERNIE 4.5, GLM, Seed-OSS, MiniCPM5, Ling, ZAYA1 and DeepSeek.

### Gaps
- gpt-oss-20b context length (widely reported as 128k) was not confirmed on a page fetched this session (openai.com returned 403).
- The ERNIE 4.5 release date (believed to be 2025-06-30) and the Falcon-H1R release date were not confirmed on a fetched page.
- No Apple open-weight general LLM release for 2025–2026 was found. Apple's open releases appear to be research or vision-only, but this was not searched in depth.
- The Tiny Aya licence is unconfirmed, and the OpenMDW-1.1 licence text was not reviewed.
- The "Nemotron 3.5 Bolt" specs come from unopened roundup pages.

## Q2. Unsloth / PEFT QLoRA support and T4 (16 GB) feasibility

### Takeaway
Three candidates are documented to fit a free T4: **gpt-oss-20b QLoRA (≈14 GB VRAM)**, **Nemotron 3 Nano 4B ("fits on a free Colab GPU")**, and by size Olmo 3 7B, SmolLM3-3B, MiniCPM5-2B and Nemotron Nano 9B v2 (standard dense models, not explicitly verified here). Every 30B-total MoE (GLM-4.7-Flash, Nemotron 3 Nano 30B / 3.5 Lightning, Trinity Mini, Kimi Linear, Kolibri-1) is **not T4-trainable**; Unsloth puts Nemotron 3 Nano 30B on an 80 GB A100. gpt-oss adds harmony-template and MXFP4-export complexity.

### Cited Findings

**gpt-oss-20b**
- QLoRA needs about 14 GB VRAM for gpt-oss-20b, and BF16 LoRA about 44 GB. Unsloth recommends ≥16 GB for stable local runs. Data is a `messages` list rendered with `tokenizer.apply_chat_template` into Harmony (`<|start|>`, `<|message|>`, `<|return|>`), and Unsloth says it fixed the chat template. `reasoning_effort` is low, medium or high (default low) and is passed to `apply_chat_template`. The example dataset is HuggingFaceH4/Multilingual-Thinking, which contains reasoning traces (official) — [Unsloth gpt-oss fine-tune tutorial](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)
- Export works through `save_pretrained_merged(..., save_method="mxfp4")` or `"merged_16bit"`, then `convert_hf_to_gguf.py` and `llama-quantize`. **Internal contradiction:** the FAQ on the same page says no library supports exporting to MXFP4 after training — [Unsloth gpt-oss fine-tune tutorial](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)
- The free Colab notebook is the quickstart for gpt-oss-20b. The ~14 GB QLoRA fits the T4's ~15 GB usable memory — [Unsloth Colab install](https://unsloth.ai/docs/get-started/install/google-colab); [Unsloth gpt-oss blog, Aug 2025](https://unsloth.ai/blog/gpt-oss)
- Community report: running inference on Colab/Kaggle T4 after fine-tuning threw errors, so the author tested on another GPU — [Zenn write-up (community)](https://zenn.dev/prgckwb/articles/gpt-oss-finetuning-unsloth)
- OpenAI states the 20B "can be fine-tuned on consumer hardware" — [HF openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

**NVIDIA Nemotron**
- Unsloth supports fine-tuning all Nemotron models. **The 4B fits on a free Colab GPU; the 30B does not.** The 30B uses an 80 GB A100 notebook, and 16-bit LoRA on Nano 30B takes about 60 GB. The MoE router is frozen by default. To keep reasoning ability, mix ≥75% reasoning and ≤25% non-reasoning examples (official) — [Unsloth Nemotron 3 guide](https://unsloth.ai/docs/models/nemotron-3)
- Nemotron 3.5 Lightning is supported for fine-tuning, but Unsloth gives no VRAM figure. It warns that memory depends on context length, optimizer and batch size, not just the 3B active parameters — [Unsloth Nemotron 3.5 guide](https://unsloth.ai/docs/models/nemotron-3.5)
- The Nemotron 3 Nano 4B card has no fine-tuning instructions — [HF Nemotron-3-Nano-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)

**Others**
- Hunyuan's card documents LLaMA-Factory SFT with think/answer tags and `/no_think` prefixes for fast-mode examples. The required Transformers code sits on a companion branch, pending review — [HF tencent/Hunyuan-7B-Instruct](https://huggingface.co/tencent/Hunyuan-7B-Instruct)
- Unsloth's model catalog lists an ERNIE 4.5 21B-A3B Thinking GGUF, but I found no ERNIE fine-tuning notebook — [Unsloth model catalog](https://unsloth.ai/docs/get-started/all-our-models)
- SmolLM3 needs Transformers ≥4.53, and the full nanotron training recipe is public — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)

### Inferences
- **gpt-oss-20b is the only MoE in scope with a documented T4 QLoRA path.** The T4's ~15 GB is near the 14 GB requirement, so seq len 2048 and batch 1 with gradient accumulation will be needed. The 500–1500 token prompts should fit at 2048.
- The harmony format makes the training target design matter. Train only the `final` channel (intensity token + line) with an empty or minimal `analysis` channel, and set `reasoning_effort=low`. Otherwise the model keeps generating analysis tokens at inference and breaks the latency budget. That this works cleanly is not confirmed (see Gaps).
- Unsloth's "≥75% reasoning examples" advice for Nemotron is for keeping reasoning. We want non-reasoning output, so train every example in `enable_thinking=False` / no-think format. The same applies to SmolLM3 (`/no_think`), Hunyuan (`/no_think`) and MiniCPM5 (hybrid thinking).
- A 21B-total MoE such as ERNIE 21B-A3B in 4-bit is roughly 11–12 GB of weights (an estimate), which makes T4 QLoRA borderline and untested by Unsloth. Treat it as infeasible without evidence. 30B+ MoEs are clearly infeasible on a T4. A Modal L4 (24 GB) would also be tight for 30B-A3B QLoRA, so an A100 would be needed.
- Models with Mamba-2 / Gated DeltaNet layers (Nemotron, Falcon-H1, Olmo Hybrid, ZAYA1, Kimi Linear) depend on custom kernels in training. Unsloth explicitly covers Nemotron. The others have no documented Unsloth path.

### Gaps
- Unsloth T4-specific confirmation for Nemotron 3 Nano 4B ("free Colab GPU" is presumed to be a T4) and whether its Mamba kernels run in fp16 on Turing (T4 has no bf16).
- Whether gpt-oss-20b can be fine-tuned to skip the analysis channel entirely, i.e. go straight to `final`. No official guidance was found; the Unsloth tutorial only shows thinking data.
- Unsloth support status for SmolLM3, Olmo 3, MiniCPM5, Ling-3.0-tiny, ZAYA1 and Olmo Hybrid was not checked page-by-page this session.

## Q3. llama.cpp / GGUF / Ollama support and CPU-serving fit (i7-13700K, 32 GB RAM)

### Takeaway
gpt-oss-20b (Ollama `gpt-oss:20b`, 16 GB footprint) and Nemotron 3 Nano 4B (Unsloth GGUF, about 3 GB at 4-bit, llama.cpp listed officially) have first-class local support. ERNIE 4.5, GLM-4-9B, Hunyuan, Command R7B and Falcon-H1 have community quants. GLM-4.7-Flash needed Jan-2026 llama.cpp fixes. Kimi Linear needed an unmerged PR in Dec 2025. Olmo Hybrid has no confirmed llama.cpp support. For our latency budget, a 3–4B dense or ≤3.6B-active MoE is the right size class on CPU.

### Cited Findings
- gpt-oss-20b runs via `ollama run gpt-oss:20b`, LM Studio, Transformers, vLLM and SGLang, and llama.cpp is listed for quantized use. It "runs within 16 GB of memory" thanks to MXFP4 (official) — [HF openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- Nemotron 3 Nano 4B lists HF, vLLM (≥0.15.1 plus a custom reasoning parser), **llama.cpp**, TRT-LLM and SGLang, with community quants for Ollama and LM Studio (official) — [HF Nemotron-3-Nano-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16). Unsloth's GGUF needs about 3 GB at 4-bit and 5 GB at 8-bit. Build the latest llama.cpp, with `-DGGML_CUDA=OFF` for CPU — [Unsloth Nemotron 3 guide](https://unsloth.ai/docs/models/nemotron-3)
- Nemotron 3 Nano 30B-A3B GGUF needs about 24 GB at 4-bit. Nemotron 3.5 Lightning GGUF needs about 20 GB at 4-bit and 33 GB at 8-bit, and llama.cpp is supported — [Unsloth Nemotron 3 guide](https://unsloth.ai/docs/models/nemotron-3); [Unsloth Nemotron 3.5 guide](https://unsloth.ai/docs/models/nemotron-3.5)
- Falcon-H1 works "out of the box" with llama.cpp, and official GGUFs are published — [GitHub tiiuae/Falcon-H1](https://github.com/tiiuae/Falcon-H1)
- GLM-4.7-Flash GGUF is usable but requires a recent llama.cpp; a Jan-2026 fix addressed looping and output quality (secondary) — [dev.co](https://dev.co/ai/llms/glm-4-7-flash-gguf)
- Kimi-Linear GGUFs required a llama.cpp build with `kimi_linear` arch support. A Dec-2025 PR (#18381) made it work, and early users reported slowdowns past 3k tokens (community) — [APEX GGUF card](https://huggingface.co/myric/Kimi-Linear-48B-A3B-Instruct-APEX-GGUF); [r/LocalLLaMA snapshot, 2025-12-26](https://reddit.sentinel-team.org/posts/1pvxmqt/snapshots/2025-12-26T13%3A07%3A59.337519Z)
- ERNIE-4.5-21B-A3B and 0.3B, GLM-4-9B-0414, Hunyuan-7B and Command R7B cards each link quantized builds for llama.cpp, Ollama and LM Studio — [ERNIE 21B-A3B](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT); [ERNIE 0.3B](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT); [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414); [Hunyuan-7B](https://huggingface.co/tencent/Hunyuan-7B-Instruct); [Command R7B](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024)
- MiniCPM5-2B shipped GGUF and MLX builds, plus a speculative-decoding draft model, on release day — [OrcaRouter](https://www.orcarouter.ai/blog/minicpm5-2b-open-weights-release)
- Olmo Hybrid: the only deployment docs found are for vLLM; nothing was found on llama.cpp — [Lambda](https://lambda.ai/inference-models/allenai/olmo-hybrid-instruct-dpo-7b)

### Inferences
- These are unmeasured rules of thumb and must be checked with `llama-bench` on the target i7-13700K. CPU decode speed scales roughly with active-parameter bytes per token, and prefill scales with active compute.
  - A dense 8B at Q4 (Qwen3-8B class) will likely struggle to produce 30–45 output tokens **plus** prefill 500–1500 prompt tokens within 2–3 s on CPU.
  - A 3–4B dense model (Nemotron 3 Nano 4B, SmolLM3-3B) or a ~3–3.6B-active MoE (gpt-oss-20b, ERNIE 21B-A3B) should be about 2× faster per token.
  - **Prefill, not decode, is likely to dominate.** Keep a stable prompt prefix (system prompt + match card) and append only new events, so llama.cpp's prompt cache reuses the KV cache.
- Memory fit on 32 GB: gpt-oss-20b (~13–16 GB), any 4B (~3 GB) and ERNIE 21B-A3B Q4 (estimated ~12–13 GB) fit easily. 30B-A3B models (~20–24 GB) fit but leave little headroom. Kimi Linear 48B and Kolibri 78B do not fit comfortably at Q4.
- Hybrid recurrent models (Nemotron Mamba-2, Falcon-H1, Olmo Hybrid, Kimi Linear) carry a specific risk: llama.cpp's prefix KV-cache reuse works differently for recurrent state. If partial-prefix reuse is unsupported, every request re-prefills the full 500–1500 tokens. Mamba prefill is cheap per token, which may offset this, but it needs testing.

### Gaps
- No independent CPU tokens/s figures for gpt-oss-20b, Nemotron 3 Nano 4B or ERNIE 21B-A3B on a 13700K-class CPU were collected this session.
- llama.cpp prompt-cache behaviour was not verified for hybrid Mamba models or for gpt-oss's sliding-window attention layers.
- Ollama native support for Nemotron 3 Nano 4B, MiniCPM5 and Ling-3.0-tiny was not verified; only "community quants" links are known.

## Q4. Relevant benchmarks (IFEval, creative writing, hallucination, chat) and quirks (always-reasoning, refusals)

### Takeaway
Little published data covers what we need: casual or hype English register, format discipline and grounding. The best official numbers among these models are Nemotron 3 Nano 4B's **reasoning-off** scores: IFEval 82.8 (prompt) / 88.0 (instruction), HaluEval 62.2 and EQ-Bench3 63.2. Olmo 3 7B Instruct roughly matches Qwen3-8B on IFEval (85.6 vs 86.3) but trails it on chat (AlpacaEval 2 LC 40.9 vs 49.8). gpt-oss-20b appears weak on creative writing in an aggregator's EQ-Bench CW v3 table and has over-refusal claims. Most new models default to thinking, and the R1 distills, Falcon-H1R and the GLM-4.7-Flash/Z1 line are reasoning-first.

### Cited Findings

**IFEval / chat / hallucination (official unless noted)**
- Nemotron 3 Nano 4B reasoning-off: IFEval-Prompt 82.8, IFEval-Instruction 88, IFBench 43.2/44.2, **HaluEval 62.2**, **EQ-Bench3 63.2**, BFCL v3 61.1, RULER 128k 91.1. Reasoning-on: IFEval 87.9/92 — [HF Nemotron-3-Nano-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- Nemotron Nano 9B v2: IFEval 90.3, but only with **reasoning on**. No reasoning-off IFEval is reported — [HF Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- Olmo 3-Instruct 7B vs Qwen3 8B (no reasoning), per Ai2's own table: IFEval 85.6 vs 86.3, IFBench 32.3 vs 29.3, **AlpacaEval 2 LC 40.9 vs 49.8**, MMLU 69.1 vs 80.4, Safety 87.3 vs 78.0 — [Ai2 blog: Olmo 3](https://allenai.org/blog/olmo3)
- Hunyuan-7B-Instruct: IF-Eval 79.3, SysBench 72.7 — [HF Hunyuan-7B-Instruct](https://huggingface.co/tencent/Hunyuan-7B-Instruct)
- SmolLM3: no no-think IFEval figure in the launch blog. It is described as competitive with Qwen3-4B and Gemma3-4B, with Qwen3-4B "generally" highest. AIME25 is 36.7 with thinking vs 9.3 without — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- ERNIE 4.5 21B-A3B-PT: no benchmarks on the card — [HF](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT)
- GLM-4.7-Flash's published benchmarks are all reasoning, agentic or coding (AIME25, GPQA, SWE-bench, τ²); none are on chat or creative writing — [HF GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)

**Creative writing (EQ-Bench Creative Writing v3)**
- The official leaderboard page rendered no data rows when fetched on 2026-11-02 — [eqbench.com](https://eqbench.com/creative_writing.html)
- Aggregator, sampled 2026-05-28, layout garbled: **gpt-oss-20b 721.7 vs gpt-oss-120b 1034.5** on the Creative Writing v3 score overview (likely Elo). Treat as unverified independent data — [benchmarklist.com](https://benchmarklist.com/benchmarks/creative_writing_v3/)
- The benchmark's own repo says scores need its runs file and a judge model, and should be treated "as a guide rather than absolute truth" — [GitHub EQ-bench/creative-writing-bench](https://github.com/EQ-bench/creative-writing-bench)

**Quirks: reasoning by default**
- gpt-oss: reasoning effort can only be set to low, medium or high. The card gives no "off" switch — [HF openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b); [Unsloth tutorial](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)
- These default to thinking but can be switched off:
  - Nemotron Nano 9B v2 (`/no_think`) — [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
  - Nemotron 3 Nano 4B (`enable_thinking=False`; harder tasks "may lose some accuracy") — [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
  - SmolLM3 (`/no_think`) — [HF blog](https://huggingface.co/blog/smollm3)
  - Hunyuan (`/no_think`) — [HF](https://huggingface.co/tencent/Hunyuan-7B-Instruct)
  - Seed-OSS (thinking budget 0) — [ByteDance Seed blog](https://seed.bytedance.com/en/blog/seed-oss-open-source-models-release)
- Nemotron 3 / 3.5 30B-A3B: the chat template's generation prompt ends with `<think>`, so thinking is on by default. Unsloth recommends 32K–262K max new tokens — [Unsloth Nemotron 3 guide](https://unsloth.ai/docs/models/nemotron-3); Lightning has separate thinking and instruct sampling modes — [Unsloth Nemotron 3.5](https://unsloth.ai/docs/models/nemotron-3.5)
- DeepSeek-R1-0528-Qwen3-8B is a chain-of-thought distill evaluated with 16 samples at temperature 0.6, i.e. a reasoning model — [HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B). Falcon-H1R-7B is likewise a reasoning model — [HF](https://huggingface.co/tiiuae/Falcon-H1R-7B)

**Quirks: safety and refusals around violent words such as "kill"**
- The OpenAI model card notes that gpt-oss-20b underperforms o4-mini on the illicit/violent category of its production safety benchmarks. That is about harm avoidance, not over-refusal — [arXiv 2508.10925](https://arxiv.org/pdf/2508.10925)
- A blog test claims gpt-oss-20b left 54% of English questions unanswered vs 1.4% for GPT-5 ("overaligned"). The methodology is unverified and the questions were not game or creative prompts — [Substack, M. Sukhareva](https://msukhareva.substack.com/p/another-big-problem-of-gpt-oss-erratic)
- Community "abliterated" gpt-oss-20b builds with refusals removed are popular, e.g. 41.3K pulls on Ollama, which indirectly suggests users find the refusals limiting — [Ollama huihui_ai/gpt-oss-abliterated](https://registry.ollama.ai/huihui_ai/gpt-oss-abliterated:20b)
- Olmo 3-Instruct 7B scores higher than Qwen3-8B on Ai2's safety aggregate (87.3 vs 78.0) — [Ai2 blog](https://allenai.org/blog/olmo3)
- The Nemotron 4B licence/card warns against circumventing safety guardrails — [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)

### Inferences
- For this task the reasoning-off benchmark row is what counts. Nemotron 3 Nano 4B is the only non-excluded small model whose official card reports reasoning-off IFEval, hallucination (HaluEval) **and** EQ-Bench3 together, which makes it the best-documented fit. Its EQ-Bench3 (emotional intelligence/dialogue) of 63.2 is not a creative-writing score.
- gpt-oss-20b combines weak creative writing (aggregator), heavy reasoning-first post-training and over-refusal claims. That is a register and tone risk for hype casting. QLoRA on 3–5k examples usually fixes format and register, but a safety prior that reacts to words like "kill", "execute" or "headshot" in a game context could surface as hedging. Test it before committing.
- Olmo 3's higher safety score and lower AlpacaEval suggest a more conservative, less chatty prior than Qwen3-8B. That is mildly negative for hype register.
- Any model that thinks by default needs every training example in no-think format, plus `enable_thinking=False` or `/no_think` at serving time. Otherwise a stray `<think>` block blows the 2–3 s budget. Pure reasoning distills (R1-Distill, R1-0528-Qwen3-8B, Falcon-H1R, GLM-Z1) give no benefit here and their base models are better starting points.

### Gaps
- No reliable EQ-Bench Creative Writing v3 scores were retrieved for Nemotron 3 Nano 4B, Olmo 3, SmolLM3, MiniCPM5, ERNIE 4.5 or GLM. LMArena open-model rankings for 2026 were not retrievable this session.
- No direct evidence was found, for any model, of refusals on esports or game violence wording. A 20-prompt smoke test before choosing would settle it.
- No independent (non-vendor) IFEval re-measurements were found for these models.

## Q5. USP for our use and verdict

### Takeaway
Only two outside models are worth a head-to-head against Qwen3-8B and Gemma 4 E4B. **NVIDIA Nemotron 3 Nano 4B** (Mar 2026) is the strongest: a 4B English-first hybrid Mamba model with official reasoning-off IFEval and HaluEval scores, a free-Colab fine-tune path, 3 GB GGUF and a commercial-use licence. **gpt-oss-20b** is fast on CPU thanks to 3.6B active, Apache 2.0 and T4-trainable in 14 GB, but it is riskier because of harmony, an always-reasoning design, MXFP4 export issues and refusals and weak creative writing. Everything else is niche or unsuitable.

### Cited Findings
- See Q1–Q4 for the facts behind each verdict. Key anchors:
  - Nemotron 3 Nano 4B card — [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
  - Unsloth: 4B fits free Colab — [Unsloth Nemotron 3](https://unsloth.ai/docs/models/nemotron-3)
  - gpt-oss QLoRA ≈14 GB — [Unsloth gpt-oss tutorial](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)
  - Olmo 3 vs Qwen3-8B table — [Ai2](https://allenai.org/blog/olmo3)
  - Hunyuan territorial exclusion — [HF LICENSE commit](https://huggingface.co/tencent/Hunyuan-1.8B-Instruct/commit/21ab9fd367ee99ba8001d34a182252ddb2ed255c)
  - Command R7B CC-BY-NC — [HF](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024)

### Inferences

Verdict table, as of 2026-11-02. Licence, size and support facts are cited in Q1–Q4; the verdicts are my judgement.

| Model (release) | Size / arch | Licence | T4 QLoRA | CPU GGUF fit | USP for CS2 casting | Verdict |
|---|---|---|---|---|---|---|
| **NVIDIA Nemotron 3 Nano 4B** (2026-03-16) | 3.97B dense hybrid Mamba-2 (4 attention layers) | NVIDIA Nemotron Open Model License (commercial OK, guardrail clause) | Yes (Unsloth: free Colab) | ~3 GB Q4, llama.cpp official | English-first. Reasoning-off IFEval 82.8/88, HaluEval 62.2, EQ-Bench3 63.2. Likely about 2× faster than an 8B on CPU | **Serious candidate**: add it to the bake-off as the "fast small" option vs Gemma 4 E4B |
| **OpenAI gpt-oss-20b** (2025-08-05) | 21B MoE, 3.6B active, MXFP4 experts | Apache 2.0 | Yes, ~14 GB (Unsloth) | ~13–16 GB, Ollama native | Biggest knowledge for the active cost, fast CPU decode | **Serious but risky**: harmony template, no reasoning "off" (only low), MXFP4→GGUF export confusion, weak CW v3 per aggregator, over-refusal claims. Smoke-test "kill" prompts first |
| Nemotron Nano 9B v2 (2025-08-18) | 9B dense hybrid Mamba-2 | NVIDIA Open Model License | Probably (dense 9B) | ~5–6 GB Q4 (est.) | Strong IFEval (reasoning on) | **Niche / superseded** by the 4B for our latency |
| Ai2 Olmo 3 7B Instruct (2025-11-20) | 7B dense | Apache 2.0, fully open data | Yes (dense 7B, est.) | Standard | Fully open data (good for a student write-up). IFEval ≈ Qwen3-8B | **Niche**: weaker chat (AlpacaEval 40.9 vs 49.8) and a more conservative safety prior than Qwen3-8B, with no latency gain |
| Ai2 Olmo Hybrid 7B (2026-03-05) | 7B, Gated DeltaNet + attention 3:1 | Apache 2.0 | Unknown | No llama.cpp evidence | Long-context efficiency, irrelevant at 1.5k tokens | **Unsuitable (for now)**: serving path unproven |
| HF SmolLM3-3B (2025-07-08) | 3B dense | Apache 2.0, fully open | Yes (est.) | Small | Fastest fully open option. `/no_think` mode | **Niche fallback** if 4B is too slow; generally below Qwen3-4B |
| OpenBMB MiniCPM5-2B (2026-09-07) | 2.52B dense, hybrid thinking | Apache 2.0 | Yes (est.) | GGUF day one | Newest small dense model, very fast | **Niche**: English casual register unproven, vendor-only benchmarks |
| Ant Ling-3.0-tiny (listed 2026) | 7.9B MoE, 1.3B active | MIT | Unknown | ~4.5 GB | Very low active parameter count | **Niche**: tooling and llama.cpp status unverified |
| Zyphra ZAYA1-8B (May 2026) | 8.4B MoE, 0.76B active, hybrid | Apache 2.0 | Unknown | Unknown | On-device | **Niche / experimental** |
| Arcee Trinity Nano Preview (2025-12-01) | 6B MoE, ~1B active | Apache 2.0 (possibly OpenMDW since 2026-05) | Unknown | Unknown | Small active count | **Niche**: labelled "experimental preview" |
| Baidu ERNIE 4.5 21B-A3B (2025) | 21B MoE, 3B active | Apache 2.0 | Borderline / unproven | ~12–13 GB Q4 (est.) | Fast CPU decode, 131k context | **Niche**: no English chat or creative benchmarks, no Unsloth fine-tune guide |
| ERNIE 4.5 0.3B | 0.36B dense | Apache 2.0 | Yes | Tiny | Possible speculative-decoding draft | **Unsuitable** as the main model (too small) |
| Zhipu GLM-4-9B-0414 (Apr 2025) | 9B dense | MIT | Yes (est.) | Quants exist | Permissive | **Niche / superseded**: Chinese-English focus, 9B is slow on CPU |
| Zhipu GLM-4.7-Flash (2026-01-20) | 30B MoE, 3B active | MIT | **No** (30B) | ~20 GB+, needed llama.cpp fixes | Strong agentic and coding | **Unsuitable**: not T4-trainable, reasoning or agent focus |
| NVIDIA Nemotron 3 Nano 30B-A3B / 3.5 Lightning (2025-12 / 2026-08-11) | 30B hybrid MoE, 3B active | NVIDIA OML / OpenMDW-1.1 | **No** (A100) | ~20–24 GB Q4 | Strong reasoning, 1M context | **Unsuitable for T4**. Only an option for an A100 fine-tune with CPU serving |
| Tencent Hunyuan 0.5–7B (2025-07-30) | Dense, 256K | Hunyuan Community (excludes EU/UK/KR) | Yes (LLaMA-Factory) | Quants exist | IF-Eval 79.3 (7B) | **Unsuitable** if the user is in the EU, UK or KR. Otherwise niche (thinks by default) |
| ByteDance Seed-OSS-36B (2025-08-20) | 36B dense | Apache 2.0 | **No** | Too slow on CPU | Thinking budget 0 | **Unsuitable** (size) |
| Moonshot Kimi-Linear-48B-A3B (2025-10) | 48B MoE, 3B active, linear attention | MIT | **No** | ~28 GB+ Q4 (est.), experimental llama.cpp | 1M context | **Unsuitable** |
| DeepSeek R1 distills / R1-0528-Qwen3-8B (2025) | Qwen/Llama bases | MIT (+ base licences) | Yes | Yes | None: always reasons | **Unsuitable** (latency, and the base Qwen is covered elsewhere) |
| Cohere Command R7B (2024-12-13) | 7–8B dense | **CC-BY-NC-4.0** + AUP, gated | Yes | Quants exist | 128K, RAG-tuned | **Unsuitable**: non-commercial licence and dated |
| Cohere Tiny Aya (Feb 2026) | Small multilingual | Unclear | n/a | n/a | 70+ languages (not needed) | **Unsuitable** |
| TII Falcon-H1 / Falcon-H1R-7B | Hybrid Mamba | Falcon-LLM License (custom) | Unknown | llama.cpp official | H1R is a reasoner | **Niche / unsuitable** |
| Large-total, small-active MoEs from Oct 2026 (Kolibri-1 78B-A3.5B, AliceAI 80B-A3B, Raon-1-mini 28.8B-A4.3B) | ≥28B total | Apache 2.0 (Raon gated) | **No** | Kolibri and AliceAI too big for 32 GB RAM | — | **Unsuitable** |

- Recommendation for the report writer: keep Qwen3-8B vs Gemma 4 E4B as the main shortlist and add **Nemotron 3 Nano 4B** as a third arm. It is a plausible replacement for Gemma 4 E4B as the latency-safe choice. Add **gpt-oss-20b** only if CPU latency for 8B-class dense models proves too slow and the 4B models lack fluency, and only after a quick refusal and harmony-export smoke test.
- Pass this to the Gemma researcher: NVIDIA's **Nemotron 3.5 Bolt-E4B** is listed as a Gemma 4 E4B fine-tune (Oct 2026) — [xigh/open-weight-models](https://github.com/xigh/open-weight-models).

### Gaps
- No head-to-head data, official or independent, compares Nemotron 3 Nano 4B or gpt-oss-20b with Qwen3-8B or Gemma 4 E4B on short-form casual or hype English generation. That has to come from the project's own bake-off.
- r/LocalLLaMA consensus on 2026 small models could not be read directly. Search returned only SEO roundups, some with obvious errors (e.g. a nonexistent "Llama 3.3 8B") — [SiliconFlow roundup (low quality)](https://www.siliconflow.com/articles/best-small-llms-under-10b-parameters)
- Nothing found on Apple open releases, and nothing beyond Command R7B, Tiny Aya and North-Mini-Code for Cohere.
