# Llama / Meta Muse, Mistral, IBM Granite and Liquid AI LFM as QLoRA base models for a CPU-served CS2 caster-line generator (as of 2 Nov 2026)

Scope note: research run on 2026-11-02. "Vendor" = official claim from the model maker (including when the vendor reports competitor scores it measured itself). "Community" = third-party, unverified. Facts with no date are flagged. Qwen/Gemma appear only where a cited source compares against them.

## Q1. Release dates, sizes, architecture, params, context, licence (per family)

### Takeaway
By Nov 2026 all four families have moved on a lot. Meta has released no new Llama since Llama 4 (Apr 2025). Its 2026 open models are branded **Muse**, and the smallest is the 30B dense **Muse Glimmer**, which is Apache 2.0. Mistral's in-range line is **Ministral 3 (3B/8B/14B, Dec 2025, Apache 2.0)**. IBM went back to plain dense transformers with **Granite 4.1 (3B/8B/30B, Apr 2026, Apache 2.0)**, after the **Granite 4.0** hybrid Mamba-2 models (Oct 2025). Liquid AI ships very often: LFM2.5 at 1.2B, 2.6B, 8B-A1B and 24B-A2B through 2026, and **LFM3-1.7B on 27 Oct 2026**, all under the LFM Open License, which is Apache-based and free below $10M revenue. Older Llama 3.x and Mistral 7B/NeMo/Ministral-2410 are 2023–24 vintage and superseded.

### Cited Findings

**Meta (Llama, then Muse)**
- Llama 3.1-8B: released 23 Jul 2024 (dated; 2024 vintage) — [Meta blog](https://ai.meta.com/blog/meta-llama-3-1/)
- Llama 3.2 1B/3B text plus 11B/90B vision: licence "Version Release Date: September 25, 2024" — [HF LICENSE.txt](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt)
- Llama 3.2 licence §1.b.i: anyone redistributing a derivative must "prominently display 'Built with Llama'…" and "include 'Llama' at the beginning of any such AI model name". §2 requires a separate licence above 700M monthly active users. The licence text itself has no EU clause — [HF LICENSE.txt](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt)
- The EU restriction in Llama 3.2 applies only to the **multimodal** models. Rights are not granted to individuals or companies domiciled in the EU, but end users of products built on them are unaffected. The 1B/3B text models are not covered by this restriction — [HF Llama 3.2 blog](https://huggingface.co/blog/llama32); [Meta FAQ](https://www.llama.com/faq/)
- Meta has released no new Llama weights since Llama 4 (April 2025). 2026 open-weight releases use the "Muse" name (aggregator plus Wikipedia, via search summary) — [Wikipedia: Meta AI](https://en.wikipedia.org/wiki/Meta_AI); [aicomparison.ai](https://aicomparison.ai/meta-llama-models-guide/)
- Muse Spark launched 8 Apr 2026 as a proprietary model. Muse Spark Open was released 15 Oct 2026 as an MoE with 816B total / 42B active (Wikipedia; not independently verified here) — [Wikipedia: Meta AI](https://en.wikipedia.org/wiki/Meta_AI)
- **Muse Glimmer** (10 Aug 2026): 30B **dense** model (2B Perception-Encoder ViT plus 28B text decoder). The decoder has 52 layers in 13 repeats of 3 sliding-window layers + 1 full-attention layer. Distilled from a larger Muse. **Apache 2.0**; the HF post mentions no AUP or attribution clause. Context is not stated in the post (an example config uses 32,768). No smaller Muse was released — [HF blog: Muse Glimmer](https://huggingface.co/blog/muse-glimmer); [VentureBeat](https://venturebeat.com/technology/meta-returns-to-open-source-with-muse-glimmer-an-apache-2-0-licensed-30b-parameter-ai-model-optimized-for-agents-available-now)
- Muse Glimmer needs over 55 GB at full precision and under 20 GB at about 4-bit (Meta, via press) — [VentureBeat](https://venturebeat.com/technology/meta-returns-to-open-source-with-muse-glimmer-an-apache-2-0-licensed-30b-parameter-ai-model-optimized-for-agents-available-now)

**Mistral**
- Mistral 7B: Sept 2023, Apache 2.0 (dated, superseded) — [Mistral news](https://mistral.ai/news/announcing-mistral-7b); [resumelens summary](https://www.resumelens.org/blog/ai/mistral-model-family)
- Mistral NeMo 12B (Instruct-2407; built with NVIDIA): 12B, 128k context, Tekken tokenizer (~128k vocab), **Apache 2.0**, recommended temperature 0.3, billed as a "drop-in replacement of Mistral 7B". MMLU 68.0 (5-shot, vendor). The page gives no explicit date; "2407" implies July 2024 — [HF model card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- Ministral 8B (2410, Oct 2024): **Mistral Research License** (research/non-commercial only; commercial use needs a separate licence). Ministral 3B was not openly downloadable for self-deployment without a commercial licence — [TechCrunch 16 Oct 2024](https://techcrunch.com/2024/10/16/mistral-releases-new-ai-models-optimized-for-edge-devices); [HF card](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/blob/refs%2Fpr%2F7/README.md?code=true)
- **Ministral 3** (released 2 Dec 2025): 3B, 8B and 14B dense, each in Base, Instruct and Reasoning variants, all with image understanding, all **Apache 2.0**. Up to 256K context — [Mistral: Introducing Mistral 3](https://mistral.ai/news/mistral-3/); [HF 14B Base card](https://huggingface.co/mistralai/Ministral-3-14B-Base-2512/blob/main/README.md)
- Ministral 3 8B Instruct 2512: about 8.4B language model plus 0.4B vision encoder (HF lists 9B). 256k context (vLLM default max-model-len 262,144). Instruct weights ship as **FP8**; Base and Reasoning ship as BF16. 11 languages — [HF card](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- The Ministral 3 tech report (arXiv 2601.08584, Jan 2026) says the models come from Mistral Small 3.1 via "Cascade Distillation". One version names Small 3.2 as the parent, so the sources conflict — [arXiv 2601.08584](https://arxiv.org/html/2601.08584v1)
- I found no new Ministral release in 2026 — [search summary incl. Wikipedia](https://en.wikipedia.org/wiki/Mistral_AI)
- **Mistral Small 4** (16 Mar 2026): about 119B-total MoE with ~6B active, Apache 2.0, ~262K context. It merges reasoning, multimodal and coding (secondary sources) — [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/mistral-small-4-four-models-unified-in-one-open-source-moe/); [OpenRouter](https://openrouter.ai/mistralai/mistral-small-2603)
- Mistral Large 4 (29 Oct 2026) appears in only one Wikipedia snippet and is unverified — [Wikipedia: Mistral AI](https://en.wikipedia.org/wiki/Mistral_AI)

**IBM Granite**
- **Granite 4.0** (2 Oct 2025):
  - H-Small: hybrid MoE, 32B total / 9B active
  - H-Tiny: hybrid MoE, 7B / 1B active
  - H-Micro: dense hybrid, 3B
  - Micro: dense transformer, 3B, "for platforms that don't yet support hybrid architectures"
  - Mamba-2 to transformer blocks in a **9:1** ratio. Tiny and Small use MoE with shared experts. No positional encoding (NoPE).
  - Trained on samples up to 512K tokens, validated to 128K. **Apache 2.0**, ISO 42001, cryptographically signed checkpoints.
  - Claims >70% less RAM for long inputs and concurrent batches — [IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- Granite 4.0 Nano 350M and 1B (hybrid and non-hybrid variants) are listed by Unsloth. The exact release date was not verified here — [Unsloth Granite 4.0 docs](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0)
- All Granite 4.0 models were trained on 15T tokens — [Unsloth Granite 4.0 docs](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0)
- **Granite 4.1** (29 Apr 2026 per the HF card; press article dated 1 May 2026):
  - Dense, decoder-only transformers at **3B, 8B and 30B**, base and instruct, plus FP8 variants. **Apache 2.0**.
  - About 15T tokens over 5 pre-training phases, then SFT and multi-stage RL. The RL stages target instruction following, **conversation quality**, factual accuracy and math.
  - IBM claims the 8B instruct matches or beats Granite 4.0 H-Small (32B MoE) and has a "simpler architecture that is easier to fine-tune" — [HF granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b); [Techstrong.ai](https://techstrong.ai/features/ibms-granite-4-1-smaller-models-smarter-design/)
- Granite 4.1 8B architecture: 40 layers, d=4096, 32 heads, 8 KV heads (GQA), RoPE, SwiGLU, RMSNorm, tied embeddings. The architecture table says 8B, but HF metadata shows 9B. Context **131,072** per the HF card, which conflicts with Techstrong's "512K". 12 languages — [HF granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b); contradicted by [Techstrong.ai](https://techstrong.ai/features/ibms-granite-4-1-smaller-models-smarter-design/)
- Granite Switch 4.1 (14 or 15 Oct 2026; sources disagree): modular 8B and 30B bundles of a base model, adapters and a speech-to-text encoder behind one endpoint. Benchmarks are vendor-reported only — [Unite.AI](https://www.unite.ai/ibm-releases-modular-granite-switch-4-1-models-in-8b-and-30b-sizes/); [TechJack](https://techjacksolutions.com/ai-brief/ibm-granite-switch-4-1-open-weight-agentic-endpoint/)
- No Granite 5 was found as of Nov 2026 — [search summary incl. Wikipedia](https://en.wikipedia.org/wiki/IBM_Granite)

**Liquid AI LFM**
- Release timeline from Liquid's own news index:
  - LFM2-2.6B: 23 Sep 2025
  - LFM2-8B-A1B: 7 Oct 2025
  - LFM2.5 (1.2B family): 5 Jan 2026
  - LFM2.5-1.2B-Thinking: 20 Jan 2026
  - LFM2-24B-A2B: 24 Feb 2026
  - LFM2.5-350M: 31 Mar 2026
  - LFM2.5-8B-A1B: 28 May 2026
  - LFM2.5-230M: 25 Jun 2026
  - LFM2.5-2.6B: 4 Aug 2026
  - LFM2.5-24B-A2B: 13 Oct 2026
  - LFM3: 27 Oct 2026

  — [Liquid AI news/models](https://www.liquid.ai/news/models)
- **LFM2.5-1.2B** (Base/Instruct, 5 Jan 2026): hybrid conv-plus-attention LFM2 architecture. Pre-training raised from 10T to 28T tokens, then SFT, preference alignment and multi-stage RL. Siblings: JP, VL-1.6B, Audio-1.5B — [Liquid blog LFM2.5](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- **LFM2.5-2.6B** (4 Aug 2026): Base plus a post-trained "agentic" model, ~34T tokens, 128K vocab, 128K context-extension phase. The post does not say whether it is reasoning-only — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-2-6b)
- **LFM2.5-8B-A1B** (28 May 2026): MoE with 8B total / ~1B active, GQA plus gated short-conv blocks. 128K context (LFM2-8B-A1B had 32,768). 128K vocab, 38T tokens. **Reasoning-only (emits explicit CoT)** — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b)
- **LFM2.5-24B-A2B** (13 Oct 2026): 24B total / 2B active, 40 layers, 64 experts with top-4 routing, conv plus GQA hybrid, 131,072 context. **Reasoning-only**. API price $0.10 in / $1.00 out per 1M tokens — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-24b-a2b)
- **LFM3** (27 Oct 2026): six ~2B models branched from LFM3-1.7B-Base (text, VL, audio, omni, two "decision" models).
  - LFM3-1.7B has 24 layers: 13 gated short convolutions (kernel 7, up from 3), 5 sliding-window attention layers (256-token window) and 6 global attention layers. Head-wise gated attention, partial RoPE, sandwich norm, 128K vocab.
  - 22T tokens of pre-training plus a 3.5T mid-train. **256K context**. About 1 GB at 4-bit — [Liquid blog LFM3](https://www.liquid.ai/blog/lfm3); [aiweekly](https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context)
- **LFM Open License v1.0**:
  - Text is identical to Apache 2.0 except for conditional grants, a commercial-use limit, a defined licensor and automatic termination.
  - Free commercial use while annual revenue stays **below $10M USD**. Non-commercial and research use is not subject to the threshold.
  - Fine-tuned weights may be redistributed (no copyleft). Liquid trademarks may not be used except for customary origin references. There is no naming rule for derivatives — [LFM license](https://www.liquid.ai/lfm-license)
- The LFM2.5-24B-A2B post links that licence. The LFM2.5 1.2B, 2.6B and 8B-A1B posts and the LFM3 post name no licence; they only say "download, fine-tune, and deploy without restrictions" — [Liquid LFM2.5-24B-A2B](https://www.liquid.ai/blog/lfm2-5-24b-a2b); [Liquid LFM2.5](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)

**Quick reference table (in-range or near-range candidates)**

| Model | Released | Type | Total / active | Context | Licence | Source |
|---|---|---|---|---|---|---|
| Llama 3.2 1B / 3B | 25 Sep 2024 | dense | 1B / 3B | 128K (not re-verified) | Llama 3.2 Community ("Llama" prefix, "Built with Llama") | [LICENSE](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt) |
| Llama 3.1 8B | 23 Jul 2024 | dense | 8B | 128K (not re-verified) | Llama 3.1 Community | [Meta](https://ai.meta.com/blog/meta-llama-3-1/) |
| Muse Glimmer | 10 Aug 2026 | dense + vision | 30B (28B decoder) | n/s | Apache 2.0 | [HF blog](https://huggingface.co/blog/muse-glimmer) |
| Mistral NeMo 12B | Jul 2024 | dense | 12B | 128K | Apache 2.0 | [HF](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| Ministral 3 3B/8B/14B | 2 Dec 2025 | dense + 0.4B vision | 3/8.4/14B | 256K | Apache 2.0 | [Mistral](https://mistral.ai/news/mistral-3/) |
| Mistral Small 4 | 16 Mar 2026 | MoE | ~119B / ~6B | ~262K | Apache 2.0 | [RITS](https://rits.shanghai.nyu.edu/ai/mistral-small-4-four-models-unified-in-one-open-source-moe/) |
| Granite 4.0 H-Micro / Micro | 2 Oct 2025 | hybrid Mamba-2 / dense | 3B | 128K validated | Apache 2.0 | [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models) |
| Granite 4.0 H-Tiny | 2 Oct 2025 | hybrid MoE | 7B / 1B | 128K validated | Apache 2.0 | [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models) |
| Granite 4.1 3B/8B | 29 Apr 2026 | dense transformer | 3B / 8B (9B meta) | 128K (card) vs 512K (press) | Apache 2.0 | [HF](https://huggingface.co/ibm-granite/granite-4.1-8b) |
| LFM2.5-1.2B-Instruct | 5 Jan 2026 | conv + attn hybrid | 1.2B | n/s (32,768 used in Unsloth guide) | LFM Open v1.0 | [Liquid](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai) |
| LFM2.5-2.6B | 4 Aug 2026 | conv + attn hybrid | 2.6B | 128K phase | LFM Open (assumed) | [Liquid](https://www.liquid.ai/blog/lfm2-5-2-6b) |
| LFM2.5-8B-A1B | 28 May 2026 | conv + attn MoE, reasoning-only | 8B / ~1B | 128K | LFM Open (assumed) | [Liquid](https://www.liquid.ai/blog/lfm2-5-8b-a1b) |
| LFM3-1.7B | 27 Oct 2026 | conv + SWA + global attn | 1.7B | 256K | n/s on blog | [Liquid](https://www.liquid.ai/blog/lfm3) |

### Inferences
- **Licence ranking for a public student project:**
  - **Best:** Apache 2.0 (Ministral 3, NeMo, Mistral 7B, Granite 3/4/4.1, Muse Glimmer).
  - **Then:** LFM Open v1.0. It is free for a student and only asks for Apache-style notices, but it is not OSI "open source" and has a termination clause.
  - **Then:** Llama 3.x Community. Any released fine-tune must be named "Llama-…", show "Built with Llama" and follow the AUP. This is workable but adds friction.
  - **Avoid:** Ministral-8B-2410 (Mistral Research Licence). It is superseded anyway by Ministral 3 8B (Apache).
- The EU multimodal restriction is irrelevant if a text-only Llama 3.2 1B/3B or 3.1 8B is used.
- Muse Glimmer is Meta's notable licence change (Apache, no "Llama" naming rule), but at 30B dense it is outside both the T4 training and CPU serving budgets (see Q3).
- Mistral Small 4 (~119B MoE) cannot fit in 32 GB RAM even at 4-bit (~60+ GB estimated), so the "Mistral Small" line is out. Mistral Small 3.x (24B dense, Apache 2.0, 2025) is above range: ~14 GB at Q4_K_M would fit in RAM but decode only ~4–5 tok/s on a desktop CPU (estimate scaled from the ~10 tok/s 8B figure in Q3). It also cannot be QLoRA-trained comfortably at seq 2048 on a 16 GB T4.

### Gaps
- Exact release dates and licences for Mistral Small 3/3.1/3.2 were not re-verified this session. From background knowledge: Jan, Mar and Jun 2025, Apache 2.0, 24B.
- No official LFM3 licence statement was found on the blog. It is likely the LFM Open License, but the HF model card should be checked.
- The Llama 4 licence EU clause and naming terms were not fetched. Llama 4 (Scout 109B / Maverick 400B MoE, 17B active) is far outside this size range anyway.
- Granite 4.1 context (128K vs 512K) is unresolved; the HF card is more authoritative.
- I did not fetch the LFM2-8B-A1B (Oct 2025) card, so its exact active parameters and licence are unverified here.

## Q2. Fine-tuning: Unsloth / PEFT / TRL QLoRA support, T4 fit at seq ~2048, hybrid-architecture gotchas

### Takeaway
Standard dense transformers are the easy path on a free T4: Llama 3.x, Ministral 3, NeMo and Granite 4.1. LFM2.5-1.2B also has first-class Unsloth support and "fits comfortably" on a free T4. Granite 4.0 hybrids are supported by Unsloth (Micro and Tiny fit a T4), but Mamba layers bring kernel and adapter-saving pitfalls. Ministral 3 needs transformers v5, and its Instruct weights ship in FP8. Muse Glimmer needs ~24 GB VRAM. The reasoning-only LFM2.5 MoEs would have to be SFT'd out of chain-of-thought.

### Cited Findings
- **Unsloth generic VRAM minimums:** 14B QLoRA (4-bit) ≈ 8.5 GB, 14B 16-bit LoRA ≈ 33 GB. Some models need more; OOM is usually fixed by batch size 1–3 — [Unsloth requirements (de)](https://unsloth.ai/docs/de/los-gehts/fine-tuning-for-beginners/unsloth-requirements)
- **LFM2.5 + Unsloth:**
  - The 1.2B model "fits comfortably" on a free Colab T4 (as does VL-1.6B); claimed 2x faster with 50% less VRAM.
  - Example config is 16-bit LoRA (load_in_4bit=False), max_seq_length 4096, r=16, alpha=16, adamw_8bit, lr 2e-4.
  - Target modules: q_proj, k_proj, v_proj, out_proj, **in_proj**, w1, w2, w3 (in_proj/out_proj include the conv blocks).
  - ChatML-like template (`<|startoftext|>`, `<|im_start|>`, `<|im_end|>`); llama.cpp runs use `--jinja`.
  - GGUF export via `save_pretrained_gguf(..., "q4_k_m")`.
  - Recommended inference settings: temp 0.1, top_p 0.1, top_k 50, repetition_penalty 1.05.
  - "Not recommended for knowledge-intensive tasks or programming" — [Unsloth LFM2.5 tutorial](https://unsloth.ai/docs/models/tutorials/lfm2.5.md)
- Liquid's docs state LFM2.5 is fully supported by Unsloth. Liquid recommends LFM2.5-1.2B-**Base** for heavy fine-tuning — [Liquid docs: Unsloth](https://docs.liquid.ai/lfm/fine-tuning/unsloth.md); [Liquid LFM2.5 blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- LFM3: "every model can be fine-tuned on a single GPU". No framework is named (5 days old at time of research) — [Liquid LFM3 blog](https://www.liquid.ai/blog/lfm3)
- LFM2.5-8B-A1B and 24B-A2B are **reasoning-only** (explicit CoT before the answer). Both blogs link to "run and fine-tune locally" docs without details — [8B-A1B](https://www.liquid.ai/blog/lfm2-5-8b-a1b); [24B-A2B](https://www.liquid.ai/blog/lfm2-5-24b-a2b)
- **Ministral 3 + Unsloth:**
  - Base, Instruct and Reasoning at 3B/8B/14B are supported. The 14B "should fit on a free Colab GPU" for fine-tuning.
  - Needs **transformers v5** plus the latest unsloth / unsloth_zoo.
  - Notebooks: 3B vision SFT and 3B GRPO.
  - Instruct template has a long default "Le Chat" system prompt with `{today}`/`{yesterday}` placeholders that must be filled. Reasoning uses `[THINK]…[/THINK]`.
  - Instruct recommended temp 0.1–0.15 — [Unsloth Ministral 3](https://unsloth.ai/docs/models/ministral-3)
- Ministral 3 Instruct ships in FP8. Transformers can load it as BF16 via `FineGrainedFP8Config(dequantize=True)`, and Base/Reasoning ship as BF16 — [HF Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- **Granite 4.0 + Unsloth:**
  - All sizes are supported. "Granite-4.0 micro and tiny fit comfortably in a 15GB VRAM T4."
  - The example loads `unsloth/granite-4.0-h-micro` with `load_in_4bit=True`.
  - IBM recommends temp 0.0, top_k 0, top_p 1.0.
  - Micro (non-hybrid) exists "for when Mamba2 isn't fully supported" — [Unsloth Granite 4.0](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0)
- **Hybrid Mamba gotchas:**
  - Without `mamba_ssm` and `causal_conv1d` installed, HF transformers falls back to a slow eager path.
  - Mamba checkpoints recommend keeping the model in float32 when fine-tuning with PEFT (stated for pure Mamba checkpoints) — [HF state-spaces/mamba-790m-hf card](https://huggingface.co/state-spaces/mamba-790m-hf/blob/9e8ad0bf9cbdad8193b5b6c29c6f3339fcbf8ee2/README.md)
- A Feb 2026 preprint on LoRA-tuning **Granite-4.0-H-Micro** found that the PEFT adapter-only save **silently discarded trained SSM parameters** (A_log, D, dt_bias). It also reports that co-training those SSM parameters gave consistent gains (preprint, not peer-reviewed) — [Zenodo 18819932](https://zenodo.org/records/18819932)
- One community Granite-4.0-H-Micro + Unsloth tutorial installs `causal_conv1d==1.5.2 --no-build-isolation` first — [Towards AI (member-only)](https://pub.towardsai.net/ibms-granite-4-0-fine-tuning-made-simple-create-custom-ai-models-with-python-and-unsloth-4fc11b529c1f)
- **Granite 4.1:** the HF card lists Transformers, vLLM, SGLang and Docker Model Runner, and does not mention Unsloth. IBM's own cookbook has an Unsloth QLoRA page, but the visible content is for Granite 3.0 — [HF granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b); [IBM Granite docs: Unsloth](https://www.ibm.com/granite/docs/fine-tune/unsloth)
- **Muse Glimmer:** day-0 TRL support (SFT and GRPO, LoRA or full). Unsloth says it trains locally on **24 GB VRAM** with LoRA/QLoRA and exports GGUF/FP8/NVFP4 — [HF blog](https://huggingface.co/blog/muse-glimmer); [Unsloth Muse Glimmer train](https://unsloth.ai/docs/models/muse-glimmer/train)

### Inferences
- **T4 feasibility at seq 2048 with QLoRA, batch 1–2 plus gradient accumulation:**
  - Comfortable: LFM2.5-1.2B/2.6B, LFM3-1.7B (if Unsloth/transformers support it), Llama 3.2 1B/3B, Ministral 3 3B, Granite 4.0 H-Micro/Micro/H-Tiny, Granite 4.1 3B.
  - Fits: 8B-class dense (Llama 3.1 8B, Ministral 3 8B, Granite 4.1 8B), consistent with Unsloth's 14B≈8.5 GB QLoRA minimum.
  - Tight but likely: NeMo 12B and Ministral 3 14B.
  - Does not fit: Muse Glimmer (24 GB), Mistral Small 3.x (24B), LFM2.5-24B-A2B and Granite H-Small.
  - With 3–5k examples × ~1.5k tokens (~6M tokens per epoch), 1B–3B models should do multiple epochs inside Kaggle's 30 h/week. 8B models will be slower but feasible.
- T4 (Turing) has no native bf16 or FP8 (background knowledge, not sourced above). Training will run in fp16, and Ministral 3 Instruct's FP8 weights must be dequantized, or Unsloth's 4-bit/BF16 re-uploads used.
- For Ministral 3, fine-tune the **Base** or Instruct with your own system prompt and drop the "Le Chat" default. Leave the vision tower frozen or unloaded for text-only SFT; there is no explicit guidance, so this is untested.
- Avoid reasoning-only LFM2.5 MoEs (8B-A1B, 24B-A2B) unless willing to SFT away the CoT. Every wasted thinking token costs CPU latency. Prefer LFM2.5-1.2B-Instruct, LFM2.5-2.6B or LFM3-1.7B (instruct status to verify).
- **Granite 4.1** removes all Mamba gotchas; it is a standard Llama-like dense transformer. That makes it the lowest-risk IBM option for QLoRA. With the Granite 4.0 hybrids, verify that the trained SSM parameters (if unfrozen) survive adapter save and merge before converting to GGUF.

### Gaps
- No Unsloth page for Granite 4.1 or LFM3 was found. Day-0 support for LFM3's new layers (SWA, head-wise gated attention, kernel-7 conv) in Unsloth, PEFT or transformers is unverified.
- No measured peak VRAM at seq 2048 on a T4 was found for any candidate. The figures above are generic Unsloth minimums.
- Whether `mamba_ssm`/`causal_conv1d` CUDA kernels build and run on T4 (sm_75) for Granite 4.0-H was not verified.

## Q3. llama.cpp / GGUF / Ollama support and CPU speed

### Takeaway
GGUF/llama.cpp support exists for every in-range candidate. Liquid ships official GGUFs; Granite 4.0 hybrids work in llama.cpp, though IBM said throughput optimisation was "ongoing" in Oct 2025; Ministral 3 and Muse Glimmer have Unsloth GGUFs; Granite 4.1 has community quantisations. Liquid publishes the only concrete x86 CPU numbers: LFM2.5-1.2B on an AMD Ryzen AI 9 HX 370 runs ~2,975 tok/s prefill and 116 tok/s decode at Q4_0. No source gives Intel desktop (i7-13700K) numbers for any of these models. Community estimates put an 8B Q4_K_M at only ~10 tok/s decode on an i7 with DDR5.

### Cited Findings
- **LFM2.5-1.2B-Instruct** (llama.cpp, Q4_0, 1K prefill / 100 decode):
  - AMD Ryzen AI 9 HX 370: **prefill 2,975 tok/s, decode 116 tok/s, 856 MB**. Qwen3-1.7B on the same chip: 2,008 / 62 tok/s, 1,465 MB.
  - Galaxy S25 Ultra: 335 / 70 tok/s.
  - GGUF checkpoints for all models; also MLX, vLLM, ONNX, NexaSDK, LEAP. No Intel results (vendor) — [Liquid LFM2.5 blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- Unsloth quotes LFM2.5 at "239 tok/s decode on an AMD CPU" in under 1 GB RAM (CPU and quant unspecified) — [Unsloth LFM2.5](https://unsloth.ai/docs/models/tutorials/lfm2.5.md)
- **LFM2.5-2.6B:** ~220 tok/s on Apple M5 Max and ~113 tok/s on AMD Ryzen AI Max+ 395, both under 2.5 GB; quant not stated. GGUF for llama.cpp (vendor) — [Liquid](https://www.liquid.ai/blog/lfm2-5-2-6b)
- **LFM2.5-8B-A1B:** 253 tok/s on M5 Max and 146 tok/s on Ryzen AI Max+ 395, under 6 GB; quant not stated. Day-one llama.cpp GGUF (vendor) — [Liquid](https://www.liquid.ai/blog/lfm2-5-8b-a1b)
- **LFM2.5-24B-A2B:** 184 tok/s on M5 Max and 114 tok/s on Ryzen AI Max+ 395 (not stated to be CPU-only). GGUF available (vendor) — [Liquid](https://www.liquid.ai/blog/lfm2-5-24b-a2b)
- **LFM3-1.7B:**
  - Speed tests were run with llama.cpp at Q4_0 with 32K context.
  - It decodes 1.8–2.8× faster than the next-fastest compared model on three unnamed devices, and ~25% faster than LFM2-1.2B at 8K context.
  - KV cache at 128K context is 1.6 GB, vs 4.3 GB for Qwen3.5-4B (vendor) — [Liquid LFM3](https://www.liquid.ai/blog/lfm3)
- **Granite 4.0:**
  - "llama.cpp and MLX support the hybrid architecture, but work to fully optimize throughput … is still ongoing" (as of 2 Oct 2025).
  - Ollama and LM Studio are launch partners. vLLM 0.10.2 has full hybrid support — [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
  - Unsloth gives llama.cpp CPU build flags and an Ollama command (`ollama run hf.co/unsloth/granite-4.0-h-small-GGUF:UD-Q4_K_XL`) — [Unsloth Granite 4.0](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0)
- **Granite 4.1:** no GGUF in IBM's repo; the card links an 82-model quantisations listing for llama.cpp, Ollama and LM Studio — [HF granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b)
- **Ministral 3:** Unsloth GGUFs (UD-Q4_K_XL) for Instruct and Reasoning. Requires "latest llama.cpp" with `--jinja`. Ollama is not mentioned — [Unsloth Ministral 3](https://unsloth.ai/docs/models/ministral-3)
- **Muse Glimmer:** day-0 llama.cpp, with calibrated GGUFs from Meta plus Unsloth quants and a DFlash speculative-decoding drafter. Q4_K_M is preferred. No CPU speed figures — [HF blog](https://huggingface.co/blog/muse-glimmer); [HF GGUF repo](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
- **Generic x86 CPU (community):**
  - Llama 3.1 8B Q4_K_M ≈ 5.5 GB, ~**10 tok/s** on an i7 with DDR5. Memory bandwidth, not core count, sets decode speed, and only the P-cores matter on an i7-13700 — [InsiderLLM CPU-only guide 2026](https://insiderllm.com/guides/cpu-only-llms-what-actually-works/)
  - Another source gives 6–10 tok/s for 7B on CPU-only DDR5 — [mustafa.net](https://mustafa.net/llm-tokens-per-second-benchmarks/)
  - An i5-13400 DDR5 user saw 6–8 tok/s, with occasional 0.9 tok/s from bad thread settings — [HF discussion](https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/discussions/3)
- A CEUR paper on an i7-13700H laptop found 6–8B models offer the best quality/latency balance on CPU-only llama.cpp; exact numbers were not extracted — [CEUR Vol-4164 paper11](https://ceur-ws.org/Vol-4164/paper11.pdf)

### Inferences
- **Latency budget (rough estimate, not measured):** one line ≈ prefill of new prompt tokens + decode of ~30–45 tokens (intensity token + 20–40 words-ish tokens).
  - **8B dense at Q4_K_M** (Llama 3.1 8B, Ministral 3 8B, Granite 4.1 8B, and also the shortlist's Qwen3-8B):
    - Decode at ~10 tok/s means ~3–4.5 s for decode alone, before prefill. Uncached prefill of ~1,000 tokens likely adds a few seconds more, because the 13700K is AVX2-only, with no AVX-512 on Raptor Lake (background knowledge).
    - So **8B dense likely misses 2–3 s on CPU** unless the prompt is mostly cached (llama.cpp KV/prompt-cache reuse of the static system prompt and match card) and outputs stay short.
  - **~1–3B active models** (LFM2.5-1.2B, LFM3-1.7B, LFM2.5-2.6B, Granite 4.0 H-Micro/H-Tiny, Granite 4.1 3B, Ministral 3 3B, Llama 3.2 3B):
    - Scaling Liquid's HX 370 figures, a 1.2B model would prefill 1,000 tokens in ~0.3–0.5 s and decode 40 tokens in ~0.4 s. 3B dense would be roughly 2.5× slower, still about 1.5–2 s.
    - These comfortably meet the target on CPU.
- Liquid's "Ryzen AI Max+ 395" numbers come from a chip with a 256-bit LPDDR5X bus. From background knowledge, that is roughly 2.5–3× the bandwidth of dual-channel DDR5 on a desktop i7-13700K, so expect proportionally lower decode on the user's machine. The HX 370 (dual-channel-class LPDDR5X) figures are the most transferable.
- MoE with ~1B active (LFM2.5-8B-A1B, Granite H-Tiny) decodes like a ~1B model but needs RAM for all weights (8B ≈ 5–6 GB at Q4). That is fine in 32 GB, and gives better quality per ms than dense 8B, if reasoning output can be suppressed.
- **Ollama:** any HF GGUF can be pulled via `hf.co/...`, but Ollama's bundled llama.cpp may lag mainline for brand-new architectures (LFM3, possibly Granite hybrid kernels). Plan to serve via `llama-server` directly if Ollama lags.

### Gaps
- No vendor or independent benchmark gives tok/s on an Intel 13th-gen desktop for any of these models. The user should run `llama-bench` (pp512/pp1024 and tg64) on Q4_K_M / Q4_0 for 2–3 shortlisted models.
- No independent numbers were found for Granite 4.0-H CPU speed in llama.cpp after the Oct 2025 "optimization ongoing" caveat.
- Mainline llama.cpp and Ollama support status for LFM3's new architecture is unverified. Liquid used llama.cpp internally, which implies at least a Liquid-supported build exists.

## Q4. Benchmarks (IFEval / format following, creative writing, hallucination, chat quality) and known quirks

### Takeaway
On instruction-following benchmarks, Liquid's models report very high scores for their size: LFM3-1.7B IFEval 96.7 / IFBench 73.0 and LFM2.5-1.2B IFEval 86.2. Granite 4.1 8B reports IFEval 87.1 and Arena-Hard 69.0. Ministral 3 8B reports roughly Qwen3-VL-8B parity on Arena-Hard and WildBench. Almost all of these are vendor numbers. No usable EQ-Bench Creative Writing scores were found for Ministral, Granite or LFM. Llama 3.1 8B sits near the bottom of that board (~713 Elo, aggregator), and Meta's Muse proprietary models sit near the top. Casual "hype" register quality for these families is essentially unmeasured.

### Cited Findings
- **LFM3-1.7B vs others** (Liquid-measured; includes the user's shortlist family):

  | Model | IFEval | IFBench | GPQA-D | BFCL v4 |
  |---|---|---|---|---|
  | LFM3-1.7B | 96.66 | 73.03 | 63.37 | 71.39 |
  | LFM2.5-2.6B | 95.39 | 60.48 | — | — |
  | Gemma 4 E4B | 90.02 | 44.56 | — | — |
  | Qwen3.5 4B | 92.52 | 38.16 | — | — |
  | Qwen3.5 9B | 92.08 | 48.84 | — | — |

  LFM3 "trails larger models on agentic coding and knowledge" — [Liquid LFM3](https://www.liquid.ai/blog/lfm3)
- **LFM2.5-1.2B-Instruct:**
  - IFEval 86.23, IFBench 47.33, Multi-IF 60.98, MMLU-Pro 44.35, GPQA 38.89.
  - Liquid-measured comparators:
    - Llama 3.2 1B Instruct: IFEval 52.37, IFBench 15.93
    - Granite-4.0-h-1b: IFEval 80.08, IFBench 24.93
    - Granite-4.0-1b: IFEval 79.61
    - Qwen3-1.7B (instruct mode): IFEval 73.68

  — [Liquid LFM2.5](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- **LFM2.5-8B-A1B:**
  - IFEval 91.84, IFBench 56.47, Multi-IF 79.93.
  - **AA-Omniscience** index −24.70 (accuracy 8.67, **non-hallucination rate 63.47**).
  - Liquid-measured comparators:
    - **Granite-4.0-H-Tiny:** IFEval 82.23, IFBench 21.28
    - Gemma-4-26B-A4B: IFEval 91.40
    - Qwen3.5-4B: IFEval 87.80

  — [Liquid](https://www.liquid.ai/blog/lfm2-5-8b-a1b)
- **LFM2.5-2.6B:**
  - IFBench 59.17 (vs Qwen3.5-9B 56.47, Gemma-4-E4B 39.24), Multi-IF 80.07, IFStruct 85.49.
  - AA-Omniscience −29.50 vs Qwen3.5-9B −50.43 (higher is better, i.e. fewer confident wrong answers).
  - "Hallucination control" was a training focus (vendor) — [Liquid](https://www.liquid.ai/blog/lfm2-5-2-6b)
- **LFM2.5-24B-A2B:** IFEval 96.75, IFBench 74.69 vs Qwen3.6-35B-A3B 93.06 / 64.21 (vendor) — [Liquid](https://www.liquid.ai/blog/lfm2-5-24b-a2b)
- **LFM quirk:** Unsloth says LFM2.5 is "not recommended for knowledge-intensive tasks or programming"; it is aimed at extraction, RAG, agentic and tool use. Recommended temp is 0.1 — [Unsloth LFM2.5](https://unsloth.ai/docs/models/tutorials/lfm2.5.md)
- **Granite 4.1** (IBM-reported, no independent evaluation):

  | Benchmark | 8B | 3B | 30B |
  |---|---|---|---|
  | IFEval (avg) | 87.06 | 82.30 | 89.65 |
  | ArenaHard | 68.98 | — | — |
  | AlpacaEval 2.0 | 50.08 | — | — |
  | MT-Bench | 8.61 | — | — |
  | MMLU | 73.84 | 67.02 | 80.16 |
  | SimpleQA | 4.82 | — | — |

  — [HF granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b); [Techstrong (no independent eval)](https://techstrong.ai/features/ibms-granite-4-1-smaller-models-smarter-design/)
- One RL stage of Granite 4.1 explicitly targets "conversation quality". IBM positions the family for enterprise assistants, RAG and tool calling, and pairs it with Granite Guardian — [Techstrong](https://techstrong.ai/features/ibms-granite-4-1-smaller-models-smarter-design/); [HF card](https://huggingface.co/ibm-granite/granite-4.1-8b)
- IBM claims Granite-4.0-H-Small beats all open-weight models except Llama 4 Maverick on IFEval (per Stanford HELM), and that 4.0 Micro beats Granite 3.3 8B. IBM recommends temp 0 — [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models); [Unsloth Granite 4.0](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0)
- **Ministral 3 8B Instruct:** Arena Hard 0.509 (Qwen3-VL-8B-Instruct 0.528), WildBench 66.8 (66.3), MM-MTBench 8.08 (8.00), MATH 0.876 (0.946). No IFEval reported. Mistral recommends temp < 0.1 for production and "higher values for creative tasks" (vendor) — [HF card](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- **Mistral NeMo 12B:** MMLU 68.0, TruthfulQA 50.3 (base, vendor, 2024); temp 0.3 recommended — [HF card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- **EQ-Bench Creative Writing v3:**
  - Aggregator lists Llama3.1-8B-Instruct at **713 Elo, rank 111 of 116** (updated 25 Oct 2026) — [Sophon/aggregator via search](https://sophon.at/evals/eqbench-creative-writing); [llm-stats](https://llm-stats.com/benchmarks/creative-writing-v3)
  - The scale is anchored with ministral-3b (old 2410) at 200 and DeepSeek-R1 at 1500 — [EQ-Bench about](https://eqbench.com/about.html)
- Meta's Muse proprietary models sit near the top of EQ-Bench Creative Writing (Muse Wonder 1606, Muse Spark 1.1 1542.6, Spark 1.3 1525.5, Spark 1.2 1473.8). Muse Glimmer was not in the visible top 21 of 109 (aggregator snapshot, "Aug 2026") — [Sophon](https://sophon.at/evals/eqbench-creative-writing)
- **Muse Glimmer** (vendor, thinking mode): IFBench 77.0 (vs Gemma4-31B and Qwen3.6-27B), GPQA-D 83.5, SWE-Bench Verified 76.0. No creative-writing or IFEval scores — [HF blog](https://huggingface.co/blog/muse-glimmer)
- Llama 3.x is dated: there have been no new Llama weights since Apr 2025, and Meta's focus moved to Muse — [aicomparison.ai](https://aicomparison.ai/meta-llama-models-guide/); [Wikipedia](https://en.wikipedia.org/wiki/Meta_AI)

### Inferences
- **Format following** (intensity token + one short line): LFM3-1.7B, the LFM2.5 family and Granite 4.1 all report IFEval in the high 80s to 90s. After SFT on 3–5k examples, format adherence should be near-perfect for any candidate. Base IFEval matters less than how the model's prose sounds after fine-tuning.
- **Grounding and hallucination:** Liquid explicitly trains for hallucination control and reports AA-Omniscience non-hallucination rates. But that metric measures abstention on knowledge questions, not faithfulness to a provided game-state frame. The task is input-grounded (players, weapons, score are all in the frame), so the LFM and Granite "low world knowledge" weaknesses (e.g. Granite 4.1 8B SimpleQA 4.82) matter little. They may even help, since the model is less inclined to invent esports trivia.
- **Register risk:**
  - Granite: IBM tunes and markets for enterprise, defaulting to temp 0 — a dry default tone.
  - LFM: tuned for extraction and agents with temp 0.1.
  - Ministral 3: recommends temp < 0.1 for production.
  - None has a published creative-writing score, so the "hype caster" voice must come almost entirely from the fine-tuning data. Plan an A/B blind evaluation of fine-tuned outputs rather than trusting leaderboards.
- Llama 3.1 8B's very low EQ-Bench creative-writing rank, plus its 2024 vintage, makes it a weak challenger. Llama 3.2 1B's IFEval of 52 (Liquid-measured) is far below LFM2.5-1.2B's 86 at the same size.

### Gaps
- EQ-Bench Creative Writing scores for Ministral 3, NeMo, Granite 4.x and LFM2.5/LFM3 could not be extracted (JS-rendered leaderboard).
- No independent (non-vendor) IFEval, Arena or LMArena numbers were found for Granite 4.1, LFM2.5, LFM3 or Ministral 3.
- No studies were found on hallucination or faithfulness to structured inputs for these models.
- No community evidence (r/LocalLLaMA) on casual or "hype" English register quality was retrieved this session.
- NeMo's community reputation for creative and role-play writing is background knowledge only and unsourced here.

## Q5. USP and verdict for each model for this specific use case (challengers to Qwen3-8B / Gemma 4 E4B)

### Takeaway
For a CPU-served 2–3 s budget, the strongest challengers from these families are the small, fast Liquid models: **LFM2.5-1.2B-Instruct** (proven Unsloth/T4 path, vendor CPU numbers) and **LFM3-1.7B** (best reported instruction following, but 5 days old with tooling risk). **Granite 4.1 3B/8B** is the "safe, Apache-2.0, plain-transformer" alternative. **Ministral 3 3B/8B** is the Apache-2.0 option with Mistral's chattier style. Llama 3.x, Mistral 7B, NeMo and Muse Glimmer are not recommended.

### Cited Findings
- **LFM2.5-1.2B-Instruct:**
  - ~116 tok/s decode, ~2,975 tok/s prefill, 856 MB on an x86 laptop CPU (vendor).
  - IFEval 86 vs Llama 3.2 1B's 52.
  - Free-T4 fine-tuning is documented by Unsloth. Licence is free for a student (< $10M revenue) — [Liquid](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai); [Unsloth](https://unsloth.ai/docs/models/tutorials/lfm2.5.md); [LFM license](https://www.liquid.ai/lfm-license)
- **LFM3-1.7B:** IFEval 96.66 and IFBench 73.03, ahead of Gemma 4 E4B (90.02 / 44.56) in Liquid's table. About 1 GB at 4-bit, 256K context, released 27 Oct 2026 (vendor) — [Liquid LFM3](https://www.liquid.ai/blog/lfm3)
- **LFM2.5-2.6B:** IFBench 59.17 vs Gemma-4-E4B 39.24 and Qwen3.5-9B 56.47; under 2.5 GB (vendor) — [Liquid](https://www.liquid.ai/blog/lfm2-5-2-6b)
- **Granite 4.1 8B/3B:**
  - Apache 2.0 dense transformer. IBM calls it "easier to fine-tune" and RL-tuned for conversation quality.
  - 8B: IFEval 87.06, Arena-Hard 68.98 (vendor) — [HF](https://huggingface.co/ibm-granite/granite-4.1-8b); [Techstrong](https://techstrong.ai/features/ibms-granite-4-1-smaller-models-smarter-design/)
- **Granite 4.0 H-Tiny** (7B / 1B active hybrid MoE): fits a T4 for fine-tuning, but the Liquid-measured IFEval is 82.23 and IFBench 21.28. Hybrid kernel and SSM-save gotchas apply — [Unsloth](https://unsloth.ai/docs/models/tutorials/ibm-granite-4.0); [Liquid](https://www.liquid.ai/blog/lfm2-5-8b-a1b); [Zenodo](https://zenodo.org/records/18819932)
- **Ministral 3 8B:** Apache 2.0. Arena-Hard and WildBench at parity with Qwen3-VL-8B-Instruct (vendor). Base variant available for clean SFT — [HF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- **Muse Glimmer:** Apache 2.0 Meta model with day-0 llama.cpp/TRL/Unsloth support, but 30B dense, needs a 24 GB GPU to train and is under 20 GB at 4-bit — [HF blog](https://huggingface.co/blog/muse-glimmer); [Unsloth](https://unsloth.ai/docs/models/muse-glimmer/train)

### Inferences
**Per-model USP and verdict for CS2 caster lines**

- **LFM2.5-1.2B-Instruct (or -Base for SFT): TOP CPU CHALLENGER.**
  - USP: the fastest credible option on CPU (well under 1 s per line, estimated). It has the most mature Unsloth/T4 and GGUF path of the four families.
  - Risks: limited prose richness and variety at 1.2B. Vendor-recommended low temperature may cause repetitive lines, so mitigate with data diversity and a repetition penalty.
  - Ideal for testing whether a tiny model can carry the caster voice once fine-tuned.
- **LFM3-1.7B: HIGH-UPSIDE, HIGH-RISK.**
  - USP: best reported instruction following of any model here, newest architecture, small KV cache.
  - Risks: released 5 days before this research. Unsloth, PEFT and llama.cpp mainline support for the new layers is unverified. Licence on the blog is unstated, and whether it emits thinking is unknown.
  - Re-check in a few weeks; a strong candidate if tooling lands.
- **LFM2.5-2.6B: MIDDLE GROUND.**
  - More capacity than 1.2B and still fast (under 2.5 GB). Agentic post-training; check that it is not reasoning-only.
- **LFM2.5-8B-A1B: NOT IDEAL AS-IS.**
  - CPU-fast MoE (~1B active), but reasoning-only. CoT would blow the latency budget unless SFT removes it.
  - MoE QLoRA on a T4 is also less proven.
- **Granite 4.1 8B: SAFE ENTERPRISE CHALLENGER to Qwen3-8B.**
  - USP: Apache 2.0, ISO 42001, plain transformer (no Mamba gotchas). Vendor IFEval and Arena-Hard are respectable.
  - Risks: dry enterprise tone, and the same CPU latency problem as any 8B dense model.
- **Granite 4.1 3B: CPU-FRIENDLY SAFE PICK.**
  - Plain transformer, Apache 2.0, IFEval 82.3 (vendor).
- **Granite 4.0 H-Micro / H-Tiny: SECONDARY.**
  - USP: hybrid Mamba memory efficiency and a fast 1B-active MoE.
  - Risks: lower reported instruction following, fiddlier fine-tuning (kernels, SSM params) and possibly unoptimised llama.cpp throughput.
  - Superseded for this use by Granite 4.1 dense.
- **Ministral 3 8B / 3B: GOOD APACHE ALTERNATIVE.**
  - USP: Mistral models' conversational style. Base, Instruct and Reasoning variants; 3B gives a CPU-fast option.
  - Risks: FP8 Instruct weights on a T4, a vision tower to ignore, a transformers v5 requirement and a "Le Chat" system-prompt template to override.
  - 14B is too slow for CPU serving.
- **Mistral NeMo 12B: DATED** (Jul 2024). Apache 2.0, but too slow for the CPU target and superseded by Ministral 3 14B.
- **Mistral 7B:** obsolete (2023).
- **Mistral Small 3.x / 4:** out of range for both T4 training and CPU serving.
- **Llama 3.1 8B / 3.2 1B-3B: NOT RECOMMENDED.**
  - 2024 vintage and weak creative-writing rank (3.1 8B). 3.2 1B instruction following is far below LFM2.5-1.2B.
  - Licence naming and attribution friction. Meta has moved to Muse.
- **Muse Glimmer: OUT OF BUDGET.**
  - Interesting only as a possible teacher or data generator (Apache 2.0) on a rented GPU. Not usable as a T4-trained, CPU-served student.
- **Overall vs the shortlist:**
  - If the 2–3 s CPU target is firm, an 8B dense model (Qwen3-8B, Granite 4.1 8B, Ministral 3 8B) is borderline at best.
  - A 1–3B model from these families (LFM2.5-1.2B/2.6B, LFM3-1.7B, Granite 4.1 3B, Ministral 3 3B) is the realistic CPU class, competing with Gemma 4 E4B.
  - Recommended bake-off: fine-tune LFM2.5-1.2B (or LFM3-1.7B if tooling is ready) and Granite 4.1 3B or Ministral 3 3B alongside the shortlist. Compare blind on caster-line quality and measured `llama-bench` latency on the i7-13700K.

### Gaps
- No head-to-head (vendor or independent) evaluation of any of these models on short sports or esports commentary, or on casual "hype" register, was found.
- Whether LFM2.5-2.6B and LFM3-1.7B are non-reasoning by default needs checking on their HF model cards.
- Real latency on the user's i7-13700K (DDR4 vs DDR5 unknown) must be measured.
