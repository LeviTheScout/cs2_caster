# Qwen open-weight family (1B–14B dense + small-active MoE) as QLoRA base for a live CS2 caster-line generator — status as of 2 Nov 2026

Scope reminder: base model for QLoRA fine-tune on ~3–5k examples (structured game-state frame of ~500–1500 tokens, output: intensity token + 20–40-token English caster line). Train on free T4 (16 GB); serve on CPU (i7-13700K, 32 GB RAM) via llama.cpp/Ollama GGUF 4–5 bit, target 2–3 s per line; fallback Modal L4. Current project shortlist: Qwen3-8B, thinking off.

Research date: 2 Nov 2026. "Official" = Qwen model cards, blog or GitHub. "Independent" = Artificial Analysis, Unsloth, papers. "Community" = GitHub issues, blogs, forums.

---

## 1. Which Qwen models actually exist in range, with release dates, sizes, architecture, context and licence? (Confirm "Qwen 3.5 9B" and "Qwen 3.6 27B".)

### Takeaway
Both rumoured names are real. **Qwen3.5-9B** came out 2 Mar 2026: dense, natively multimodal, hybrid Gated-DeltaNet attention, Apache-2.0. **Qwen3.6-27B** came out 22 Apr 2026: dense 27B, multimodal, Apache-2.0. Qwen3.6 has no open model below 27B. Qwen3.7 is API-only. Qwen3.8's smallest open model is 27B. No Qwen4 weights have been published as of late Oct 2026. Within the project's size window, the newest Qwen open models are still the **Qwen3.5 small series (0.8B/2B/4B/9B, Mar 2026)** and the **3B-active MoEs (Qwen3.5-35B-A3B Feb 2026, Qwen3.6-35B-A3B Apr 2026)**.

### Cited Findings

**Release timeline (official dates unless noted)**
- **Qwen3 (28 Apr 2025):** dense 0.6B, 1.7B, 4B, 8B, 14B and 32B, plus MoE 30B-A3B and 235B-A22B, all Apache 2.0. — [Wikipedia: Qwen](https://en.wikipedia.org/wiki/Qwen) (citing the HF Qwen3 collection)
- **Qwen3 technical report:** arXiv 2505.09388 v1, submitted 14 May 2025. Qwen3 is a hybrid model where "by default, the model operates in thinking mode", and you switch modes with "/think and /no_think flags in the user query or system message". — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388)
- **Qwen3-30B-A3B-Instruct-2507:** released 30 Jul 2025; the Thinking-2507 variant followed on 31 Jul 2025. **Qwen3-4B-Instruct-2507** was released 6 Aug 2025 (official Qwen3 README news, relayed via search). No 8B or 14B "2507" refresh appears in that list. — [QwenLM/Qwen3 README](https://raw.githubusercontent.com/QwenLM/Qwen3/main/README.md)
- **Qwen3-30B-A3B-Instruct-2507 specs:** 30.5B total / 3.3B active, 128 experts with 8 active, 262,144 native context, Apache-2.0. "Supports only non-thinking mode and does not generate `<think></think>` blocks." — [HF model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- **Qwen3-4B-Instruct-2507 specs:** 4.0B (3.6B non-embedding), 262,144 native context, Apache-2.0. It is non-thinking only, so `enable_thinking=False` is "no longer required". — [HF model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Qwen3-Next-80B-A3B:** released 11 Sep 2025. — [QwenLM/Qwen3.8 GitHub news](https://github.com/QwenLM/Qwen3.8)
- **Qwen3.5 flagship:** 397B-A17B, 16 Feb 2026. **Qwen3.5-122B-A10B, 35B-A3B and 27B** followed on 24 Feb 2026. **Qwen3.5-9B, 4B, 2B and 0.8B** followed on 2 Mar 2026. — [QwenLM/Qwen3.8 GitHub news](https://github.com/QwenLM/Qwen3.8); [Qwen announcement on X, 2 Mar 2026](https://x.com/Alibaba_Qwen/status/2028460046510965160). Qwen's post says base models were released alongside the post-trained ones.
- **Qwen3.5-9B architecture:** 32 layers, hidden size 4096, layout "8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))". Causal LM with a vision encoder (image-text-to-text). Native context 262,144, extensible to ~1.01M with YaRN. Apache-2.0. Trained with multi-token prediction. The HF safetensors metadata reports about 10B params. — [HF Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Qwen3.6-35B-A3B:** 16 Apr 2026. **Qwen3.6-27B:** 22 Apr 2026. — [QwenLM/Qwen3.8 GitHub news](https://github.com/QwenLM/Qwen3.8)
- **Qwen3.6-35B-A3B specs:** 35B total / 3B active, 256 experts (8 routed + 1 shared active), 40 layers, hidden size 2048, hybrid "10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))". Vision encoder (image + video). 262,144 native context. Apache-2.0. — [HF Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Qwen3.6-27B** is a dense multimodal 27B with flagship-level coding claims. — [Qwen blog: Qwen3.6-27B](https://qwen.ai/blog?id=qwen3.6-27b). Artificial Analysis (independent) scores Qwen3.6-27B at 46 on its Intelligence Index and 35B-A3B at 43, but notes the 27B used ~3.7x the output tokens of Gemma 4 31B. — [Artificial Analysis on X](https://x.com/ArtificialAnlys/status/2049881951260283097)
- **Qwen3.7 (May–Jun 2026):** only Qwen3.7-Max and Qwen3.7-Plus, both proprietary/API. No open small models. — [Wikipedia: Qwen](https://en.wikipedia.org/wiki/Qwen). A targeted search found no Qwen3.7 open checkpoints.
- **Qwen3.8 open models:** Qwen3.8-2.4T-A95B (weights 12 Aug 2026; per Wikipedia, under a custom licence requiring a commercial licence above a US$50M revenue threshold) and **Qwen3.8-27B (14 Aug 2026, Apache-2.0)**. No Qwen3.8 below 27B is listed. — [QwenLM/Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8); [Wikipedia: Qwen](https://en.wikipedia.org/wiki/Qwen); [HF Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- **Qwen3.8-Flash-Next:** a ~125B MoE under the "Qwen Community 1.0" licence, framed as a preview of the Qwen4 architecture. Dates conflict: 26 Aug 2026 per [Wikipedia](https://en.wikipedia.org/wiki/Qwen) vs 28 Aug 2026 per [Yotta Labs](https://www.yottalabs.ai/post/qwen-3-8-flash-next-specs-qwen-4-preview-2026). Wikipedia is also internally inconsistent about which Flash variant is open.
- **Qwen4:** the family was reportedly announced at Apsara (22 Sep 2026) with Max/Plus/Flash tiers and a 27B open-weights tier. As of ~24–29 Oct 2026 no Qwen4 repository or weights existed on HF. An attendee reported, second-hand, that Alibaba's CTO said "next week" on 29 Oct. — [Yotta Labs: Qwen 4 release date](https://www.yottalabs.ai/post/qwen-4-release-date-what-is-known-how-to-prepare-2026); [CellCog](https://cellcog.ai/blog/qwen-4-release-date/); [Manifold market](https://manifold.markets/Terminator2/will-alibaba-publish-open-weights-f). All third-party and unconfirmed.

**Candidate inventory (only models relevant to the size window)**

| Model | Released | Params (dense / MoE total-active) | Native ctx | Licence | Thinking | Modality | Source |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B / 4B / 8B / 14B | 28 Apr 2025 | dense | ~32–40K native, 128K via YaRN | Apache-2.0 | hybrid, thinking on by default | text | [Wikipedia](https://en.wikipedia.org/wiki/Qwen), [Unsloth](https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune) |
| Qwen3-30B-A3B (orig.) | 28 Apr 2025 | 30.5B / 3.3B | as above | Apache-2.0 | hybrid | text | [Wikipedia](https://en.wikipedia.org/wiki/Qwen) |
| Qwen3-30B-A3B-Instruct-2507 | 30 Jul 2025 | 30.5B / 3.3B | 262K | Apache-2.0 | non-thinking only | text | [HF](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| Qwen3-4B-Instruct-2507 | 6 Aug 2025 | 4.0B dense | 262K | Apache-2.0 | non-thinking only | text | [HF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Qwen3-Next-80B-A3B | 11 Sep 2025 | 80B / 3B | — | Apache-2.0 | separate Instruct/Thinking | text | [GitHub](https://github.com/QwenLM/Qwen3.8), [Wikipedia](https://en.wikipedia.org/wiki/Qwen) |
| Qwen3.5-0.8B / 2B / 4B / 9B | 2 Mar 2026 | dense, hybrid DeltaNet | 262K | Apache-2.0 | hybrid; default disputed (see Q2) | image+text | [GitHub](https://github.com/QwenLM/Qwen3.8), [HF 9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Qwen3.5-27B / 35B-A3B | 24 Feb 2026 | 27B dense / 35B-3B MoE | 262K | Apache-2.0 | hybrid, thinking default | image+text | [GitHub](https://github.com/QwenLM/Qwen3.8) |
| Qwen3.6-35B-A3B | 16 Apr 2026 | 35B / 3B | 262K | Apache-2.0 | hybrid, thinking default | image+video+text | [HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.6-27B | 22 Apr 2026 | 27B dense | 262K (third-party) | Apache-2.0 | hybrid | multimodal | [Qwen blog](https://qwen.ai/blog?id=qwen3.6-27b), [InsiderLLM](https://insiderllm.com/guides/qwen-3-6-local-ai-guide/) |
| Qwen3.8-27B | 14 Aug 2026 | 27B | 262K (serving flags) | Apache-2.0 | reasoning_effort / preserve_thinking | unclear | [GitHub](https://github.com/QwenLM/Qwen3.8) |

### Inferences
- **The "newest small Qwen" in the 1–14B range is still Qwen3.5 (Mar 2026).** Nothing in Qwen3.6, 3.7 or 3.8 replaced the 4B/9B tier as of 2 Nov 2026. The 27B models (3.6, 3.8) are newer but sit outside the CPU 2–3 s budget. At 4-bit the 27B is ~17 GB, and dense generation at that size on desktop DDR5 is likely a few tokens/s.
- **The only "newer than Qwen3" models whose CPU speed fits the budget** are the 3B-active MoEs (Qwen3.5-35B-A3B, Qwen3.6-35B-A3B) plus Qwen3-30B-A3B-Instruct-2507. Per Unsloth, the 35B-A3Bs need ~22 GB at 4-bit, which fits in 32 GB RAM.
- **Qwen3-Next-80B-A3B is out for CPU serving.** 80B total params at 4-bit is roughly 45 GB or more, beyond 32 GB RAM. This is an estimate, not a sourced measurement.
- **Licence direction:** everything in range is Apache-2.0, which is fine for a public student release. Qwen started using custom/community licences for its largest and newest open models (Qwen3.8-2.4T, Qwen3.8-Flash-Next), so any Qwen4 small model's licence must be checked before switching.

### Gaps
- Official release date and full text-benchmark table for Qwen3.6-27B were not fetched. The date comes from Qwen's GitHub news; the 262K context comes from a third-party guide.
- Qwen3-VL small models (2B/4B/8B/30B-A3B, ~Oct 2025) were not verified from primary sources. Artificial Analysis only mentions "Qwen3 VL 8B" (index 17).
- Whether Qwen3.8-27B is text-only is unconfirmed. One HF user comment says it is; there is no official statement.
- Qwen4 status could change any day. Re-check huggingface.co/Qwen before finalising.

---

## 2. Non-thinking / instruct variants: how to switch thinking off, and does thinking leak after fine-tuning?

### Takeaway
There are three regimes:
- **Qwen3 (Apr 2025, incl. 8B) is hybrid.** Turn thinking off with `enable_thinking=False` in the chat template, or `/no_think` in the prompt. The template then still emits an empty `<think>\n\n</think>` block.
- **Qwen3-2507 Instruct checkpoints (4B, 30B-A3B) are non-thinking only** and never emit `<think>` tags. This is the cleanest option for a format-strict fine-tune.
- **Qwen3.5/3.6 drop the `/no_think` soft switch.** Only `enable_thinking: false` via chat-template kwargs works. Ollama's toggles have been buggy.

There is no documented case of "thinking leaks back after LoRA". The documented risk is template mismatch: the train template must match the inference template exactly, including empty think tags.

### Cited Findings
- **Qwen3 (hybrid):** thinking is on by default, and "/think and /no_think flags in the user query or system message" switch modes. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388)
- **Unsloth on Qwen3:** set `enable_thinking=False` in `apply_chat_template` (Transformers/vLLM), or add `/no_think` in llama.cpp/Ollama. "Non-thinking prompts include empty `<think>` tags." Recommended non-thinking sampling is T=0.7, top_p=0.8, top_k=20, min_p=0. — [Unsloth Qwen3 guide](https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune)
- **Unsloth fine-tuning advice for Qwen3:** "Use 75% reasoning and 25% non-reasoning in your dataset to make the model retain its reasoning capabilities". Training on non-reasoning data alone "may affect its reasoning ability", which is irrelevant here because reasoning is not needed. — [Unsloth Qwen3 guide](https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune)
- **Qwen3 model-card behaviour (relayed via search):** with `enable_thinking=True`, the model "will always output a block wrapped in `<think>...</think>`", which may be empty if thinking is disabled. — [Qwen3-4B-FP8 card](https://huggingface.co/Qwen/Qwen3-4B-FP8)
- **Ollama:** an issue asks to strip the empty `<think>` tag from Qwen3 non-thinking output. — [ollama#10496](https://github.com/ollama/ollama/issues/10496)
- **Empty think tags get learned (independent paper, ICDM 2026):** when every training label contained an empty think block, LoRA-tuned Qwen models reproduced it in every prediction. The same paper found Qwen "Think" vs "Nothink" training templates differed by <0.004 F1 on a structured extraction task. — [arXiv 2606.08051](https://arxiv.org/abs/2606.08051); detail on empty blocks relayed from the [PDF via search](https://arxiv.org/pdf/2606.08051)
- **Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507:** "supports only non-thinking mode and does not generate `<think></think>` blocks". — [HF 4B-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [HF 30B-A3B-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- **Qwen3.5-9B card:** thinking is the default and output begins with `<think>`. Disable it via `chat_template_kwargs: {"enable_thinking": False}`. "Qwen3.5 does not officially support the soft switch of Qwen3, i.e., /think and /nothink." Non-thinking ("Instruct") general sampling: T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5. — [HF Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Conflict on Qwen3.5 small-model default:** Unsloth says "For Qwen3.5 0.8B, 2B, 4B and 9B, reasoning is disabled by default". The llama.cpp flag is `--chat-template-kwargs '{"enable_thinking":false}'`. — [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5.md). This contradicts the 9B HF card's "thinking by default" text. The 2B card reportedly says non-thinking by default ([HF Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), via search snippet).
- **Qwen3.6-35B-A3B:** thinking is on by default. Disable it with `enable_thinking: False`; `/think` and `/nothink` are not supported. There is a new `preserve_thinking` kwarg. — [HF Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). Qwen3.8 adds `reasoning_effort` and `preserve_thinking`. — [QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8)
- **Community reports on Qwen3.5 in Ollama:**
  - Format/structured output is reportedly ignored when thinking is disabled, possibly because of a missing empty think tag in the template.
  - Ollama's Go runner reportedly drops repetition/presence penalties, even though Qwen relies on presence_penalty=1.5 against loops.
  - Sources: [ollama#14493](https://github.com/ollama/ollama/issues/14493), [ollama#14645](https://github.com/ollama/ollama/issues/14645). Reporter claims, not confirmed by Ollama.
  - A blog says `enable_thinking` could not turn off thinking for qwen3.5 in Ollama, and describes infinite loops inside the reasoning block. — [lyn.one](https://lyn.one/reasoning-control-flow) (community)

### Inferences
- **For a strict "intensity token + one line" output, a non-thinking-only checkpoint removes a failure class.** Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507 have no think tags to leak and no `/no_think` to forget. With hybrid Qwen3-8B, train and serve with the identical template (`enable_thinking=False`, empty think block included). Also set a stop sequence or grammar so a stray `<think>` can never reach the caster output.
- **With 3–5k purely non-thinking examples, the fine-tune will itself strongly condition the output format.** Leak risk mainly arises if serving uses a different template, e.g. Ollama's default Modelfile vs the training template. Ship a custom Modelfile/template with the GGUF.
- **For Qwen3.5 the thinking default is ambiguous across sources.** Explicitly pass `enable_thinking:false` in both training formatting and the llama.cpp server rather than relying on defaults.

### Gaps
- No report was found of thinking re-emerging after LoRA on non-thinking data. The absence of reports is not proof.
- The Qwen3.5 small-model default (thinking vs non-thinking) remains contradictory between the HF card and Unsloth. Check `chat_template.jinja` in the repo directly.

---

## 3. Unsloth / PEFT / TRL QLoRA support and free-T4 (16 GB) fit at seq len ~2048

### Takeaway
**Qwen3 dense up to 14B is the sweet spot for free-T4 QLoRA.** Unsloth says Qwen3-14B "fits comfortably" on a 16 GB T4, so 8B and 4B do too.

**Qwen3.5 is a poor fit for a free T4.** Unsloth explicitly says QLoRA 4-bit "is not recommended" for any Qwen3.5 model because of quantization error. The recommended bf16 LoRA needs ~22 GB for 9B, which is beyond a T4. Only Qwen3.5-0.8B/2B/4B (3/5/10 GB) fit.

**MoE 30B-A3B / 35B-A3B do not fit a free T4.** Qwen3-30B-A3B needs 17.5 GB with Unsloth; Qwen3.5-35B-A3B bf16 LoRA needs 74 GB. Qwen3-30B-A3B-Instruct-2507 is trainable on a 24 GB L4.

### Cited Findings
- "Qwen3 (14B) fits comfortably in a Google Colab 16GB VRAM Tesla T4 GPU". Free Colab notebooks exist for Qwen3-14B (Reasoning + Conversational) and Qwen3-4B (GRPO). — [Unsloth Qwen3 guide](https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune)
- "Qwen3-30B-A3B works on just 17.5GB VRAM with Unsloth". — [Unsloth Qwen3 guide](https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune)
- **Qwen3.5 bf16 LoRA VRAM:** 0.8B 3 GB, 2B 5 GB, 4B 10 GB, 9B 22 GB, 27B 56 GB, 35B-A3B 74 GB. — [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- **Unsloth on Qwen3.5 QLoRA:** "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense", due to "higher than normal quantization differences". For MoE, "MoE QLoRA 4-bit is not recommended due to BitsandBytes limitations". — [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- **Unsloth Qwen3.5 notebooks:** free Colab notebooks for 0.8B/2B/4B bf16 LoRA (plus a 4B GRPO notebook); A100 notebooks for 27B and 35B-A3B. No 9B text notebook. Kernel compilation "can take longer than normal, especially on T4 GPUs". Text-only SFT on the multimodal checkpoint works, and `finetune_vision_layers=False` freezes the vision part. Qwen3.5 requires Transformers v5. GGUF export works via `save_pretrained_gguf` (q4_k_m, q8_0, f16). "You must use the same chat template you trained with." — [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- **Independent evidence that LoRA on small Qwen3.5 works for structured tasks:** Qwen 3.5 4B LoRA reached 96.60% F1 / 91.67% strict exact-match on merchant extraction, vs 96.95% for LoRA Llama-3.1-8B, at ~3.8x lower inverse throughput. Qwen 3.5 0.8B reached 94.75% F1. — [arXiv 2606.08051 (ICDM 2026)](https://arxiv.org/abs/2606.08051)

### Inferences
- **Qwen3-8B QLoRA at seq len 2048 on a T4 should fit** with Unsloth (4-bit base, gradient checkpointing, batch 1–2 with grad accumulation), since 14B is documented to fit. Expect ~6–8 GB+ headroom. This is an estimate; no exact 8B/2048 VRAM figure was found.
- **Qwen3-4B-Instruct-2507 QLoRA is trivially T4-feasible** and roughly 2x faster per step than 8B (estimate).
- **Qwen3.5-9B is effectively ruled out for free-T4 training** if you follow Unsloth's guidance: no QLoRA, and 22 GB for bf16 LoRA. You could ignore the advice and QLoRA it anyway at some quality risk, or rent an L4 (24 GB), where 22 GB is tight.
- **T4s (Turing) have no native bf16.** Training falls back to fp16, which slightly undercuts the "bf16 LoRA" recipe for Qwen3.5 4B on free Colab. This is general hardware knowledge, not from the sources above.
- **Qwen3-30B-A3B-Instruct-2507 QLoRA (17.5 GB) is a good Modal L4 option** if a stronger base is wanted. 3k–5k short examples should take on the order of an hour or less (rough estimate).

### Gaps
- No measured VRAM figure for Qwen3-8B QLoRA at 2048 tokens on a T4. No figure for Qwen3-4B-Instruct-2507 either.
- A Hugging Face forum thread titled "Qwen3.5-4B loss exploding" exists ([HF forum](https://discuss.huggingface.co/t/qwen3-5-4b-loss-exploding/174057)) but was not read. It may point to Qwen3.5 training-stability issues.
- No evidence was found on whether Unsloth has since relaxed its Qwen3.5 QLoRA warning, or on Qwen3.6-35B-A3B fine-tuning VRAM. The Unsloth Qwen3.5 page does not mention Qwen3.6.

---

## 4. llama.cpp / GGUF / Ollama support and CPU serving (i7-13700K, 32 GB, 2–3 s per line)

### Takeaway
**Qwen3 and Qwen3-2507 (standard transformer / standard MoE)** have mature llama.cpp and Ollama support. The main gotcha is the empty-think template.

**Qwen3.5/3.6 (hybrid Gated DeltaNet + vision encoder)** are supported in llama.cpp, text and vision. However, Unsloth says its Qwen3.5 GGUFs do not work in Ollama because of separate mmproj files, and Ollama users reported Qwen3.5 running far slower than llama.cpp (issue open as of late Mar 2026).

For CPU, prompt processing of the ~1000-token frame, not generation, is the likely latency bottleneck for all candidates.

### Cited Findings
- "llama.cpp supports the Qwen3.5 open model series (text & vision)." — [QwenLM/Qwen3.8 README](https://github.com/QwenLM/Qwen3.8)
- **Unsloth on Qwen3.5 and Ollama:** "Currently no Qwen3.5 GGUF works in Ollama due to separate mmproj vision files. Use llama.cpp compatible backends." (undated). — [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5.md)
- **Qwen3.5 GGUF memory (Unsloth):** 4B 5.5 GB @4-bit; 9B 6.5 GB @4-bit / 9 GB @6-bit; 27B 17 GB @4-bit; 35B-A3B 22 GB @4-bit. Chat-template fixes were shipped 5 Mar for the larger models. If you see gibberish, the context length may be too low; try `--cache-type-k bf16 --cache-type-v bf16`. — [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5.md)
- **Ollama vs llama.cpp on Qwen3.5 ([ollama#14579](https://github.com/ollama/ollama/issues/14579), opened 3 Mar 2026, still open 27 Mar 2026):**
  - Qwen3.5-35B-A3B Q4 ran at ~15–20 tok/s in Ollama vs ~100 tok/s in llama.cpp on an RTX 3090 Ti.
  - One user reported qwen3.5:27b 67x slower than gemma3:27b in Ollama.
  - The suspected cause is memory estimation for the "hybrid Mamba+Attention architecture". A fix commit exists in a fork; no merge was confirmed.
- **Community CPU-adjacent data:**
  - Qwen3.5-9B on an M2 16 GB Mac mini: 53.8 tok/s prompt eval, 9.8 tok/s generation, multimodal prompt. — [llama.cpp#20704](https://github.com/ggml-org/llama.cpp/issues/20704)
  - A CPU user of Qwen3.5-4B found prompt processing dominates wall time. — [askwell#661](https://github.com/Rumeasiyan/askwell/issues/661)
- **Qwen3 on llama.cpp/Ollama (Unsloth):** use `/no_think` or `/nothink` to disable thinking. For MoE, `-ot ".ffn_.*_exps.=CPU"` offloads experts to CPU. For 6+ tok/s, "available memory should match or exceed the size of the model". 128K GGUFs use YaRN. — [Unsloth Qwen3 guide](https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune)
- **Artificial Analysis (5 Mar 2026):** 4-bit footprint about 6 GB for Qwen3.5-9B and about 3 GB for 4B. — [AA: Qwen3.5 small models](https://artificialanalysis.ai/articles/qwen3-5-small-models)

### Inferences (estimates — must be measured with `llama-bench -p 1024 -n 40` on the target i7-13700K)
- **Generation on DDR5 desktop CPUs is memory-bandwidth-bound.**
  - An 8–9B model at Q4_K_M (~5–6 GB) plausibly gives ~10–14 tok/s, so a 30–40-token line takes ~3 s.
  - A 4B (~2.5–3 GB) is roughly 2x faster.
  - A 3B-active MoE (Qwen3-30B-A3B-2507, Qwen3.5/3.6-35B-A3B) generates at roughly 4B-dense speed but needs ~18–22 GB resident RAM.
- **Prompt processing of 500–1500 tokens on CPU is likely several seconds or more for 8–9B.** This exceeds the 2–3 s target unless the static prefix (system prompt, match card) is cached in llama.cpp's server and only the new events are processed each line, or unless the frame is shrunk.
- **For Qwen3.5/3.6, prompt-prefix reuse may behave differently.** These are hybrid recurrent (DeltaNet) models, so KV-style reuse needs state checkpoints. This is a risk to verify.
- **For the CPU budget, smaller dense non-thinking models (Qwen3-4B-Instruct-2507) are the safest bet.** The 3B-active MoEs are the "quality at speed" option if RAM allows.
- **Prefer llama.cpp `llama-server` over Ollama for Qwen3.5/3.6.** Reasons: the Ollama compatibility and performance issues above, the reported dropping of presence_penalty, and thinking-toggle bugs. For Qwen3 and Qwen3-2507, Ollama is fine if you supply a custom template.

### Gaps
- No i7-13700K (or similar desktop x86) llama.cpp benchmarks were found for Qwen3-8B, Qwen3-4B-2507, Qwen3.5-9B or the 3B-active MoEs. No CPU prompt-processing numbers specific to Gated DeltaNet were found either.
- Whether Ollama fixed the Qwen3.5 speed issue (#14579), or its own library builds now handle Qwen3.5 mmproj, was not confirmed after Mar 2026.
- llama.cpp prompt-cache / context-checkpoint behaviour for hybrid DeltaNet models was not verified.

---

## 5. Benchmarks relevant to the use case (instruction following, creative writing, hallucination/grounding, chat quality) and known weaknesses

### Takeaway
**Qwen-reported numbers (official):**
- **Qwen3-2507 Instruct checkpoints are far better on writing than original Qwen3 non-thinking.** Qwen3-4B-Instruct-2507 scores Creative Writing v3 83.5 vs 53.6 for Qwen3-4B non-thinking, and WritingBench 83.4. Qwen3-30B-A3B-Instruct-2507 scores Creative Writing v3 86.0.
- **Qwen3-8B non-thinking scores only 64.5 on Creative Writing v3 and 83.0 on IFEval.**
- **Qwen3.5-9B reports the highest instruction-following numbers** (IFEval 91.5, IFBench 64.5, MultiChallenge 54.5, likely thinking mode) but publishes no creative-writing numbers.

**Independent measurements:**
- Artificial Analysis rates Qwen3.5-9B the most intelligent model under 10B, but extremely token-hungry and with an 82% hallucination rate on knowledge recall.
- No reliable EQ-Bench or LMArena entries were retrievable for these small models.

### Cited Findings

**Official — Qwen3 tech report (May 2025), non-thinking mode** ([arXiv 2505.09388](https://arxiv.org/html/2505.09388), Tables 16 and 18):

| Benchmark | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3-30B-A3B | Gemma-3-12B-IT (ref) | Llama-3.1-8B (ref) |
|---|---|---|---|---|---|---|
| IFEval (strict prompt) | 81.2 | 83.0 | 84.8 | 83.7 | 80.2 | 75.0 |
| Arena-Hard | 66.2 | 79.6 | 86.3 | 88.0 | 82.6 | 30.1 |
| Creative Writing v3 | 53.6 | 64.5 | 73.1 | 68.1 | 79.9 | 52.8 |
| WritingBench (0–10) | 6.85 | 7.15 | 7.24 | 7.22 | 7.05 | 4.57 |
| Multi-IF | 61.3 | 69.2 | 72.9 | 70.8 | 65.6 | 52.1 |

- **Qwen3-8B in thinking mode (Table 17):** IFEval 85.0, Arena-Hard 85.8, Creative Writing v3 75.0, WritingBench 7.59. — [arXiv 2505.09388](https://arxiv.org/html/2505.09388). A search-engine summary had the 8B thinking/non-thinking creative-writing values swapped; the fetched tables above are authoritative.

**Official — 2507 model cards (Jul/Aug 2025; Arena-Hard v2 judged by GPT-4.1; WritingBench here on a 0–100 scale):**

| Benchmark | Qwen3-4B non-think | Qwen3-4B-Instruct-2507 | Qwen3-30B-A3B non-think | Qwen3-30B-A3B-Instruct-2507 | GPT-4.1-nano (ref) | GPT-4o-0327 (ref) |
|---|---|---|---|---|---|---|
| IFEval | 81.2 | 83.4 | 83.7 | 84.7 | 74.5 | 83.9 |
| Arena-Hard v2 | 9.5 | 43.4 | 24.8 | 69.0 | 15.9 | 61.9 |
| Creative Writing v3 | 53.6 | 83.5 | 68.1 | 86.0 | 72.7 | 84.9 |
| WritingBench | 68.5 | 83.4 | 72.2 | 85.5 | 66.9 | 75.5 |
| MultiIF | 61.3 | 69.0 | 70.8 | 67.9 | 60.7 | 70.4 |

Sources: [HF Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [HF Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

**Official — Qwen3.5 small (Mar 2026)** ([HF Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B); compared against Qwen3-30B-A3B-Thinking-2507 and GPT-OSS-20B):
- Qwen3.5-9B / 4B: IFEval 91.5 / 89.8; IFBench 64.5 / 59.2; MultiChallenge 54.5 / 49.0; MMLU-Pro 82.5 / 79.1.
- Qwen3-30B-A3B-Thinking-2507: IFEval 88.9, IFBench 51.5, MultiChallenge 46.5.
- Vision-side HallusionBench: Qwen3.5-9B 69.3.
- No Creative Writing, WritingBench or Arena-Hard numbers are given, and there is no direct Qwen3-8B comparison.

**Independent — Artificial Analysis, 5 Mar 2026** ([AA article](https://artificialanalysis.ai/articles/qwen3-5-small-models)):
- Intelligence Index (reasoning variants): Qwen3.5-9B 32, Qwen3.5-4B 27, Qwen3 4B 2507 18, Qwen3 VL 8B 17.
- Output tokens to run the index: Qwen3.5-9B ~260M, 4B ~240M, vs Qwen3.5-27B 98M. Small-model token usage is "significantly higher" than predecessors.
- AA-Omniscience (knowledge and hallucination): Qwen3.5-9B score −56, hallucination rate 82%, accuracy 14.7%. Qwen3.5-4B: −57 / 80% / 12.8%. Qwen3-4B-2507: −61 / 84% / 12.7%.
- AA's X post adds: "most intelligent model under 10B", with 200M+ output tokens. — [AA on X](https://x.com/ArtificialAnlys/status/2029496068615057758)

**Independent / community — creative writing:**
- An aggregator lists Qwen3.5-9B at 59.2 on "EQ-Bench Creative" (metric undefined) and a POLARIS fine-tune of it at 70.3. — [SOTA2 aggregator](https://www.sota2.com/research/sota/creative-writing-evaluation-on-eq-bench-creative). Low confidence.
- The eqbench.com Creative Writing v3 table could not be retrieved (rendered client-side). — [EQ-Bench CW v3](https://eqbench.com/creative_writing.html)

**Known weaknesses (sourced):**
- **Repetition/looping:**
  - Qwen recommends presence_penalty 0–2 "to reduce endless repetition" on 2507 models. — [HF 4B-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
  - For Qwen3.5 non-thinking, presence_penalty 1.5. — [HF Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
  - Unsloth warns that higher presence_penalty gives a "slight decrease in performance". — [Unsloth Qwen3.5](https://unsloth.ai/docs/models/qwen3.5.md)
  - Community reports describe infinite loops inside Qwen3.5 thinking. — [lyn.one](https://lyn.one/reasoning-control-flow)
- **Verbosity:** Qwen3.5 small models are very token-hungry in reasoning mode (AA, above). Qwen3.6-27B uses ~3.7x the output tokens of Gemma 4 31B. — [AA on X](https://x.com/ArtificialAnlys/status/2049881951260283097)
- **Original Qwen3 non-thinking is weak on creative writing:** Qwen3-8B scores 64.5 vs Gemma-3-12B 79.9 in Qwen's own table. — [arXiv 2505.09388](https://arxiv.org/html/2505.09388)

### Inferences
- **For short, punchy English caster lines, the "writing/chat" benchmarks (Creative Writing v3, WritingBench, Arena-Hard) matter more than reasoning.** On these, the non-thinking 2507 checkpoints are Qwen's best small models by a wide margin in Qwen's own evals. Qwen3-4B-Instruct-2507 (CW v3 83.5) beats Qwen3-8B non-thinking (64.5) and even Qwen3-8B thinking (75.0). Caveat: the 2507 cards may use a different eval harness or version than the May 2025 report (WritingBench scale changed from 0–10 to 0–100), so the comparison is only indicative.
- **AA-Omniscience measures closed-book knowledge recall,** not grounding on supplied context. Its high hallucination rates for all small Qwens mainly say "don't let it invent facts". For a CS2 caster grounded on the game-state frame, the fine-tune plus input-grounded training data matter more. Train lines that only reference entities present in the frame, and add a post-hoc check that player names and weapons exist in the input.
- **Fine-tuning on 3–5k examples will largely override base register.** Base-model writing quality matters most for fluency and variety, where 2507 > original Qwen3. Base IF scores matter less once the format is trained in.

### Gaps
- No LMArena (text) ratings were found for Qwen3-4B/8B, the 2507 small models, or Qwen3.5-4B/9B. Small models are mostly absent from LMArena.
- No eqbench.com Creative Writing v3 or Longform rows for small Qwens could be retrieved. Only Qwen-reported CW v3 numbers are available.
- No grounding/faithfulness benchmarks (e.g. FACTS Grounding, context-faithfulness) were found for small Qwen models.
- Qwen3.5-9B non-thinking-mode IFEval/IFBench numbers were not published separately. The card table appears to be the default (thinking) mode.

---

## 6. Is the newest Qwen small model meaningfully better than Qwen3-8B for this use, and what is its USP?

### Takeaway
**Not clearly, for this specific pipeline.**

Qwen3.5-9B (Mar 2026) is the newest Qwen under 14B. Its USP is much stronger general intelligence and instruction following for its size (AA Intelligence Index 32, IFEval 91.5, IFBench 64.5), plus native vision and 262K context. But it comes with costs:
- Unsloth discourages QLoRA on it, and its bf16 LoRA (22 GB) does not fit a free T4.
- Its hybrid DeltaNet + vision architecture had Ollama incompatibility and slowdowns.
- It publishes no creative-writing numbers, and its reasoning-mode verbosity is extreme.

The strongest Qwen alternative to Qwen3-8B for a short English caster line on CPU is arguably **Qwen3-4B-Instruct-2507** (Aug 2025):
- non-thinking only, so there is no think-tag leak;
- Qwen-reported Creative Writing v3 83.5 / WritingBench 83.4 / IFEval 83.4;
- trivially T4-trainable and roughly 2x faster on CPU.

With a cheap L4 for training and ~20 GB RAM to spare, **Qwen3-30B-A3B-Instruct-2507** is the "quality" option (CW v3 86.0, Arena-Hard v2 69.0) with 3.3B-active CPU speed.

### Cited Findings
- **Qwen3.5-9B vs prior small Qwens (AA):** 9B (reasoning) scores 32 on the Intelligence Index vs 17 for Qwen3 VL 8B and 18 for Qwen3-4B-2507. Token usage is very high (~260M). — [AA article](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- **Qwen3.5-9B official:** IFEval 91.5, IFBench 64.5, MultiChallenge 54.5, beating Qwen3-30B-A3B-Thinking-2507 (88.9 / 51.5 / 46.5). It is natively multimodal with 262K context. — [HF Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Qwen3.5 QLoRA is "not recommended"; 9B bf16 LoRA needs 22 GB.** — [Unsloth Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- **Qwen3.5 Ollama incompatibility and slowdown.** — [Unsloth Qwen3.5](https://unsloth.ai/docs/models/qwen3.5.md); [ollama#14579](https://github.com/ollama/ollama/issues/14579)
- **Qwen3-8B non-thinking:** IFEval 83.0, Creative Writing v3 64.5, Arena-Hard 79.6. — [arXiv 2505.09388](https://arxiv.org/html/2505.09388)
- **Qwen3-4B-Instruct-2507:** CW v3 83.5, WritingBench 83.4, IFEval 83.4, Arena-Hard v2 43.4; non-thinking only; 262K context. — [HF 4B-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Qwen3-30B-A3B-Instruct-2507:** CW v3 86.0, WritingBench 85.5, IFEval 84.7, Arena-Hard v2 69.0; 3.3B active; non-thinking only. — [HF 30B-A3B-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507). The base Qwen3-30B-A3B needs 17.5 GB for Unsloth QLoRA. — [Unsloth Qwen3](https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune)
- **Qwen3.6 (Apr 2026)** adds only 27B and 35B-A3B. Its card emphasises agentic coding (SWE-bench Verified 73.4 for 35B-A3B) and gives no IFEval or writing benchmarks. — [HF Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Small LoRA'd Qwen3.5 models are competitive on structured output in an independent study:** 4B reached 96.6% F1 vs 96.95% for Llama-3.1-8B. — [arXiv 2606.08051](https://arxiv.org/abs/2606.08051)

### Inferences
- **Suggested ranking of Qwen bases for this project** (project-specific judgement from the above, not a sourced ranking):
  1. **Qwen3-4B-Instruct-2507:** best fit to all constraints (T4 QLoRA, CPU latency, no thinking, strong writing scores, Apache-2.0).
  2. **Qwen3-8B (hybrid, thinking off):** the current shortlist. Solid IF and well supported, but weaker on Qwen's own creative-writing metric and with an empty-think-tag template quirk. A reasonable A/B partner for #1.
  3. **Qwen3-30B-A3B-Instruct-2507:** the quality option. Train on an L4 rather than a T4; serve on CPU with ~18–20 GB RAM.
  4. **Qwen3.5-4B (bf16 LoRA fits a T4 at ~10 GB, fp16 on T4) or Qwen3.5-9B (needs ≥24 GB GPU, no QLoRA):** for experimenting if instruction-following headroom is needed. Use llama.cpp, not Ollama, and pin `enable_thinking:false`.
  5. **Qwen3.5/3.6-35B-A3B, Qwen3.6-27B, Qwen3.8-27B:** not practical to fine-tune on the project's hardware. Possibly useful as zero-shot teachers to generate or clean synthetic training lines, if licence terms allow (all Apache-2.0).
- **The 2–3 s CPU target is likely the binding constraint.** Model choice should be decided by a 30-minute `llama-bench` on the i7-13700K with the real ~1000-token prompt, comparing 4B-2507 Q4_K_M, 8B Q4_K_M and 30B-A3B-2507 Q4_K_M. Prefix caching is likely required for any of them.
- **If Qwen4 (reportedly with a 27B open tier) lands in Nov 2026,** it is unlikely to change the sub-10B choice unless a small dense tier is also released and its licence is permissive.

### Gaps
- No head-to-head (official or independent) of Qwen3.5-9B vs Qwen3-8B on writing/chat-style benchmarks in non-thinking mode.
- No public fine-tuning reports for short-form sports/esports commentary on any Qwen model.
- CPU latency on the target machine is unmeasured (see Q4).
