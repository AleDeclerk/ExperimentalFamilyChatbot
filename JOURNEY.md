# Journey: Building an Emirati Arabic Family Chatbot

The complete story of how this project evolved, every model we tried, every problem we hit, and how we solved it.

---

## The Goal

Build a chatbot that speaks Emirati Arabic dialect (not formal/MSA Arabic), specialized in family affairs — parenting, cooking, traditions, adolescent support — and runs on edge devices with just 8GB of RAM.

Dev machine: Apple M3 Max with 48GB RAM.

---

## Phase 1: Environment Setup (the hard part nobody warns you about)

### Python 3.14 disaster
Started with Python 3.14 (latest). Immediately hit a wall: `dill` (a dependency of `datasets`) had a breaking change that made it incompatible. The entire HuggingFace ecosystem couldn't load.

**Fix:** Downgraded to Python 3.11. Lesson: ML ecosystem lags behind Python releases by 1-2 major versions.

### Unsloth won't work on Apple Silicon
The original spec called for Unsloth (fast QLoRA). Turns out Unsloth only supports NVIDIA/AMD/Intel GPUs. No Apple Silicon support.

**Fix:** Rewrote the fine-tuning script from scratch using PEFT + TRL directly. More boilerplate but works on MPS.

### TRL 0.24 breaking changes
TRL had recently shipped breaking changes:
- `SFTTrainer` config moved to `SFTConfig`
- `tokenizer` parameter renamed to `processing_class`
- `max_seq_length` renamed to `max_length`
- `use_mps_device` removed entirely

Took a while to debug because the error messages weren't clear.

### llama.cpp build
Needed cmake (not just make) for the Metal backend. Quick `brew install cmake` and a clean build fixed it.

---

## Phase 2: First Model — Falcon H1R 7B (failed)

### Why Falcon H1R
The original spec targeted Falcon H1R 7B because:
- Hybrid Mamba-Transformer architecture = theoretically faster inference
- Strong reasoning benchmarks (88.1% AIME-24)
- Apache 2.0-ish license
- Official GGUF support

### What happened
Trained on 125 examples. Loss barely moved: 4.77 -> 4.67.

The model's responses were a mess — mixed Arabic and English, couldn't maintain the Emirati dialect, and the reasoning-oriented architecture seemed to fight against the conversational format. It kept trying to output `<think>` tags instead of actual responses.

### The quantization nightmare
Falcon H1R has Mamba layers (SSM) alongside transformer layers. During GGUF conversion, the SSM_A parameters (state-space matrices) had values that went to infinity after `exp(A_log)` in layers 38 and 43. This completely broke quantization.

We wrote a fix (`fix_ssm_weights.py`) that clamped `A_log` values, which let quantization succeed, but the model quality was already too poor to justify continuing.

**Decision: Falcon H1R is the wrong model for this task.** Good at math, bad at Arabic conversation.

---

## Phase 3: The Search for an Arabic-Native Model

### Falcon H1 Arabic — phantom model
TII (the Falcon team) had announced "Falcon H1 Arabic" but it wasn't on HuggingFace. Couldn't find it anywhere. Maybe internal-only or not yet released.

### Jais 7B Chat — the right choice
Found Jais, built by Inception/G42 (a UAE company). Specifically designed for Arabic including Gulf dialects. Key facts:
- Llama architecture (no exotic layers = clean quantization)
- 7B parameters
- Gated repo on HuggingFace (had to accept terms of use)
- Actually trained on Arabic internet data including dialectal text

### Also evaluated
- **Qwen2.5-3B-Instruct:** Good multilingual model, smaller (3B). We ran a full pipeline with it but Aya 8B was the better bet for quality.

---

## Phase 4: Jais Fine-Tuning — Iterative Improvement

### Round 1: 260 examples
- Loss: 4.3 -> 1.75
- Accuracy jumped from 36% to 70%
- Arabic was coherent but generic (MSA-ish, not Emirati)
- Refusals didn't work at all — the model would happily help with programming questions

### Round 2: 1370 examples
Expanded the dataset significantly. Added multi-turn conversations.
- Loss: 4.36 -> 0.37
- Accuracy: 35% -> 93%
- Major improvement in Emirati dialect usage
- Multi-turn conversations worked naturally

### Round 3: 2268 examples (v4 dataset)
This round focused on fixing specific problems:

**Problem 1: Persona confusion.** The chatbot would respond as if it were a family member ("ya waladi" = my son, "habibi"). Users would say "ya yimma" (mom) and the bot would play along.

**Fix:** Rewrote all training data to enforce the assistant persona. Created `fix_persona.py` to clean the entire dataset. Added explicit persona boundary examples ("I'm your virtual assistant, not your mother, but I'm here to help!").

**Problem 2: Adolescent topics.** No coverage of teen issues — puberty, menstruation, emotional support for teenagers.

**Fix:** Generated 78 examples covering adolescent topics with cultural sensitivity. The menstruation examples were particularly important — many Emirati families need support discussing this topic.

**Problem 3: Weak refusals.** The model would semi-refuse but then answer anyway ("this isn't my specialty, but here's how to code a website...").

**Fix:** Added 30 firm-but-kind refusal examples. The pattern: acknowledge the request, clearly state it's outside scope, redirect to family topics, end with "if you need family advice, I'm here!"

---

## Phase 5: Quantization and Edge Testing (Jais)

### Pipeline
1. Fuse LoRA adapter into base model (`merge_and_unload`)
2. Convert to GGUF f16 with llama.cpp's `convert_hf_to_gguf.py`
3. Quantize from f16 to Q5_K_M with `llama-quantize`

Unlike Falcon H1R, Jais (pure Llama architecture) quantized without any issues.

### Results
- **GGUF Q5_K_M size:** 4.7GB (target was ~5.4GB — came in under!)
- **Tokens/sec:** 60+ on M3 Max (target was >5 tok/s)
- **TTFT:** 60-150ms (target was <2s)
- **RAM peak:** ~5.7GB (target was <7GB)

All targets exceeded by a wide margin on the dev machine.

---

## Phase 6: Preparing for Aya Expanse 8B

### Why upgrade from Jais
Jais 7B Chat works well but has limitations:
- Sometimes falls back to MSA Arabic
- Responses can be verbose (ChatGPT-style lists)
- 7B model from a smaller lab vs 8B from Cohere

### Why Aya Expanse 8B
- CohereLabs model, specifically designed for multilingual use
- 23 languages with strong Arabic support
- 8B parameters (slightly larger but better quality-per-param)
- Active development and community
- Cohere-style chat template (well-structured)

### Weakness scan
Before fine-tuning Aya, we ran a systematic scan of how the base model handles our use cases. Found 5 categories of weakness:

1. **Dialect:** Responds in MSA instead of Emirati (says "يمكنك" instead of "تقدر")
2. **Persona:** Assumes family member role when addressed as "yimma" or "baba"
3. **Refusals:** Doesn't refuse out-of-scope topics (programming, investments)
4. **Verbose tone:** Long responses with markdown lists, feels like ChatGPT
5. **Partial scope:** Semi-refuses but gives info anyway

### Dataset v5: targeted fixes
Generated 487 additional examples specifically targeting these weaknesses:
- 130 examples of pure Emirati dialect responses
- 95 persona correction examples
- 149 firm refusal examples
- 115 concise/warm tone examples

**Total dataset: 2755 examples**

### Training config adjustments for Aya
Compared to Jais config, we adjusted:
- **LR: 1e-4** (lower than Jais's 2e-4 — larger model needs gentler learning)
- **LoRA alpha: 32** (doubled from 16 — higher alpha for stronger adaptation)
- **Max seq length: 1024** (doubled from 512 — Aya handles longer context)
- **Epochs: 3** (reduced from 5 — more data needs fewer passes)

---

## Phase 7: Pipeline and Deployment (current)

### What's built
- `run_aya_pipeline.sh` — End-to-end pipeline: data prep -> fine-tune -> fuse -> quantize -> test
- `deploy/ui.py` — Gradio chat interface
- `deploy/server.py` — FastAPI wrapper for llama.cpp
- GitHub repo: public at AleDeclerk/ExperimentalFamilyChatbot

### What's next
1. Run the Aya fine-tuning (~2-3 hours on M3 Max)
2. Evaluate against test set
3. Compare Aya vs Jais quality
4. Gradio demo with final model

---

## Dataset Evolution Summary

```
v1 (125 examples)
 └─> Basic 10 categories, manual creation
      │
v2 (260 examples)
 └─> Expanded with variations, more examples per category
      │
v3 (1262 examples)
 └─> Multi-turn conversations added
 └─> Persona cleaned (assistant, not family member)
      │
v4 (2268 examples)
 └─> Adolescent topics (puberty, menstruation, emotional support)
 └─> Robust refusals (programming, investments, poetry)
 └─> 17 categories total
      │
v5 (2755 examples)
 └─> Weakness fixes for Aya base model
 └─> Pure Emirati dialect reinforcement
 └─> Firmer refusals, concise tone
 └─> 21 categories total
```

---

## Models Tried

| Model | Params | Result | Why stopped |
|-------|--------|--------|-------------|
| Falcon H1R 7B | 7B | Failed | Poor Arabic, SSM quantization issues, outputs `<think>` tags |
| Jais 7B Chat | 7B | Good (93% acc) | Functional but MSA fallback, verbose |
| Qwen2.5-3B-Instruct | 3B | Pipeline tested | Smaller model, used for pipeline validation |
| Aya Expanse 8B | 8B | In progress | Best multilingual Arabic support |

---

## Key Lessons

1. **The ML ecosystem is fragile.** Python version, CUDA/MPS compatibility, library breaking changes — expect to spend 30% of your time on environment issues.

2. **Model selection > training tricks.** Switching from Falcon to Jais was worth more than any hyperparameter tuning.

3. **Data quality scales better than data quantity.** Going from 260 to 1370 examples with the same quality gave diminishing returns. Going from generic data to targeted weakness fixes gave outsized improvements.

4. **Persona enforcement needs explicit training data.** System prompts alone are not enough — the model needs to see examples of maintaining boundaries.

5. **Refusals must be firm.** A half-refusal ("this isn't my area but here are some tips...") is worse than no refusal at all. Train on clean refusals.

6. **Quantization is model-dependent.** Llama-architecture models (Jais, Aya) quantize cleanly. Exotic architectures (Mamba hybrid) can have numerical issues.

7. **Apple Silicon is viable for fine-tuning** but requires different tooling (no Unsloth, no BitsAndBytes 4-bit, float16 instead of NF4).
