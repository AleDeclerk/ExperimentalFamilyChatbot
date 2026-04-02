# Edge Chatbot: Project Specification

## 1. Overview

Fine-tune a Falcon H1R 7B model with QLoRA to build a domain-specific chatbot that runs on devices with 8GB of RAM. Development happens locally on an M3 Max (48GB); the deliverable is a quantized GGUF model (~5GB) that serves inference on constrained hardware via llama.cpp.

---

## 2. Goals & Non-Goals

### Goals

- Fine-tune Falcon H1R 7B on a custom domain dataset using QLoRA
- Produce a quantized GGUF artifact (Q5_K_M) deployable on 8GB devices
- Serve the chatbot via a lightweight API (llama.cpp server or FastAPI)
- Validate quality and latency under simulated edge constraints
- Document the full pipeline so it's reproducible for other domains/clients

### Non-Goals

- Full fine-tune (we use parameter-efficient QLoRA only)
- GPU inference on the target device (CPU-only)
- Multi-model serving or model routing
- Production-grade frontend (Gradio demo is sufficient for v1)

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DEV ENVIRONMENT (Mac M3 Max 48GB)         │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Dataset   │───▶│ QLoRA    │───▶│ Evaluate │              │
│  │ Prep      │    │ Fine-tune│    │ & Iterate│              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                        │                                    │
│                        ▼                                    │
│               ┌──────────────┐    ┌──────────────┐         │
│               │ Fuse Adapter │───▶│ Quantize to  │         │
│               │ + Base Model │    │ GGUF Q5_K_M  │         │
│               └──────────────┘    └──────────────┘         │
│                                          │                  │
│                        ┌─────────────────┘                  │
│                        ▼                                    │
│               ┌──────────────────┐                          │
│               │ Simulate Edge    │                          │
│               │ (4 threads, 2048 │                          │
│               │  ctx, CPU-only)  │                          │
│               └──────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                         │
                         │  .gguf artifact (~5GB)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              TARGET DEVICE (8GB RAM, CPU-only)               │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ llama.cpp    │───▶│ FastAPI /    │───▶│ Gradio UI /  │  │
│  │ server       │    │ REST API     │    │ Client App   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack

| Component | Tool | Version / Notes |
|-----------|------|-----------------|
| Base model | Falcon H1R 7B | `tiiuae/Falcon-H1R-7B` from HuggingFace |
| Fine-tuning | Unsloth + TRL | QLoRA, 4-bit base, LoRA rank 16 |
| Dataset format | JSONL | ChatML / messages format |
| Quantization | llama.cpp | `convert_hf_to_gguf.py` → `llama-quantize` |
| Target quant | Q5_K_M | ~5.4GB file, fits in 8GB with headroom |
| Inference server | llama.cpp server | CPU-only, OpenAI-compatible API |
| Demo UI | Gradio | Minimal chat interface |
| Containerization | Docker | Optional, for reproducible edge deploy |
| Language | Python 3.11+ | All scripts |
| Dev machine | Mac M3 Max 48GB | Apple Silicon, MLX available as fallback |

---

## 5. Model Selection Rationale

**Why Falcon H1R 7B over alternatives:**

- Hybrid Transformer-Mamba architecture: faster inference than pure-transformer 7B models
- Strong reasoning (88.1% AIME-24) means it retains more capability post-quantization
- Official GGUF support from TII
- Apache 2.0-based license (Falcon TII License) permits commercial use
- 7B params in Q5_K_M = ~5.4GB, leaving ~2.5GB for OS + KV cache on 8GB device

**Quantization target: Q5_K_M**

| Quant | File Size | RAM Needed | Quality | Verdict |
|-------|-----------|------------|---------|---------|
| Q8_0 | 8.07 GB | ~9-10 GB | Excellent | Too tight for 8GB device |
| Q6_K | 6.23 GB | ~7.5 GB | Very good | Risky, no headroom |
| Q5_K_M | 5.39 GB | ~6.5 GB | Good | **Selected — safe margin** |
| Q4_K_M | 4.60 GB | ~5.5 GB | Acceptable | Fallback if Q5 is too tight |

---

## 6. Dataset Specification

### Format

```jsonl
{
  "messages": [
    {"role": "system", "content": "System prompt defining chatbot persona and domain."},
    {"role": "user", "content": "User question or request."},
    {"role": "assistant", "content": "Expected response."}
  ]
}
```

### Requirements

- Minimum 500 examples for tone/format adaptation
- Recommended 1,000–3,000 for domain knowledge injection
- Train/test split: 90/10
- Max sequence length: 2048 tokens per conversation
- All examples manually reviewed or generated via Claude API with human validation

### Dataset Sources (choose based on domain)

1. **Existing documentation** → Parse FAQs, knowledge bases, manuals into Q&A pairs
2. **Synthetic generation** → Use Claude API to expand docs into conversational pairs
3. **Real conversations** → Anonymized support tickets or chat logs
4. **Hybrid** → Seed with real data, expand with synthetic, validate manually

### Data Quality Checklist

- [ ] No PII in training data
- [ ] Consistent system prompt across all examples
- [ ] Responses are concise (under 300 tokens average for edge latency)
- [ ] Balanced topic coverage across the domain
- [ ] Adversarial examples included (out-of-scope questions with polite refusals)

---

## 7. Fine-Tuning Configuration

### QLoRA Hyperparameters

```python
# LoRA config
r = 16                    # Rank — start here, increase to 32 if underfitting
lora_alpha = 16           # Scale factor — keep equal to r
lora_dropout = 0          # Unsloth recommends 0
target_modules = [        # All linear layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training config
per_device_train_batch_size = 4
gradient_accumulation_steps = 4     # Effective batch size = 16
learning_rate = 2e-4
num_train_epochs = 3
warmup_steps = 50
max_seq_length = 2048
lr_scheduler_type = "cosine"
weight_decay = 0.01
fp16 = True                         # BF16 on Apple Silicon if using MLX
```

### Expected Resource Usage (M3 Max)

- Model in 4-bit: ~4GB
- LoRA adapters + optimizer states: ~2-4GB
- Activations + batch: ~4-8GB
- **Total: ~10-16GB** (well within 48GB)
- Training time estimate: ~30-60 min for 1,000 examples, 3 epochs

---

## 8. Pipeline Steps

### Step 1: Environment Setup

```bash
# Create project structure
mkdir -p edge-chatbot/{data/{raw,processed},scripts,models/{base,adapters,fused,quantized},deploy,eval}
cd edge-chatbot

# Python environment
python3 -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install unsloth transformers datasets accelerate peft trl
pip install llama-cpp-python gradio fastapi uvicorn

# llama.cpp for quantization
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc) && cd ..

# Download base model
huggingface-cli download tiiuae/Falcon-H1R-7B --local-dir ./models/base/falcon-h1r-7b
```

### Step 2: Prepare Dataset

```bash
python scripts/prepare_dataset.py \
  --input data/raw/conversations.jsonl \
  --output-train data/processed/train.jsonl \
  --output-test data/processed/test.jsonl \
  --test-size 0.1
```

### Step 3: Fine-Tune

```bash
python scripts/finetune.py \
  --model-name tiiuae/Falcon-H1R-7B \
  --train-data data/processed/train.jsonl \
  --output-dir models/adapters/v1 \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --lora-rank 16
```

### Step 4: Evaluate

```bash
python scripts/evaluate.py \
  --model-path tiiuae/Falcon-H1R-7B \
  --adapter-path models/adapters/v1 \
  --test-data data/processed/test.jsonl \
  --output eval/results_v1.json
```

### Step 5: Fuse & Quantize

```bash
# Fuse adapter into base model
python scripts/fuse_model.py \
  --base-model tiiuae/Falcon-H1R-7B \
  --adapter-path models/adapters/v1 \
  --output-path models/fused/v1

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py \
  models/fused/v1 \
  --outtype f16 \
  --outfile models/quantized/falcon-chatbot-f16.gguf

# Quantize to Q5_K_M
./llama.cpp/llama-quantize \
  models/quantized/falcon-chatbot-f16.gguf \
  models/quantized/falcon-chatbot-q5km.gguf \
  Q5_K_M
```

### Step 6: Simulate Edge & Validate

```bash
# Simulate 8GB device constraints
./llama.cpp/llama-server \
  -m models/quantized/falcon-chatbot-q5km.gguf \
  -t 4 \
  -c 2048 \
  --host 0.0.0.0 \
  --port 8080

# Run latency benchmark
./llama.cpp/llama-bench \
  -m models/quantized/falcon-chatbot-q5km.gguf \
  -t 4 -p 128 -n 256
```

### Step 7: Deploy to Target

```bash
# Copy artifact to device
scp models/quantized/falcon-chatbot-q5km.gguf user@device:/opt/chatbot/

# On device: run server
./llama-server \
  -m /opt/chatbot/falcon-chatbot-q5km.gguf \
  -t 4 -c 2048 \
  --host 0.0.0.0 --port 8080 --mlock
```

---

## 9. Evaluation Criteria

### Quality Metrics

| Metric | Method | Target |
|--------|--------|--------|
| Domain accuracy | LLM-as-judge (Claude API) on test set | > 85% correct |
| Tone consistency | Manual review of 50 random outputs | Matches persona |
| Refusal rate | Out-of-scope questions correctly refused | > 90% |
| Hallucination | Domain-specific factual questions | < 5% hallucination |

### Performance Metrics (on target device)

| Metric | Target |
|--------|--------|
| Time to first token | < 2 seconds |
| Tokens per second | > 5 tok/s (CPU-only) |
| Memory usage | < 7GB peak |
| Context window | 2048 tokens functional |

### Comparison Baseline

Run the same test set on the **base Falcon H1R 7B Q5_K_M without fine-tuning** to measure the delta. The fine-tuned version should show measurable improvement in domain accuracy and tone.

---

## 10. Project Structure

```
edge-chatbot/
├── data/
│   ├── raw/                    # Source materials (docs, FAQs, tickets)
│   └── processed/
│       ├── train.jsonl          # Training split
│       └── test.jsonl           # Evaluation split
├── scripts/
│   ├── prepare_dataset.py       # Raw → JSONL, train/test split
│   ├── generate_synthetic.py    # Claude API → synthetic training data
│   ├── finetune.py              # QLoRA fine-tuning with Unsloth
│   ├── fuse_model.py            # Merge adapter + base
│   ├── evaluate.py              # Run test set, compute metrics
│   └── benchmark_edge.sh        # Latency/throughput on simulated edge
├── models/
│   ├── base/                    # Falcon H1R 7B (FP16, ~15GB)
│   ├── adapters/                # LoRA adapters per version (~50-100MB)
│   │   ├── v1/
│   │   └── v2/
│   ├── fused/                   # Adapter merged into base
│   └── quantized/               # GGUF files for deployment
│       ├── falcon-chatbot-q5km.gguf
│       └── falcon-chatbot-q4km.gguf  # Fallback
├── eval/
│   ├── results_v1.json          # Evaluation results per version
│   └── benchmarks/              # Latency/throughput logs
├── deploy/
│   ├── server.py                # FastAPI wrapper
│   ├── ui.py                    # Gradio chat interface
│   ├── Dockerfile               # Container for edge device
│   └── docker-compose.yml
├── llama.cpp/                   # Cloned repo for quantization tools
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quality degrades after quantization | Medium | Evaluate at each quant level; fall back to Q4_K_M or use Q6_K |
| Fine-tune overfits on small dataset | High | Early stopping, eval loss monitoring, augment with synthetic data |
| Falcon H1R hybrid arch not supported by tool | Medium | Verify Unsloth/TRL compatibility before starting; fall back to HF Transformers + PEFT |
| Inference too slow on target CPU | Medium | Reduce context to 1024; try Q4_K_M; optimize llama.cpp build flags for target arch |
| Mamba layers lose quality with quantization | Low-Medium | Benchmark reasoning tasks specifically pre/post quant; compare with Falcon 3-7B (pure transformer) as backup model |

---

## 12. Milestones

| # | Milestone | Deliverable | Est. Time |
|---|-----------|-------------|-----------|
| M1 | Environment ready | All deps installed, base model downloaded, inference verified | Day 1-2 |
| M2 | Dataset v1 | 500+ examples in JSONL, train/test split, quality reviewed | Day 3-5 |
| M3 | First fine-tune | Adapter v1 trained, basic eval shows improvement over base | Day 6-7 |
| M4 | Quantized artifact | GGUF Q5_K_M produced, inference validated | Day 8 |
| M5 | Edge simulation | Benchmarks pass on simulated constraints (speed, RAM, quality) | Day 9-10 |
| M6 | Demo ready | Gradio UI running on simulated edge, end-to-end demo | Day 11-12 |
| M7 | Deploy to device | Model running on actual target hardware | Day 13-14 |

---

## 13. Future Iterations

- **v2:** Expand dataset to 3,000+ examples, add multi-turn conversations
- **v3:** Experiment with DPO (Direct Preference Optimization) for response quality
- **v4:** Add RAG layer for dynamic knowledge (llama.cpp supports embeddings)
- **v5:** Test Falcon Edge models for even smaller devices (2-4GB)
- **Productize:** Package as a repeatable offering — client provides docs/data, pipeline produces deployable chatbot
