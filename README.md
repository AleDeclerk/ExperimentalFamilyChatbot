# Emirati Family Chatbot

A domain-specific chatbot that speaks Emirati Arabic dialect, specialized in family affairs. Fine-tuned with QLoRA and quantized to GGUF for edge deployment on devices with 8GB of RAM.

## What it does

- Answers family-related questions in Emirati Arabic dialect (not MSA)
- Covers: parenting, cooking, traditions, adolescent issues, celebrations, daily life
- Politely refuses out-of-scope topics (programming, investments, etc.)
- Maintains a warm but professional assistant persona (never role-plays as a family member)

## Tech Stack

| Component | Tool |
|-----------|------|
| Base model | Aya Expanse 8B (CohereLabs/aya-expanse-8b) |
| Fine-tuning | QLoRA via PEFT + TRL |
| Quantization | llama.cpp (GGUF Q5_K_M) |
| Inference | llama.cpp server (OpenAI-compatible API) |
| UI | Gradio |
| Dev machine | Apple M3 Max 48GB |

## Quick Start

### Prerequisites

- Python 3.11+
- llama.cpp compiled with Metal (macOS) or CUDA (Linux)
- ~16GB disk for model files

### Run inference (after fine-tuning)

```bash
# Terminal 1: Start llama.cpp server
./llama.cpp/build/bin/llama-server \
  -m models/quantized/aya-chatbot-q5km.gguf \
  -t 4 -c 2048 --port 8080

# Terminal 2: Start Gradio UI
python deploy/ui.py
# Open http://localhost:7860
```

### Run the full pipeline

```bash
# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download Aya Expanse 8B to models/base/aya-expanse-8b/
huggingface-cli download CohereForAI/aya-expanse-8b --local-dir models/base/aya-expanse-8b

# Run pipeline (prepare data -> fine-tune -> fuse -> quantize -> test)
bash scripts/run_aya_pipeline.sh
```

## Dataset

2755 conversations in Emirati Arabic across 21 categories, generated with Claude API and manually validated. Includes multi-turn conversations and targeted weakness fixes.

See [TRAINING_DATA.md](TRAINING_DATA.md) for full dataset documentation.

## Project History

This project went through multiple model iterations (Falcon H1R 7B -> Jais 7B Chat -> Aya Expanse 8B) and 5 dataset versions. See [JOURNEY.md](JOURNEY.md) for the complete development story.

## Project Structure

```
├── data/
│   ├── raw/                    # Raw JSONL (v1-v5)
│   └── processed/              # Train/test splits per model
├── scripts/
│   ├── finetune.py             # QLoRA fine-tuning (generic, any HF model)
│   ├── fuse_model.py           # Merge LoRA adapter + base
│   ├── evaluate.py             # Evaluate against test set
│   ├── prepare_dataset.py      # Validate + split data
│   ├── run_aya_pipeline.sh     # End-to-end pipeline (Aya)
│   └── generate_*.py           # Data generation scripts
├── models/
│   ├── base/                   # Downloaded base models
│   ├── adapters/               # LoRA adapters
│   ├── fused/                  # Merged models
│   └── quantized/              # GGUF files for deployment
├── deploy/
│   ├── server.py               # FastAPI wrapper
│   └── ui.py                   # Gradio chat interface
├── eval/                       # Results and logs
└── llama.cpp/                  # Quantization tools
```

## Performance (Jais 7B, previous model)

| Metric | Result |
|--------|--------|
| Tokens/s | 60+ tok/s (M3 Max) |
| TTFT | 60-150ms |
| GGUF size | 4.7GB (Q5_K_M) |
| RAM peak | ~5.7GB |

## License

MIT
