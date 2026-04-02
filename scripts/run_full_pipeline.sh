#!/usr/bin/env bash
# Pipeline completo: generar datos → preparar → fine-tune Jais → fusionar → cuantizar → test
set -euo pipefail

cd "/Users/alejandrodeclerk/Desktop/Repos/Falcon chatbot"
VENV=".venv/bin/python"
LOG="eval/pipeline_log.txt"
mkdir -p eval

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "========================================="
log "PIPELINE COMPLETO - Jais 7B Chat"
log "========================================="

# === PASO 1: Datos ya generados y limpios ===
log "PASO 1: Usando datos limpios v4 (asistente virtual + adolescentes + refusals)..."
wc -l data/raw/all_conversations_v4.jsonl | tee -a "$LOG"

# === PASO 2: Preparar dataset ===
log "PASO 2: Preparando train/test split..."
$VENV scripts/prepare_dataset.py \
  --input data/raw/all_conversations_v4.jsonl \
  --output-train data/processed/train_jais.jsonl \
  --output-test data/processed/test_jais.jsonl \
  --test-size 0.1 2>&1 | tee -a "$LOG"

# === PASO 3: Fine-tune Jais 7B Chat ===
log "PASO 3: Fine-tuning Jais 7B Chat..."
log "  Modelo: models/base/jais-7b-chat"
log "  Epochs: 3, Batch: 1, Grad Accum: 16, LR: 2e-4, LoRA rank: 16"

# Limpiar adapters anteriores
rm -rf models/adapters/jais-v1

$VENV scripts/finetune.py \
  --model-name ./models/base/jais-7b-chat \
  --train-data data/processed/train_jais.jsonl \
  --output-dir models/adapters/jais-v1 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --learning-rate 2e-4 \
  --lora-rank 16 \
  --max-seq-length 512 2>&1 | tee -a "$LOG"

log "Fine-tuning completado!"

# === PASO 4: Fusionar adapter + modelo base ===
log "PASO 4: Fusionando adapter con modelo base..."
rm -rf models/fused/jais-v1

$VENV scripts/fuse_model.py \
  --base-model ./models/base/jais-7b-chat \
  --adapter-path models/adapters/jais-v1 \
  --output-path models/fused/jais-v1 2>&1 | tee -a "$LOG"

log "Fusión completada!"

# === PASO 5: Convertir a GGUF y cuantizar ===
log "PASO 5: Convirtiendo a GGUF f16..."
rm -f models/quantized/jais-chatbot-f16.gguf models/quantized/jais-chatbot-q5km.gguf

$VENV llama.cpp/convert_hf_to_gguf.py \
  models/fused/jais-v1 \
  --outtype f16 \
  --outfile models/quantized/jais-chatbot-f16.gguf 2>&1 | tee -a "$LOG"

log "Cuantizando a Q5_K_M..."
./llama.cpp/build/bin/llama-quantize \
  models/quantized/jais-chatbot-f16.gguf \
  models/quantized/jais-chatbot-q5km.gguf \
  Q5_K_M 2>&1 | tee -a "$LOG"

log "Cuantización completada!"
ls -lh models/quantized/jais-chatbot-q5km.gguf | tee -a "$LOG"

# === PASO 6: Test rápido con llama-server ===
log "PASO 6: Test rápido con llama.cpp..."

# Iniciar servidor en background
./llama.cpp/build/bin/llama-server \
  -m models/quantized/jais-chatbot-q5km.gguf \
  -t 4 -c 2048 --host 127.0.0.1 --port 8080 &
SERVER_PID=$!
sleep 10

# Test 1: Saludo
log "Test 1: Saludo..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "أنت مساعد عائلي ودود يتحدث باللهجة الإماراتية. أجب مباشرة باللهجة الإماراتية فقط."},
      {"role": "user", "content": "هلا يمه شخبارك؟"}
    ],
    "max_tokens": 512, "temperature": 0.7
  }' 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d['timings']
print(f'Response: {c.get(\"content\",\"\")[:300]}')
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"

# Test 2: Comida
log "Test 2: Comida..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "أنت مساعد عائلي ودود يتحدث باللهجة الإماراتية. أجب مباشرة باللهجة الإماراتية فقط."},
      {"role": "user", "content": "شو نطبخ حق الغدا اليوم؟"}
    ],
    "max_tokens": 512, "temperature": 0.7
  }' 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d['timings']
print(f'Response: {c.get(\"content\",\"\")[:300]}')
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"

# Test 3: Crianza
log "Test 3: Crianza..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "أنت مساعد عائلي ودود يتحدث باللهجة الإماراتية. أجب مباشرة باللهجة الإماراتية فقط."},
      {"role": "user", "content": "ولدي ما يسمع الكلام شو أسوي؟"}
    ],
    "max_tokens": 512, "temperature": 0.7
  }' 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d['timings']
print(f'Response: {c.get(\"content\",\"\")[:300]}')
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"

# Test 4: Refusal (out-of-scope)
log "Test 4: Refusal..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "أنت مساعد عائلي ودود يتحدث باللهجة الإماراتية. أجب مباشرة باللهجة الإماراتية فقط."},
      {"role": "user", "content": "ساعدني أبرمج موقع"}
    ],
    "max_tokens": 512, "temperature": 0.7
  }' 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d['timings']
print(f'Response: {c.get(\"content\",\"\")[:300]}')
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"

# Matar servidor
kill $SERVER_PID 2>/dev/null || true

log "========================================="
log "PIPELINE COMPLETADO!"
log "========================================="
log "Artefactos:"
log "  Adapter: models/adapters/jais-v1/"
log "  Fused:   models/fused/jais-v1/"
log "  GGUF:    models/quantized/jais-chatbot-q5km.gguf"
log "  Log:     eval/pipeline_log.txt"
log "========================================="
