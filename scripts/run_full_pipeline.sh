#!/usr/bin/env bash
# Pipeline completo: generar datos → preparar → fine-tune Jais → fusionar → cuantizar → test
set -euo pipefail

cd "/Users/alejandrodeclerk/Desktop/Repos/Falcon chatbot"
VENV=".venv/bin/python"
LOG="eval/pipeline_log.txt"
mkdir -p eval

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "========================================="
log "PIPELINE COMPLETO - Qwen2.5-3B-Instruct"
log "========================================="

MODEL_BASE="./models/base/qwen2.5-3b-instruct"
MODEL_NAME="qwen-chatbot"
ADAPTER_DIR="models/adapters/qwen-v1"
FUSED_DIR="models/fused/qwen-v1"
SYSTEM_MSG="أنت مساعد افتراضي ودود متخصص في الشؤون العائلية، تتحدث باللهجة الإماراتية. تقدم النصائح والمساعدة لأفراد العائلة في حياتهم اليومية بأسلوب دافئ ومحترم. أنت دائماً مساعد وليس فرداً من العائلة. استخدم التعبيرات الإماراتية الشائعة."

# === PASO 1: Datos ya generados y limpios ===
log "PASO 1: Usando datos limpios v4 (asistente virtual + adolescentes + refusals)..."
wc -l data/raw/all_conversations_v4.jsonl | tee -a "$LOG"

# === PASO 2: Preparar dataset ===
log "PASO 2: Preparando train/test split..."
$VENV scripts/prepare_dataset.py \
  --input data/raw/all_conversations_v4.jsonl \
  --output-train data/processed/train_qwen.jsonl \
  --output-test data/processed/test_qwen.jsonl \
  --test-size 0.1 2>&1 | tee -a "$LOG"

# === PASO 3: Fine-tune Qwen2.5-3B-Instruct ===
log "PASO 3: Fine-tuning Qwen2.5-3B-Instruct..."
log "  Modelo: $MODEL_BASE"
log "  Epochs: 5, Batch: 2, Grad Accum: 8, LR: 2e-4, LoRA rank: 32"

rm -rf "$ADAPTER_DIR"

$VENV scripts/finetune.py \
  --model-name "$MODEL_BASE" \
  --train-data data/processed/train_qwen.jsonl \
  --output-dir "$ADAPTER_DIR" \
  --epochs 5 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --learning-rate 2e-4 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --max-seq-length 512 2>&1 | tee -a "$LOG"

log "Fine-tuning completado!"

# === PASO 4: Fusionar adapter + modelo base ===
log "PASO 4: Fusionando adapter con modelo base..."
rm -rf "$FUSED_DIR"

$VENV scripts/fuse_model.py \
  --base-model "$MODEL_BASE" \
  --adapter-path "$ADAPTER_DIR" \
  --output-path "$FUSED_DIR" 2>&1 | tee -a "$LOG"

log "Fusión completada!"

# === PASO 5: Convertir a GGUF y cuantizar ===
log "PASO 5: Convirtiendo a GGUF f16..."
rm -f models/quantized/${MODEL_NAME}-f16.gguf models/quantized/${MODEL_NAME}-q5km.gguf

$VENV llama.cpp/convert_hf_to_gguf.py \
  "$FUSED_DIR" \
  --outtype f16 \
  --outfile models/quantized/${MODEL_NAME}-f16.gguf 2>&1 | tee -a "$LOG"

log "Cuantizando a Q5_K_M..."
./llama.cpp/build/bin/llama-quantize \
  models/quantized/${MODEL_NAME}-f16.gguf \
  models/quantized/${MODEL_NAME}-q5km.gguf \
  Q5_K_M 2>&1 | tee -a "$LOG"

log "Cuantización completada!"
ls -lh models/quantized/${MODEL_NAME}-q5km.gguf | tee -a "$LOG"

# === PASO 6: Test rápido con llama-server ===
log "PASO 6: Test rápido con llama.cpp..."

./llama.cpp/build/bin/llama-server \
  -m models/quantized/${MODEL_NAME}-q5km.gguf \
  -t 4 -c 2048 --host 127.0.0.1 --port 8080 &
SERVER_PID=$!
sleep 10

# Test 1: Saludo
log "Test 1: Saludo..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM_MSG\"},
      {\"role\": \"user\", \"content\": \"هلا، شخبارك؟\"}
    ],
    \"max_tokens\": 256, \"temperature\": 0.7
  }" 2>/dev/null | $VENV -c "
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
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM_MSG\"},
      {\"role\": \"user\", \"content\": \"شو نطبخ حق الغدا اليوم؟\"}
    ],
    \"max_tokens\": 256, \"temperature\": 0.7
  }" 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d['timings']
print(f'Response: {c.get(\"content\",\"\")[:300]}')
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"

# Test 3: Adolescente
log "Test 3: Adolescente..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM_MSG\"},
      {\"role\": \"user\", \"content\": \"بنتي عمرها ١٣ وصارت عصبية وايد شو أسوي؟\"}
    ],
    \"max_tokens\": 256, \"temperature\": 0.7
  }" 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d['timings']
print(f'Response: {c.get(\"content\",\"\")[:300]}')
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"

# Test 4: Menstruación
log "Test 4: Menstruación..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM_MSG\"},
      {\"role\": \"user\", \"content\": \"بنتي يتها أول دورة وخايفة، شو أسوي؟\"}
    ],
    \"max_tokens\": 256, \"temperature\": 0.7
  }" 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d['timings']
print(f'Response: {c.get(\"content\",\"\")[:300]}')
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"

# Test 5: Refusal
log "Test 5: Refusal..."
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"$SYSTEM_MSG\"},
      {\"role\": \"user\", \"content\": \"ساعدني أبرمج موقع\"}
    ],
    \"max_tokens\": 256, \"temperature\": 0.7
  }" 2>/dev/null | $VENV -c "
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
