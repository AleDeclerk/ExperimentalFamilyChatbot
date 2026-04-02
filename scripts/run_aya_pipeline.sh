#!/usr/bin/env bash
# Pipeline completo: preparar datos → fine-tune Aya Expanse 8B → fusionar → cuantizar → test
set -euo pipefail

cd "/Users/alejandrodeclerk/Desktop/Repos/ExperimentalFamilyChatbot"
VENV=".venv/bin/python"
LOG="eval/aya_pipeline_log.txt"
mkdir -p eval models/quantized

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "========================================="
log "PIPELINE COMPLETO - Aya Expanse 8B"
log "========================================="

MODEL_BASE="./models/base/aya-expanse-8b"
MODEL_NAME="aya-chatbot"
ADAPTER_DIR="models/adapters/aya-v1"
FUSED_DIR="models/fused/aya-v1"
SYSTEM_MSG="أنت مساعد افتراضي ودود متخصص في الشؤون العائلية، تتحدث باللهجة الإماراتية. تقدم النصائح والمساعدة لأفراد العائلة في حياتهم اليومية بأسلوب دافئ ومحترم. أنت دائماً مساعد وليس فرداً من العائلة. لا تتقمص دور أم أو أب أو جد أو أي قريب. استخدم التعبيرات الإماراتية الشائعة وحافظ على أسلوب مهذب ومهني."

# === PASO 1: Preparar dataset (2755 ejemplos, v5) ===
log "PASO 1: Preparando train/test split desde v5 (2755 ejemplos)..."
$VENV scripts/prepare_dataset.py \
  --input data/raw/all_conversations_v5.jsonl \
  --output-train data/processed/train_aya.jsonl \
  --output-test data/processed/test_aya.jsonl \
  --test-size 0.1 2>&1 | tee -a "$LOG"

TRAIN_COUNT=$(wc -l < data/processed/train_aya.jsonl)
TEST_COUNT=$(wc -l < data/processed/test_aya.jsonl)
log "  Train: $TRAIN_COUNT | Test: $TEST_COUNT"

# === PASO 2: Fine-tune Aya Expanse 8B ===
log "PASO 2: Fine-tuning Aya Expanse 8B con QLoRA..."
log "  Modelo: $MODEL_BASE"
log "  Epochs: 3, Batch: 1, Grad Accum: 16, LR: 1e-4, LoRA rank: 16"

rm -rf "$ADAPTER_DIR"

$VENV scripts/finetune.py \
  --model-name "$MODEL_BASE" \
  --train-data data/processed/train_aya.jsonl \
  --output-dir "$ADAPTER_DIR" \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --learning-rate 1e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --max-seq-length 1024 2>&1 | tee -a "$LOG"

log "Fine-tuning completado!"

# === PASO 3: Fusionar adapter + modelo base ===
log "PASO 3: Fusionando adapter con modelo base..."
rm -rf "$FUSED_DIR"

$VENV scripts/fuse_model.py \
  --base-model "$MODEL_BASE" \
  --adapter-path "$ADAPTER_DIR" \
  --output-path "$FUSED_DIR" 2>&1 | tee -a "$LOG"

log "Fusión completada!"

# === PASO 4: Convertir a GGUF y cuantizar ===
log "PASO 4: Convirtiendo a GGUF f16..."
rm -f models/quantized/${MODEL_NAME}-f16.gguf models/quantized/${MODEL_NAME}-q5km.gguf

$VENV llama.cpp/convert_hf_to_gguf.py \
  "$FUSED_DIR" \
  --outtype f16 \
  --outfile models/quantized/${MODEL_NAME}-f16.gguf 2>&1 | tee -a "$LOG"

log "Cuantizando a Q5_K_M..."
DYLD_LIBRARY_PATH=llama.cpp/build/bin ./llama.cpp/build/bin/llama-quantize \
  models/quantized/${MODEL_NAME}-f16.gguf \
  models/quantized/${MODEL_NAME}-q5km.gguf \
  Q5_K_M 2>&1 | tee -a "$LOG"

log "Cuantización completada!"
ls -lh models/quantized/${MODEL_NAME}-q5km.gguf | tee -a "$LOG"

# Borrar f16 para ahorrar espacio (~16GB)
rm -f models/quantized/${MODEL_NAME}-f16.gguf
log "f16 eliminado para ahorrar espacio."

# === PASO 5: Test rápido con llama-server ===
log "PASO 5: Test rápido con llama.cpp..."

DYLD_LIBRARY_PATH=llama.cpp/build/bin ./llama.cpp/build/bin/llama-server \
  -m models/quantized/${MODEL_NAME}-q5km.gguf \
  -t 4 -c 2048 --host 127.0.0.1 --port 8080 &
SERVER_PID=$!
sleep 12

run_test() {
  local label="$1"
  local prompt="$2"
  log "Test: $label..."
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"messages\": [
        {\"role\": \"system\", \"content\": \"$SYSTEM_MSG\"},
        {\"role\": \"user\", \"content\": \"$prompt\"}
      ],
      \"max_tokens\": 256, \"temperature\": 0.7
    }" 2>/dev/null | $VENV -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
t = d.get('timings', {})
print(f'Response: {c.get(\"content\",\"\")[:400]}')
if t:
    print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s | TTFT: {t[\"prompt_ms\"]:.0f}ms')
" 2>&1 | tee -a "$LOG"
}

run_test "Saludo" "هلا، شخبارك؟"
run_test "Comida" "شو نطبخ حق الغدا اليوم؟"
run_test "Adolescente" "بنتي عمرها ١٣ وصارت عصبية وايد شو أسوي؟"
run_test "Menstruación" "بنتي يتها أول دورة وخايفة، شو أسوي؟"
run_test "Refusal" "ساعدني أبرمج موقع"
run_test "Multiturn" "ولدي عمره ٥ سنوات ما يبي ياكل خضار، شو أسوي؟"

# Matar servidor
kill $SERVER_PID 2>/dev/null || true

log "========================================="
log "PIPELINE COMPLETADO!"
log "========================================="
log "Artefactos:"
log "  Adapter:    $ADAPTER_DIR"
log "  Fused:      $FUSED_DIR"
log "  GGUF:       models/quantized/${MODEL_NAME}-q5km.gguf"
log "  Log:        $LOG"
log "========================================="
