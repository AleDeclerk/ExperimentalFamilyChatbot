#!/usr/bin/env bash
# Benchmark de latencia/throughput simulando condiciones edge (4 threads, 2048 ctx).
#
# Uso:
#   bash scripts/benchmark_edge.sh models/quantized/falcon-chatbot-q5km.gguf

set -euo pipefail

MODEL="${1:?Uso: $0 <path-to-gguf>}"
THREADS=4
CTX=2048
PROMPT_TOKENS=128
GEN_TOKENS=256

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Modelo no encontrado: $MODEL"
    exit 1
fi

echo "=== Benchmark Edge ==="
echo "Modelo: $MODEL"
echo "Threads: $THREADS | Context: $CTX"
echo "Prompt tokens: $PROMPT_TOKENS | Gen tokens: $GEN_TOKENS"
echo ""

# Benchmark con llama-bench
if command -v llama-bench &>/dev/null; then
    llama-bench -m "$MODEL" -t "$THREADS" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS"
elif [ -f "llama.cpp/llama-bench" ]; then
    ./llama.cpp/llama-bench -m "$MODEL" -t "$THREADS" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS"
else
    echo "ERROR: llama-bench no encontrado. Compilá llama.cpp primero."
    exit 1
fi
