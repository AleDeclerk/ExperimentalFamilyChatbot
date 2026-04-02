# Chatbot Familiar Emirati

Chatbot de dominio especifico que habla en dialecto arabe emirati, especializado en asuntos familiares. Fine-tuneado con QLoRA y cuantizado a GGUF para deployment en dispositivos con 8GB de RAM.

## Que hace

- Responde preguntas familiares en dialecto arabe emirati (no en arabe estandar/MSA)
- Cubre: crianza, cocina, tradiciones, temas de adolescentes, celebraciones, vida diaria
- Rechaza educadamente temas fuera de alcance (programacion, inversiones, etc.)
- Mantiene una persona de asistente profesional y calido (nunca se hace pasar por un familiar)

## Stack Tecnologico

| Componente | Herramienta |
|------------|-------------|
| Modelo base | Aya Expanse 8B (CohereLabs/aya-expanse-8b) |
| Fine-tuning | QLoRA via PEFT + TRL |
| Cuantizacion | llama.cpp (GGUF Q5_K_M) |
| Inferencia | llama.cpp server (API compatible con OpenAI) |
| UI | Gradio |
| Maquina de desarrollo | Apple M3 Max 48GB |

## Inicio Rapido

### Requisitos

- Python 3.11+
- llama.cpp compilado con Metal (macOS) o CUDA (Linux)
- ~16GB de disco para archivos del modelo

### Correr inferencia (despues del fine-tuning)

```bash
# Terminal 1: Levantar llama.cpp server
./llama.cpp/build/bin/llama-server \
  -m models/quantized/aya-chatbot-q5km.gguf \
  -t 4 -c 2048 --port 8080

# Terminal 2: Levantar Gradio UI
python deploy/ui.py
# Abrir http://localhost:7860
```

### Correr el pipeline completo

```bash
# Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Descargar Aya Expanse 8B a models/base/aya-expanse-8b/
huggingface-cli download CohereForAI/aya-expanse-8b --local-dir models/base/aya-expanse-8b

# Correr pipeline (preparar datos -> fine-tune -> fusionar -> cuantizar -> test)
bash scripts/run_aya_pipeline.sh
```

## Dataset

2755 conversaciones en arabe emirati en 21 categorias, generadas con Claude API y validadas manualmente. Incluye conversaciones multi-turn y correcciones targetadas de debilidades.

Ver [TRAINING_DATA.md](TRAINING_DATA.md) para documentacion completa del dataset.

## Historia del Proyecto

Este proyecto paso por multiples iteraciones de modelos (Falcon H1R 7B -> Jais 7B Chat -> Aya Expanse 8B) y 5 versiones de dataset. Ver [JOURNEY.md](JOURNEY.md) para la historia completa de desarrollo.

## Estructura del Proyecto

```
├── data/
│   ├── raw/                    # JSONL crudos (v1-v5)
│   └── processed/              # Splits train/test por modelo
├── scripts/
│   ├── finetune.py             # Fine-tuning QLoRA (generico, cualquier modelo HF)
│   ├── fuse_model.py           # Fusionar adapter LoRA + base
│   ├── evaluate.py             # Evaluar contra test set
│   ├── prepare_dataset.py      # Validar + dividir datos
│   ├── run_aya_pipeline.sh     # Pipeline end-to-end (Aya)
│   └── generate_*.py           # Scripts de generacion de datos
├── models/
│   ├── base/                   # Modelos base descargados
│   ├── adapters/               # Adapters LoRA
│   ├── fused/                  # Modelos fusionados
│   └── quantized/              # Archivos GGUF para deployment
├── deploy/
│   ├── server.py               # Wrapper FastAPI
│   └── ui.py                   # Interfaz de chat Gradio
├── eval/                       # Resultados y logs
└── llama.cpp/                  # Herramientas de cuantizacion
```

## Rendimiento (Aya Expanse 8B fine-tuned)

| Metrica | Resultado |
|---------|-----------|
| Tokens/s | 57 tok/s (M3 Max) |
| TTFT | 64-256ms |
| Tamano GGUF | 5.4GB (Q5_K_M) |
| RAM pico | ~6.3GB |
| Training | ~3h en M3 Max (MPS) |
| Dataset | 2755 ejemplos (v5) |

## Licencia

MIT
