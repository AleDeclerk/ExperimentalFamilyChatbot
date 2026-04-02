# Edge Chatbot: Proceso de Desarrollo

## Resumen

Chatbot familiar en arabe emirati fine-tuneado con QLoRA, cuantizado a GGUF Q5_K_M para deployment en dispositivos con 8GB de RAM.

**Modelo actual:** Aya Expanse 8B (CohereLabs/aya-expanse-8b) -> QLoRA -> GGUF Q5_K_M
**Modelos anteriores:** Falcon H1R 7B (descartado), Jais 7B Chat (funcional, superado)
**Dataset:** 2755 conversaciones en arabe emirati (v5)
**Rendimiento (Jais):** 60+ tok/s en Apple M3 Max, TTFT ~60-150ms
**Dev machine:** Apple M3 Max 48GB

---

## M1: Environment Setup

### Estructura del proyecto
```
ExperimentalFamilyChatbot/
├── data/raw/              # Datos crudos (JSONL, v1-v5)
├── data/processed/        # Train/test splits por modelo
├── scripts/               # Pipeline scripts
├── models/base/           # Modelos base descargados
├── models/adapters/       # LoRA adapters
├── models/fused/          # Modelo fusionado
├── models/quantized/      # GGUF cuantizados
├── deploy/                # Server y UI
├── eval/                  # Resultados y logs
└── llama.cpp/             # Build de llama.cpp
```

### Dependencias
- Python 3.11 (3.14 era incompatible con el ecosistema ML: dill breaking change)
- PyTorch 2.10 con MPS (Apple Silicon)
- PEFT 0.18 + TRL 0.24 (Unsloth no soporta Apple Silicon)
- llama.cpp compilado con Metal

### Problemas resueltos
1. **Unsloth incompatible con Apple Silicon** -> reescritura del script de fine-tuning usando PEFT + TRL directamente
2. **Python 3.14 incompatible** con dill -> downgrade a Python 3.11
3. **TRL 0.24 breaking changes**: SFTTrainer -> SFTConfig, tokenizer -> processing_class, max_seq_length -> max_length, se elimino use_mps_device
4. **llama.cpp necesitaba cmake** -> instalacion via Homebrew

---

## M2: Dataset

### Evolucion del dataset

| Version | Ejemplos | Notas |
|---------|----------|-------|
| v1 | 125 | Primera iteracion, 10 categorias |
| v2 | 260 | Expandido con variaciones |
| v3 | 1262 | +multi-turn, limpieza de persona |
| v4 | 2268 | +adolescentes, +pubertad/menstruacion, +refusals robustos |
| v5 | 2755 | +weakness fixes para Aya (dialecto, persona, refusals, tono) |

### Categorias (21 categorias en v5)

**Base (v1-v4):** Saludos, Comida/cocina, Crianza, Salud/adultos mayores, Celebraciones, Religion/tradiciones, Relaciones familiares, Vida diaria, Tecnologia moderna, Tradiciones/cultura, Viajes, Problemas/soluciones, Refusals basicos, Adolescentes, Pubertad/menstruacion, Refusals robustos, Soporte emocional

**Weakness Fixes (v5, +487 ejemplos):** Dialecto emirati puro, Correccion de persona, Refusals firmes, Tono conciso y calido

### Generacion de datos
- Single-turn: Generados con Claude API, validados manualmente
- Multi-turn: Combinacion automatica de turnos con validacion de coherencia
- Weakness fixes: Generados a partir de scan de debilidades del modelo Aya base

### Decision de persona
El chatbot es siempre un asistente virtual profesional y calido. Nunca asume el rol de familiar (mama, papa, abuelo). Esto se refuerza en el system prompt y en datos de training.

---

## M3: Fine-Tuning

### Intento 1: Falcon H1R 7B (fallo total)
- Loss: 4.77 -> 4.67 (125 ejemplos, practicamente no aprendio)
- **Resultado: 0% usable.** Generaba `<think>` tags en loop infinito en vez de respuestas
- 3 de 3 tests fallidos: output literal era `<think>` (256 tokens de basura)
- Mezclaba arabe e ingles, no entendia que tenia que conversar
- Cuantizacion rota: capas SSM con valores inf en layers 38 y 43
- Velocidad: 6.5 tok/s pero generando tokens inutiles

### Intento 2: Jais 7B Chat (260 ejemplos)
- Loss: 4.3 -> 1.75 | Accuracy: 36% -> 70%
- Arabe coherente pero en MSA generico (formal), no emirati
- Refusals inexistentes: le pedias programar y programaba alegremente
- Sin personalidad ni dialecto

### Intento 3: Jais 7B Chat (1370 ejemplos)
- Loss: 4.36 -> 0.37 | Accuracy: 35% -> 93%
- Gran mejora en dialecto emirati y multi-turn
- PERO: confundia persona (le decian "yimma" y se hacia la mama)
- Refusals debiles: "no es mi area pero aca van tips de programacion..."
- Tono verboso, listas estilo ChatGPT

### Intento 4: Jais 7B Chat (2268 ejemplos)
- Persona corregida + adolescentes + refusals firmes
- Mejora notable pero aun caia en MSA ocasionalmente
- Scope parcial: semi-rechazaba pero daba info de todas formas

### Intento 5: Aya Expanse 8B (2755 ejemplos) - COMPLETADO
- Modelo multilingue de Cohere, 8B params, excelente soporte arabe
- Dataset v5 con weakness fixes especificos (dialecto, persona, refusals, tono)
- Training: 3 epochs, 465 steps, ~3h en M3 Max
- GGUF Q5_K_M: 5.4GB | 57 tok/s | TTFT 64-256ms
- Refusals: perfectos. Persona: se mantiene. Dialecto: muy natural
- Debil en menstruacion (respuesta incoherente) y artifacts multilingues ocasionales

### Configuracion de entrenamiento (Aya)
```
Modelo base: CohereLabs/aya-expanse-8b (local)
Metodo: QLoRA (LoRA rank=16, alpha=32, dropout=0.05)
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Batch size: 1
Gradient accumulation: 16
Effective batch size: 16
Learning rate: 1e-4 (cosine scheduler)
Epochs: 3
Max sequence length: 1024
Dispositivo: Apple M3 Max 48GB (MPS, float16)
```

---

## M4: Fusion y Cuantizacion

### Pipeline
1. Fusionar LoRA adapter con modelo base (merge_and_unload)
2. Guardar modelo fusionado en float16
3. Convertir a GGUF f16 con convert_hf_to_gguf.py
4. Cuantizar a Q5_K_M con llama-quantize

### Resultados

| Modelo | GGUF Q5_K_M | Cabe en 8GB |
|--------|-------------|-------------|
| Jais 7B | 4.7GB | Si, con margen |
| Aya 8B | 5.4GB | Si, justo |

---

## M5: Simulacion Edge

### Resultados de performance

| Metrica | Target (spec) | Jais 7B Q5_K_M | Aya 8B Q5_K_M |
|---------|--------------|----------------|---------------|
| Tokens/s | >5 tok/s | **60+ tok/s** | **57 tok/s** |
| TTFT | <2s | **60-150ms** | **64-256ms** |
| Tamano GGUF | ~5.4GB | **4.7GB** | **5.4GB** |
| RAM peak | <7GB | ~5.7GB | ~6.3GB |

---

## M6: Demo

### Componentes
- **llama.cpp server** (puerto 8080): API OpenAI-compatible
- **FastAPI server** (puerto 8000): Wrapper con validacion
- **Gradio UI** (puerto 7860): Interfaz de chat web

### Ejecucion
```bash
# Terminal 1: Servidor
llama.cpp/build/bin/llama-server -m models/quantized/aya-chatbot-q5km.gguf -t 4 -c 2048 --port 8080

# Terminal 2: UI
python deploy/ui.py
# Abrir http://localhost:7860
```

---

## Scripts del Pipeline

| Script | Funcion |
|--------|---------|
| prepare_dataset.py | Valida JSONL, filtra invalidos, split train/test |
| finetune.py | QLoRA fine-tuning con PEFT + TRL (MPS compatible) |
| fuse_model.py | Fusiona LoRA adapter con modelo base |
| evaluate.py | Evalua contra test set |
| benchmark_edge.sh | Benchmark con llama-bench |
| generate_1500.py | Genera datos base (13 categorias) |
| generate_adolescents.py | Genera datos de adolescentes/pubertad/refusals |
| generate_more_data.py | Genera datos adicionales |
| generate_weakness_fixes.py | Genera fixes para debilidades detectadas |
| fix_persona.py | Limpia persona del asistente en todos los datos |
| run_full_pipeline.sh | Pipeline end-to-end (Qwen) |
| run_aya_pipeline.sh | Pipeline end-to-end (Aya Expanse 8B) |
| deploy/server.py | FastAPI wrapper para llama.cpp |
| deploy/ui.py | Gradio chat interface |

---

## Decisiones Clave

1. **Aya sobre Jais sobre Falcon:** Falcon H1R tiene ventaja arquitectonica pero es pobre en arabe. Jais es nativo arabe y funcional. Aya Expanse 8B es multilingue con excelente arabe y mayor capacidad (8B vs 7B).

2. **Python 3.11 sobre 3.14:** El ecosistema ML no es compatible con Python 3.14 todavia.

3. **PEFT+TRL sobre Unsloth:** Unsloth solo soporta NVIDIA/AMD/Intel GPUs. En Apple Silicon usamos PEFT + TRL directamente.

4. **Persona de asistente:** El chatbot es un asistente virtual, nunca un familiar. Reforzado en system prompt y datos.

5. **Q5_K_M como target:** Balance optimo entre calidad y tamano. Q4_K_M como fallback para devices mas limitados.

6. **Dataset iterativo:** Cada version del dataset resuelve problemas especificos detectados en evaluacion. v5 incluye weakness fixes targetados para Aya.
