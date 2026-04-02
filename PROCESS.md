# Edge Chatbot: Proceso de Desarrollo

## Resumen

Chatbot familiar en arabe emirati fine-tuneado con QLoRA sobre Jais 7B Chat, cuantizado a GGUF Q5_K_M (~4.7GB) para deployment en dispositivos con 8GB de RAM.

**Modelo final:** Jais 7B Chat -> QLoRA -> GGUF Q5_K_M
**Rendimiento:** 60+ tok/s en Apple M3 Max, TTFT ~60-150ms
**Dataset:** ~2268 conversaciones en arabe emirati

---

## M1: Environment Setup

### Estructura del proyecto
```
edge-chatbot/
+-- data/raw/              # Datos crudos (JSONL)
+-- data/processed/        # Train/test splits
+-- scripts/               # Pipeline scripts
+-- models/base/           # Modelos base descargados
+-- models/adapters/       # LoRA adapters
+-- models/fused/          # Modelo fusionado
+-- models/quantized/      # GGUF cuantizados
+-- deploy/                # Server y UI
+-- eval/                  # Resultados y logs
+-- llama.cpp/             # Build de llama.cpp
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

| Version | Ejemplos | Categorias | Notas |
|---------|----------|-----------|-------|
| v1 | 125 | 10 | Primera iteracion manual |
| v2 | 260 | 10 | Expandido con variaciones |
| v3 | 1262 | 13 + multi-turn | Limpieza de persona (asistente virtual) |
| v4 | 2268 | 17 + multi-turn | +adolescentes, +pubertad/menstruacion, +refusals robustos |

### Categorias del dataset final (v4)

**Categorias base:**
1. Saludos y check-ins (35 ejemplos)
2. Comida y cocina (40 ejemplos)
3. Crianza y ninos (30 ejemplos)
4. Salud y adultos mayores (20 ejemplos)
5. Celebraciones y eventos (20 ejemplos)
6. Religion y tradiciones (14 ejemplos)
7. Relaciones familiares (15 ejemplos)
8. Vida diaria y hogar (15 ejemplos)
9. Tecnologia y vida moderna (10 ejemplos)
10. Tradiciones y cultura (10 ejemplos)
11. Viajes y paseos (10 ejemplos)
12. Problemas y soluciones (10 ejemplos)
13. Refusals basicos (15 ejemplos)

**Categorias nuevas (v4):**
14. Adolescentes general (30 ejemplos)
15. Pubertad y menstruacion (36 ejemplos)
16. Refusals robustos (30 ejemplos)
17. Soporte emocional adolescentes (12 ejemplos)

**Multi-turn:** ~1200 conversaciones de 2-3 turnos generadas por combinacion

### System prompt

```
Eres un asistente virtual amable especializado en asuntos familiares, hablas en dialecto emiratí.
Ofreces consejos y ayuda a los miembros de la familia en su vida diaria con un estilo calido y respetuoso.
Siempre eres un asistente, nunca un miembro de la familia. No asumas el rol de madre, padre, abuelo ni ningun pariente.
Usa expresiones emiratíes comunes y mantén un estilo educado y profesional.
```

### Decision de persona
Inicialmente el chatbot respondia como familiar ("ya waladi", "habibti"). Se corrigio para que mantenga siempre su postura de asistente virtual profesional pero calido.

---

## M3: Fine-Tuning

### Intento 1: Falcon H1R 7B (descartado)
- **Problema:** Modelo orientado a razonamiento (hybrid Mamba-Transformer), pobre en arabe
- **Resultado:** Loss 4.77->4.67 (125 ejemplos), respuestas mezcladas arabe/ingles
- **Decision:** Cambiar a un modelo nativo en arabe

### Investigacion de modelos arabes
- **Falcon H1 Arabic:** Anunciado por TII pero no disponible en HuggingFace
- **Jais 7B Chat:** De Inception/G42 (UAE), entrenado especificamente en arabe incluyendo dialectos del Golfo. Arquitectura Llama. Gated repo -> requirio aceptacion de terminos.

### Intento 2: Jais 7B Chat (260 ejemplos)
- **Loss:** 4.3 -> 1.75
- **Accuracy:** 36% -> 70%
- **Resultado:** Arabe coherente pero generico, refusals no funcionan

### Intento 3: Jais 7B Chat (1370 ejemplos)
- **Loss:** 4.36 -> 0.37
- **Accuracy:** 35% -> 93%
- **Resultado:** Arabe emirati natural, mucho mejor calidad

### Intento 4: Jais 7B Chat (2268 ejemplos) - FINAL
- Dataset con persona corregida + adolescentes + menstruacion + refusals
- Mejora esperada en consistencia de persona y manejo de refusals

### Configuracion de entrenamiento
```
Modelo base: inceptionai/jais-adapted-7b-chat
Metodo: QLoRA (LoRA rank=16, alpha=16, dropout=0.05)
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Batch size: 1
Gradient accumulation: 16
Effective batch size: 16
Learning rate: 2e-4 (cosine scheduler)
Epochs: 3
Max sequence length: 512
Dispositivo: Apple M3 Max 48GB (MPS, float16)
```

---

## M4: Fusion y Cuantizacion

### Pipeline
1. Fusionar LoRA adapter con modelo base (merge_and_unload)
2. Guardar modelo fusionado en float16 (~14GB)
3. Convertir a GGUF f16 con convert_hf_to_gguf.py
4. Cuantizar a Q5_K_M con llama-quantize

### Problema con Falcon H1R (resuelto al cambiar a Jais)
Las capas Mamba (SSM_A) de Falcon H1R tenian valores inf tras exp(A_log) que hacian fallar la cuantizacion. Se resolvio clampeando A_log en layers 38 y 43. Con Jais (arquitectura Llama pura) no hay este problema.

### Resultado
- GGUF Q5_K_M: **4.7GB** (cabe en 8GB device con margen)
- Cuantizacion sin errores

---

## M5: Simulacion Edge

### Configuracion
```bash
llama-server -m jais-chatbot-q5km.gguf -t 4 -c 2048 --host 0.0.0.0 --port 8080
```

### Resultados de performance

| Metrica | Target (spec) | Resultado |
|---------|--------------|-----------|
| Tokens/s | >5 tok/s | **60+ tok/s** |
| Time to first token | <2s | **60-150ms** |
| Tamano GGUF | ~5.4GB | **4.7GB** |
| RAM peak | <7GB | ~5.7GB |

---

## M6: Demo

### Componentes
- **llama.cpp server** (puerto 8080): API OpenAI-compatible
- **Gradio UI** (puerto 7860): Interfaz de chat web

### Ejecucion
```bash
# Terminal 1: Servidor
llama.cpp/build/bin/llama-server -m models/quantized/jais-chatbot-q5km.gguf -t 4 -c 2048 --port 8080

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
| fix_persona.py | Limpia persona del asistente en todos los datos |
| run_full_pipeline.sh | Pipeline end-to-end automatizado |
| deploy/server.py | FastAPI wrapper para llama.cpp |
| deploy/ui.py | Gradio chat interface |

---

## Decisiones Clave

1. **Jais sobre Falcon H1R:** Falcon H1R tiene ventaja arquitectonica (Mamba hybrid) pero Jais es nativo en arabe y produce resultados superiores para este caso de uso.

2. **Python 3.11 sobre 3.14:** El ecosistema ML (dill, datasets, torch) no es compatible con Python 3.14 todavia.

3. **PEFT+TRL sobre Unsloth:** Unsloth solo soporta NVIDIA/AMD/Intel GPUs. En Apple Silicon usamos PEFT + TRL directamente.

4. **Packing desactivado:** El packing de secuencias requiere Flash Attention que no esta disponible en MPS. Sin packing el training es mas lento pero funciona correctamente.

5. **Persona de asistente:** El chatbot es un asistente virtual, nunca un familiar. Esto se refuerza en el system prompt y en los datos de training.

6. **Q5_K_M como target:** Balance optimo entre calidad y tamano (4.7GB). Q4_K_M disponible como fallback si 8GB device es muy ajustado.
