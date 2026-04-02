# Recorrido: Construyendo un Chatbot Familiar en Arabe Emirati

La historia completa de como evoluciono este proyecto, cada modelo que probamos, cada problema que encontramos, y como lo resolvimos.

---

## El Objetivo

Construir un chatbot que hable en dialecto arabe emirati (no arabe formal/MSA), especializado en asuntos familiares — crianza, cocina, tradiciones, soporte adolescente — y que corra en dispositivos edge con solo 8GB de RAM.

Maquina de desarrollo: Apple M3 Max con 48GB de RAM.

---

## Fase 1: Setup del Entorno (la parte dificil de la que nadie te avisa)

### Desastre con Python 3.14
Arrancamos con Python 3.14 (lo ultimo). Chocamos contra la pared inmediatamente: `dill` (una dependencia de `datasets`) tenia un breaking change que lo hacia incompatible. Todo el ecosistema de HuggingFace no podia cargar.

**Fix:** Downgrade a Python 3.11. Leccion: el ecosistema ML va 1-2 versiones mayores atras de Python.

### Unsloth no funciona en Apple Silicon
El spec original pedia Unsloth (QLoRA rapido). Resulta que Unsloth solo soporta GPUs NVIDIA/AMD/Intel. Nada de Apple Silicon.

**Fix:** Reescribimos el script de fine-tuning desde cero usando PEFT + TRL directo. Mas boilerplate pero funciona en MPS.

### Breaking changes de TRL 0.24
TRL habia sacado breaking changes recientemente:
- La config de `SFTTrainer` se mudo a `SFTConfig`
- El parametro `tokenizer` se renombro a `processing_class`
- `max_seq_length` se renombro a `max_length`
- `use_mps_device` se elimino completamente

Tomo un rato debuggear porque los mensajes de error no eran claros.

### Build de llama.cpp
Necesitaba cmake (no solo make) para el backend Metal. Un rapido `brew install cmake` y un build limpio lo resolvio.

---

## Fase 2: Primer Modelo — Falcon H1R 7B (fallo)

### Por que Falcon H1R
El spec original apuntaba a Falcon H1R 7B porque:
- Arquitectura hibrida Mamba-Transformer = teoricamente inferencia mas rapida
- Benchmarks fuertes en razonamiento (88.1% AIME-24)
- Licencia tipo Apache 2.0
- Soporte oficial de GGUF

### Que paso
Entrenamos con 125 ejemplos. El loss apenas se movio: 4.77 -> 4.67.

Las respuestas del modelo eran un desastre — mezclaba arabe e ingles, no podia mantener el dialecto emirati, y la arquitectura orientada a razonamiento parecia pelear contra el formato conversacional. Seguia intentando generar tags `<think>` en vez de respuestas reales.

### La pesadilla de la cuantizacion
Falcon H1R tiene capas Mamba (SSM) junto con capas transformer. Durante la conversion a GGUF, los parametros SSM_A (matrices de estado) tenian valores que iban a infinito despues de `exp(A_log)` en las capas 38 y 43. Esto rompia la cuantizacion completamente.

Escribimos un fix (`fix_ssm_weights.py`) que clampeaba los valores de `A_log`, lo que permitio que la cuantizacion funcione, pero la calidad del modelo ya era demasiado pobre para justificar continuar.

**Decision: Falcon H1R es el modelo equivocado para esta tarea.** Bueno en matematicas, malo en conversacion arabe.

---

## Fase 3: La Busqueda de un Modelo Nativo en Arabe

### Falcon H1 Arabic — modelo fantasma
TII (el equipo de Falcon) habia anunciado "Falcon H1 Arabic" pero no estaba en HuggingFace. No lo pudimos encontrar en ningun lado. Quizas era solo interno o no se habia lanzado todavia.

### Jais 7B Chat — la eleccion correcta
Encontramos Jais, construido por Inception/G42 (una empresa de EAU). Disenado especificamente para arabe incluyendo dialectos del Golfo. Datos clave:
- Arquitectura Llama (sin capas exoticas = cuantizacion limpia)
- 7B parametros
- Repo gated en HuggingFace (habia que aceptar terminos de uso)
- Realmente entrenado con datos de internet en arabe incluyendo texto dialectal

### Tambien evaluamos
- **Qwen2.5-3B-Instruct:** Buen modelo multilingue, mas chico (3B). Corrimos un pipeline completo con el pero Aya 8B era la mejor apuesta para calidad.

---

## Fase 4: Fine-Tuning de Jais — Mejora Iterativa

### Ronda 1: 260 ejemplos
- Loss: 4.3 -> 1.75
- Accuracy salto de 36% a 70%
- El arabe era coherente pero generico (tirando a MSA, no emirati)
- Los refusals no funcionaban para nada — el modelo alegremente ayudaba con preguntas de programacion

### Ronda 2: 1370 ejemplos
Expandimos el dataset significativamente. Agregamos conversaciones multi-turn.
- Loss: 4.36 -> 0.37
- Accuracy: 35% -> 93%
- Mejora importante en el uso del dialecto emirati
- Las conversaciones multi-turn funcionaban naturalmente

### Ronda 3: 2268 ejemplos (dataset v4)
Esta ronda se enfoco en arreglar problemas especificos:

**Problema 1: Confusion de persona.** El chatbot respondia como si fuera un familiar ("ya waladi" = hijo mio, "habibi"). Los usuarios decian "ya yimma" (mama) y el bot le seguia el juego.

**Fix:** Reescribimos todos los datos de training para reforzar la persona de asistente. Creamos `fix_persona.py` para limpiar todo el dataset. Agregamos ejemplos explicitos de limites de persona ("Soy tu asistente virtual, no tu mama, pero estoy aca para ayudarte!").

**Problema 2: Temas de adolescentes.** No habia cobertura de temas teen — pubertad, menstruacion, soporte emocional para adolescentes.

**Fix:** Generamos 78 ejemplos cubriendo temas adolescentes con sensibilidad cultural. Los ejemplos de menstruacion eran particularmente importantes — muchas familias emiraties necesitan apoyo para discutir este tema.

**Problema 3: Refusals debiles.** El modelo semi-rechazaba pero despues respondia igual ("esto no es mi especialidad, pero aca te cuento como programar un sitio web...").

**Fix:** Agregamos 30 ejemplos de rechazo firme pero amable. El patron: reconocer el pedido, decir claramente que esta fuera de alcance, redirigir a temas familiares, terminar con "si necesitas consejo familiar, aca estoy!"

---

## Fase 5: Cuantizacion y Testing Edge (Jais)

### Pipeline
1. Fusionar adapter LoRA con modelo base (`merge_and_unload`)
2. Convertir a GGUF f16 con `convert_hf_to_gguf.py` de llama.cpp
3. Cuantizar de f16 a Q5_K_M con `llama-quantize`

A diferencia de Falcon H1R, Jais (arquitectura Llama pura) cuantizo sin ningun problema.

### Resultados
- **Tamano GGUF Q5_K_M:** 4.7GB (el target era ~5.4GB — quedo por debajo!)
- **Tokens/seg:** 60+ en M3 Max (el target era >5 tok/s)
- **TTFT:** 60-150ms (el target era <2s)
- **RAM pico:** ~5.7GB (el target era <7GB)

Todos los targets superados por amplio margen en la maquina de desarrollo.

---

## Fase 6: Preparacion para Aya Expanse 8B

### Por que mejorar desde Jais
Jais 7B Chat funciona bien pero tiene limitaciones:
- A veces cae en arabe estandar (MSA)
- Las respuestas pueden ser verbosas (listas estilo ChatGPT)
- Modelo 7B de un lab mas chico vs 8B de Cohere

### Por que Aya Expanse 8B
- Modelo de CohereLabs, disenado especificamente para uso multilingue
- 23 idiomas con soporte fuerte de arabe
- 8B parametros (un poco mas grande pero mejor calidad por parametro)
- Desarrollo activo y comunidad
- Chat template estilo Cohere (bien estructurado)

### Scan de debilidades
Antes de fine-tunear Aya, corrimos un scan sistematico de como el modelo base maneja nuestros casos de uso. Encontramos 5 categorias de debilidad:

1. **Dialecto:** Responde en MSA en vez de emirati (dice "يمكنك" en vez de "تقدر")
2. **Persona:** Asume rol de familiar cuando le dicen "yimma" o "baba"
3. **Refusals:** No rechaza temas fuera de alcance (programacion, inversiones)
4. **Tono verboso:** Respuestas largas con listas markdown, se siente como ChatGPT
5. **Scope parcial:** Semi-rechaza pero da info de todas formas

### Dataset v5: fixes targetados
Generamos 487 ejemplos adicionales apuntando especificamente a estas debilidades:
- 130 ejemplos de respuestas en dialecto emirati puro
- 95 ejemplos de correccion de persona
- 149 ejemplos de rechazo firme
- 115 ejemplos de tono conciso y calido

**Dataset total: 2755 ejemplos**

### Ajustes de configuracion de training para Aya
Comparado con la config de Jais, ajustamos:
- **LR: 1e-4** (mas bajo que el 2e-4 de Jais — modelo mas grande necesita aprendizaje mas suave)
- **LoRA alpha: 32** (duplicado desde 16 — alpha mas alto para adaptacion mas fuerte)
- **Max seq length: 1024** (duplicado desde 512 — Aya maneja contexto mas largo)
- **Epochs: 3** (reducido desde 5 — mas datos necesitan menos pasadas)

---

## Fase 7: Pipeline y Deployment (actual)

### Lo que esta construido
- `run_aya_pipeline.sh` — Pipeline end-to-end: prep datos -> fine-tune -> fusionar -> cuantizar -> test
- `deploy/ui.py` — Interfaz de chat Gradio
- `deploy/server.py` — Wrapper FastAPI para llama.cpp
- Repo GitHub: publico en AleDeclerk/ExperimentalFamilyChatbot

### Lo que sigue
1. Correr el fine-tuning de Aya (~2-3 horas en M3 Max)
2. Evaluar contra test set
3. Comparar calidad Aya vs Jais
4. Demo Gradio con modelo final

---

## Resumen de la Evolucion del Dataset

```
v1 (125 ejemplos)
 └─> 10 categorias basicas, creacion manual
      │
v2 (260 ejemplos)
 └─> Expandido con variaciones, mas ejemplos por categoria
      │
v3 (1262 ejemplos)
 └─> Conversaciones multi-turn agregadas
 └─> Persona limpiada (asistente, no familiar)
      │
v4 (2268 ejemplos)
 └─> Temas adolescentes (pubertad, menstruacion, soporte emocional)
 └─> Refusals robustos (programacion, inversiones, poesia)
 └─> 17 categorias en total
      │
v5 (2755 ejemplos)
 └─> Fixes de debilidades para modelo base Aya
 └─> Refuerzo de dialecto emirati puro
 └─> Refusals mas firmes, tono conciso
 └─> 21 categorias en total
```

---

## Modelos Probados

| Modelo | Params | Resultado | Por que se detuvo |
|--------|--------|-----------|-------------------|
| Falcon H1R 7B | 7B | Fallo | Arabe pobre, problemas de cuantizacion SSM, genera tags `<think>` |
| Jais 7B Chat | 7B | Bueno (93% acc) | Funcional pero cae en MSA, verboso |
| Qwen2.5-3B-Instruct | 3B | Pipeline testeado | Modelo mas chico, usado para validar pipeline |
| Aya Expanse 8B | 8B | En progreso | Mejor soporte multilingue de arabe |

---

## Lecciones Clave

1. **El ecosistema ML es fragil.** Version de Python, compatibilidad CUDA/MPS, breaking changes de librerias — espera gastar 30% de tu tiempo en problemas de entorno.

2. **Seleccion de modelo > trucos de training.** Cambiar de Falcon a Jais valio mas que cualquier ajuste de hiperparametros.

3. **La calidad de los datos escala mejor que la cantidad.** Pasar de 260 a 1370 ejemplos con la misma calidad dio retornos decrecientes. Pasar de datos genericos a fixes targetados de debilidades dio mejoras desproporcionadas.

4. **El refuerzo de persona necesita datos de training explicitos.** Los system prompts solos no son suficientes — el modelo necesita ver ejemplos de mantener limites.

5. **Los refusals deben ser firmes.** Un medio-rechazo ("esto no es mi area pero aca van unos tips...") es peor que no rechazar nada. Entrena con rechazos limpios.

6. **La cuantizacion depende del modelo.** Los modelos con arquitectura Llama (Jais, Aya) cuantizan limpio. Arquitecturas exoticas (hibrido Mamba) pueden tener problemas numericos.

7. **Apple Silicon es viable para fine-tuning** pero requiere herramientas diferentes (no Unsloth, no BitsAndBytes 4-bit, float16 en vez de NF4).
