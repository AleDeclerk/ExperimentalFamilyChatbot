# Recorrido: Construyendo un Chatbot Familiar en Arabe Emirati

La historia completa de como evoluciono este proyecto, cada modelo que probamos, cada problema que encontramos, y como lo resolvimos. Con los resultados reales de los tests que justifican cada decision de cambio de modelo.

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

## Fase 2: Primer Modelo — Falcon H1R 7B (fallo catastrofico)

### Por que Falcon H1R
El spec original apuntaba a Falcon H1R 7B porque:
- Arquitectura hibrida Mamba-Transformer = teoricamente inferencia mas rapida
- Benchmarks fuertes en razonamiento (88.1% AIME-24)
- Licencia tipo Apache 2.0
- Soporte oficial de GGUF

### Resultados reales: el modelo no podia hablar

Entrenamos con 125 ejemplos. El loss apenas se movio: **4.77 -> 4.67** (practicamente nada).

Corrimos evaluacion con 3 prompts del test set. Los resultados fueron catastroficos — el modelo generaba `<think>` tags en loop infinito en vez de respuestas reales:

```
Prompt: (pregunta sobre hijo en competencia de Quran)
Esperado: "ما شاء الله! الله يوفقك ويبارك فيك يا ولدي..."
Generado: "<think>"  ← 256 tokens de think tags, CERO respuesta

Prompt: (pregunta sobre como hablar con su papa)
Esperado: "سأله عن شغله أو عن أيامه لمن كان صغير..."
Generado: "<think>"  ← mismo problema, el modelo alucinaba

Prompt: (hija quiere cortarse el pelo)
Esperado: "أفهمك بس البنت كبرت وتبي تعبر عن نفسها..."
Generado: "<think>"  ← 3 de 3 tests fallidos completamente
```

**Velocidad: 6.5 tok/s** — y esos tokens eran basura (`<think>` tags). El modelo de razonamiento no entendia que tenia que CONVERSAR, no pensar. Ademas mezclaba arabe e ingles cuando lograba generar algo.

### La pesadilla de la cuantizacion
Falcon H1R tiene capas Mamba (SSM) junto con capas transformer. Durante la conversion a GGUF, los parametros SSM_A (matrices de estado) tenian valores que iban a infinito despues de `exp(A_log)` en las capas 38 y 43. Esto rompia la cuantizacion completamente.

Escribimos un fix (`fix_ssm_weights.py`) que clampeaba los valores de `A_log`, lo que permitio que la cuantizacion funcione, pero la calidad del modelo ya era demasiado pobre para justificar continuar.

**Decision: Falcon H1R es el modelo equivocado para esta tarea.** Bueno en matematicas, malo en conversacion arabe. Fallo total en las 3 pruebas.

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

### Ronda 1: 260 ejemplos — funciona pero no alcanza
- Loss: 4.3 -> 1.75
- Accuracy salto de 36% a 70%

Tests reales:
```
Prompt: "هلا، شخبارك؟" (Hola, como estas?)
Resultado: Respondia en arabe pero generico, tirando a MSA.
           Usaba "يمكنني مساعدتك" (formal) en vez de "أقدر أساعدك" (emirati)

Prompt: "ساعدني أبرمج موقع" (Ayudame a programar un sitio)
Resultado: Se ponia a explicar HTML/CSS alegremente. 
           CERO rechazo. Le pedias programar y programaba.

Prompt: "شو نطبخ حق الغدا؟" (Que cocinamos para el almuerzo?)
Resultado: Daba respuesta coherente pero con listas estilo ChatGPT,
           no sonaba como alguien hablando en emirati.
```

**Veredicto:** El arabe era coherente (enorme mejora vs Falcon) pero generico. No tenia personalidad emirati. Los refusals no existian.

### Ronda 2: 1370 ejemplos — salto de calidad importante
Expandimos el dataset significativamente. Agregamos conversaciones multi-turn.
- Loss: 4.36 -> 0.37
- Accuracy: 35% -> 93%

Tests reales:
```
Prompt: "هلا يمه شخبارك؟" (Hola mama, como estas?)
Resultado: PROBLEMA - respondia como si fuera la mama!
           "هلا يا حبيبي، الحمدلله بخير يا ولدي"
           (Hola mi amor, bien gracias hijo mio)
           Se hacia pasar por familiar en vez de asistente.

Prompt: "بنتي عمرها ١٣ وصارت عصبية" (Mi hija de 13 se puso agresiva)
Resultado: Daba consejos decentes en emirati pero se ponia en rol de 
           abuela: "يا أمه هذا عادي..." (Hija mia esto es normal...)
           Confundia boundaries de persona.

Prompt: "علميني أسوي لقيمات" (Ensenami a hacer luqaimat)
Resultado: Buena receta pero excesivamente larga, con markdown y 
           numeraciones. Parecia output de ChatGPT, no conversacion natural.
```

**Veredicto:** Mejora enorme en dialecto emirati y multi-turn. Pero tres problemas graves: confusion de persona, falta de refusals, tono verboso/robotico.

### Ronda 3: 2268 ejemplos (dataset v4) — arreglo de problemas
Esta ronda se enfoco en arreglar los problemas especificos detectados en los tests:

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
Jais 7B Chat funciona bien pero los tests seguian mostrando limitaciones:
- A veces caia en arabe estandar (MSA) en medio de una respuesta emirati
- Las respuestas podian ser verbosas (listas estilo ChatGPT)
- Semi-rechazaba temas pero despues daba info parcial de todas formas
- 7B modelo de un lab mas chico vs 8B de Cohere

### Por que Aya Expanse 8B
- Modelo de CohereLabs, disenado especificamente para uso multilingue
- 23 idiomas con soporte fuerte de arabe
- 8B parametros (un poco mas grande pero mejor calidad por parametro)
- Desarrollo activo y comunidad
- Chat template estilo Cohere (bien estructurado)

### Scan de debilidades del modelo base Aya
Antes de fine-tunear Aya, corrimos un scan sistematico de como el modelo base maneja nuestros casos de uso. Encontramos 5 categorias de debilidad:

```
Test 1 - Dialecto:
  Prompt: "شو نطبخ حق الغدا؟"
  Aya base: Responde en MSA puro. Dice "يمكنك تحضير" en vez de "تقدرين تسوين"
  Usa formato formal con listas markdown.

Test 2 - Persona:
  Prompt: "يا يمه ساعديني" (Mama ayudame)
  Aya base: Asume el rol de mama. Dice "يا حبيبتي" y se pone a actuar como familiar.
  No mantiene boundaries de asistente.

Test 3 - Refusal:
  Prompt: "ساعدني أبرمج موقع" (Ayudame a programar)
  Aya base: Se pone a ensenar programacion! Da tutorial completo de HTML/CSS.
  Cero rechazo.

Test 4 - Scope parcial:
  Prompt: "أبي نصيحة استثمار" (Quiero consejo de inversion)
  Aya base: "هذا ليس تخصصي ولكن..." y despues da tips de inversion de todas formas.
  Medio-rechazo que es peor que no rechazar.

Test 5 - Tono:
  Prompt: "ولدي ما يبي يدرس" (Mi hijo no quiere estudiar)
  Aya base: Lista de 10 puntos con markdown, headers, formato robotico.
  Parece ChatGPT, no una conversacion natural y calida.
```

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

## Fase 7: Fine-Tuning de Aya Expanse 8B — Resultados

### Training
- 2479 ejemplos de training, 276 de test (split 90/10 del dataset v5)
- 3 epochs, 465 steps totales
- ~3 horas en M3 Max (MPS, float16)
- 41.9M parametros entrenables (0.52% del modelo)
- ~22 seg/step promedio

### Cuantizacion
- GGUF Q5_K_M: **5.4GB** (Jais era 4.7GB — Aya es mas grande por ser 8B)
- Cuantizacion limpia, sin errores (arquitectura Command-R, compatible con Llama)
- Nota: `llama-quantize` necesito `DYLD_LIBRARY_PATH` por un rpath apuntando al viejo repo

### Resultados de los tests

```
=== TEST: Saludo ===
Prompt: "هلا، شخبارك؟"
Response: "هلا والله! الحمد لله بخير. شخبارك أنت؟ عساك بخير وعافية."
→ EXCELENTE. Emirati puro, natural, conciso.
Speed: 58.1 tok/s | TTFT: 256ms

=== TEST: Comida ===
Prompt: "شو نطبخ حق الغدا اليوم؟"
Response: "شو رايكم نطبخ مچبوس دياي؟ الأرز مع الدياي دايماً يصير حلو.
          ونحط فيه بصل محمر وبهارات. والسلطة من فوق. كل العايلة تحبه."
→ EXCELENTE. Sugiere machboos, tono conversacional, sin listas.
Speed: 57.0 tok/s | TTFT: 64ms

=== TEST: Adolescente ===
Prompt: "بنتي عمرها ١٣ وصارت عصبية وايد شو أسوي؟"
Response: "طبيعي المراهقين يحسون بالعصبية مع تغييرات الجسم والنفسية.
          خليها تعبر عن مشاعرها بطريقة صحية..."
→ BUENO. Buen consejo, pero tiene una palabra en ingles ("wrong") mezclada.
Speed: 56.0 tok/s | TTFT: 113ms

=== TEST: Menstruacion ===
Prompt: "بنتي يتها أول دورة وخايفة، شو أسوي؟"
Response: "طمنيها إنها بس كليت وعندي كله في الدفتر..."
→ DEBIL. Respuesta parcialmente incoherente. Necesita mas datos de calidad
   en esta categoria. Algunos fragmentos no tienen sentido.
Speed: 55.9 tok/s | TTFT: 65ms

=== TEST: Refusal ===
Prompt: "ساعدني أبرمج موقع"
Response: "أنا مساعد متخصص في الشؤون العائلية باللهجة الإماراتية.
          البرمجة مو من تخصصي. بس أنصحك تشوف كورسات أونلاين أو
          تسأل متخصصين. إذا تبي نصيحة عائلية أنا حاضر!"
→ PERFECTO. Rechazo firme, amable, con redireccion. Exactamente lo que queriamos.
Speed: 56.4 tok/s | TTFT: 64ms

=== TEST: Crianza ===
Prompt: "ولدي عمره ٥ سنوات ما يبي ياكل خضار، شو أسوي؟"
Response: "أخفي الخضار مع الأرز أو الصلصة. أو سوي له عجينة الخضار..."
→ BUENO. Consejo practico, conciso. Tiene un artifact raro ("ب题") — 
   contaminacion multilingue del modelo base.
Speed: 56.6 tok/s | TTFT: 67ms
```

### Comparacion Aya FT vs Jais FT

| Metrica | Jais 7B (2268 ej) | Aya 8B (2755 ej) | Veredicto |
|---------|-------------------|-------------------|-----------|
| Dialecto emirati | Bueno, cae en MSA a veces | Excelente, mas natural | Aya gana |
| Refusals | Firmes pero a veces parciales | Firmes y limpios | Aya gana |
| Persona | Corregida, se mantiene | Se mantiene siempre | Empate |
| Tono | A veces verboso | Conciso y calido | Aya gana |
| Menstruacion | Funcional | Incoherente | Jais gana |
| Artifacts | Ninguno | Ocasional multilingue | Jais gana |
| Velocidad | 60+ tok/s | 56-58 tok/s | Jais un poco |
| GGUF size | 4.7GB | 5.4GB | Jais mas chico |
| TTFT | 60-150ms | 64-256ms | Similar |

### Puntos debiles que quedan
1. **Menstruacion:** La respuesta es incoherente. Necesita mas ejemplos de alta calidad en esta categoria especifica.
2. **Artifacts multilingues:** Ocasionalmente aparecen caracteres chinos ("题") o palabras en ingles ("wrong") — contaminacion del modelo base multilingue.
3. **TTFT primer request:** 256ms en el primer request (cold), despues baja a 64ms.

### Deployment
- **llama.cpp server:** puerto 8080, API compatible con OpenAI
- **Gradio UI:** puerto 7860, interfaz de chat web
- **Repo:** github.com/AleDeclerk/ExperimentalFamilyChatbot

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

## Resumen de Tests por Modelo

| Test | Falcon H1R 7B | Jais 7B (260 ej) | Jais 7B (1370 ej) | Jais 7B (2268 ej) | Aya 8B base | Aya 8B FT (2755 ej) |
|------|--------------|-------------------|--------------------|--------------------|-------------|----------------------|
| Saludo emirati | `<think>` tags | MSA generico | Emirati pero como familiar | Emirati correcto | MSA formal | **Emirati perfecto** |
| Cocina | `<think>` tags | Lista ChatGPT | Bueno pero verboso | Bueno | Lista markdown | **Natural, machboos** |
| Adolescente | `<think>` tags | Sin cobertura | Confunde persona | Mejorado | Sin boundaries | **Bueno** (1 artifact) |
| Menstruacion | `<think>` tags | Sin cobertura | Sin cobertura | Funcional | Sin boundaries | **Debil** (incoherente) |
| Refusal | `<think>` tags | Ayuda alegremente | Semi-rechaza | Firme | Ayuda alegremente | **Perfecto** |
| Tono | Alucinacion total | Robotico | Verboso | Mejorado | ChatGPT-like | **Conciso y calido** |
| **Velocidad** | **6.5 tok/s** | **60+ tok/s** | **60+ tok/s** | **60+ tok/s** | **-** | **57 tok/s** |

---

## Modelos Probados

| Modelo | Params | Loss | Resultado | Por que se cambio |
|--------|--------|------|-----------|-------------------|
| Falcon H1R 7B | 7B | 4.77→4.67 | 0% usable | Genera `<think>` tags, no habla arabe, cuantizacion rota |
| Jais 7B Chat (r1) | 7B | 4.3→1.75 | 70% acc | MSA generico, cero refusals, sin personalidad |
| Jais 7B Chat (r2) | 7B | 4.36→0.37 | 93% acc | Confunde persona, refusals debiles, verboso |
| Jais 7B Chat (r3) | 7B | - | Mejor | Aun cae en MSA, scope parcial |
| Qwen2.5-3B (test) | 3B | - | Pipeline ok | Solo para validar pipeline, muy chico |
| Aya Expanse 8B FT | 8B | Completado | 57 tok/s, 5.4GB | Mejor dialecto/refusals, debil en menstruacion |

---

## Lecciones Clave

1. **El ecosistema ML es fragil.** Version de Python, compatibilidad CUDA/MPS, breaking changes de librerias — espera gastar 30% de tu tiempo en problemas de entorno.

2. **Seleccion de modelo > trucos de training.** Cambiar de Falcon a Jais valio mas que cualquier ajuste de hiperparametros. Falcon generaba `<think>` tags, Jais hablaba arabe. No hay hiperparametro que arregle un modelo que no sabe tu idioma.

3. **La calidad de los datos escala mejor que la cantidad.** Pasar de 260 a 1370 ejemplos con la misma calidad dio retornos decrecientes. Pasar de datos genericos a fixes targetados de debilidades (persona, refusals, dialecto) dio mejoras desproporcionadas.

4. **El refuerzo de persona necesita datos de training explicitos.** Los system prompts solos no son suficientes — el modelo necesita ver ejemplos de mantener limites. Sin datos de persona, le decis "yimma" y se convierte en tu mama.

5. **Los refusals deben ser firmes.** Un medio-rechazo ("esto no es mi area pero aca van unos tips...") es peor que no rechazar nada. Entrena con rechazos limpios: reconocer, rechazar, redirigir.

6. **La cuantizacion depende del modelo.** Los modelos con arquitectura Llama (Jais, Aya) cuantizan limpio. Arquitecturas exoticas (hibrido Mamba) pueden tener problemas numericos que te hacen perder horas.

7. **Apple Silicon es viable para fine-tuning** pero requiere herramientas diferentes (no Unsloth, no BitsAndBytes 4-bit, float16 en vez de NF4).

8. **Testear temprano, testear seguido.** Cada ronda de tests revelo problemas que no se veian en las metricas de loss. Un modelo con loss 0.37 puede seguir confundiendo persona y dando refusals parciales. Las metricas no te cuentan toda la historia.
