# Auditoría Técnica Exhaustiva de `cuda_engine.py` (Línea Base)

**Fecha:** 2026-10-06  
**Rama:** `feat/arm-pipeline-v2`  
**Archivo analizado:** `frontend/experiments/digital_arm/cuda_engine.py` (1,100 líneas)  
**Entorno de ejecución:** Windows 10/11, Python 3.13, PyTorch 2.13.0+cu130, NVIDIA RTX A1000 6GB Laptop GPU  

---

## 1. Resumen Ejecutivo

El motor actual logra inferencia acelerada mediante PyTorch CUDA y soporte multi-modelo (YOLO11, FastSAM, RVM ResNet50/MobileNetV3 y Resta de Fondo). No obstante, la arquitectura presenta cuellos de botella severos de sincronización GPU↔CPU, condiciones de carrera en el estado compartido, duplicación de contratos de API, falta de validación de entradas y una fuerte dependencia de heurísticas geométricas dispersas.

Esta auditoría clasifica los hallazgos en cinco categorías normativas: **BUG**, **RENDIMIENTO**, **ROBUSTEZ**, **SEGURIDAD** y **MANTENIBILIDAD**.

---

## 2. Hallazgos por Severidad

### 🔴 BUG (Crítico / Alto)

| ID | Línea(s) | Descripción del Hallazgo | Impacto en Producción |
|---|---|---|---|
| **BUG-01** | `789` | **Timestamps de MediaPipe no monótonos:** Se calcula `frame_timestamp_ms = int(time.perf_counter() * 1000)`. En bucles rápidos (>1000 FPS en etapas cortas o frames omitidos), el entero en milisegundos se repite. MediaPipe `detect_for_video` exige estricta monotonía (`ts > last_ts`), lanzando excepciones internas de C++. | Pérdida intermitente del tracking de mano y cursor bloqueado. |
| **BUG-02** | `963, 977, 1024, 1040, 1043` | **Condiciones de carrera en variables de estado sin candado:** `state.arm_only`, `state.guided_filter_on` y `state.show_hud` se leen en el bucle principal de inferencia (hilo 1) sin adquirir `state.lock`, mientras los hilos HTTP (hilos 2..N) en `/api/settings` los mutan concurrentemente. Además, los atajos de teclado (`[G]`, `[H]`) en líneas 1040 y 1043 mutan estas banderas directamente sin lock. | Lecturas atómicas corruptas o desincronización entre la UI web y el render. |
| **BUG-03** | `216, 232, 264, 476-478` | **Duplicación de cabeceras HTTP CORS:** Los endpoints individuales envían `Access-Control-Allow-Origin: *` explícitamente y luego el método sobrescrito `end_headers()` (línea 476) vuelve a emitir `Access-Control-Allow-Origin: *`. | Cabeceras duplicadas rechazadas por navegadores modernos bajo modo estricto. |
| **BUG-04** | `829-830, 873-874` | **Desalineación espacial de máscaras en YOLO / FastSAM:** Se usa `F.interpolate` bilineal directo para escalar la máscara predicha `raw_mask` hacia `(h, w)` sin compensar el padding de letterbox original de Ultralytics. No se usa `retina_masks=True`. | Desfase de varios píxeles entre la mano real capturada y la silueta proyectada. |
| **BUG-05** | `828, 872` | **Fusión destructiva de instancias con `sum(dim=0)`:** Se suman todas las máscaras predichas por la red. Si hay un robot, fondo o artefacto detectado como persona junto a la mano, se fusionan indiscriminadamente en un solo tensor. | Falsos positivos proyectados en la pared. |

---

### 🟠 RENDIMIENTO (Latencia / Cuello de Botella)

| ID | Línea(s) | Descripción del Hallazgo | Impacto en Producción |
|---|---|---|---|
| **PERF-01** | `134, 187, 1001` | **Idas y vueltas síncronas GPU ↔ CPU en el camino crítico:** El pipeline genera tensores CUDA, luego hace `.cpu().numpy()` para `cv2.connectedComponentsWithStats`, luego recrea tensores con `torch.from_numpy(...).to(device)`, ejecuta shader CUDA, y vuelve a hacer `.cpu().numpy()` para `cv2.imencode`. | Bloqueo del pipeline con múltiples `cudaMemcpy` síncronos, elevando la latencia en 15–25 ms. |
| **PERF-02** | `832, 875` | **Llamadas bloqueantes a `.item()` en GPU:** En `raw_mask.sum().item() > 50` se fuerza una sincronización total del hilo CUDA con la CPU en cada cuadro para evaluar una condición simple. | Rotura del pipeline asíncrono de PyTorch CUDA; GPU entra en espera ociosa. |
| **PERF-03** | `181-182` | **Construcción de grids de resolución completa con `np.ogrid[:h, :w]`:** Se crea una cuadrícula de coordenadas 1280×720 en memoria principal en cada frame que hay manos detectadas. | Consumo de memoria transitoria y coste innecesario de 3–6 ms en CPU. |
| **PERF-04** | `808-815, 862-869` | **Inferencia de FastSAM a resolución completa (640x640) sin ROI:** FastSAM procesa el cuadro entero en lugar de un recorte ajustado a la extremidad activa, operando en FP32 sin `half=True`. | Caída a <25 FPS en la GPU RTX A1000 Laptop. |
| **PERF-05** | `745, 1018` | **Doble codificación JPEG incondicional:** Se ejecuta `cv2.imencode(".jpg")` dos veces por frame (uno para `latest_raw_jpeg` y otro para `latest_jpeg`), incluso cuando no hay ningún cliente HTTP ni visor web conectado. | Pérdida fija de 8–12 ms por frame en codificación de software CPU. |
| **PERF-06** | `313, 338, 363` | **Bucles de streaming con `time.sleep` estático que saturan hilos:** Los endpoints `/stream`, `/raw_stream` y `/cursor_stream` mantienen bucles infinitos con `time.sleep(0.016)` a `0.033`, reteniendo hilos de `ThreadingHTTPServer` indefinidamente. | Bloqueo potencial del pool de hilos y jitter temporal en el render del navegador. |

---

### 🟡 ROBUSTEZ (Estabilidad / Manejo de Fallas)

| ID | Línea(s) | Descripción del Hallazgo | Impacto en Producción |
|---|---|---|---|
| **ROB-01** | `91, 766, 801, 382` | **Bloques `except Exception: pass` silenciosos:** Múltiples bloques capturan genéricamente `Exception` y la descartan con `pass`. En la detección de MediaPipe (línea 801), si el framework falla o los tensores se desalinean, el motor continúa en silencio sin traza. | Dificultad extrema para depurar fallos en campo; fallas silenciosas de componentes. |
| **ROB-02** | `604, 610` | **`torch.hub.load` sin bloqueo de versión local ni commit SHA fijo:** En cada arranque en frío se invoca `torch.hub.load("PeterL1n/RobustVideoMatting", ...)` confiando en la caché de `torch.hub`. Si la caché está limpia y no hay conexión a internet, el motor colapsa. | Imposibilidad de arranque offline garantizado en entornos de exhibición. |
| **ROB-03** | `641-650` | **Dependencia exclusiva de cámara física en vivo:** No existe argumento `--source` para reproducir un archivo `.mp4`, directorio de imágenes o secuencia sintética. No se puede hacer benchmarking reproducible ni testing CI. | Imposibilidad de pruebas unitarias automáticas y comparación científica A/B. |
| **ROB-04** | `539-544` | **Sondeo de cámara con `time.sleep(0.03)` bloqueante durante el inicio:** Se leen hasta 40 fotogramas por backend para cada índice de cámara (hasta 1.2 segundos por prueba de cámara), ralentizando el arranque hasta en 8 segundos. | Tiempo de inicio excesivo. |

---

### 🔵 SEGURIDAD (CORS / Validación)

| ID | Línea(s) | Descripción del Hallazgo | Impacto en Producción |
|---|---|---|---|
| **SEC-01** | `390, 437` | **Ausencia de límite en `Content-Length` en peticiones POST:** Se lee directamente `content_length = int(self.headers.get("Content-Length", 0))` y se ejecuta `self.rfile.read(content_length)` sin cota superior. Un payload malformado de gran tamaño puede agotar la memoria del proceso (DoS). | Vulnerabilidad DoS por memoria no acotada. |
| **SEC-02** | `476` | **Cabeceras CORS permisivas globales (`*`):** Aunque el servicio escucha en `127.0.0.1`, cualquier sitio web externo abierto en el navegador del usuario puede enviar peticiones cross-origin y conmutar modelos o leer el stream de la cámara. | Exposición del flujo de video a orígenes no autorizados en el navegador. |

---

### ⚪ MANTENIBILIDAD (Arquitectura / Deuda Técnica)

| ID | Línea(s) | Descripción del Hallazgo | Impacto en Producción |
|---|---|---|---|
| **MAN-01** | `202, 267-271, 401-405, 618-625` | **Nombres de modelos y FPS hardcodeados duplicados en 4 sitios:** La lista de nombres (ej. `"YOLO11-seg (CUDA - 59 FPS)"`) está duplicada en el estado inicial, en `/api/status`, en `/api/model` y en `main()`. Si cambia un texto o se agrega un modelo, se desincronizan las respuestas. | Fragilidad ante adición o renombrado de motores de IA. |
| **MAN-02** | `211` | **Variable huérfana no utilizada (`state.dsr`):** En `EngineState.__init__` se define `self.dsr = 0.25`, pero en la inferencia de RVM (línea 953) se utiliza `cur_dsr = 0.30 if rvm_key == "ResNet50" else 0.50`, dejando `state.dsr` como código muerto. | Confusión sobre qué resolución de downsampling se está aplicando realmente. |
| **MAN-03** | `Globales` | **Ausencia de configuración centralizada (`EngineConfig`):** Umbrales de confianza (`0.08`, `0.18`, `0.20`), tamaños de entrada (`640`), persistencias temporales (`10`, `12`), límites de área y parámetros de guided filter están dispersos como números mágicos en el código. | Dificultad para calibrar parámetros desde archivos de configuración o CLI. |
| **MAN-04** | `Globales` | **Uso de `print` en lugar de `logging` estructurado:** Mensajes de depuración, errores y avisos se emiten vía `print()` sin niveles de severidad ni timestamps unificados. | Imposibilidad de filtrar logs en pruebas automatizadas o entornos productivos. |

---

## 3. Matriz de Remediación por Fases

| ID | Hallazgo | Fase de Resolución | Módulo / Clase Responsable |
|---|---|---|---|
| **BUG-01** | Timestamp MediaPipe monótono | **Fase 0** (Línea base) | `safe_timestamp_ms()` |
| **ROB-03** | Modo offline `--source` | **Fase 0** (Línea base) | `VideoSourceReader` |
| **MAN-03** | Números mágicos dispersos | **Fase 0** / **Fase 1** | `EngineConfig` (dataclass) |
| **MAN-01** | Modos duplicados | **Fase 0** / **Fase 1** | `ModelRegistry` / Enum de modos |
| **MAN-02** | `state.dsr` huérfano | **Fase 0** / **Fase 4** | Unificación en `EngineConfig` |
| **BUG-02** | Condiciones de carrera en `state` | **Fase 1** / **Fase 6** | `ThreadSafeState` |
| **BUG-03** | Cabeceras CORS duplicadas | **Fase 0** / **Fase 6** | `ProjectionHTTPHandler.end_headers` |
| **BUG-04** | Desalineación `F.interpolate` | **Fase 2** | `retina_masks=True` en YOLO |
| **BUG-05** | Fusión `sum(dim=0)` en YOLO/SAM | **Fase 1** / **Fase 2** / **Fase 3** | `ArmMaskSelector` |
| **PERF-01** | Roundtrips GPU↔CPU | **Fase 1** / **Fase 4** | Post-proceso en GPU o baja resolución |
| **PERF-02** | `.item()` bloqueante | **Fase 1** / **Fase 3** | Tensor boolean reductions en GPU |
| **PERF-03** | `np.ogrid` completo en frame | **Fase 1** | Atenuación en ROI / tensor GPU |
| **PERF-04** | FastSAM frame completo FP32 | **Fase 3** | FastSAM ROI + FP16 |
| **PERF-05** | Doble `imencode` incondicional | **Fase 6** | Codificación bajo demanda de clientes |
| **PERF-06** | Hilos bloqueados en streaming | **Fase 6** | Event-driven frame distribution |
| **ROB-01** | `except: pass` | **Fase 0** / **Fase 1** | Logging tipado estructurado |
| **ROB-02** | Descarga en vivo RVM | **Fase 0** / **Fase 4** | Carga local de checkpoint verificado |
| **SEC-01** | POST body no acotado | **Fase 0** / **Fase 6** | Límite estricto de payload (64 KB) |

---

## 4. Conclusión de la Auditoría

El motor tiene bases sólidas de aceleración de hardware en la RTX A1000, pero sufre de contención de hilos, desacoplamiento entre detección de mano y segmentación, y roundtrips innecesarios entre VRAM y RAM. La ejecución secuencial del plan de refactorización (`feat/arm-pipeline-v2`) resolverá estos puntos de forma medible.
