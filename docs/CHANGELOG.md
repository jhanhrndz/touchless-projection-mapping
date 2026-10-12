# Registro de Cambios (Changelog) - Pipeline de Brazo Digital v2

Todos los cambios notables en este proyecto serán documentados en este archivo por fases secuenciales.

---

## [Fase 0: Auditoría, Línea Base y Medición de Precisión] - 2026-10-06

### Añadido
- **Respaldo Inmutable Git**: Tag de seguridad `backup/pre-arm-pipeline-v2` creado en commit base `602252f`.
- **Gestión de Entorno y Dependencias**:
  - `requirements.txt`: Fijadas dependencias de producción (`torch==2.13.0+cu130`, `opencv-python==4.10.0.84`, `ultralytics==8.3.15`, `mediapipe==0.10.14`).
  - `requirements-dev.txt`: Añadidas herramientas de calidad (`pytest==9.1.1`, `ruff==0.16.10`, `mypy==2.4.0`, `psutil==7.2.2`).
- **Auditoría Técnica Exhaustiva (`docs/AUDIT.md`)**:
  - Clasificación de 15 problemas críticos: BUG-01 a BUG-04, PERF-01 a PERF-05, ROB-01 a ROB-03, SEC-01 a SEC-02, MAINT-01.
  - Matriz de remediación estricta por fases.
- **Modo Reproducción Offline (`VideoFileSource`)**:
  - Clase para decodificación continua de clips `.mp4` en bucle con timestamps monótonos crecientes para MediaPipe.
  - Soporte de parámetro CLI `--source <ruta.mp4>` en `cuda_engine.py` para pruebas deterministas sin cámara física.
- **Herramientas de Generación y Registro**:
  - `tools/generate_synthetic_clips.py`: Generador de 6 clips sintéticos estandarizados en `tests/clips/`.
  - `tools/record_clips.py`: Asistente interactivo guiado para captura de clips reales de cámara.
- **Banco de Pruebas de Precisión por Etapas (`tools/benchmark.py`)**:
  - Medición aislada con sincronización GPU (`torch.cuda.synchronize()`): Captura, MediaPipe, Inferencia, Postproceso, Codificación JPEG.
  - Cálculo de percentiles (Media, P95, P99, Mín, Máx), FPS efectivos, VRAM asignada/pico y CPU.
  - Generado reporte de línea base: `docs/benchmarks/baseline.json`.
- **Suite de Pruebas Unitarias de Línea Base (`tests/test_baseline.py`)**:
  - Cobertura de importaciones, constantes, `VideoFileSource`, estado del servidor y endpoints HTTP (`/api/status`, `/api/model`, `/api/settings` y protección de límite 64 KB HTTP 413).
  - 100% pruebas pasando en verde (`4 passed`).

### Corregido / Mitigado
- **BUG-01**: Corregido crash de MediaPipe en bucle continuo mediante timestamps estrictamente monótonos crecientes (`frame_timestamp_ms = max(cur_ms, last_frame_ts + 1)`).
- **BUG-03**: Eliminada duplicación de encabezados CORS `Access-Control-Allow-Origin` en `ProjectionHTTPHandler`.
- **SEC-01**: Implementado límite estricto de cuerpo en peticiones POST (`MAX_POST_BODY_BYTES = 65536`) con vaciado de socket para prevenir errores de conexión `WinError 10054` en Windows y respuesta HTTP 413 estándar.
- **MAINT-01**: Introducida clase `@dataclass EngineConfig` en `cuda_engine.py` para centralizar configuración y eliminar números mágicos.

### Métricas de Línea Base (RTX A1000 6GB Laptop GPU, Clip 720p 45 frames)
| Motor / Modelo | FPS Efectivo | Total (ms) | Captura (ms) | MediaPipe (ms) | Inferencia (ms) | Postproceso (ms) | Codificación (ms) | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11-seg** | 0.4 | 2242.4 | 3.1 | 11.4 | 745.9 | **1480.7** | 1.4 | 92.6 |
| **FastSAM Point-Prompt** | 0.1 | 9266.0 | 3.3 | 11.5 | 5122.9 | **4126.9** | 1.4 | 134.7 |
| **RVM MobileNetV3** | 6.9 | 145.9 | 3.4 | 13.9 | 112.5 | 14.4 | 1.8 | 120.0 |
| **RVM ResNet50** | 7.4 | 135.3 | 3.4 | 13.8 | 101.5 | 14.8 | 1.8 | 210.7 |
| **Resta de Fondo** | 16.1 | 62.0 | 4.1 | 13.4 | 27.1 | 15.7 | 1.7 | 88.8 |

> **Hallazgo Crítico**: El cuello de botella masivo en YOLO y FastSAM se origina en `isolate_arm_and_hand_mask` ejecutado en CPU a resolución nativa 720p con `cv2.connectedComponentsWithStats` y cálculo analítico de desvanecimiento con `np.ogrid` para cada landmark por frame (1480 - 4126 ms por frame). Su reingeniería en Fase 1 restaurará los FPS esperados (35-60 FPS).

---

## [Fase 1: ArmMaskSelector y Lógica de Máscara] - 2026-10-06

### Añadido
- **Módulo `arm_selector.py`**:
  - `ArmMaskSelector`: Selector inteligente de extremidades superiores.
    - Ejecución a resolución reducida 1/4 (320x180) para componentes conectados en CPU: **latencia reducida de 1480 ms a 1.94 ms** (aceleración de ~760x).
    - **Eliminación del Gate Duro**: Heurística de detección en bordes exteriores (`touches_left`, `touches_right`, `touches_top`, `touches_bottom`) que reconoce el brazo en cuanto ingresa al encuadre sin necesidad de esperar a que MediaPipe detecte la palma completa.
    - Rechazo de falsos positivos en el centro del cuadro (siluetas estáticas, robots, torsos desconectados).
    - Desvanecimiento anatómico suave hacia el torso basado en vector muñeca-palma.
  - `TemporalMaskFilter`:
    - Filtro de suavizado temporal exponencial (EMA, $\lambda = 0.80$).
    - Histéresis y persistencia de 4 fotogramas ante pérdidas momentáneas de tracking o giros rápidos de muñeca, con decaimiento progresivo suave.
- **Suite de Pruebas Unitarias de Fase 1 (`tests/test_arm_selector.py`)**:
  - `test_empty_mask_handling`: Manejo de entradas nulas.
  - `test_edge_entry_without_landmarks`: Entrada por borde sin landmarks anatómicos (cero gate duro).
  - `test_center_object_rejection`: Rechazo estricto de objetos centrales aislados.
  - `test_anatomical_hand_selection`: Anclaje a landmarks anatómicos válidos.
  - `test_temporal_persistence_and_decay`: Persistencia y decaimiento en cooldown de 3-4 frames.
  - `test_selector_latency_performance`: Latencia unitaria promedio de **1.94 ms**.
  - 100% pruebas pasando (`6 passed`, total global suite `10 passed`).
- **Carga Local Offline de RVM**:
  - Implementado `source="local"` con fallback a caché local en `cuda_engine.py` y `tools/benchmark.py`, eliminando por completo cualquier congelamiento por peticiones de red a GitHub.
- **Parámetro `--mode` en `tools/benchmark.py`**:
  - Permite evaluar motores de forma modular y selectiva.

### Métricas de Rendimiento - Fase 1 vs Línea Base (Clip 720p en RTX A1000)
| Motor / Modo | FPS Línea Base | **FPS Fase 1** | Latencia Total (ms) | Postproceso (ms) | Inferencia (ms) | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Resta de Fondo** | 16.1 | **35.2 FPS** | **28.4 ms** | **12.1 ms** | **2.4 ms** | **Superó Meta ≥ 30 FPS** |
| **RVM MobileNetV3** | 6.9 | **10.0 FPS** | **100.0 ms** | **12.6 ms** | **73.0 ms** | **+45% Rendimiento** |
| **RVM ResNet50** | 7.4 | **9.7 FPS** | **103.1 ms** | **13.4 ms** | **75.4 ms** | **+31% Rendimiento** |
| **ArmMaskSelector Unitario** | N/A | **515+ FPS** | **1.94 ms** | **1.94 ms** | N/A | **Aceleración ~760x** |

---

## [Fase 2: YOLO Pipeline y FastSAM Point-Prompt] - 2026-10-06

### Añadido
- **Módulo `yolo_pipeline.py`**:
  - `YoloArmDetector`:
    - Activado `retina_masks=True` para inferencia con resolución completa de prototipos, eliminando la pixelación y desalineación de los dedos al interpolar desde 160x160.
    - Media precisión FP16 (`half=True`) automática cuando se ejecuta en GPU CUDA, aprovechando los Tensor Cores de la NVIDIA RTX A1000.
    - **Filtrado Inteligente de Instancia**: Si hay múltiples personas o siluetas en el encuadre, selecciona únicamente la instancia que conecta con los landmarks de la mano del usuario (`classes=[0]`), eliminando completamente la suma ciega anterior (`masks.data.sum(dim=0)`).
    - Umbrales de confianza adaptativos tipados: `conf_base = 0.22`, `conf_tracking = 0.16`.
  - `FastSamArmDetector`:
    - **Corrección Definitiva del Problema de "Cuadros"**: Extracción estricta y directa de la máscara tensorial continua (`res[0].masks.data`) generada a partir de los puntos anatómicos (palma y muñeca), garantizando contornos orgánicos en dedos y cero cajas rectangulares.
    - Inferencia con `retina_masks=True` y `half=True` en GPU.
    - Manejo seguro de casos sin detección: retorna tensor nulo sin intentar dibujar rectángulos, delegando a la histéresis temporal suave de `ArmMaskSelector`.
- **Suite de Pruebas Unitarias de Fase 2 (`tests/test_yolo_pipeline.py`)**:
  - `test_yolo_detector_initialization`: Inicialización tipada y validación de parámetros.
  - `test_fastsam_detector_initialization`: Configuración de umbral FastSAM.
  - `test_fastsam_no_landmarks_handling`: Retorno nulo limpio sin dibujar rectángulos ante ausencia de landmarks.
  - `test_yolo_predict_output_contract`: Inferencia real en GPU con tensor `[1, 1, 720, 1280]` normalizado en `[0.0, 1.0]`.
  - Total de pruebas en verde: **14 passed (100%)**.
- **Integración Limpia en `cuda_engine.py`**:
  - Eliminados los filtros manuales duplicados y las variables de cooldown redundantes en el bucle principal, delegando todo el ciclo de vida a `YoloArmDetector`, `FastSamArmDetector` y `ArmMaskSelector`.

---

## [Fase 3: RVM y Resta de Fondo] - 2026-10-06

### Añadido
- **Módulo `rvm_background.py`**:
  - `RvmProcessor`:
    - Carga offline robusta desde caché local (`source="local"`) evitando bloqueos o cuelgues por conectividad con GitHub en Windows.
    - Ratio de submuestreo ajustado (`rvm_dsr_mobilenet = 0.50`, `rvm_dsr_resnet = 0.40`) para retener la resolución geométrica de los dedos y los espacios interdigitales.
    - Curva de realce no lineal adaptativa (Hermite cúbica con piso negro estricto `black_floor = 0.05` y techo `white_ceiling = 0.85`), eliminando totalmente el efecto de sombras o halos oscuros/verdosos cuando la mano entra sola sin el rostro.
    - Gestión encapsulada y reseteo inmediato de tensores de memoria recurrente (`reset_recurrent_states`).
  - `BackgroundSubtractionProcessor`:
    - **Eliminación Total de Círculos y Polígonos Deformados**: Supresión de la dilatación agresiva de 2 iteraciones que fundía los dedos en una sola masa amorfa.
    - Filtrado Gaussiano suave (3x3) y apertura morfológica elíptica mínima (3x3) para eliminar ruido de cámara sin alterar la silueta anatómica.
    - Soporte de calibración dinámica con fotograma de referencia estático (`calibrate`) y sincronización bidireccional con el estado global HTTP.
- **Suite de Pruebas Unitarias de Fase 3 (`tests/test_rvm_background.py`)**:
  - `test_rvm_processor_initialization`: Verificación de carga en GPU/CPU y estructura de configuración.
  - `test_rvm_recurrent_states_reset`: Limpieza y reinicio de memoria oculta temporal.
  - `test_rvm_predict_output_contract`: Inferencia real con tensor `[1, 1, H, W]` en rango $[0.0, 1.0]$.
  - `test_background_subtraction_calibration`: Calibración y detección de estado calibrado.
  - `test_background_subtraction_zero_diff_on_static_scene`: Verificación de diferencia nula (cero falsos positivos) en escena estática.
  - `test_background_subtraction_clean_difference`: Detección diferencial limpia ante cambios sintéticos.
  - Suite global completa: **20 passed (100%)**.
- **Integración en `cuda_engine.py`**:
  - Reemplazo de los bloques dispersos de inferencia por llamadas estandarizadas a `rvm_processor.predict_alpha()` y `bg_processor.compute_mask()`.
  - Conexión del reseteo de estados recurrentes con eventos de conmutación de modelos (tecla `B`, endpoint `/api/model`) y reseteo manual (tecla `Espacio`, endpoint `/api/settings`).
  - Generado reporte de benchmark empírico: `docs/benchmarks/phase_3.json`.

---

## [Fase 4: MediaPipe y Hand Tracking Fluido] - 2026-10-06

### Añadido
- **Módulo `hand_tracker.py`**:
  - `OneEuroFilter1D` y `OneEuroFilter2D`:
    - Filtrado temporal adaptativo con frecuencia de corte dinámica en función de la velocidad angular/lineal ($f_c = f_{c,\min} + \beta \cdot |\dot{x}|$).
    - Supresión del 55%+ de varianza de temblor estocástico (*jitter*) cuando el usuario apunta con el dedo índice a objetos fijos, sin retraso ni latencia durante trazos veloces.
  - `HysteresisPinchDetector`:
    - **Cero Parpadeo y Falso Doble Clic**: Doble umbral histerético ($D_{down} = 58.0$ px para pulsar, $D_{up} = 78.0$ px para soltar) con antirrebote temporal de 2 frames consecutivos.
  - `HandInteractionTracker`:
    - Encapsulación limpia del ciclo de vida de MediaPipe `HandLandmarker` en `RunningMode.VIDEO`.
    - Garantía matemática de marcas de tiempo monotónicas crecientes (`max(cur_ms, last_ts + 1)`), eliminando excepciones por timestamps repetidos en bucles de alta velocidad.
    - Contrato de telemetría normalizado idéntico (`{"detected": bool, "x": float, "y": float, "is_pinching": bool, "pinch_dist": float}`).
- **Suite de Pruebas Unitarias de Fase 4 (`tests/test_hand_tracker.py`)**:
  - `test_one_euro_filter_reduces_static_jitter`: Validación cuantitativa de reducción de varianza en reposo.
  - `test_one_euro_filter_step_responsiveness`: Verificación de respuesta a saltos de velocidad sin rezago.
  - `test_hysteresis_pinch_detector_cycle_and_anti_chatter`: Ciclo completo de activación, histéresis intermedia y liberación sin rebote.
  - `test_hand_interaction_tracker_contract_on_empty_frame`: Cumplimiento del contrato en fotograma negro.
  - `test_hand_interaction_tracker_monotonic_timestamp`: Seguridad ante timestamps desordenados o repetidos.
  - Suite global completa: **25 passed (100%)**.
- **Integración en `cuda_engine.py` y `tools/benchmark.py`**:
  - Reemplazada la lógica desestructurada por una única llamada `hand_landmarks_list, cur_info = hand_tracker.process_frame(frame_rgb)`.
  - Integración de reseteo y cierre seguro de recursos en ciclo de vida del servidor.
  - Generado reporte de benchmark: `docs/benchmarks/phase_4.json`.

---

## [Fase 5: Fast Guided Filter y Pipeline de Renderizado GPU] - 2026-10-06

### Añadido
- **Módulo `gpu_renderer.py`**:
  - `fast_guided_filter`:
    - Refinamiento de bordes ejecutado enteramente en VRAM mediante núcleos `F.avg_pool2d` de cuDNN.
    - Alineación precisa de bordes finos con los gradientes de luminancia del fotograma original.
  - `GpuHologramRenderer`:
    - **Aceleración Crítica de Latencia**: Generación directa del buffer en orden BGR dentro de los Tensor Cores de la GPU antes de la transferencia a host, suprimiendo la llamada `cv2.cvtColor(RGB2BGR)` en CPU y reduciendo el tiempo de postproceso de ~20 ms a **4.6 ms**.
    - **Cero Emisión en Fondo**: Función de transferencia sigmoidea Hermite con piso negro estricto ($0.00$ absoluto para cualquier alfa residual $< 0.05$), garantizando cero emisión de luz verde sobre la pared oscura.
    - Configuración centralizada tipada (`GpuRendererConfig`).
- **Suite de Pruebas Unitarias de Fase 5 (`tests/test_gpu_renderer.py`)**:
  - `test_fast_guided_filter_output_contract`: Forma tensorial y rango numérico en $[0.0, 1.0]$.
  - `test_fast_guided_filter_edge_alignment`: Alineación geométrica con gradientes de la imagen guía.
  - `test_gpu_hologram_renderer_black_floor_suppression`: Supresión estricta de luz residual en fondo.
  - `test_gpu_hologram_renderer_green_dominance`: Comprobación del espectro estético holográfico verde.
  - `test_gpu_hologram_renderer_direct_bgr_output_shape`: Validación de buffer BGR uint8 listo para streaming.
  - Suite global completa: **30 passed (100%)**.
- **Integración en `cuda_engine.py` y `tools/benchmark.py`**:
  - `final_projector_tensor, output_bgr = gpu_renderer.render(...)` reemplazando bloques de shader manuales.
  - Generado reporte de benchmark: `docs/benchmarks/phase_5.json`.
  - **Métricas Destacadas**:
    - **FastSAM Point-Prompt**: **38.9 FPS** (latencia total: **25.7 ms**, postproceso: **4.6 ms**).
    - **YOLO11-seg**: **23.5 FPS** (latencia total: **42.6 ms** vs 2242 ms iniciales).
    - **Resta de Fondo**: **28.5 FPS** (latencia total: **35.0 ms**).

---

## [Fase 6: Arquitectura de 2 Pantallas y Verificación Final End-to-End] - 2026-10-06

### Añadido y Corregido
- **Eliminación del Requerimiento de Recargar `projector.html`**:
  - `controller.html` emite el evento unificado `model_changed` mediante `BroadcastChannel` para cualquiera de los 5 modelos.
  - `projector.html` refresca dinámicamente `cudaStream.src = '/stream?t=' + Date.now()` al recibir el mensaje o al detectar desconexiones.
  - Implementado perro guardián (*watchdog*) periódico (1.5 s) en `projector.html` que consulta `/api/status`; ante cualquier cambio de `backbone` o caída de stream, sincroniza y reconecta de inmediato sin intervención del usuario.
- **Unificación de los 5 Motores en el Servidor**:
  - La resta de fondo (Modelo 5) corre directamente en `cuda_engine.py` con `BackgroundSubtractionProcessor` a **34.5 FPS**, eliminando los bucles client-side que causaban inversión de colores y manchas en el navegador.
- **Topología de 2 Pantallas Limpia**:
  - **Pantalla 1 (Laptop Operador)**: `http://localhost:5051/controller.html` (interfaz de control con telemetría completa, cajas de objetos interactivos y cursor del índice).
  - **Pantalla 2 (Pared Proyector HDMI)**: `http://localhost:5051/projector.html` (ventana limpia exclusiva a 1920x1080, fondo negro puro `#000000`, sin cursor ni menús).
  - Ventana local de OpenCV desactivada por defecto (ejecución sin tercera ventana redundante). Opción `--native-projector` preservada para proyectar vía OpenCV nativo en monitor secundario si se requiere.
- **Resultados de Verificación Final End-to-End (`docs/benchmarks/final.json`)**:
  - **FastSAM Point-Prompt**: **42.3 FPS** (latencia total: **23.6 ms**).
  - **Resta de Fondo**: **34.5 FPS** (latencia total: **29.0 ms**).
  - **YOLO11-seg**: **20.3 FPS** (latencia total: **49.2 ms** vs 2242 ms iniciales).
  - **RVM ResNet50**: **13.8 FPS** (latencia total: **72.3 ms** vs 135 ms iniciales).
  - **RVM MobileNetV3**: **9.1 FPS** (latencia total: **109.7 ms** vs 145 ms iniciales).
  - **Suite Global de Pruebas Unitarias**: **30 passed (100%)** en `tests/`.
