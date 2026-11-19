# Touchless Projection Mapping 🖐️✨

> Sistema profesional de proyección interactiva holográfica potenciado por visión artificial acelerada por hardware (NVIDIA CUDA), suite de 8 modelos de segmentación neuronal y automatización IoT física en tiempo real.

<p align="center">
  <img src="docs/assets/touchless-projection-mapping.jpg" alt="Touchless Projection Mapping Cover" width="100%" />
</p>

---

## 🌟 Descripción General

**Touchless Projection Mapping** transforma cualquier pared u objeto físico en una superficie interactiva de control sin necesidad de sensores vestibles, pantallas táctiles ni costosas cámaras de profundidad.

A través de una cámara web estándar y una GPU con aceleración CUDA, el sistema:
1. **Rastrea las manos** en tiempo real con **MediaPipe Hand Landmarker** y estabilización cinemática mediante filtro **One-Euro**.
2. **Segmenta los brazos y el cuerpo** utilizando modelos fundacionales de vanguardia (**FastSAM**, **YOLO11-seg**, **Robust Video Matting** o algoritmos diferenciales).
3. **Proyecta una silueta holográfica verde neón** sobre fondos de video dinámicos con fusión aditiva (`mix-blend-mode: screen`) y filtrado guiado en GPU (**Fast Guided Filter**).
4. **Detecta gestos de clic táctil por pellizco (*pinch*)** con histeresis calibrada a 65 px y respuesta inmediata en 1 fotograma.
5. **Activa dispositivos físicos reales** (enchufes inteligentes TP-Link Kasa / Tapo, luces, relés y actuadores) mediante **WebSockets** y un editor visual de polígonos interactivos.

---

## 🏛️ Arquitectura del Proyecto (`src-layout`)

```text
touchless-projection-mapping/
├── config/                      ← CONFIGURACIÓN CENTRALIZADA Y ESCENAS
│   ├── devices.yaml             (Mapeo de dispositivos IoT físicos reales)
│   ├── devices.dev.yaml         (Modo desarrollo / simulación mock segura)
│   ├── objects.json             (Polígonos y geometrías interactivas proyectadas)
│   └── examples/                (Escenas de ejemplo: lámpara, estantería, etc.)
│
├── src/                         ← CÓDIGO FUENTE UNIFICADO (src-layout estándar)
│   ├── backend/                 (FastAPI, WebSockets, control IoT y editor de geometrías)
│   │   ├── devices/             (Controladores: wifi_switch, kasa, http, udp, mock)
│   │   ├── static/              (Editor visual de polígonos y estilos en navegador)
│   │   ├── config.py & server.py (Núcleo y API REST del runtime de dispositivos)
│   │   └── editor.py            (Lógica geométrica y serialización de objetos)
│   │
│   └── vision/                  (Motor neuronal acelerado por GPU / CUDA)
│       ├── engine.py            (Bucle de visión, shaders GPU, telemetría y gestos)
│       ├── server.py            (Servidor HTTP, streaming MJPEG y API REST)
│       ├── hand_tracker.py      (MediaPipe Hand Landmarker + Filtro One-Euro + Pinch)
│       ├── gpu_renderer.py      (Shader holográfico BGR cuDNN acelerado en GPU)
│       ├── models/              (Pesos locales: FastSAM-s, YOLO11, Hand Landmarker)
│       └── pipelines/           (Pipelines para RVM, YOLO11-seg, FastSAM, Resta GPU)
│
├── frontend/                    ← INTERFACES DE USUARIO Y PROYECCIÓN
│   ├── controller.html          (Panel del operador: vista de cámara, selector de modelos)
│   └── projector.html           (Ventana de proyección en pared: negro puro, cero HUD)
│
├── tests/                       ← BANCO DE PRUEBAS AUTOMATIZADO (581 tests unitarios)
│   ├── backend/                 (Pruebas de servidor, dispositivos, editor y WebSocket)
│   └── vision/                  (Pruebas de modelos IA, shaders GPU, tracking y matting)
│
├── scripts/                     ← HERRAMIENTAS Y SCRIPTS OPERATIVOS
│   ├── benchmark.py             (Medición aislada de latencia y FPS reales)
│   ├── download_models.py       (Descarga y verificación de pesos neuronales)
│   ├── generate_synthetic_clips.py (Generador de secuencias de video sintéticas)
│   ├── record_clips.py          (Grabador interactivo de clips de prueba)
│   └── list_displays.py         (Enumerador multiplataforma de proyectores y pantallas)
│
├── docs/                        ← DOCUMENTACIÓN TÉCNICA Y ARQUITECTURA
│   ├── assets/                  (Imágenes y diagramas de arquitectura)
│   ├── ARCHITECTURE.md          (Especificación técnica del sistema)
│   ├── AUDIT.md & CHANGELOG.md  (Auditorías y registro de cambios)
│   └── SETUP_GUIDE.md           (Guía paso a paso de puesta en marcha)
│
├── run.py                       ← ORQUESTADOR NATIVO PROFESIONAL (Gestión de host y procesos)
├── iniciar_proyecto.bat         ← ARRANQUE CON 1 CLIC (Lanzador para Windows)
├── setup.bat                    ← INSTALADOR AUTOMÁTICO DE ENTORNO, CUDA Y MODELOS
├── pyproject.toml               ← Configuración unificada del proyecto y pytest
├── requirements.txt             ← Dependencias consolidadas del sistema
├── .env.example                 ← Plantilla de configuración de variables de entorno
└── README.md
```

---

## 🎯 Suite de 8 Modelos de Segmentación

El sistema permite alternar en caliente entre 8 métodos de segmentación para adaptarse al hardware y entorno disponible:

| # | Modelo | Dispositivo | FPS | Latencia | Características Principales |
| :-: | :--- | :---: | :---: | :---: | :--- |
| **1** | **FastSAM Point-Prompt** | CUDA (FP16) | **33 FPS** | ~30 ms | **Modo Recomendado.** Extracción orgánica por punto anatómico, contornos finos sin cajas. |
| **2** | **YOLO11-seg** | CUDA (FP16) | **37 FPS** | ~27 ms | Máxima velocidad pura con `retina_masks=True` y filtrado por instancia de mano. |
| **3** | **RVM MobileNetV3** | CUDA | **34 FPS** | ~29 ms | Video matting temporal recurrente con memoria de bordes y preservación interdigital. |
| **4** | **RVM ResNet50** | CUDA | **28 FPS** | ~35 ms | Máxima fidelidad en siluetas anatómicas y dedos finos. |
| **5** | **Resta de Fondo GPU** | GPU / CPU | **34 FPS** | ~29 ms | Resta diferencial contra pared física calibrada con tecla `[Espacio]`. Cero parpadeo. |
| **6** | **Resta de Fondo Web** | Navegador | **40 FPS** | ~25 ms | Fallback 100% en navegador con semillas anatómicas de MediaPipe. |
| **7** | **MediaPipe Pose Web** | Navegador | **28 FPS** | ~35 ms | Fallback web independiente basado en pose anatómica (v3). |
| **8** | **RVM ONNX Web** | Navegador | **22 FPS** | ~45 ms | Fallback web basado en ONNX Runtime WebAssembly (v4). |

---

## 🚀 Instalación y Puesta en Marcha

### Requisitos de Hardware
- **Sistema Operativo:** Windows 10/11 (64-bit)
- **Python:** Python 3.10, 3.11, 3.12 o 3.13 (64-bit)
- **GPU (Recomendada):** NVIDIA GeForce (GTX 1650 Ti+, RTX 2060+, RTX 30/40 series o Quadro) con soporte CUDA. Si no dispone de GPU dedicada, arrancará automáticamente en modo CPU seguro.
- **Periféricos:** Cámara web USB estándar (720p/1080p a 30 o 60 FPS) y proyector HDMI (o monitor secundario extendido).

---

### Método 1: Instalación Rápida (Recomendado en Windows)

1. **Configurar el entorno y descargar modelos (Solo la primera vez):**
   Haz doble clic en:
   ```cmd
   setup.bat
   ```
   *Crea automáticamente el entorno virtual `.venv`, instala PyTorch con aceleración CUDA, instala todas las librerías de `requirements.txt`, descarga los modelos neuronales y verifica la GPU.*

2. **Iniciar el Sistema Completo:**
   Haz doble clic en:
   ```cmd
   iniciar_proyecto.bat
   ```
   *Ejecuta el orquestador profesional `run.py`, realiza autodiagnósticos de hardware y abre las interfaces web sincronizadas.*

---

### Método 2: Ejecución mediante Orquestador Python (`run.py`)

El orquestador nativo [`run.py`](run.py) es el estándar profesional Pro-AV para controlar el ciclo de vida de los servicios con pre-flight checks y apagado limpio (*Graceful Shutdown*):

```powershell
# 1. Activar entorno virtual
.venv\Scripts\activate

# 2. Verificar estado del hardware y puertos sin arrancar
python run.py --check-only

# 3. Iniciar el sistema completo en modo interactivo normal
python run.py

# Opciones avanzadas de arranque:
python run.py --model yolo11                # Iniciar con un modelo IA específico
python run.py --source data/demo.mp4        # Modo offline usando un clip de video (sin cámara web)
python run.py --no-browser                  # No abrir pestañas de navegador automáticamente
python run.py --backend-port 8000 --vision-port 5051  # Personalizar puertos
```

---

## ⚙️ Configuración y Variables de Entorno (`.env`)

El sistema incluye una plantilla completa en [`.env.example`](.env.example). Para personalizar tu entorno local:

1. **Crear tu archivo `.env`:**
   ```powershell
   # En Windows PowerShell:
   Copy-Item .env.example .env

   # En Bash:
   cp .env.example .env
   ```

2. **Parámetros configurables:**

| Variable | Por Defecto | Descripción |
| :--- | :---: | :--- |
| `APP_HOST` | `127.0.0.1` | Dirección IP de escucha del Backend IoT y Editor. |
| `APP_PORT` | `8000` | Puerto HTTP/WebSocket del Backend. Editor en: `http://localhost:8000/editor`. |
| `VISION_PORT` | `5051` | Puerto HTTP del Motor de Visión y vistas de proyección. |
| `VISION_MODEL` | `rvm_mobilenet` | Modelo neuronal inicial (`rvm_mobilenet`, `yolo11`, `fastsam`, `bg_sub_gpu`). |
| `VISION_ARM_ONLY` | `false` | `false`: silueta de cuerpo completo; `true`: solo mano/antebrazo. |
| `VISION_PINCH_DOWN` | `65.0` | Umbral en píxeles para activar clic por pellizco. |
| `VISION_PINCH_UP` | `85.0` | Umbral en píxeles para liberar el clic (histeresis anti-rebote). |
| `VISION_PINCH_DEBOUNCE`| `1` | Fotogramas de confirmación (1 = respuesta táctil instantánea). |
| `SMART_SWITCH_HOST` | `10.0.0.42` | IP local del enchufe o interruptor inteligente TP-Link Kasa. |
| `KASA_USERNAME` | — | Correo de cuenta TP-Link (para dispositivos con protocolo KLAP). |
| `KASA_PASSWORD` | — | Contraseña de cuenta TP-Link. |
| `RUNTIME_PROJECTOR_SIZE`| `1920x1080` | Resolución nativa del proyector (Ancho x Alto en px). |

---

## 🖥️ Arquitectura de Pantallas y Vistas Web

El sistema opera con **tres ventanas web coordinadas**:

1. **Panel del Operador (Laptop / Monitor Principal):**
   - URL: **`http://localhost:5051/controller.html`**
   - Vista previa de cámara en tiempo real, selector de modelos IA, telemetría de FPS/latencia de GPU, controles de video de fondo y calibración de pared.

2. **Pantalla Proyector (Pared de Proyección):**
   - URL: **`http://localhost:5051/projector.html`**
   - Arrastra esta ventana a la pantalla extendida del proyector y presiona **F11** (o tecla **F**) para pantalla completa.
   - **Regla óptica estricta:** Fondo negro puro `#000000` (100% transparente ante la luz del proyector), proyectando exclusivamente el holograma verde neón y los videos de mapeo sin ningún elemento de interfaz de usuario.

3. **Editor de Polígonos y Hardware IoT:**
   - URL: **`http://localhost:8000/editor`**
   - Lienzo interactivo en resolución 1:1 con la pared. Permite dibujar polígonos sobre objetos físicos (lámparas, cuadros, interruptores) y vincularlos a dispositivos de hardware reales o simulados.

---

## 🎮 Controles Rápidos y Atajos de Teclado

| Tecla | Acción |
| :---: | :--- |
| **`[Espacio]`** o **`[C]`** | **Calibrar fondo:** Captura la pared vacía como referencia limpia (Modelo 5). |
| **`[B]`** | **Alternar modelo:** Conmuta secuencialmente entre los 8 modelos de IA/visión. |
| **`[G]`** | **Fast Guided Filter:** Activa o desactiva el filtrado de bordes finos en GPU. |
| **`[V]`** | **Video de fondo:** Pausa o reanuda el video de mapeo proyectado. |
| **`[F]`** o **`[F11]`** | **Pantalla completa:** En la ventana de proyección. |
| **`[P]`** | **Abrir proyector:** Abre la ventana secundaria de proyección. |
| **`[Q]`** o **`[ESC]`** | **Detener:** Cierra el motor de visión y libera la memoria VRAM. |

---

## 🧪 Pruebas Unitarias y Calidad de Código

El proyecto cuenta con un banco integral de **581 pruebas automatizadas** que validan el backend FastAPI, los controladores de hardware, el motor de visión, los filtros cinemáticos y la detección de gestos:

```powershell
# Ejecutar la suite completa de pruebas
pytest tests/ -v

# Ejecutar únicamente las pruebas del motor de visión y gestos
pytest tests/vision/test_hand_tracker.py -v

# Ejecutar benchmark de rendimiento por etapas en GPU
python scripts/benchmark.py --mode all --frames 60
```

---

## 📦 Integración en Repositorio Central (`Audacia`)

Para integrar este proyecto como subcarpeta dentro del monorepo central de la organización:

```text
Audacia/
├── proyecto1/
├── proyecto2/
└── touchless-projection-mapping/   ← Este módulo
```

Se recomienda utilizar **`git subtree` con `--squash`** desde una bifurcación (*Fork*), lo que permite incorporar la suite completa en su propia carpeta manteniendo una historia limpia y modular.

---

## 🤝 Créditos y Reconocimientos

- Módulos de automatización de hardware y hub WebSocket en `backend/` basados en [interactive-projection-mapping](https://github.com/buildwbrendan/interactive-projection-mapping) de **Brendan**.
- Motor de visión unificado (`src/vision/`), suite multimodelo (FastSAM, YOLO11, RVM), orquestador nativo `run.py`, calibración de gestos táctiles, sincronización de doble ventana, fusión de video de fondo y shaders holográficos desarrollados por **Jhan Hernández**.

---

## 📄 Licencia

Este proyecto está distribuido bajo la Licencia GPL-3.0 / MIT. Consulta el archivo [pyproject.toml](pyproject.toml) para más detalles.
