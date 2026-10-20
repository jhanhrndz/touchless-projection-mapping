# Orquestador Nativo Profesional y Calibración de Sensibilidad de Pellizco

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un orquestador de procesos nativo en Python (`run.py`) con control de ciclo de vida limpio (Ctrl+C sin procesos huérfanos), verificación de hardware (CUDA, cámara, puertos), y calibrar con precisión la sensibilidad del detector de pellizco MediaPipe en `src/vision/hand_tracker.py` y `src/vision/config.py`.

**Architecture:** 
1. Subpipeline de Gestos: Parametrizar en `EngineConfig` los umbrales de histéresis anatómica (`pinch_down_px: 65.0`, `pinch_up_px: 85.0`, `debounce_frames: 1`) conectándolos a `HandInteractionTracker` y permitiendo configuración por variables de entorno y endpoint `/api/settings`.
2. Orquestador `run.py`: Script de orquestación en Python usando `subprocess.Popen`, detección de señales `SIGINT`/`SIGTERM` con cierre gracioso coordinado, validación de puertos libres (8000 y 5051), y atajo de un solo clic desde `iniciar_proyecto.bat`.

**Tech Stack:** Python 3.11+, PyTorch (CUDA), MediaPipe, FastAPI, Uvicorn, Windows PowerShell / Batch.

**Spec:** Requerimiento del usuario para:
1. Eliminar la sensación de pérdida de precisión en el pellizco ajustando la sensibilidad geométrica.
2. Adoptar la decisión profesional de ejecución nativa en Host sustituyendo la fragilidad de `.bat` con un runner profesional en Python, reservando Docker solo para CI/CD.

## Global Constraints

- Compatibilidad estricta con Windows 10/11 y PowerShell/cmd.
- Respetar `src-layout` con `PYTHONPATH=src`.
- Cero regresión en los 580 tests automatizados del banco de pruebas.
- Mantener la separación de puertos: Backend en 8000, Visión en 5051.

## Review Focus

1. `PinchConfig` en `src/vision/hand_tracker.py` y `src/vision/config.py`: Si el umbral inferior de pellizco es muy bajo (<58px), personas con manos más alejadas de la cámara no logran activar el clic con comodidad. Elevar a 65px y reducir debounce a 1 frame restaura la inmediatez táctil.
2. Muerte limpia de subprocesos en Windows: `Popen.terminate()` en Windows a veces no propaga la señal a procesos hijos de Python. `run.py` debe manejar `SIGINT` o `taskkill /F /T` para asegurar que los puertos 8000 y 5051 queden libres.
3. Validación de puertos antes de arrancar: Si el puerto 8000 o 5051 ya está ocupado por una instancia previa, `run.py` debe informar claramente el PID que lo bloquea en lugar de crashear crípticamente con `OSError: [Errno 10048]`.

---

### Task 1: Calibración y Parametrización de Sensibilidad de Pellizco

**Files:**
- Modify: `src/vision/config.py`
- Modify: `src/vision/hand_tracker.py`
- Modify: `src/vision/engine.py`
- Test: `tests/vision/test_hand_tracker.py`

**Interfaces:**
- `EngineConfig`: Añadir `pinch_down_px: float = 65.0`, `pinch_up_px: float = 85.0`, `pinch_debounce_frames: int = 1`.
- `HandTrackerConfig`: Exponer `debounce_frames: int = 1`, `pinch_down_px: float = 65.0`, `pinch_up_px: float = 85.0`.
- `VisionEngine.__init__`: Propagar `self.cfg.pinch_down_px`, etc., hacia `HandInteractionTracker`.

- [ ] **Step 1: Escribir prueba unitaria para la nueva sensibilidad y configuración en `tests/vision/test_hand_tracker.py`**
  - Validar que con `debounce_frames=1` el clic active inmediatamente en el primer fotograma $\le 65.0\text{ px}$.
  - Validar que `HandInteractionTracker` acepte y aplique los umbrales de `HandTrackerConfig`.

- [ ] **Step 2: Ejecutar pytest para verificar que la prueba falle si aún no está implementada**
  - Ejecutar: `.venv\Scripts\python.exe -m pytest tests/vision/test_hand_tracker.py -v`

- [ ] **Step 3: Actualizar `src/vision/config.py`**
  - Añadir campos en `EngineConfig` y soporte en `from_env()` para `PINCH_DOWN_PX`, `PINCH_UP_PX`, `PINCH_DEBOUNCE`.

- [ ] **Step 4: Actualizar `src/vision/hand_tracker.py`**
  - Modificar `HandTrackerConfig` con los nuevos valores por defecto (`pinch_down_px=65.0`, `pinch_up_px=85.0`, `debounce_frames=1`).
  - Pasar `debounce_frames` al inicializar `HysteresisPinchDetector`.

- [ ] **Step 5: Conectar `self.cfg` en `VisionEngine` (`src/vision/engine.py`)**
  - Instanciar `HandInteractionTracker` inyectando `task_path=self.cfg.hand_landmarker_path` y `config=HandTrackerConfig(...)`.

- [ ] **Step 6: Ejecutar pytest para confirmar que pase**
  - Ejecutar: `.venv\Scripts\python.exe -m pytest tests/vision/test_hand_tracker.py -v`

---

### Task 2: Implementación del Orquestador Profesional `run.py`

**Files:**
- Create: `run.py`
- Modify: `iniciar_proyecto.bat`
- Test: `tests/backend/test_server.py`

**Interfaces:**
- `run.py`: Script ejecutable CLI con argumentos: `--no-browser`, `--source`, `--model`, `--port-backend`, `--port-vision`.

- [ ] **Step 1: Crear `run.py` con las siguientes capacidades:**
  1. Detección automática del entorno virtual `.venv` y re-ejecución transparente con el binario de `.venv`.
  2. Banner ANSI con estética de estudio Pro-AV.
  3. Comprobación preliminar de salud:
     - Detección de CUDA en GPU.
     - Detección de cámara física (OpenCV).
     - Comprobación de puertos 8000 y 5051 libres (con socket bind test).
  4. Lanzamiento asíncrono y coordinado de:
     - `src.backend` (FastAPI + WebSockets en puerto 8000).
     - `src.vision` (Motor neuronal + Stream en puerto 5051).
  5. Manejador de señal `signal.signal(signal.SIGINT, ...)` y `atexit` para terminar limpiamente ambos subprocesos sin dejar zombis en Windows (`taskkill` seguro como fallback).
  6. Apertura automática opcional del navegador en `http://localhost:5051/controller.html` y `http://localhost:5051/projector.html` tras verificar que los puertos respondan HTTP 200.

- [ ] **Step 2: Probar `run.py --help` para verificar opciones CLI**
  - Ejecutar: `.venv\Scripts\python.exe run.py --help`

- [ ] **Step 3: Actualizar `iniciar_proyecto.bat`**
  - Delegar la ejecución a `run.py %*` garantizando que un doble clic en Windows inicie el orquestador profesional.

---

### Task 3: Verificación Completa y Commit

**Files:**
- Suite: `tests/backend/` y `tests/vision/`
- Git: Commit estructurado

- [ ] **Step 1: Ejecutar la suite completa de pruebas de backend y visión**
  - Ejecutar: `.venv\Scripts\python.exe -m pytest tests/ -v`
  - Esperar 0 fallos.

- [ ] **Step 2: Verificar `git status` y `git diff`**
  - Asegurar que solo se hayan tocado los archivos planeados.

- [ ] **Step 3: Commit en Git**
  - Mensaje: `feat(core): professional python orchestrator run.py and high-precision pinch tuning`
