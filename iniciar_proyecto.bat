@echo off
cd /d "%~dp0"

echo ======================================================================
echo   INICIANDO SISTEMA DE PROYECCION INTERACTIVA TOUCHLESS
echo ======================================================================
echo.
echo   [1] Backend Hardware/Objetos:  http://localhost:8000/
echo   [2] Laptop Operador:           http://localhost:5051/controller.html
echo   [3] Pared Proyector:           http://localhost:5051/projector.html
echo.
echo   Atajos de teclado en vivo:
echo     - [Espacio] o [C] : Calibrar fondo limpio (Modelo 5)
echo     - [B]             : Alternar entre los 5 motores de IA / Matemática
echo     - [G]             : Activar / Desactivar Fast Guided Filter
echo     - [V]             : Pausar / Reanudar video de fondo
echo     - [Q] o [ESC]     : Detener motor
echo ======================================================================
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [AVISO] Entorno virtual .venv no encontrado. Usando python del sistema...
    set "PYTHON_EXE=python"
)

:: Iniciar el Backend (Runtime en puerto 8000)
echo [*] Levantando Servidor Backend (FastAPI / Dispositivos en puerto 8000)...
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=backend\src"
start /b "Backend-Runtime" %PYTHON_EXE% -m runtime --devices backend\data\devices.dev.yaml --objects frontend\objects.json --port 8000

:: Iniciar el Motor de Vision e Interaccion (Puerto 5051)
echo [*] Iniciando Motor de Vision Neuronal (Puerto 5051)...
%PYTHON_EXE% -m vision_engine %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] El motor de vision se detuvo con un error.
    pause
)
