@echo off
cd /d "%~dp0"

echo ======================================================================
echo   INICIANDO MOTOR DE PROYECCION INTERACTIVA (SISTEMA 2 PANTALLAS)
echo ======================================================================
echo.
echo   Pantalla 1 (Laptop Operador):  http://localhost:5051/controller.html
echo   Pantalla 2 (Pared Proyector):   http://localhost:5051/projector.html
echo.
echo   Opciones en vivo:
echo     - [Espacio] o [C] : Calibrar fondo (Modelo 5)
echo     - [B]             : Alternar entre los 8 motores de IA
echo     - [G]             : Activar / Desactivar Fast Guided Filter
echo     - [V]             : Pausar / Reanudar video de fondo
echo     - [Q] o [ESC]     : Detener motor
echo ======================================================================
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [AVISO] Entorno virtual .venv no encontrado. Usando python del sistema...
    echo Si falta alguna libreria, ejecuta primero setup.bat.
    echo.
    set "PYTHON_EXE=python"
)

%PYTHON_EXE% -m vision_engine %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] El motor de vision se detuvo con un error.
    pause
)
