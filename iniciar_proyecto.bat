@echo off
chcp 65001 > nul
echo ======================================================================
echo   🚀 INICIANDO MOTOR DE PROYECCIÓN INTERACTIVA (SISTEMA 2 PANTALLAS)
echo ======================================================================
echo.
echo   🖥️ Pantalla 1 (Laptop Operador):  http://localhost:5051/controller.html
echo   📽️ Pantalla 2 (Pared Proyector):   http://localhost:5051/projector.html
echo.
echo   Opciones en vivo:
echo     - [Espacio] o [C] : Calibrar fondo (Modelo 5)
echo     - [B]             : Alternar entre los 5 motores de IA
echo     - [G]             : Activar / Desactivar Fast Guided Filter
echo     - [Q] o [ESC]     : Detener motor
echo ======================================================================
echo.

if exist .venv\Scripts\python.exe (
    set PYTHON_EXE=.venv\Scripts\python.exe
) else (
    set PYTHON_EXE=python
)

%PYTHON_EXE% -m vision_engine %*
pause
