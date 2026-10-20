@echo off
setlocal
cd /d "%~dp0"

:: Touchless Projection Mapping — Lanzador Windows
:: Delega la orquestacion al runner profesional nativo run.py

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [AVISO] Entorno virtual .venv no encontrado. Usando python del sistema...
    set "PYTHON_EXE=python"
)

%PYTHON_EXE% run.py %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] El sistema se detuvo con codigo de error %errorlevel%.
    pause
)
