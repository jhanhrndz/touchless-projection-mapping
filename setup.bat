@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ======================================================================
echo   🛠️ CONFIGURACIÓN AUTOMÁTICA DEL PROYECTO (CUDA + VISION ENGINE)
echo ======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python no está instalado o no se encuentra en el PATH del sistema.
    echo Por favor instala Python 3.10, 3.11, 3.12 o 3.13 (64-bit) marcando la casilla:
    echo "Add python.exe to PATH".
    echo Descarga desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Comprobando entorno virtual en .venv...
if not exist .venv (
    echo Creando entorno virtual .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo ✓ Entorno virtual .venv creado con éxito.
) else (
    echo ✓ Entorno virtual .venv ya existe.
)

echo.
echo [2/5] Actualizando instalador pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo.
echo [3/5] Instalando PyTorch con aceleración CUDA para GPU NVIDIA...
echo (Intentando PyTorch con soporte CUDA 12.4 / 12.1...)
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% neq 0 (
    echo.
    echo [AVISO] Falló la descarga de PyTorch CUDA 12.4. Intentando PyTorch estándar...
    .venv\Scripts\pip.exe install torch torchvision
)

echo.
echo [4/5] Instalando dependencias del sistema...
.venv\Scripts\pip.exe install -r requirements.txt
if exist requirements-dev.txt (
    .venv\Scripts\pip.exe install -r requirements-dev.txt
)

echo.
echo [5/5] Verificando modelos neuronales...
.venv\Scripts\python.exe tools\download_models.py

echo.
echo ======================================================================
echo   🔍 VERIFICACIÓN DE ACELERACIÓN HARDWARE:
echo ======================================================================
.venv\Scripts\python.exe -c "import torch; print('  ✓ PyTorch version:', torch.__version__); print('  ✓ Aceleración GPU CUDA activa:', torch.cuda.is_available()); print('  ✓ Dispositivo detectado:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Modo seguro)')"
echo ======================================================================
echo.
echo [LISTO] Configuración finalizada con éxito.
echo Para iniciar el sistema en cualquier momento, solo ejecuta:
echo   --> iniciar_proyecto.bat
echo.
pause
