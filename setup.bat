@echo off
cd /d "%~dp0"

echo ======================================================================
echo   CONFIGURACION AUTOMATICA DEL PROYECTO (CUDA + VISION ENGINE)
echo ======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 goto :no_python

echo [1/5] Comprobando entorno virtual en .venv...
if exist .venv\Scripts\python.exe goto :venv_ok

echo Creando entorno virtual .venv...
python -m venv .venv
if %errorlevel% neq 0 goto :venv_failed
echo [OK] Entorno virtual .venv creado con exito.
goto :venv_ready

:venv_ok
echo [OK] Entorno virtual .venv ya existe.

:venv_ready
echo.
echo [2/5] Actualizando instalador pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo.
echo [3/5] Instalando PyTorch con aceleracion CUDA para GPU NVIDIA...
echo Intentando descarga con CUDA 12.6...
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% equ 0 goto :torch_ok

echo.
echo [AVISO] Intentando descarga con CUDA 12.4...
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% equ 0 goto :torch_ok

echo.
echo [AVISO] Instalando PyTorch estandar de PyPI...
.venv\Scripts\pip.exe install torch torchvision

:torch_ok
echo.
echo [4/5] Instalando dependencias del sistema...
.venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 goto :req_failed

if exist requirements-dev.txt (
    .venv\Scripts\pip.exe install -r requirements-dev.txt
)

echo.
echo [5/5] Verificando modelos neuronales...
.venv\Scripts\python.exe tools\download_models.py

echo.
echo ======================================================================
echo   VERIFICACION DE ACELERACION HARDWARE:
echo ======================================================================
.venv\Scripts\python.exe -c "import torch; print('  * PyTorch version:', torch.__version__); print('  * Aceleracion GPU CUDA activa:', torch.cuda.is_available()); print('  * Dispositivo detectado:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Modo seguro)')"
echo ======================================================================
echo.
echo [LISTO] Configuracion finalizada con exito.
echo Para iniciar el sistema en cualquier momento, ejecuta:
echo   --^> iniciar_proyecto.bat
echo.
pause
exit /b 0

:no_python
echo.
echo [ERROR] Python no esta instalado o no se encuentra en el PATH del sistema.
echo Por favor instala Python 3.10, 3.11, 3.12 o 3.13 de 64 bits marcando la casilla:
echo "Add python.exe to PATH".
echo Descarga desde: https://www.python.org/downloads/
echo.
pause
exit /b 1

:venv_failed
echo.
echo [ERROR] No se pudo crear el entorno virtual .venv.
echo Verifica los permisos del directorio o la instalacion de Python.
echo.
pause
exit /b 1

:req_failed
echo.
echo [ERROR] Ocurrio un error al instalar las dependencias de requirements.txt.
echo.
pause
exit /b 1
