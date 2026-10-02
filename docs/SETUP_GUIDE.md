# Guía de Instalación y Calibración

## 1. Requisitos del Sistema
- **Sistema Operativo:** Windows 10/11, macOS, o Linux.
- **Python:** Versión 3.10 o superior.
- **Cámara Web:** 720p o 1080p con buena iluminación frontal.
- **Proyector (Opcional):** Configurado como pantalla extendida en modo 1920x1080.
- **Navegador Web:** Google Chrome, Microsoft Edge o navegadores basados en Chromium (recomendados para aceleración WebGL y MediaPipe).

---

## 2. Instalación de Dependencias

### Entorno Python
En la raíz del proyecto:
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuración de Variables de Entorno
Copia la plantilla de entorno:
```bash
cp .env.example .env
```
Edita `.env` para ajustar los puertos o credenciales de dispositivos si utilizas hardware inteligente (TP-Link Kasa o ESP32).

---

## 3. Puesta en Marcha

### Terminal 1: Servidor de Visión (Frontend)
```bash
cd frontend
python server.py
```
El servidor arrancará en `http://localhost:5050`.

### Terminal 2: Servidor de Automatización (Backend)
```bash
cd backend
python -m runtime --devices data/devices.dev.yaml
```
El backend WebSocket escuchará en `ws://localhost:8080/ws`.

---

## 4. Procedimiento de Calibración Paso a Paso

1. Abre `http://localhost:5050/` en tu navegador.
2. Otorga permisos de acceso a la cámara.
3. **Calibración de Fondo:**
   - Asegúrate de que **no haya personas, brazos ni objetos móviles** en el encuadre de la cámara.
   - Presiona la barra espaciadora **`[ESPACIO]`**. Verás un mensaje de confirmación en el panel de diagnóstico indicando que el fondo fue registrado.
4. **Abrir Proyector:**
   - Haz clic en **"🖥️ Abrir Proyección"**.
   - Arrastra la ventana emergente a la pantalla del proyector.
   - Presiona **F11** para ponerla en pantalla completa.
5. **(Opcional) Cargar Video de Fondo:**
   - En el controlador, haz clic en **"🎬 Video Fondo"**.
   - Selecciona tu video de animación o ambientación.
   - Ajusta el deslizador de brillo si deseas que el brazo verde resalte más sobre el fondo.
6. **Interacción:**
   - Extiende tu brazo frente a la cámara. Verás tu brazo digitalizado proyectarse sobre la pared.
   - Apunta a las zonas mapeadas y junta el dedo índice con el pulgar para activar dispositivos físicos o acciones interactivas.
