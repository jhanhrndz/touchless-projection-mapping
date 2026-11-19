# Roadmap del Proyecto — Estado Actual y Pasos a Seguir 🗺️

> Documento técnico y operativo que detalla los módulos implementados, la arquitectura preparada para escalabilidad y las tareas pendientes para completar el 100% del sistema de proyección interactiva.

---

## 📊 1. Resumen del Estado de Madurez del Sistema

El proyecto **Touchless Projection Mapping (v2.0)** cuenta con un núcleo de visión artificial, renderizado holográfico, editor de geometrías y orquestación de hardware completamente funcional y respaldado por **581 pruebas automatizadas**.

| Componente | Estado | Madurez | Descripción |
| :--- | :---: | :---: | :--- |
| **Visión e IA (Tracking + 4 Motores)** |  **Completado** | 100% | MediaPipe + Filtro One-Euro, RVM MobileNet, RVM ResNet, YOLO11-seg y Resta de Fondo GPU. |
| **Interacción Táctil por Gestos (Pinch)**|  **Completado** | 100% | Histeresis calibrada (65px down, 85px up) con debounce instantáneo de 1 fotograma. |
| **Shaders y Renderizado Holográfico** |  **Completado** | 100% | Silueta verde neón, Fast Guided Filter en GPU y fusión aditiva con video de fondo. |
| **Orquestador Host Profesional (`run.py`)**|  **Completado** | 100% | Pre-flight checks de GPU CUDA y puertos (8000/5051), apagado limpio y sincronización web. |
| **Editor Visual de Polígonos (`/editor`)** |  **Completado** | 100% | Dibujo de geometrías interactivas 1:1 en lienzo web con persistencia en `objects.json`. |
| **Drivers de Hardware IoT (Kasa / HTTP)** |  **Preparado** | 85% | Controladores desacoplados con soporte Mock y Kasa KLAP. Listo para enlazar hardware físico. |
| **Calibración Automática Cámara-Proyector** | ⏳ **Pendiente** | 30% | Funciona mediante coordenadas 1:1. Pendiente módulo de auto-homografía con ArUco. |
| **Gestos Avanzados Multi-Mano** | ⏳ **Pendiente** | 20% | Clic por pellizco operativo. Pendiente swipe, rotación y palma extendida. |

---

## 🛠️ 2. Pasos a Seguir para la Conclusión al 100%

### Fase 1: Enlace Físico de Dispositivos IoT (Lámpara Inteligente)
* **Objetivo:** Pasar del modo simulación seguro (`devices.dev.yaml` / `MockDevice`) al control de una lámpara o bombilla inteligente física real en la pared proyectada.
* **Arquitectura actual:** El sistema ya está diseñado de forma modular y desacoplada en `src/backend/devices/`. El motor de visión ya emite los eventos `enter`, `leave` y `click` hacia el WebSocket del backend.
* **Tareas pendientes:**
  1. Adquirir o configurar el enchufe inteligente WiFi (ej. TP-Link Kasa KP115, Tapo P110 o Shelly Plus Plug).
  2. Asignar una dirección IP fija o reserva DHCP al enchufe dentro de la red local del centro (ej. `192.168.1.150`).
  3. En `config/devices.yaml`, vincular el objeto a la lámpara:
     ```yaml
     - object_id: lamp_main
       type: kasa
       label: "Lámpara de Escritorio"
       host: "192.168.1.150"
     ```
  4. Desde el editor web (`http://localhost:8000/editor`), dibujar el polígono exacto sobre la lámpara física proyectada y asignarle el `Device ID: lamp_main`.

---

### Fase 2: Calibración Automática de Perspectiva (Auto-Homografía)
* **Objetivo:** Eliminar la necesidad de alinear manualmente la cámara web con el área de proyección.
* **Tareas pendientes:**
  1. Diseñar una rutina en el proyector que proyecte un patrón de calibración óptico (tablero de ajedrez o marcadores ArUco de 4 esquinas).
  2. Capturar el patrón con la cámara web mediante OpenCV (`cv2.findChessboardCorners` o `cv2.aruco.detectMarkers`).
  3. Calcular la matriz de homografía bidimensional de 3x3 ($H$) que mapea píxeles de cámara a píxeles de proyección.
  4. Aplicar la transformación de perspectiva en tiempo real a las coordenadas de los landmarks de la mano (`cv2.perspectiveTransform`), garantizando precisión milimétrica aunque la cámara esté inclinada respecto a la pared.

---

### Fase 3: Catálogo Extendido de Gestos Touchless
* **Objetivo:** Ampliar la capacidad interactiva más allá del clic por pellizco básico.
* **Tareas pendientes:**
  1. **Deslizamiento (*Swipe Gesture*):** Detectar desplazamientos rápidos de la mano en los ejes $X$ e $Y$ para cambiar de diapositiva o alternar escenas de mapping.
  2. **Arrastre y Soltar (*Drag & Drop*):** Mantener el estado de pellizco mientras se mueve la mano para arrastrar objetos virtuales sobre la pared.
  3. **Palma Abierta Sostenida (*Hold to Reset*):** Gesto de seguridad con los 5 dedos extendidos durante 2 segundos para reiniciar o apagar todos los dispositivos vinculados.

---

### Fase 4: Retroalimentación Sonora Espacializada (Audio FX)
* **Objetivo:** Reforzar la inmersión del usuario cuando interactúa con objetos holográficos.
* **Tareas pendientes:**
  1. Aprovechar el endpoint `/style` y la carpeta `data/sounds/` ya existente en el backend.
  2. Implementar sintetizador de audio en el navegador (Web Audio API) para reproducir un pulso sónico (*tick/chime*) al hacer *hover* sobre un objeto interactivo y un sonido de confirmación al hacer clic.

---

### Fase 5: Empaquetado para Instalaciones Permanentes
* **Objetivo:** Desplegar el sistema en ferias, museos o exposiciones sin requerir terminal abierta ni configuración técnica por parte del operador.
* **Tareas pendientes:**
  1. Crear un instalador ejecutable de Windows mediante InnoSetup o PyInstaller que incluya el entorno virtual pre-configurado y los modelos neuronales verificados.
  2. Configurar arranque automático como servicio o acceso directo de inicio en Windows con inicio en pantalla completa secundario forzado.

---

## 📋 3. Guía de Ejecución para Nuevos Desarrolladores

Si un compañero de equipo o evaluador continúa con el desarrollo de los módulos pendientes:

1. **Revisar la arquitectura:** Leer [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) y [`docs/SETUP_GUIDE.md`](SETUP_GUIDE.md).
2. **Ejecutar en modo simulación (Mock):**
   ```powershell
   python run.py
   ```
   *El sistema funciona inmediatamente sin necesidad de hardware físico real gracias al controlador mock.*
3. **Ejecutar las pruebas unitarias antes de cualquier cambio:**
   ```powershell
   pytest tests/ -v
   ```
4. **Verificar rendimiento de GPU:**
   ```powershell
   python scripts/benchmark.py --mode all --frames 60
   ```
