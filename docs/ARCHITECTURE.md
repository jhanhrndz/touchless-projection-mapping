# Arquitectura del Sistema - Touchless Projection Mapping

## 1. Visión General del Sistema

**Touchless Projection Mapping** es una plataforma de interacción humano-computadora (HCI) y automatización espacial que transforma superficies físicas en interfaces táctiles sin contacto. Utiliza visión artificial para detectar manos y extremidades superiores, resta de fondo para aislar la silueta humana, y proyectores convencionales para mapear digitalmente acciones sobre el mundo real.

```mermaid
flowchart TB
    subgraph VisionCore["Motor de Visión (Laptop: index.html)"]
        Cam["Cámara Web"] --> MP["MediaPipe Hands (60 FPS)"]
        Cam --> Sub["Resta de Fondo (Canvas 1920x1080)"]
        MP --> Pinch["Detección de Pellizco (Pinch Clic)"]
        Pinch --> Poly["Colisión Punto-Polígono (Ray-Casting)"]
        Sub --> FX["Filtro Digital (Verde Neón Matrix)"]
    end

    subgraph IPC["Canal de Sincronización Local"]
        BC["BroadcastChannel (hand_projection_channel)"]
    end

    subgraph ProjectionWindow["Ventana de Proyección (projection.html)"]
        BG["Capa de Video de Fondo (z-index: 1)"]
        Arm["Capa de Silueta Digital (z-index: 2, mix-blend-mode: screen)"]
        BG --> Proj["Proyector Físico en Pared"]
        Arm --> Proj
    end

    subgraph HardwareBackend["Backend de Dispositivos (backend: 8080)"]
        WS["WebSocket Hub (/ws)"]
        Dev["Controlador de Dispositivos (Kasa, ESP32, Mock)"]
        WS --> Dev
    end

    FX -->|Stream JPEG ~25 FPS| BC
    VisionCore -->|Eventos de Video (Blob, Play, Brillo)| BC
    BC --> Arm
    BC --> BG
    Poly -->|Eventos de Clic / Dispositivo| WS
```

---

## 2. Componentes Principales

### 2.1. Ventana 1: Controlador de Operador (`frontend/index.html`)
- **Propósito:** Brinda al usuario operador control completo sobre la cámara, la calibración de fondo, la carga de videos y el estado de la red.
- **Resta de Fondo:** Captura un frame inicial de referencia con la tecla `[ESPACIO]`. Cualquier objeto nuevo (el brazo del usuario) se aísla por diferencia euclidiana de color RGB.
- **MediaPipe Hands:** Procesa 21 puntos clave anatómicos para calcular el vector del pulgar al índice. Una distancia euclidiana inferior a un umbral configurado activa el evento de "pellizco" (*click*).
- **Control de Fondo:** Permite seleccionar archivos locales de video (`.mp4`, `.webm`), pausar/reanudar, limpiar y controlar el nivel de brillo para calibración óptica.

### 2.2. Ventana 2: Proyección Limpia (`frontend/projection.html`)
- **Propósito:** Diseñada para desplegarse en pantalla completa en el monitor secundario (proyector).
- **Cero Elementos Obstructivos:** No contiene cursores, cajas de diálogo, menús ni polígonos interactivos. Solo muestra la proyección luminosa final.
- **Composición Óptica:**
  - Capa 1: `<video id="bg-video">` en bucle con ajuste de brillo CSS (`filter: brightness()`).
  - Capa 2: `<img id="stream-view">` con `mix-blend-mode: screen`. El fondo negro (`#000000`) se vuelve ópticamente nulo, dejando pasar el video intacto mientras el brazo holográfico verde se superpone con resplandor aditivo.

### 2.3. IPC entre Ventanas (`BroadcastChannel`)
La sincronización se realiza completamente dentro del navegador utilizando la API estándar de `BroadcastChannel`:
- Nombre del canal: `"hand_projection_channel"`
- Sin peticiones HTTP ni retrasos de servidor: la latencia de paso de mensajes es sub-milisegundo.
- Capacidad para transferir objetos `Blob` y `File` de video directamente en memoria entre pestañas del mismo origen.

### 2.4. Backend de Dispositivos (`backend/`)
- Desarrollado sobre **FastAPI** y **WebSockets**.
- Recibe cargas útiles JSON como:
  ```json
  {
    "type": "click",
    "id": "smart_plug_1",
    "polygon_id": "fan_switch"
  }
  ```
- Despacha comandos hacia dispositivos TP-Link Kasa, microcontroladores ESP32/Arduino vía Wi-Fi/UDP, o el bus simulado en entornos de desarrollo.
