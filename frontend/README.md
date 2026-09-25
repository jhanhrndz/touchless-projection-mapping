# Prototipo: Brazo Digitalizado V2 (MediaPipe + Resta de Fondo + WebSocket)

Este es un experimento avanzado e independiente dentro de `interactive-projection-mapping`.

## ¿Qué hace?
- **Resta de Fondo:** Aísla el brazo real en tiempo real eliminando el entorno de la habitación para proyectar una silueta limpia.
- **Estética Holográfica:** Aplica un filtro de color verde digitalizado eliminando texturas de piel para un efecto de brazo de energía.
- **Detección por MediaPipe Hands:** Rastrela la posición del dedo índice para la interacción matemática con los polígonos sin mostrar el esqueleto en pantalla.
- **Gesto de Clic:** Reconoce el pellizco (*pinch* entre pulgar e índice) para activar objetos.
- **Doble Ventana (Sin Desfase):**
  - `index.html` (Laptop): Controlador con el video filtrado, cursor táctico y polígonos interactivos.
  - `projection.html` (Proyector): Vista limpia que proyecta exclusivamente el brazo verde sobre fondo negro, sin figuras ni cursores.
- **WebSocket:** Envía eventos en tiempo real a `ws://localhost:8080/ws` para interactuar con los dispositivos físicos del backend.

## Cómo ejecutarlo:
1. Asegúrate de que el contenedor de Docker esté activo (`docker compose up -d`).
2. En una terminal, inicia el servidor dedicado del experimento:
   ```bash
   python experiments/digital_arm_v2/server.py
   ```
3. Abre en tu navegador:
   `http://localhost:5050` (Controlador en tu laptop).
4. Otorga permiso a la cámara web.
5. Sal del encuadre de la cámara y presiona la tecla **[ESPACIO]** para capturar y calibrar el fondo vacío.
6. Abre la ventana de proyección con el botón en pantalla y arrástrala al proyector (F11 para pantalla completa).
