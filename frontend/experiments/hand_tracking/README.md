# Prototipo Aislado: Control por Mano (MediaPipe + WebSocket)

Este es un experimento independiente del proyecto `interactive-projection-mapping`.

## ¿Qué hace?
- Utiliza la cámara web de tu laptop para detectar tu mano en tiempo real mediante **Google MediaPipe Hands**.
- Coloca un cursor luminoso en la punta de tu **dedo índice**.
- Detecta la entrada/salida (hover) sobre el polígono de la lámpara.
- Detecta el gesto de pellizco (*pinch* entre el pulgar y el índice) como un **clic**.
- Se conecta mediante **WebSocket** nativo a `ws://localhost:8080/ws` para accionar el backend sin haber modificado nada del código fuente original.

## Cómo ejecutarlo:
1. Asegúrate de que el contenedor de Docker del proyecto esté activo (`docker compose up -d`).
2. En una terminal, inicia el servidor estático en el puerto 5050:
   ```bash
   python -m http.server 5050 --directory experiments/hand_tracking
   ```
3. Abre en tu navegador:
   `http://localhost:5050`
4. Otorga permiso a la cámara web.
