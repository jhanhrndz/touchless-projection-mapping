# Experimentos de Control por Gestos y Visión

Esta carpeta contiene prototipos y extensiones independientes para interactuar con el sistema de proyección interactiva utilizando visión por computadora (cámara web + MediaPipe) sin modificar el núcleo del proyecto original.

## Prototipos Disponibles

1. **`digital_arm_v2/` (Versión Recomendada):**
   - Brazo digitalizado mediante algoritmo de resta de fondo (background subtraction).
   - Estética verde holográfica.
   - Proyección limpia en la pared (únicamente el brazo, sin UI ni cursores).
   - Controlador en laptop con precisión matemática 1:1 y puntero táctico.

2. **`hand_tracking/` (Prototipo Inicial):**
   - Primer prototipo funcional de control por gestos con visualización del esqueleto de MediaPipe Hands y retícula clásica.

3. **`digital_arm/` (V1):**
   - Primera iteración experimental de silueta digital.
