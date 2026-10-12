# Guía del Pipeline V3: WebGL 2.0, MediaPipe Pose y Shaders

## 1. Motivación y Arquitectura

La versión V3 surge para resolver los problemas fundamentales de la resta de fondo tradicional:
1. **Inmunidad a la Iluminación:** No depende de un fotograma inicial estático (`[ESPACIO]`). Es insensible a la auto-exposición de la cámara web, sombras de la habitación y rebotes de luz del proyector sobre la pared.
2. **Aislamiento Exclusivo de Brazos:** Utiliza la inferencia anatómica de **MediaPipe Pose** para calcular las articulaciones de los brazos (hombro, codo, muñeca) y combina esa información en un **Shader WebGL 2.0** con la máscara de segmentación corporal. El torso, la cabeza y el fondo se descartan en la GPU.
3. **Fluidez Extrema (50-60 FPS):** Sustituye la compresión síncrona `canvas.toDataURL('image/jpeg')` por transferencia de texturas acelerada por hardware mediante `createImageBitmap()` sobre `BroadcastChannel`.

---

## 2. Acceso y Convivencia de Versiones (Rollback Garantizado)

Ambas versiones se encuentran desplegadas y disponibles en el servidor local:

| Versión | URL del Controlador | URL del Proyector | Tecnología |
| :--- | :--- | :--- | :--- |
| **V2 (Estable)** | `http://localhost:5050/index.html` | `http://localhost:5050/projection.html` | Resta RGB estática + MediaPipe Hands |
| **V3 (Avanzada)** | `http://localhost:5050/v3/index.html` | `http://localhost:5050/v3/projection.html` | MediaPipe Pose + Shaders WebGL 2.0 + Manos |

---

## 3. Estructura de Archivos V3

```text
frontend/v3/
├── index.html                  # Interfaz de control del operador (Cámara + WebGL + Menú de Shaders)
├── projection.html             # Ventana limpia para el proyector (mix-blend-mode: screen)
└── js/
    ├── webgl_renderer.js       # Motor WebGL 2.0, cálculo de cápsula (SDF) y shaders de efectos
    └── vision_manager.js       # Orquestador de MediaPipe Pose (segmentación) y Hands (gestos)
```

---

## 4. Efectos Disponibles en el Shader

El operador puede cambiar el estilo visual del brazo en caliente desde la barra superior:
1. **Matrix Plasma (Verde Neón):** Núcleo fluorescente modulado por luminancia con realce de bordes mediante `smoothstep()`.
2. **Cyberpunk Cyan:** Estética cibernética en azul cian con líneas de escaneo (*scanlines*) horizontales animadas.
3. **Holograma Pulsante:** Variación sinusoidal en el brillo con parpadeo digital sci-fi.
4. **Silueta Sólida:** Verde puro minimalista de alto contraste optimizado para proyectores con baja luminosidad.

---

## 5. Control de Grosor del Brazo

El control deslizante **"💪 Grosor"** (valores de 6 a 18, por defecto 11) ajusta en tiempo real el radio de la cápsula cilíndrica del shader (`u_arm_radius`). Esto permite calibrar el volumen del brazo para usuarios con ropa holgada o distintas distancias a la cámara.
