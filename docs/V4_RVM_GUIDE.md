# Guía del Pipeline V4: RVM (Robust Video Matting) + WebGL 2.0

## 1. Visión General de la Versión V4

La versión **V4** integra el modelo de matting de nivel cinematográfico **RVM (Robust Video Matting)** de PeterL1n para lograr:
1. **Recorte Fotográfico Subpíxel:** A diferencia de las redes de segmentación estándar que procesan a 256×256 y cortan los dedos, RVM genera un mapa alfa con transparencia continua (valores float de 0.0 a 1.0) que preserva **cada milímetro de cada dedo, uña, silueta y pliegue de la piel**.
2. **Memoria Recurrente Temporal:** RVM cuenta con 4 estados recurrentes (`r1, r2, r3, r4`) que recuerdan la posición del usuario entre fotogramas, eliminando el parpadeo y los bordes inestables.
3. **Inmunidad a la Luz y Fondo:** No requiere calibrar con `[ESPACIO]`. RVM fue entrenado con millones de fondos dinámicos y cambios de iluminación.
4. **Shaders Cibernéticos Activos:** Combina la silueta de alta definición con shaders WebGL 2.0 que transforman los pliegues y nudillos en **circuitos digitales fluorescentes** mediante derivadas por hardware (`dFdx`/`dFdy`).

---

## 2. Mapa de Rutas de Todas las Versiones (Rollback Garantizado)

| Versión | URL del Controlador | URL de la Proyección | Motor de Visión |
| :--- | :--- | :--- | :--- |
| **V2 (Original)** | `http://localhost:5050/index.html` | `http://localhost:5050/projection.html` | Resta de fondo RGB estática (`[ESPACIO]`) |
| **V3 (Anatómica)** | `http://localhost:5050/v3/index.html` | `http://localhost:5050/v3/projection.html` | MediaPipe Pose + Hands + Shaders WebGL |
| **V4 (Cine RVM)** | `http://localhost:5050/v4/index.html` | `http://localhost:5050/v4/projection.html` | **Robust Video Matting (ONNX) + Shaders WebGL** |

---

## 3. Estructura de Archivos V4

```text
frontend/v4/
├── index.html                   # Controlador V4 con indicador RVM y selectores
├── projection.html              # Ventana limpia para el proyector (mix-blend-mode: screen)
├── models/
│   ├── rvm_mobilenetv3_fp32.onnx  # Pesos locales RVM (14.9 MB)
│   └── rvm_mobilenetv3_fp16.onnx  # Pesos locales RVM optimizados (7.5 MB)
└── js/
    ├── rvm_engine.js            # Motor ONNX Runtime Web y gestión de memoria recurrente
    └── webgl_compositor.js      # Compositor GPU, circuitos neón y filtrado de extremidades
```
