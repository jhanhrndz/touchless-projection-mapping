# Guía del Sistema de Video de Fondo

## 1. Fundamentos y Física Óptica de Proyección

En el mapeo de proyección física, la luz es **aditiva**. Los proyectores comerciales (DLP, 3LCD, láser) no proyectan color "negro", sino ausencia de luz:
- `#000000` (negro) = La lámpara/diodo no emite luz, dejando visible la superficie original.
- `#00FF00` (verde puro) = La lámpara emite intensidad máxima en el canal verde.

Cuando se reproduce un video con áreas blancas o muy luminosas de fondo, la lámpara proyecta luz intensa en esas regiones. Si el brazo digital se proyecta sobre una zona blanca, la luz verde se sumará al blanco, disminuyendo el contraste percibido por el ojo humano.

### Soluciones Implementadas:
1. **Modo de Fusión (`mix-blend-mode: screen`):**
   Garantiza que cualquier píxel negro proveniente del stream JPEG de la cámara sea 100% transparente para el video de fondo, sumando el valor lumínico del brazo digital sin crear artefactos de borde.
2. **Control Dinámico de Brillo (`brightness(10% - 100%)`):**
   El operador cuenta con un deslizador en el controlador (`index.html`) para atenuar el video de fondo (por defecto 70%). Esto preserva la saturación del brazo holográfico verde sin perder los detalles del video.

---

## 2. Formatos y Características Recomendadas de Video

| Parámetro | Recomendación Óptima | Nota |
| :--- | :--- | :--- |
| **Contenedor / Códec** | `.mp4` (H.264 / AAC) o `.webm` (VP9) | Soportado nativamente por hardware en navegadores modernos. |
| **Resolución** | 1920x1080 (Full HD) | Coincide 1:1 con la relación de aspecto del proyector y el canvas. |
| **Paleta de Colores** | Fondos oscuros, temas espaciales, texturas sci-fi o cibernéticas | Maximiza el impacto visual del brazo fluorescente. |
| **Cuadros por Segundo** | 30 FPS o 60 FPS | La aceleración GPU del navegador se encarga de la reproducción sin afectar la CPU de visión. |

---

## 3. Protocolo de Mensajería `BroadcastChannel`

Ambas ventanas se comunican en el canal `"hand_projection_channel"` mediante los siguientes eventos:

### 3.1. `set_bg_video`
Emitido por `index.html` al cargar un video local:
```javascript
{
  type: "set_bg_video",
  blob: File,            // Objeto Blob/File del video seleccionado
  name: "universo.mp4",  // Nombre del archivo
  brightness: 70         // Porcentaje de brillo (10..100)
}
```

### 3.2. `toggle_bg_video`
Alterna entre pausa y reproducción del video en el proyector:
```javascript
{
  type: "toggle_bg_video"
}
```

### 3.3. `update_bg_video`
Actualiza el brillo o filtros ópticos en caliente sin reiniciar la reproducción:
```javascript
{
  type: "update_bg_video",
  brightness: 60
}
```

### 3.4. `clear_bg_video`
Detiene el video, libera los recursos en memoria (`URL.revokeObjectURL`) y oculta el elemento:
```javascript
{
  type: "clear_bg_video"
}
```

### 3.5. `projection_ready`
Emitido por `projection.html` cuando se abre o recarga para solicitar el estado actual del video al controlador.
