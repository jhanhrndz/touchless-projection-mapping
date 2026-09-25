# Touchless Projection Mapping 🖐️✨

> Sistema de mapeo de proyección interactiva potenciado por visión artificial en tiempo real, detección de gestos y automatización de dispositivos físicos de hardware.

---

## 🌟 Descripción General

**Touchless Projection Mapping** convierte cualquier pared u objeto físico en una interfaz interactiva sin necesidad de contacto físico, sensores vestibles ni costosas cámaras de profundidad. 

A través de una cámara web estándar y **Google MediaPipe Hands**, el sistema rastrea las manos en tiempo real, aísla el brazo del usuario mediante algoritmos de **resta de fondo (background subtraction)**, proyecta una silueta holográfica sobre la pared y activa dispositivos físicos reales (enchufes inteligentes, luces, actuadores) mediante eventos de red en tiempo real sobre **WebSockets**.

---

## 🏛️ Arquitectura del Monorepo

El proyecto está organizado en dos componentes desacoplados pero integrados:

```text
touchless-projection-mapping/
├── frontend/                        ← NÚCLEO DE VISIÓN Y GESTOS (Cliente)
│   ├── index.html                   (Controlador táctico para laptop con cámara y puntero)
│   ├── projection.html              (Ventana limpia para el proyector sin cursores)
│   ├── server.py                    (Servidor HTTP local rápido para desarrollo en :5050)
│   ├── objects.json                 (Zonas y polígonos interactivos mapeados)
│   ├── README.md                    (Documentación específica del módulo de visión)
│   └── experiments/                 (Histórico de prototipos: hand_tracking y digital_arm)
│
├── backend/                         ← CONTROL DE HARDWARE Y DISPOSITIVOS (Servidor)
│   ├── src/                         (Servidor FastAPI, hub de WebSockets y drivers)
│   ├── data/                        (Configuración de dispositivos devices.yaml / mock)
│   ├── docker/                      (Scripts y entorno Docker)
│   ├── docker-compose.yml           (Levanta el backend en puerto 8080)
│   └── README.md                    (Documentación del sistema de dispositivos de Brendan)
│
├── .gitignore
├── LICENSE
└── README.md
```

### 1. `frontend/` (Motor de Visión y Brazo Digital)
* **Visión Artificial:** Detección de puntos articulares (landmarks) a 60 FPS con MediaPipe Hands.
* **Resta de Fondo Dinámica:** Aísla el brazo humano en tiempo real eliminando el fondo de la habitación con calibración instantánea mediante la tecla `[ESPACIO]`.
* **Brazo Holográfico:** Filtro digitalizado (verde Matrix) que oculta la textura de la piel para una estética limpia.
* **Modelo de Doble Ventana:**
  * **Controlador (`index.html`):** Para tu laptop. Muestra el video de monitoreo, retícula de puntería y zonas interactivas coloreadas.
  * **Proyección (`projection.html`):** Para la pared. Fondo negro absoluto que muestra únicamente el brazo verde proyectado.
* **Gesto de Clic:** Detección de pellizco (*pinch* entre el pulgar y el índice) para interactuar con los polígonos.

### 2. `backend/` (Automatización de Hardware)
* **WebSocket Hub:** Recibe eventos de clic y hover emitidos por el frontend.
* **Drivers de Dispositivos:** Integración nativa con enchufes inteligentes (Kasa / TP-Link), actuadores de estanterías (ESP32 / Raspberry Pi), UDP y HTTP.
* **Modo Simulación:** Ejecución segura sin hardware real conectado mediante mocks que registran eventos en consola (`devices.dev.yaml`).
* **Docker:** Listo para levantar con un solo comando.

---

## 🚀 Inicio Rápido

### Requisitos Previos
* Python 3.10+ (o Docker Desktop)
* Una cámara web
* (Opcional) Un proyector conectado como pantalla extendida

---

### Paso 1: Iniciar el Backend (Servidor de Hardware)

Con **Docker** (Recomendado):
```bash
cd backend
docker compose up -d
```
O con **Python** localmente:
```bash
cd backend
uv sync
uv run python -m runtime --devices data/devices.dev.yaml
```
El backend estará escuchando conexiones WebSocket en `ws://localhost:8080/ws`.

---

### Paso 2: Iniciar el Frontend (Visión y Gestos)

En otra terminal:
```bash
cd frontend
python server.py
```

1. Abre en tu laptop: **`http://localhost:5050/`** y autoriza el uso de la cámara web.
2. Sal del encuadre de la cámara y presiona la tecla **`[ESPACIO]`** para calibrar el fondo vacío.
3. Haz clic en el botón **"🖥️ Abrir Ventana de Proyección"** (o entra a `http://localhost:5050/projection.html`).
4. Arrastra esa ventana a la pantalla de tu proyector y presiona **F11** para pantalla completa.
5. ¡Entra con tu brazo, apunta a los objetos mapeados y junta los dedos para interactuar!

---

## 🤝 Créditos y Reconocimientos

* El motor de hardware, servidor FastAPI y drivers en `backend/` están basados en el proyecto [interactive-projection-mapping](https://github.com/buildwbrendan/interactive-projection-mapping) de **Brendan**.
* El cliente de visión artificial, algoritmo de resta de fondo, brazo digitalizado y arquitectura de doble ventana en `frontend/` fueron diseñados y desarrollados de forma independiente.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
