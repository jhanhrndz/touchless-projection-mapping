"""Servidor HTTP y streaming MJPEG de baja latencia para vision_engine."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from vision_engine.config import FRONTEND_DIR, OBJECTS_PATH, BACKEND_DATA_DIR

if TYPE_CHECKING:
    from vision_engine.config import EngineConfig

logger = logging.getLogger("vision_engine.server")

# Asegurar mimetypes para modelos ONNX y fuentes web
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/octet-stream", ".pt")
mimetypes.add_type("application/octet-stream", ".task")


class EngineState:
    """Estado global compartido entre el bucle de visión y el servidor HTTP."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.latest_jpeg: Optional[bytes] = None
        self.latest_controller_jpeg: Optional[bytes] = None
        self.latest_raw_jpeg: Optional[bytes] = None
        self.latest_raw_bgr = None
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.confidence: float = 0.0
        self.backbone_name: str = "RVM MobileNetV3 (CUDA - 60 FPS)"
        self.requested_mode: Optional[str] = None
        self.reset_requested: bool = False
        self.guided_filter_on: bool = True
        self.arm_only: bool = False
        self.bg_frame = None
        self.bg_threshold: int = 35
        self.hands_count: int = 0
        self.show_hud: bool = False
        self.running: bool = True
        self.active_monitor: int = 0
        self.frame_count: int = 0
        self.cursor = {
            "detected": False,
            "x": 0.0,
            "y": 0.0,
            "is_pinching": False,
            "pinch_dist": 0.0,
        }

    @property
    def active_model(self) -> str:
        with self.lock:
            b = (self.requested_mode or self.backbone_name).lower()
        if "parse" in b or "human_parse" in b:
            return "yolo_human_parse"
        elif "mobilenet" in b or "rvm_m" in b:
            return "rvm_mobilenet"
        elif "resnet" in b or "rvm_r" in b:
            return "rvm_resnet"
        elif "yolo" in b:
            return "yolo"
        elif "background" in b or "fondo" in b or "resta" in b:
            return "background_subtraction"
        elif "fastsam" in b:
            return "fastsam"
        return "rvm_mobilenet"

    @active_model.setter
    def active_model(self, val: str) -> None:
        with self.lock:
            self.requested_mode = str(val)

    @property
    def guided_filter(self) -> bool:
        with self.lock:
            return self.guided_filter_on

    @guided_filter.setter
    def guided_filter(self, val: bool) -> None:
        with self.lock:
            self.guided_filter_on = bool(val)

    @property
    def is_running(self) -> bool:
        with self.lock:
            return self.running

    @is_running.setter
    def is_running(self, val: bool) -> None:
        with self.lock:
            self.running = bool(val)


# Instancia compartida por defecto
STATE = EngineState()
# Aliases para compatibilidad con tests
ServerState = EngineState
SERVER_STATE = STATE


class VisionEngineHTTPHandler(SimpleHTTPRequestHandler):
    """Manejador HTTP para la API, streaming y archivos estáticos del frontend."""

    MAX_POST_BODY_BYTES: int = 65536

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_HEAD(self) -> None:
        if self.path in ["/", "/index.html"]:
            self.send_response(302)
            self.send_header("Location", "/controller.html")
            self.end_headers()
            return
        if self.path == "/projection.html":
            self.send_response(302)
            self.send_header("Location", "/projector.html")
            self.end_headers()
            return
        super().do_HEAD()

    def do_GET(self) -> None:
        # Redirigir URLs heredadas a las vistas canónicas activas
        if self.path in ["/", "/index.html"]:
            self.send_response(302)
            self.send_header("Location", "/controller.html")
            self.end_headers()
            return

        if self.path == "/projection.html":
            self.send_response(302)
            self.send_header("Location", "/projector.html")
            self.end_headers()
            return

        # 1. API de objetos
        if self.path == "/api/objects":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            if OBJECTS_PATH.exists():
                self.wfile.write(OBJECTS_PATH.read_bytes())
            elif (BACKEND_DATA_DIR / "objects.json").exists():
                self.wfile.write((BACKEND_DATA_DIR / "objects.json").read_bytes())
            else:
                self.wfile.write(b"[]")
            return

        # 2. API de telemetría de GPU y modelos
        if self.path == "/api/status":
            try:
                import torch
                has_cuda = torch.cuda.is_available()
                dev_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
                vram = round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if has_cuda else 0.0
            except Exception:
                dev_name = "CPU"
                vram = 0.0

            with STATE.lock:
                status_data = {
                    "fps": round(STATE.fps, 1),
                    "latency_ms": round(STATE.latency_ms, 1),
                    "confidence": round(STATE.confidence * 100, 1) if STATE.confidence > 0.0 else None,
                    "backbone": STATE.backbone_name,
                    "active_model": STATE.active_model,
                    "device": dev_name,
                    "vram_mb": vram,
                    "guided_filter": STATE.guided_filter_on,
                    "arm_only": STATE.arm_only,
                    "is_calibrated": STATE.bg_frame is not None,
                }
            payload = json.dumps(status_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # 3. API de posición de cursor
        if self.path == "/api/cursor":
            with STATE.lock:
                cur = dict(STATE.cursor)
            payload = json.dumps(cur).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # 4. Stream SSE para eventos de cursor
        if self.path == "/cursor_stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            while STATE.running:
                with STATE.lock:
                    cur = dict(STATE.cursor)
                try:
                    payload = f"data: {json.dumps(cur)}\n\n".encode("utf-8")
                    self.wfile.write(payload)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.016)
            return

        # 5. Stream de video MJPEG (/stream para proyector limpio, /stream?controller=1 para panel de control)
        if self.path.startswith("/stream"):
            is_controller = "controller" in self.path
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()

            while STATE.running:
                with STATE.lock:
                    if is_controller and STATE.latest_controller_jpeg is not None:
                        frame = STATE.latest_controller_jpeg
                    else:
                        frame = STATE.latest_jpeg
                if frame is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.016)
            return

        # 6. Fotograma crudo bajo demanda
        if self.path.startswith("/api/frame"):
            with STATE.lock:
                frame = STATE.latest_raw_jpeg
            if frame is not None:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(503)
                self.end_headers()
            return

        super().do_GET()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0

        if length > self.MAX_POST_BODY_BYTES:
            try:
                self.rfile.read(min(length, 1024 * 1024))
            except Exception:
                pass
            self.send_response(413)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'{"error": "Payload Too Large (max 64 KB)"}')
            self.close_connection = True
            return

        body = self.rfile.read(length) if length > 0 else b""

        # 1. Conmutar modelo
        if self.path == "/api/model":
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
                model_name = data.get("model", "")
                if model_name:
                    with STATE.lock:
                        STATE.requested_mode = model_name
                        STATE.active_model = model_name
                    resp_b = json.dumps({"status": "ok", "success": True, "requested": model_name}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_b)))
                    self.end_headers()
                    self.wfile.write(resp_b)
                    return
            except Exception as e:
                err_b = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_b)))
                self.end_headers()
                self.wfile.write(err_b)
                return

        # 2. Calibrar fondo
        if self.path == "/api/calibrate":
            with STATE.lock:
                if STATE.latest_raw_bgr is not None:
                    STATE.bg_frame = STATE.latest_raw_bgr.copy()
            resp_b = b'{"status": "calibrated", "success": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_b)))
            self.end_headers()
            self.wfile.write(resp_b)
            return

        # 3. Ajustes dinámicos
        if self.path == "/api/settings":
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
                with STATE.lock:
                    if "arm_only" in data:
                        STATE.arm_only = bool(data["arm_only"])
                    if "guided_filter" in data:
                        STATE.guided_filter_on = bool(data["guided_filter"])
                    if "bg_threshold" in data:
                        STATE.bg_threshold = int(data["bg_threshold"])
                    if data.get("reset_memory"):
                        STATE.reset_requested = True
                    gf_val = STATE.guided_filter_on
                    arm_val = STATE.arm_only
                resp_b = json.dumps({"status": "updated", "success": True, "guided_filter": gf_val, "arm_only": arm_val}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_b)))
                self.end_headers()
                self.wfile.write(resp_b)
                return
            except Exception as e:
                err_b = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_b)))
                self.end_headers()
                self.wfile.write(err_b)
                return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def end_headers(self) -> None:
        # CORS seguro para desarrollo local
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        # Suprimir logs ruidosos de /stream y /api/status para no ensuciar consola
        if args and any(p in str(args[0]) for p in ["/api/status", "/stream", "/cursor_stream", "/api/frame"]):
            return
        super().log_message(format, *args)


# Aliases para compatibilidad con código de pruebas
ProjectionHTTPHandler = VisionEngineHTTPHandler
EngineHTTPHandler = VisionEngineHTTPHandler
MAX_POST_BODY_BYTES = VisionEngineHTTPHandler.MAX_POST_BODY_BYTES


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Servidor multihilo silencioso ante desconexiones repentinas de clientes."""

    def handle_error(self, request, client_address) -> None:
        pass


def start_http_server(host: str = "0.0.0.0", port: int = 5051) -> tuple[QuietThreadingHTTPServer, int]:
    """Inicia el servidor HTTP en el puerto especificado o busca puertos alternos."""
    ports_to_try = [port, 5052, 5053, 5055, 5060, 5070]
    for p in ports_to_try:
        try:
            server = QuietThreadingHTTPServer((host, p), VisionEngineHTTPHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            logger.info("Servidor HTTP activo en: http://localhost:%d", p)
            return server, p
        except OSError:
            continue
    raise RuntimeError(f"No se pudo enlazar ningún puerto entre {ports_to_try}")
