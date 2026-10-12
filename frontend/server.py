#!/usr/bin/env python3
"""Servidor HTTP unificado para el controlador y proyector de Hand Tracking.

Sirve las interfaces del controlador y de proyección, expone la API de objetos,
telemetría en vivo, conmutación de modelos y streaming MJPEG.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from vision_engine.server import VisionEngineHTTPHandler, STATE, ServerState
    HAS_VISION_ENGINE = True
except ImportError:
    HAS_VISION_ENGINE = False

OBJECTS_PATH = FRONTEND_DIR / "objects.json"
BACKEND_OBJECTS_PATH = ROOT_DIR / "backend" / "data" / "objects.json"
HOST = os.environ.get("FRONTEND_HOST", "0.0.0.0")
PORT = int(os.environ.get("FRONTEND_PORT", os.environ.get("PORT", "5051")))


def main() -> None:
    handler = VisionEngineHTTPHandler if HAS_VISION_ENGINE else SimpleHTTPRequestHandler
    server = ThreadingHTTPServer((HOST, PORT), handler)
    print("=" * 65)
    print("  🚀 Touchless Projection Mapping - Servidor Web")
    print(f"  Servidor iniciado en: http://localhost:{PORT}/controller.html")
    print(f"  Proyector en:         http://localhost:{PORT}/projector.html")
    print(f"  Modo de motor:        {'Vision Engine (Completo)' if HAS_VISION_ENGINE else 'Estático (Básico)'}")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()
