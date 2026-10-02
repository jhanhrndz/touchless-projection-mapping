#!/usr/bin/env python3
"""Servidor HTTP aislado para el prototipo de Hand Tracking (puerto 5050).

Sirve la interfaz del controlador, la ventana de proyección y expone
el archivo real data/objects.json vía GET /api/objects sin tocar nada
del proyecto principal.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
OBJECTS_PATH = FRONTEND_DIR / "objects.json"
BACKEND_OBJECTS_PATH = ROOT_DIR / "backend" / "data" / "objects.json"
HOST = os.environ.get("FRONTEND_HOST", "0.0.0.0")
PORT = int(os.environ.get("FRONTEND_PORT", os.environ.get("PORT", "5050")))


class HandTrackingHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/objects":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()

            target_path = OBJECTS_PATH if OBJECTS_PATH.exists() else BACKEND_OBJECTS_PATH
            if target_path.exists():
                try:
                    data = target_path.read_bytes()
                    self.wfile.write(data)
                    return
                except Exception as e:
                    err = json.dumps({"error": str(e)}).encode()
                    self.wfile.write(err)
                    return
            self.wfile.write(b"[]")
            return

        super().do_GET()

    def end_headers(self) -> None:
        # Permitir CORS para desarrollo
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), HandTrackingHandler)
    print("=" * 65)
    print(f"  Prototipo Hand Tracking (2 Ventanas)")
    print(f"  Servidor iniciado en: http://localhost:{PORT}")
    print(f"  Objetos vinculados desde: {OBJECTS_PATH}")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()
