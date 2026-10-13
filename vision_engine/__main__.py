"""Punto de entrada CLI para ejecutar el motor de visión y servidor web."""

import argparse
import logging
import sys
from pathlib import Path

from vision_engine.config import EngineConfig
from vision_engine.engine import VisionEngine
from vision_engine.server import start_http_server

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision_engine")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Touchless Projection Mapping - Motor de Visión por Computadora y Holograma"
    )
    parser.add_argument("camera_index", nargs="?", type=int, default=None, help="Índice de cámara (0, 1, 2...)")
    parser.add_argument("--source", type=str, default=None, help="Ruta a archivo .mp4 offline para pruebas reproducibles")
    parser.add_argument("--port", type=int, default=5051, help="Puerto HTTP (por defecto 5051)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host HTTP (por defecto 0.0.0.0)")
    parser.add_argument("--native-projector", action="store_true", help="Abrir ventana nativa OpenCV en monitor secundario")
    parser.add_argument("--show-window", action="store_true", help="Mostrar ventana de depuración OpenCV")
    parser.add_argument("--model", type=str, default="rvm_mobilenet", help="Modelo inicial (rvm_mobilenet, rvm_resnet, yolo, background_subtraction)")
    args = parser.parse_args()

    cfg = EngineConfig.from_env()
    cfg.http_port = args.port
    cfg.host = args.host
    if args.model:
        cfg.default_model = args.model
    if args.source:
        cfg.source = args.source

    print("=" * 72)
    print("  🖐️ TOUCHLESS PROJECTION MAPPING - MOTOR DE VISIÓN Y HOLOGRAMA")
    print("=" * 72)

    # Iniciar servidor HTTP en segundo plano
    server, actual_port = start_http_server(host=cfg.host, port=cfg.http_port)
    print(f"  🖥️  Panel Operador:   http://localhost:{actual_port}/controller.html")
    print(f"  📽️  Pantalla Proyector: http://localhost:{actual_port}/projector.html")
    print("=" * 72)

    engine = VisionEngine(config=cfg)
    try:
        engine.run(
            source=args.source if args.source else args.camera_index,
            native_projector=args.native_projector,
            show_window=args.show_window,
        )
    except KeyboardInterrupt:
        print("\nMotor detenido por el usuario.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
