"""Configuración global de pytest para la suite de pruebas de vision_engine."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VISION_ENGINE_DIR = ROOT_DIR / "vision_engine"
PIPELINES_DIR = VISION_ENGINE_DIR / "pipelines"

for p in [str(ROOT_DIR), str(VISION_ENGINE_DIR), str(PIPELINES_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
