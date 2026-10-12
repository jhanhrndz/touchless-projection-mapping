#!/usr/bin/env python3
"""
Herramienta de descarga automática de modelos neuronales
=========================================================
Descarga los pesos pre-entrenados para YOLO11, FastSAM, MediaPipe y RVM
si no están presentes en el directorio local.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "vision_engine" / "models"
TORCH_HUB_CHECKPOINTS = ROOT_DIR / "torch_hub_cache" / "checkpoints"

MODEL_DOWNLOADS = [
    {
        "name": "hand_landmarker.task",
        "target_dir": MODELS_DIR,
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "expected_bytes": 7819105,
    },
    {
        "name": "FastSAM-s.pt",
        "target_dir": MODELS_DIR,
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt",
        "expected_bytes": 23851578,
    },
    {
        "name": "yolo11n-seg.pt",
        "target_dir": MODELS_DIR,
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "expected_bytes": 6182636,
    },
    {
        "name": "rvm_mobilenetv3.pth",
        "target_dir": TORCH_HUB_CHECKPOINTS,
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
        "expected_bytes": 15217721,
    },
    {
        "name": "rvm_resnet50.pth",
        "target_dir": TORCH_HUB_CHECKPOINTS,
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
        "expected_bytes": 107905875,
    },
]


def download_file(url: str, dest_path: Path, expected_bytes: int | None = None) -> bool:
    """Descarga un archivo mostrando progreso."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    print(f"📥 Descargando: {dest_path.name} desde {url}")

    def progress_callback(blocks_transferred: int, block_size: int, total_size: int) -> None:
        downloaded = blocks_transferred * block_size
        if total_size > 0:
            pct = min(100.0, (downloaded / total_size) * 100.0)
            sys.stdout.write(f"\r   [{pct:5.1f}%] {downloaded / 1024 / 1024:6.1f} MB / {total_size / 1024 / 1024:6.1f} MB")
        else:
            sys.stdout.write(f"\r   {downloaded / 1024 / 1024:6.1f} MB descargados")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=progress_callback)
        sys.stdout.write("\n")
        tmp_path.replace(dest_path)
        print(f"✅ Descarga completada: {dest_path.name}")
        return True
    except Exception as e:
        sys.stdout.write("\n")
        print(f"[ERROR] Error al descargar {dest_path.name}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def main() -> None:
    print("=" * 70)
    print("  VERIFICACION Y DESCARGA DE MODELOS NEURONALES")
    print("=" * 70)

    for item in MODEL_DOWNLOADS:
        target = item["target_dir"] / item["name"]
        if target.exists() and target.stat().st_size > 0:
            size_mb = target.stat().st_size / 1024 / 1024
            print(f"  * {item['name']:<25} presente ({size_mb:6.1f} MB)")
        else:
            print(f"  [AVISO] {item['name']:<25} no encontrado. Iniciando descarga...")
            download_file(item["url"], target, item.get("expected_bytes"))

    print("\n[LISTO] Todos los modelos han sido verificados.")


if __name__ == "__main__":
    main()
