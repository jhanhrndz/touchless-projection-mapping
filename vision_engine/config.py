"""Configuración centralizada para el motor de visión y proyección interactiva."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Rutas base del proyecto
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
MODELS_DIR = PACKAGE_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"
TORCH_HUB_DIR = ROOT_DIR / "torch_hub_cache"
BACKEND_DATA_DIR = ROOT_DIR / "backend" / "data"

# Rutas de modelos predeterminados
DEFAULT_YOLO_PATH = MODELS_DIR / "yolo11n-seg.pt"
DEFAULT_YOLO_PARSE_PATH = MODELS_DIR / "yolo-human-parse.pt"
DEFAULT_FASTSAM_PATH = MODELS_DIR / "FastSAM-s.pt"
DEFAULT_TASK_PATH = MODELS_DIR / "hand_landmarker.task"
OBJECTS_PATH = FRONTEND_DIR / "objects.json" if (FRONTEND_DIR / "objects.json").exists() else (BACKEND_DATA_DIR / "objects.json")


@dataclass
class EngineConfig:
    """Configuración tipada del motor de visión sin números mágicos."""
    # Selección de modelo
    default_model: str = "rvm_mobilenet"  # RVM MobileNetV3 por defecto (óptimo para cuerpo completo)
    arm_only: bool = False                # Segmentación nativa de cuerpo completo
    guided_filter: bool = True            # Filtro guiado rápido en GPU (activado por defecto para máxima nitidez)
    gf_radius: int = 4                    # Radio de ventana del filtro guiado
    gf_eps: float = 1e-2                  # Regularización épsilon

    # Parámetros YOLO
    yolo_conf: float = 0.28
    yolo_imgsz: int = 640
    yolo_model_path: Path = DEFAULT_YOLO_PATH

    # Parámetros YOLO-Human-Parse
    yolo_parse_conf: float = 0.05
    yolo_parse_imgsz: int = 384
    yolo_parse_model_path: Path = DEFAULT_YOLO_PARSE_PATH

    # Parámetros FastSAM
    fastsam_conf: float = 0.20
    fastsam_imgsz: int = 640
    fastsam_model_path: Path = DEFAULT_FASTSAM_PATH

    # Parámetros RVM
    rvm_dsr_mobilenet: float = 0.50
    rvm_dsr_resnet: float = 0.30
    torch_hub_dir: Path = TORCH_HUB_DIR

    # Resta de Fondo
    bg_threshold: int = 35

    # MediaPipe Hand Tracker
    hand_landmarker_path: Path = DEFAULT_TASK_PATH
    mp_min_detection_conf: float = 0.50
    mp_min_presence_conf: float = 0.50
    mp_min_tracking_conf: float = 0.50

    # Servidor HTTP y Red
    host: str = "0.0.0.0"
    http_port: int = 5051
    max_post_body_bytes: int = 65536

    # Visualización
    show_hud: bool = False
    native_projector: bool = False
    source: str = ""                # Ruta a archivo .mp4 ("" = cámara física)

    @classmethod
    def from_env(cls) -> EngineConfig:
        """Carga configuración sobrescribiendo con variables de entorno si existen."""
        cfg = cls()
        if "VISION_MODEL" in os.environ:
            cfg.default_model = os.environ["VISION_MODEL"].lower()
        if "VISION_PORT" in os.environ:
            try:
                cfg.http_port = int(os.environ["VISION_PORT"])
            except ValueError:
                pass
        if "VISION_ARM_ONLY" in os.environ:
            cfg.arm_only = os.environ["VISION_ARM_ONLY"].lower() in ("1", "true", "yes")
        if "VISION_BG_THRESHOLD" in os.environ:
            try:
                cfg.bg_threshold = int(os.environ["VISION_BG_THRESHOLD"])
            except ValueError:
                pass
        return cfg
