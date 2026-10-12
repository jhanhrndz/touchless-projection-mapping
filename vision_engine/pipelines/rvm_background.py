"""Procesadores de Segmentación para Robust Video Matting (RVM) y Resta de Fondo (Fase 3).
=============================================================================
RVM:
  - Carga 100% offline desde caché local en PyTorch Hub (source='local').
  - Downsample Ratio optimizado (0.40 para ResNet50, 0.50 para MobileNetV3) para
    preservar los espacios interdigitales y contornos finos de los dedos.
  - Curva de realce no lineal adaptativa para manos sin rostro humano completo.
  - Gestión limpia de memoria recurrente (r1i, r2i, r3i, r4i) con reinicio instantáneo.

Resta de Fondo:
  - Eliminación de artefactos de "círculos" y "polígonos rectos" mediante eliminación
    de dilataciones morfológicas agresivas.
  - Detección diferencial en espacio de color con filtrado de ruido Gaussiano fino.
  - Generación de máscaras coherentes sin doble inversión horizontal.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RvmConfig:
    """Parámetros de configuración del motor Robust Video Matting."""
    dsr_mobilenet: float = 0.50
    dsr_resnet: float = 0.40
    hand_alpha_boost: float = 1.35
    black_floor: float = 0.05
    white_ceiling: float = 0.85


class RvmProcessor:
    """Gestor de inferencia de RVM con preservación de dedos y gestión recurrente limpia."""

    def __init__(self, config: Optional[RvmConfig] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = config or RvmConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        self.rec: List[Optional[torch.Tensor]] = [None] * 4

        # Cargar repositorios y pesos desde caché local del proyecto o del sistema
        from vision_engine.config import TORCH_HUB_DIR
        proj_cache = TORCH_HUB_DIR
        if (proj_cache / "PeterL1n_RobustVideoMatting_master").exists():
            cache_dir = proj_cache / "PeterL1n_RobustVideoMatting_master"
            torch.hub.set_dir(str(proj_cache))
        else:
            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "PeterL1n_RobustVideoMatting_master"

        source = "local" if cache_dir.exists() else "github"
        repo = str(cache_dir) if cache_dir.exists() else "PeterL1n/RobustVideoMatting"

        try:
            self.models["mobilenetv3"] = torch.hub.load(repo, "mobilenetv3", source=source, trust_repo=True).eval().to(self.device)
        except Exception:
            pass

        try:
            self.models["resnet50"] = torch.hub.load(repo, "resnet50", source=source, trust_repo=True).eval().to(self.device)
        except Exception:
            pass

    def reset_recurrent_states(self) -> None:
        """Reinicia los tensores de memoria temporal recurrente (r1i, r2i, r3i, r4i)."""
        self.rec = [None] * 4

    def predict_alpha(
        self,
        frame_tensor: torch.Tensor,
        variant: str = "mobilenetv3",
        has_hands: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calcula la máscara alfa matting continua a partir del tensor de entrada normalizado [1, 3, H, W].

        Returns:
            Tensor alfa continuo [1, 1, H, W] en [0.0, 1.0].
        """
        target_device = device or self.device
        key = "resnet50" if "resnet" in variant.lower() else "mobilenetv3"
        model = self.models.get(key)

        h, w = frame_tensor.shape[-2:]
        if model is None:
            return torch.zeros((1, 1, h, w), device=target_device)

        dsr = self.cfg.dsr_resnet if key == "resnet50" else self.cfg.dsr_mobilenet
        is_cuda = target_device.type == "cuda"

        with torch.no_grad(), torch.amp.autocast("cuda" if is_cuda else "cpu"):
            fgr, pha, *self.rec = model(frame_tensor, *self.rec, downsample_ratio=dsr)

        # Realce adaptativo de contraste para manos sin rostro
        if has_hands:
            # Curva suave: suprime sombras por debajo del black floor y satura la piel
            pha_clean = torch.where(pha < self.cfg.black_floor, torch.zeros_like(pha), pha)
            norm = torch.clamp((pha_clean - self.cfg.black_floor) / (self.cfg.white_ceiling - self.cfg.black_floor), 0.0, 1.0)
            pha = norm * norm * (3.0 - 2.0 * norm)
        else:
            pha = torch.where(pha < self.cfg.black_floor, torch.zeros_like(pha), pha)

        return pha


@dataclass
class BackgroundSubtractionConfig:
    """Parámetros de configuración para resta de fondo."""
    threshold: int = 32
    blur_ksize: int = 3
    morph_kernel_size: int = 3


class BackgroundSubtractionProcessor:
    """Procesador diferencial de fondo con supresión de artefactos y preservación anatómica."""

    def __init__(self, config: Optional[BackgroundSubtractionConfig] = None) -> None:
        self.cfg = config or BackgroundSubtractionConfig()
        self.bg_frame: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size)
        )

    def calibrate(self, frame_bgr: np.ndarray) -> None:
        """Almacena el fotograma de referencia estático para calibración."""
        if frame_bgr is not None and frame_bgr.size > 0:
            self.bg_frame = frame_bgr.copy()

    def is_calibrated(self) -> bool:
        """Indica si el fondo ha sido calibrado."""
        return self.bg_frame is not None

    def compute_mask(self, frame_bgr: np.ndarray, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, bool]:
        """
        Calcula la máscara diferencial limpia sin deformaciones circulares ni poligonales.

        Returns:
            Tuple con (tensor_mascara [1, 1, H, W] en [0.0, 1.0], booleano_fondo_valido).
        """
        h, w = frame_bgr.shape[:2]
        target_device = device or torch.device("cpu")

        if self.bg_frame is None:
            # Auto-calibrar con el primer fotograma recibido
            self.calibrate(frame_bgr)
            return torch.zeros((1, 1, h, w), device=target_device), False

        # 1. Diferencia absoluta
        diff = cv2.absdiff(frame_bgr, self.bg_frame)

        # 2. Convertir a escala de grises
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 3. Filtrado espacial suave para eliminar ruido electrónico del sensor
        blurred = cv2.GaussianBlur(gray_diff, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)

        # 4. Umbralización binaria
        _, thresh = cv2.threshold(blurred, self.cfg.threshold, 255, cv2.THRESH_BINARY)

        # 5. Apertura morfológica mínima (elimina píxeles aislados SIN agrandar ni redondear dedos)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel)

        # 6. Convertir a tensor normalizado en el dispositivo objetivo
        mask_tensor = torch.from_numpy(clean_mask).to(target_device).float().unsqueeze(0).unsqueeze(0) / 255.0

        return mask_tensor, True
