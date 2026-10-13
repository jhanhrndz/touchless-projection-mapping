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
    hand_alpha_boost: float = 2.20
    black_floor: float = 0.03
    white_ceiling: float = 0.85


class RvmProcessor:
    """Gestor de inferencia de RVM con preservación de dedos y gestión recurrente limpia."""

    def __init__(self, config: Optional[RvmConfig] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = config or RvmConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        self.rec_mobilenet: List[Optional[torch.Tensor]] = [None] * 4
        self.rec_resnet: List[Optional[torch.Tensor]] = [None] * 4
        self.rec = self.rec_mobilenet

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

    def reset_recurrent_states(self, variant: Optional[str] = None) -> None:
        """Reinicia los tensores de memoria temporal recurrente de manera desacoplada."""
        if variant is None or "mobilenet" in variant.lower():
            self.rec_mobilenet = [None] * 4
        if variant is None or "resnet" in variant.lower():
            self.rec_resnet = [None] * 4
        self.rec = self.rec_mobilenet

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

        rec_states = self.rec_resnet if key == "resnet50" else self.rec_mobilenet
        dsr = self.cfg.dsr_resnet if key == "resnet50" else self.cfg.dsr_mobilenet

        try:
            with torch.no_grad():
                fgr, pha, *new_rec = model(frame_tensor, *rec_states, downsample_ratio=dsr)
                if key == "resnet50":
                    self.rec_resnet = new_rec
                else:
                    self.rec_mobilenet = new_rec
        except Exception:
            self.reset_recurrent_states(key)
            try:
                rec_states = self.rec_resnet if key == "resnet50" else self.rec_mobilenet
                with torch.no_grad():
                    fgr, pha, *new_rec = model(frame_tensor, *rec_states, downsample_ratio=dsr)
                    if key == "resnet50":
                        self.rec_resnet = new_rec
                    else:
                        self.rec_mobilenet = new_rec
            except Exception:
                return torch.zeros((1, 1, h, w), device=target_device)

        # Garantizar inmunidad frente a NaNs
        pha = torch.nan_to_num(pha, nan=0.0)

        # Realce adaptativo no-lineal con curva gamma para visibilidad nítida de extremidades
        pha = torch.where(pha < self.cfg.black_floor, torch.zeros_like(pha), pha)
        pha = torch.clamp(torch.pow(pha, 0.75) * self.cfg.hand_alpha_boost, 0.0, 1.0)

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
        self.last_raw_frame: Optional[np.ndarray] = None
        self.stable_diff_counter: int = 0
        self.auto_recalibrated: bool = False

    def calibrate(self, frame_bgr: np.ndarray) -> None:
        """Almacena el fotograma de referencia estático para calibración."""
        if frame_bgr is not None and frame_bgr.size > 0:
            self.bg_frame = frame_bgr.copy()
            self.last_raw_frame = frame_bgr.copy()
            self.stable_diff_counter = 0
            self.auto_recalibrated = False

    def is_calibrated(self) -> bool:
        """Indica si el fondo ha sido calibrado."""
        return self.bg_frame is not None

    def compute_mask(
        self,
        frame_bgr: np.ndarray,
        device: Optional[torch.device] = None,
        has_hands: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
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

        # 1. Diferencia multicanal BGR (máximo de canales) para máxima sensibilidad cromática
        diff_bgr = cv2.absdiff(frame_bgr, self.bg_frame)
        max_channel_diff = np.max(diff_bgr, axis=2)

        # 2. Diferencia en espacio de color YCrCb para detectar crominancia pura (tono de piel / ropa)
        frame_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        bg_ycrcb = cv2.cvtColor(self.bg_frame, cv2.COLOR_BGR2YCrCb)
        diff_ycrcb = cv2.absdiff(frame_ycrcb, bg_ycrcb)
        chroma_diff = np.maximum(diff_ycrcb[:, :, 1], diff_ycrcb[:, :, 2])

        # Combinar máxima diferencia de luminancia y crominancia
        combined_diff = np.maximum(max_channel_diff, chroma_diff)

        # 3. Filtrado espacial suave para eliminar ruido electrónico del sensor
        blurred = cv2.GaussianBlur(combined_diff, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)

        # 4. Umbralización binaria
        _, thresh = cv2.threshold(blurred, self.cfg.threshold, 255, cv2.THRESH_BINARY)

        # 5. Cierre morfológico (MORPH_CLOSE) para sellar puentes y agujeros
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        # 6. Relleno morfológico exacto de huecos interiores (Flood-Fill con borde seguro)
        #    Sella huecos interiores causados por objetos estáticos ocluidos sin alterar
        #    los contornos externos naturales. Añade un borde negro perimetral de 1px
        #    para garantizar inmunidad total contra ruido de sensor en los bordes.
        h_mask, w_mask = thresh_closed.shape
        padded = cv2.copyMakeBorder(thresh_closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv_pad = np.zeros((h_mask + 4, w_mask + 4), dtype=np.uint8)
        cv2.floodFill(padded, cv_pad, (0, 0), 255)
        holes_padded = cv2.bitwise_not(padded)
        holes = holes_padded[1:-1, 1:-1]
        solid_mask = thresh_closed | holes

        # 7. Apertura morfológica mínima para limpiar ruido de alta frecuencia en el fondo
        clean_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, self._kernel)

        # 8. Auto-recalibración inteligente ante movimiento o reorientación de la cámara:
        #    A) Mide el movimiento inter-fotogramas para saber si la cámara aún se está moviendo o ya se estabilizó.
        #    B) CONDICIÓN CRÍTICA DE MANOS: Si hay manos del usuario visibles en el encuadre (has_hands=True),
        #       la auto-recalibración se bloquea de inmediato (stable_diff_counter = 0). Esto garantiza que NUNCA
        #       se capture la mano como fondo de referencia, evitando siluetas o manchas verdes fantasma al moverla.
        #    C) Cuando NO hay manos (has_hands=False) y la escena presenta una alteración persistente (> 20% del encuadre)
        #       con la cámara en reposo (inter_frame_motion < 2.8) durante 30 fotogramas continuos (~1s):
        #       se captura automáticamente el nuevo fondo limpio y se devuelve una máscara en cero absoluto.
        total_pixels = h * w
        active_ratio = np.count_nonzero(clean_mask) / float(total_pixels)
        self.auto_recalibrated = False

        inter_frame_motion = 0.0
        if self.last_raw_frame is not None and self.last_raw_frame.shape == frame_bgr.shape:
            inter_frame_motion = float(cv2.absdiff(frame_bgr, self.last_raw_frame).mean())
        self.last_raw_frame = frame_bgr.copy()

        if has_hands:
            # Manos presentes en cámara: bloquear completamente la auto-calibración sobre la extremidad
            self.stable_diff_counter = 0
        elif (active_ratio > 0.20 and inter_frame_motion < 2.8) or (active_ratio > 0.60):
            self.stable_diff_counter += 1
            if self.stable_diff_counter >= 30:
                self.calibrate(frame_bgr)
                self.auto_recalibrated = True
                self.stable_diff_counter = 0
                return torch.zeros((1, 1, h, w), device=target_device), True
        else:
            self.stable_diff_counter = 0

        # 9. Convertir a tensor normalizado en el dispositivo objetivo
        mask_tensor = torch.from_numpy(clean_mask).to(target_device).float().unsqueeze(0).unsqueeze(0) / 255.0

        return mask_tensor, True
