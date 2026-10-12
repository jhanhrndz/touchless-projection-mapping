"""Pipeline Optimizado de Segmentación para YOLO11-seg y FastSAM (Fase 2).
=============================================================================
Proporciona inferencia acelerada en media precisión (FP16), alineación anatómica
sub-píxel mediante retina_masks=True, filtrado estricto por instancia para
eliminar siluetas de fondo, y extracción orgánica en FastSAM (cero cajas rectangulares).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vision_engine.config import MODELS_DIR

try:
    from ultralytics import YOLO, FastSAM
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False


@dataclass
class YoloPipelineConfig:
    """Configuración tipada del pipeline de detección YOLO / FastSAM."""
    yolo_model_path: str = "yolo11n-seg.pt"
    fastsam_model_path: str = "FastSAM-s.pt"
    imgsz: int = 640
    conf_base: float = 0.22
    conf_tracking: float = 0.16
    fastsam_conf: float = 0.20
    retina_masks: bool = True
    use_half: bool = True
    iou_threshold: float = 0.45


class YoloArmDetector:
    """Detector de extremidades superiores basado en YOLO11-seg con filtrado por instancia."""

    def __init__(self, config: Optional[YoloPipelineConfig] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = config or YoloPipelineConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracking_active: bool = False
        self.model: Optional[Any] = None

        if HAS_ULTRALYTICS:
            # Localizar archivo de modelo de forma relativa
            model_p = Path(self.cfg.yolo_model_path)
            if not model_p.exists():
                search_paths = [
                    MODELS_DIR / "yolo11n-seg.pt",
                    MODELS_DIR / "yolo11s-seg.pt",
                    Path(__file__).resolve().parent / "yolo11n-seg.pt",
                    Path(__file__).resolve().parent.parent / "models" / "yolo11n-seg.pt",
                ]
                for sp in search_paths:
                    if sp.exists():
                        model_p = sp
                        break

            try:
                self.model = YOLO(str(model_p) if model_p.exists() else "yolo11n-seg.pt")
            except Exception:
                self.model = None

    def predict_mask(
        self,
        frame_bgr: np.ndarray,
        hand_landmarks_list: Optional[List[Any]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Ejecuta la inferencia de YOLO11-seg con retina_masks y selecciona la instancia del brazo.

        Returns:
            Tuple con (tensor_mascara [1, 1, H, W], booleano_deteccion_valida).
        """
        target_device = device or self.device
        h, w = frame_bgr.shape[:2]

        if self.model is None:
            return torch.zeros((1, 1, h, w), device=target_device), False

        # Umbral adaptativo: menor si ya estamos siguiendo la extremidad
        current_conf = self.cfg.conf_tracking if self.tracking_active else self.cfg.conf_base
        is_cuda = target_device.type == "cuda"
        use_half = self.cfg.use_half and is_cuda

        try:
            results = self.model.predict(
                source=frame_bgr,
                device=target_device,
                imgsz=self.cfg.imgsz,
                conf=current_conf,
                classes=[0],             # Clase 0 = Persona / extremidades
                retina_masks=self.cfg.retina_masks,
                iou=self.cfg.iou_threshold,
                verbose=False
            )
        except Exception:
            self.tracking_active = False
            return torch.zeros((1, 1, h, w), device=target_device), False

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            self.tracking_active = False
            return torch.zeros((1, 1, h, w), device=target_device), False

        masks_tensor = results[0].masks.data  # [N, H_out, W_out]
        num_instances = masks_tensor.shape[0]

        # Si hay una sola instancia, seleccionarla directamente
        if num_instances == 1:
            best_mask = masks_tensor[0:1].unsqueeze(0).float()
            self.tracking_active = True
        else:
            # Múltiples personas/instancias: Filtrar la instancia que conecta con la mano activa
            best_idx = 0
            if hand_landmarks_list:
                hl = hand_landmarks_list[0]
                # Coordenadas de muñeca y palma
                hx = int(hl[9].x * (masks_tensor.shape[2] - 1))
                hy = int(hl[9].y * (masks_tensor.shape[1] - 1))
                for i in range(num_instances):
                    if masks_tensor[i, hy, hx] > 0.5:
                        best_idx = i
                        break
            best_mask = masks_tensor[best_idx : best_idx + 1].unsqueeze(0).float()
            self.tracking_active = True

        # Asegurar coincidencia exacta de dimensiones espaciales con el fotograma
        if best_mask.shape[-2:] != (h, w):
            best_mask = F.interpolate(best_mask, size=(h, w), mode="bilinear", align_corners=False)

        best_mask = torch.clamp(best_mask, 0.0, 1.0)
        return best_mask, True


class FastSamArmDetector:
    """Detector de segmentación guiado por puntos anatómicos con FastSAM-s."""

    def __init__(self, config: Optional[YoloPipelineConfig] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = config or YoloPipelineConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None

        if HAS_ULTRALYTICS:
            model_p = Path(self.cfg.fastsam_model_path)
            if not model_p.exists():
                search_paths = [
                    MODELS_DIR / "FastSAM-s.pt",
                    Path(__file__).resolve().parent / "FastSAM-s.pt",
                    Path(__file__).resolve().parent.parent / "models" / "FastSAM-s.pt",
                ]
                for sp in search_paths:
                    if sp.exists():
                        model_p = sp
                        break
            try:
                self.model = FastSAM(str(model_p) if model_p.exists() else "FastSAM-s.pt")
            except Exception:
                self.model = None

    def predict_mask(
        self,
        frame_bgr: np.ndarray,
        hand_landmarks_list: Optional[List[Any]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Ejecuta FastSAM Point-Prompt extrayendo estrictamente la máscara orgánica de la mano.

        Returns:
            Tuple con (tensor_mascara [1, 1, H, W], booleano_deteccion_valida).
        """
        target_device = device or self.device
        h, w = frame_bgr.shape[:2]

        if self.model is None:
            return torch.zeros((1, 1, h, w), device=target_device), False

        # Si no hay landmarks anatómicos para anclar el point-prompt
        if not hand_landmarks_list:
            return torch.zeros((1, 1, h, w), device=target_device), False

        hl = hand_landmarks_list[0]
        pts = [
            [int(hl[9].x * w), int(hl[9].y * h)],   # Palma
            [int(hl[0].x * w), int(hl[0].y * h)],   # Muñeca
        ]

        is_cuda = target_device.type == "cuda"
        use_half = self.cfg.use_half and is_cuda

        try:
            results = self.model.predict(
                source=frame_bgr,
                points=pts,
                device=target_device,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.fastsam_conf,
                retina_masks=self.cfg.retina_masks,
                half=use_half,
                verbose=False
            )
        except Exception:
            return torch.zeros((1, 1, h, w), device=target_device), False

        # Extracción explícita de máscara orgánica (NUNCA devolver bounding box rectangular)
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return torch.zeros((1, 1, h, w), device=target_device), False

        raw_mask = results[0].masks.data.sum(dim=0, keepdim=True).unsqueeze(0).float()
        raw_mask = torch.clamp(raw_mask, 0.0, 1.0)

        if raw_mask.shape[-2:] != (h, w):
            raw_mask = F.interpolate(raw_mask, size=(h, w), mode="bilinear", align_corners=False)

        return raw_mask, True
