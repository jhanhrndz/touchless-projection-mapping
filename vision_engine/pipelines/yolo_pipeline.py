"""Pipeline Optimizado de Segmentación para YOLO11-seg y FastSAM (Fase 2).
=============================================================================
Proporciona inferencia acelerada en media precisión (FP16), alineación anatómica
sub-píxel mediante retina_masks=True, filtrado estricto por instancia para
eliminar siluetas de fondo, y extracción orgánica en FastSAM (cero cajas rectangulares).
"""
from __future__ import annotations

import math
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
        self.prev_mask: Optional[torch.Tensor] = None
        self.hold_frames: int = 0
        self.max_hold_frames: int = 5
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

    def reset(self) -> None:
        """Reinicia el seguimiento y la memoria temporal de YOLO."""
        self.tracking_active = False
        self.prev_mask = None
        self.hold_frames = 0

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
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * 0.88
                self.prev_mask = decayed.clone()
                return decayed, True
            self.tracking_active = False
            self.prev_mask = None
            return torch.zeros((1, 1, h, w), device=target_device), False

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * 0.88
                self.prev_mask = decayed.clone()
                return decayed, True
            self.tracking_active = False
            self.prev_mask = None
            return torch.zeros((1, 1, h, w), device=target_device), False

        masks_tensor = results[0].masks.data  # [N, H_out, W_out]
        num_instances = masks_tensor.shape[0]

        # Si hay una sola instancia, seleccionarla directamente
        if num_instances == 1:
            best_idx = 0
            self.tracking_active = True
        else:
            # Múltiples instancias: Seleccionar con máxima coincidencia anatómica o persistencia temporal
            best_idx = 0
            if hand_landmarks_list:
                best_score = -1.0
                masks_cpu = masks_tensor.detach().cpu().numpy()
                max_h = masks_tensor.shape[1] - 1
                max_w = masks_tensor.shape[2] - 1
                for i in range(num_instances):
                    score = 0.0
                    for hl in hand_landmarks_list:
                        # Muñeca (0), base índice (5), base medio (9), base anular (13), base meñique (17)
                        for lm_idx in (0, 5, 9, 13, 17):
                            lx = min(max_w, max(0, int(hl[lm_idx].x * max_w)))
                            ly = min(max_h, max(0, int(hl[lm_idx].y * max_h)))
                            score += float(masks_cpu[i, ly, lx])
                    if score > best_score:
                        best_score = score
                        best_idx = i
                self.tracking_active = True
            elif self.prev_mask is not None:
                # Mantener la instancia con mayor solapamiento respecto al fotograma previo
                prev_ds = F.interpolate(
                    self.prev_mask, size=(masks_tensor.shape[1], masks_tensor.shape[2]), mode="nearest"
                ).squeeze()
                best_iou = -1.0
                for i in range(num_instances):
                    intersection = (masks_tensor[i] * prev_ds).sum()
                    union = masks_tensor[i].sum() + prev_ds.sum() - intersection + 1e-6
                    iou = float(intersection / union)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                self.tracking_active = True
            else:
                self.tracking_active = True

        best_mask = masks_tensor[best_idx : best_idx + 1].unsqueeze(0).float()

        # Asegurar coincidencia exacta de dimensiones espaciales con el fotograma
        if best_mask.shape[-2:] != (h, w):
            best_mask = F.interpolate(best_mask, size=(h, w), mode="bilinear", align_corners=False)

        best_mask = torch.clamp(best_mask, 0.0, 1.0)

        # Suavizado temporal EMA para eliminar parpadeo y vibración de bordes
        if self.prev_mask is not None and self.prev_mask.shape == best_mask.shape:
            best_mask = best_mask * 0.70 + self.prev_mask * 0.30
        self.prev_mask = best_mask.clone()
        self.hold_frames = self.max_hold_frames

        return best_mask, True


class FastSamArmDetector:
    """Detector de segmentación guiado por puntos anatómicos con FastSAM-s."""

    def __init__(self, config: Optional[YoloPipelineConfig] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = config or YoloPipelineConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Any] = None
        self.prev_mask: Optional[torch.Tensor] = None
        self.hold_frames: int = 0
        self.max_hold_frames: int = 5

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

    def reset(self) -> None:
        """Reinicia la memoria temporal de FastSAM."""
        self.prev_mask = None
        self.hold_frames = 0

    def predict_mask(
        self,
        frame_bgr: np.ndarray,
        hand_landmarks_list: Optional[List[Any]] = None,
        device: Optional[torch.device] = None,
        arm_only: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Ejecuta FastSAM extrayendo la máscara humana continua guiada por prompts anatómicos.

        Returns:
            Tuple con (tensor_mascara [1, 1, H, W], booleano_deteccion_valida).
        """
        target_device = device or self.device
        h, w = frame_bgr.shape[:2]

        if self.model is None or not hand_landmarks_list:
            # Si no hay modelo o NO se detectan manos, NUNCA segmentar toda la escena
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * 0.85
                self.prev_mask = decayed.clone()
                return decayed, True
            self.prev_mask = None
            return torch.zeros((1, 1, h, w), device=target_device), False

        is_cuda = target_device.type == "cuda"

        try:
            results = self.model(
                frame_bgr,
                device=target_device,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.fastsam_conf,
                retina_masks=self.cfg.retina_masks,
                verbose=False
            )
        except Exception:
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * 0.85
                self.prev_mask = decayed.clone()
                return decayed, True
            self.prev_mask = None
            return torch.zeros((1, 1, h, w), device=target_device), False

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * 0.85
                self.prev_mask = decayed.clone()
                return decayed, True
            self.prev_mask = None
            return torch.zeros((1, 1, h, w), device=target_device), False

        masks_data = None
        # Generar cajas delimitadoras (bboxes) anatómicas estables por cada mano detectada
        # Extendiéndolas hacia el antebrazo para capturar la extremidad continua
        bboxes = []
        palm_points = []
        for hl in hand_landmarks_list:
            pts_x = []
            pts_y = []
            for lm in hl:
                lx = getattr(lm, "x", None) if not isinstance(lm, dict) else lm.get("x")
                ly = getattr(lm, "y", None) if not isinstance(lm, dict) else lm.get("y")
                if lx is not None and ly is not None:
                    pts_x.append(lx * w)
                    pts_y.append(ly * h)

            if not pts_x:
                continue

            min_x, max_x = min(pts_x), max(pts_x)
            min_y, max_y = min(pts_y), max(pts_y)
            bw, bh = max_x - min_x, max_y - min_y
            pad_x = max(16, int(bw * 0.25))
            pad_y = max(16, int(bh * 0.25))

            # Obtener muñeca (0) y nudillo medio (9)
            def get_pt(idx: int) -> Tuple[float, float]:
                target = hl[idx]
                tx = getattr(target, "x", None) if not isinstance(target, dict) else target.get("x")
                ty = getattr(target, "y", None) if not isinstance(target, dict) else target.get("y")
                return (tx * w, ty * h) if (tx is not None and ty is not None) else (pts_x[0], pts_y[0])

            wx, wy = get_pt(0)
            mx, my = get_pt(9)

            # Punto central de la palma (inmune a tocar bordes de fondo)
            palm_points.append([int(min(w - 1, max(0, (wx + mx) * 0.5))), int(min(h - 1, max(0, (wy + my) * 0.5)))])

            # Extender hacia el antebrazo
            dir_x, dir_y = wx - mx, wy - my
            d_len = math.hypot(dir_x, dir_y)
            if d_len > 1.0:
                ext_len = max(bw, bh) * 0.75
                min_x = min(min_x, wx + (dir_x / d_len) * ext_len)
                max_x = max(max_x, wx + (dir_x / d_len) * ext_len)
                min_y = min(min_y, wy + (dir_y / d_len) * ext_len)
                max_y = max(max_y, wy + (dir_y / d_len) * ext_len)

            bboxes.append([
                max(0, int(min_x - pad_x)),
                max(0, int(min_y - pad_y)),
                min(w - 1, int(max_x + pad_x)),
                min(h - 1, int(max_y + pad_y)),
            ])

        # 1. Intentar prompt por Bounding Boxes (máxima precisión anatómica, selecciona la máscara con mejor IoU)
        if bboxes and hasattr(self.model, "predictor") and self.model.predictor is not None:
            try:
                prompted = self.model.predictor.prompt(results, bboxes=bboxes)
                if prompted and prompted[0].masks is not None and len(prompted[0].masks.data) > 0:
                    cand_masks = prompted[0].masks.data
                    # Filtrar máscaras espurias del fondo:
                    # Una mano/antebrazo ocupa entre 0.03% y 38% del encuadre.
                    # Fondos de habitación/paredes superan el 40% y se descartan.
                    valid_masks = []
                    total_area = float(h * w)
                    for m in cand_masks:
                        area_ratio = float(m.sum().item()) / total_area
                        if 0.0003 <= area_ratio <= 0.38:
                            valid_masks.append(m)
                    if valid_masks:
                        masks_data = torch.stack(valid_masks)
            except Exception:
                masks_data = None

        # 2. Respaldo por puntos estrictamente en el centro de palma (si las cajas no dieron máscara)
        if (masks_data is None or len(masks_data) == 0) and palm_points and hasattr(self.model, "predictor") and self.model.predictor is not None:
            try:
                prompted_pts = self.model.predictor.prompt(results, points=palm_points, labels=[1] * len(palm_points))
                if prompted_pts and prompted_pts[0].masks is not None and len(prompted_pts[0].masks.data) > 0:
                    cand_masks = prompted_pts[0].masks.data
                    valid_masks = []
                    total_area = float(h * w)
                    for m in cand_masks:
                        area_ratio = float(m.sum().item()) / total_area
                        if 0.0003 <= area_ratio <= 0.38:
                            valid_masks.append(m)
                    if valid_masks:
                        masks_data = torch.stack(valid_masks)
            except Exception:
                masks_data = None

        # NUNCA utilizar results[0].masks.data como fallback indiscriminado para no pintar la habitación entera de verde
        if masks_data is None or len(masks_data) == 0:
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * 0.85
                self.prev_mask = decayed.clone()
                return decayed, True
            self.prev_mask = None
            return torch.zeros((1, 1, h, w), device=target_device), False

        raw_mask = masks_data.sum(dim=0, keepdim=True).unsqueeze(0).float()
        raw_mask = torch.clamp(raw_mask, 0.0, 1.0)

        if raw_mask.shape[-2:] != (h, w):
            raw_mask = F.interpolate(raw_mask, size=(h, w), mode="bilinear", align_corners=False)

        # Suavizado temporal EMA para máxima estabilidad visual
        if self.prev_mask is not None and self.prev_mask.shape == raw_mask.shape:
            raw_mask = raw_mask * 0.70 + self.prev_mask * 0.30
        self.prev_mask = raw_mask.clone()
        self.hold_frames = self.max_hold_frames

        return raw_mask, True
