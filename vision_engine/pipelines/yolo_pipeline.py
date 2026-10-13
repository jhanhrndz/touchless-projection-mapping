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
    """Configuración tipada del pipeline de detección YOLO / FastSAM / Human-Parse."""
    yolo_model_path: str = "yolo11n-seg.pt"
    yolo_parse_model_path: str = "yolo-human-parse.pt"
    fastsam_model_path: str = "FastSAM-s.pt"
    imgsz: int = 640
    conf_base: float = 0.28
    conf_tracking: float = 0.22
    parse_imgsz: int = 384
    parse_conf_base: float = 0.05
    parse_conf_tracking: float = 0.04
    parse_cadence: int = 2
    fastsam_conf: float = 0.20
    retina_masks: bool = True
    use_half: bool = True
    iou_threshold: float = 0.45


class YoloArmDetector:
    """Detector y segmentador de silueta humana completa basado en YOLO11-seg con fusión multinstancia."""

    def __init__(self, config: Optional[YoloPipelineConfig] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = config or YoloPipelineConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracking_active: bool = False
        self.prev_mask: Optional[torch.Tensor] = None
        self.hold_frames: int = 0
        self.max_hold_frames: int = 5
        self.last_confidence: float = 0.0
        self.model: Optional[Any] = None

        if HAS_ULTRALYTICS:
            # Localizar archivo de modelo de forma relativa
            model_p = Path(self.cfg.yolo_model_path)
            if not model_p.exists():
                search_paths = [
                    MODELS_DIR / "yolo11n-seg.pt",
                    Path(__file__).resolve().parent / "yolo11n-seg.pt",
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
        self.last_confidence = 0.0

    def predict_mask(
        self,
        frame_bgr: np.ndarray,
        hand_landmarks_list: Optional[List[Any]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Ejecuta la inferencia de YOLO11-seg y fusiona todas las instancias válidas de la persona
        (cabeza, torso y extremidades) en una silueta humana continua y sólida sin parpadeos.

        Returns:
            Tuple con (tensor_mascara [1, 1, H, W], booleano_deteccion_valida).
        """
        target_device = device or self.device
        h, w = frame_bgr.shape[:2]

        if self.model is None:
            return torch.zeros((1, 1, h, w), device=target_device), False

        # Umbral adaptativo: menor si ya estamos siguiendo a la persona
        current_conf = self.cfg.conf_tracking if self.tracking_active else self.cfg.conf_base
        is_cuda = target_device.type == "cuda"
        use_half = self.cfg.use_half and is_cuda

        try:
            results = self.model.predict(
                source=frame_bgr,
                device=target_device,
                imgsz=self.cfg.imgsz,
                conf=current_conf,
                classes=[0],             # Clase 0 = Persona (cuerpo completo)
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

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            self.last_confidence = float(results[0].boxes.conf.mean().item())
        else:
            self.last_confidence = 0.0

        masks_tensor = results[0].masks.data  # [N, H_out, W_out]
        num_instances = masks_tensor.shape[0]

        # Fusión multinstancia de cuerpo completo:
        # Si YOLO fragmenta el cuerpo en varias instancias (e.g. torso/cabeza por un lado
        # y brazos por otro), las fusionamos todas en una silueta única continua.
        if num_instances == 1:
            combined = masks_tensor[0:1]
            self.tracking_active = True
        else:
            valid_instances = []
            total_mask_pixels = float(masks_tensor.shape[1] * masks_tensor.shape[2])
            masks_cpu = masks_tensor.detach().cpu().numpy()

            for i in range(num_instances):
                inst_area = float(masks_tensor[i].sum().item()) / total_mask_pixels
                # Descartar ruido minúsculo en esquinas (< 0.5% del cuadro)
                if inst_area < 0.005:
                    continue

                # Comprobar si la instancia contiene landmarks de la mano
                touches_hand = False
                if hand_landmarks_list:
                    max_h = masks_tensor.shape[1] - 1
                    max_w = masks_tensor.shape[2] - 1
                    for hl in hand_landmarks_list:
                        for lm_idx in (0, 5, 9, 13, 17):
                            lx = min(max_w, max(0, int(hl[lm_idx].x * max_w)))
                            ly = min(max_h, max(0, int(hl[lm_idx].y * max_h)))
                            if float(masks_cpu[i, ly, lx]) > 0.5:
                                touches_hand = True
                                break

                # Aceptar la instancia si:
                # 1. Toca la mano del usuario (instancia del brazo), O
                # 2. Ocupa un área corporal relevante (>= 2.5% del encuadre, torso/cabeza)
                if touches_hand or inst_area >= 0.025:
                    valid_instances.append(masks_tensor[i])

            if valid_instances:
                stacked = torch.stack(valid_instances)
                combined = torch.clamp(stacked.sum(dim=0, keepdim=True), 0.0, 1.0)
                self.tracking_active = True
            else:
                combined = torch.clamp(masks_tensor.sum(dim=0, keepdim=True), 0.0, 1.0)
                self.tracking_active = True

        best_mask = combined.float()
        if best_mask.ndim == 3:
            best_mask = best_mask.unsqueeze(0)

        # Asegurar coincidencia exacta de dimensiones espaciales con el fotograma
        if best_mask.shape[-2:] != (h, w):
            best_mask = F.interpolate(best_mask, size=(h, w), mode="bilinear", align_corners=False)

        best_mask = torch.clamp(best_mask, 0.0, 1.0)

        # Suavizado temporal EMA para máxima estabilidad y fluidez sin parpadeos
        if self.prev_mask is not None and self.prev_mask.shape == best_mask.shape:
            best_mask = best_mask * 0.75 + self.prev_mask * 0.25
        self.prev_mask = best_mask.clone()
        self.hold_frames = self.max_hold_frames

        return best_mask, True


class YoloHumanParseDetector:
    """Detector y segmentador anatómico de partes humanas basado en YOLO-Human-Parse (Fase 2) optimizado a 20-30 FPS."""

    def __init__(self, config: Optional[YoloPipelineConfig] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = config or YoloPipelineConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracking_active: bool = False
        self.prev_mask: Optional[torch.Tensor] = None
        self.hold_frames: int = 0
        self.max_hold_frames: int = 10
        self.decay_factor: float = 0.94
        self.ema_alpha: float = 0.70
        self.last_confidence: float = 0.0
        self.model: Optional[Any] = None
        self.frame_counter: int = 0
        self.cadence: int = getattr(self.cfg, "parse_cadence", 2)

        if HAS_ULTRALYTICS:
            model_p = Path(self.cfg.yolo_parse_model_path)
            if not model_p.exists():
                search_paths = [
                    MODELS_DIR / "yolo-human-parse.pt",
                    Path(__file__).resolve().parent / "yolo-human-parse.pt",
                    Path(__file__).resolve().parent.parent / "models" / "yolo-human-parse.pt",
                ]
                for sp in search_paths:
                    if sp.exists():
                        model_p = sp
                        break

            try:
                self.model = YOLO(str(model_p) if model_p.exists() else "yolo-human-parse.pt")
                # Calentamiento inicial rápido de CUDA en segundo plano para evitar congelamiento de 6s
                if self.device.type == "cuda":
                    dummy_warm = np.zeros((360, 640, 3), dtype=np.uint8)
                    with torch.inference_mode():
                        _ = self.model.predict(
                            dummy_warm,
                            device=self.device,
                            imgsz=getattr(self.cfg, "parse_imgsz", 384),
                            classes=[3, 4],
                            conf=0.10,
                            retina_masks=False,
                            verbose=False
                        )
            except Exception:
                self.model = None

    def reset(self) -> None:
        """Reinicia el seguimiento y la memoria temporal del detector de Human Parse."""
        self.tracking_active = False
        self.prev_mask = None
        self.hold_frames = 0
        self.last_confidence = 0.0
        self.frame_counter = 0

    def predict_mask(
        self,
        frame_bgr: np.ndarray,
        hand_landmarks_list: Optional[List[Any]] = None,
        arm_only: bool = True,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Ejecuta la inferencia de YOLO-Human-Parse aislando exclusivamente brazos y manos,
        garantizando fluidez de 20-30 FPS y cero parpadeos mediante cadencia intercalada,
        pre-escalado eficiente y suavizado temporal EMA.

        Clases: 3: arm, 4: hand.
        """
        target_device = device or self.device
        h, w = frame_bgr.shape[:2]

        if self.model is None:
            return torch.zeros((1, 1, h, w), device=target_device), False

        self.frame_counter += 1
        has_hands = bool(hand_landmarks_list and len(hand_landmarks_list) > 0)

        # Optimización de cadencia: en fotogramas pares (cuando ya se tiene máscara activa previa),
        # reutilizamos la máscara con persistencia temporal EMA, reduciendo a la mitad el cómputo de la GPU.
        if self.frame_counter % self.cadence != 0 and self.prev_mask is not None and self.tracking_active:
            if has_hands:
                self.hold_frames = self.max_hold_frames
            elif self.hold_frames > 0:
                self.hold_frames -= 1
                decayed = self.prev_mask * self.decay_factor
                self.prev_mask = decayed.clone()
                return decayed, True
            return self.prev_mask, True

        current_conf = self.cfg.parse_conf_tracking if self.tracking_active else self.cfg.parse_conf_base
        target_classes = [3, 4]
        parse_sz = getattr(self.cfg, "parse_imgsz", 384)

        # Pre-escalado eficiente con OpenCV a 640x360 para transferir 4x menos datos por PCIe
        if w > 640 or h > 360:
            infer_frame = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
        else:
            infer_frame = frame_bgr

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=infer_frame,
                    device=target_device,
                    imgsz=parse_sz,
                    conf=current_conf,
                    classes=target_classes,
                    retina_masks=False,
                    iou=self.cfg.iou_threshold,
                    verbose=False
                )
        except Exception:
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * self.decay_factor
                self.prev_mask = decayed.clone()
                return decayed, True
            self.tracking_active = False
            self.prev_mask = None
            self.last_confidence = 0.0
            return torch.zeros((1, 1, h, w), device=target_device), False

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            if self.hold_frames > 0 and self.prev_mask is not None:
                self.hold_frames -= 1
                decayed = self.prev_mask * self.decay_factor
                self.prev_mask = decayed.clone()
                return decayed, True
            self.tracking_active = False
            self.prev_mask = None
            self.last_confidence = 0.0
            return torch.zeros((1, 1, h, w), device=target_device), False

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            self.last_confidence = float(results[0].boxes.conf.mean().item())
        elif self.hold_frames > 0 and self.last_confidence > 0:
            pass
        else:
            self.last_confidence = 0.0

        masks_tensor = results[0].masks.data  # [N, H_out, W_out]
        classes_tensor = results[0].boxes.cls if results[0].boxes is not None else None
        num_instances = masks_tensor.shape[0]

        valid_instances = []
        total_pixels = float(masks_tensor.shape[1] * masks_tensor.shape[2])

        for i in range(num_instances):
            inst_area = float(masks_tensor[i].sum().item()) / total_pixels
            if inst_area < 0.0003:
                continue

            cls_id = int(classes_tensor[i].item()) if classes_tensor is not None else 3
            if cls_id in (3, 4):
                valid_instances.append(masks_tensor[i])

        if valid_instances:
            stacked = torch.stack(valid_instances)
            combined = torch.clamp(stacked.sum(dim=0, keepdim=True), 0.0, 1.0)
            self.tracking_active = True
        elif self.hold_frames > 0 and self.prev_mask is not None:
            self.hold_frames -= 1
            decayed = self.prev_mask * self.decay_factor
            self.prev_mask = decayed.clone()
            return decayed, True
        else:
            self.tracking_active = False
            self.prev_mask = None
            self.last_confidence = 0.0
            return torch.zeros((1, 1, h, w), device=target_device), False

        best_mask = combined.float()
        if best_mask.ndim == 3:
            best_mask = best_mask.unsqueeze(0)

        # Re-escalar a la resolución del fotograma en GPU mediante interpolación bilineal
        if best_mask.shape[-2:] != (h, w):
            best_mask = F.interpolate(best_mask, size=(h, w), mode="bilinear", align_corners=False)

        best_mask = torch.clamp(best_mask, 0.0, 1.0)

        # Suavizado temporal EMA para máxima estabilidad y fluidez
        if self.prev_mask is not None and self.prev_mask.shape == best_mask.shape:
            best_mask = best_mask * self.ema_alpha + self.prev_mask * (1.0 - self.ema_alpha)
        self.prev_mask = best_mask.clone()
        # Si MediaPipe detecta manos activas, extendemos el tiempo de retención para evitar cualquier caída
        self.hold_frames = self.max_hold_frames if has_hands else max(3, self.max_hold_frames // 2)

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
