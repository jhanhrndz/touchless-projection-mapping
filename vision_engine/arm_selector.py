"""Módulo de Aislamiento Orgánico de Brazo y Filtrado Temporal (ArmMaskSelector).
=============================================================================
Aísla exclusivamente manos y brazos a partir de máscaras neuronales o de resta de fondo,
eliminando torso, cara y objetos no relacionados SIN introducir artefactos geométricos.

Acelera el proceso de componentes conectados ejecutando a resolución reducida (< 1 ms en CPU),
elimina el retraso de entrada de bordes ("gate duro") detectando extremidades antes de
que MediaPipe reconozca la palma, y aplica histéresis temporal para evitar parpadeos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ArmSelectorConfig:
    """Parámetros de configuración para el selector y filtro de brazo."""
    downsample_factor: int = 4            # Reducción de escala para componentes conectados (720p -> 180p)
    min_component_area_ds: int = 35       # Área mínima en píxeles reducidos (~560 px en 720p)
    edge_margin_ds: int = 6               # Margen de borde en escala reducida para detectar entrada
    persistence_frames: int = 4           # Fotogramas de persistencia ante pérdida momentánea de tracking
    decay_rate: float = 0.70              # Tasa de decaimiento durante persistencia
    ema_weight: float = 0.80              # Ponderación del fotograma actual en EMA temporal
    fade_start_dist_ratio: float = 5.5    # Inicio de desvanecimiento hacia el torso (múltiplos de palma-muñeca)
    fade_end_dist_ratio: float = 8.5      # Fin de desvanecimiento (corte de torso)


class TemporalMaskFilter:
    """Filtro de coherencia temporal con promedio móvil exponencial (EMA) e histéresis."""

    def __init__(self, persistence_frames: int = 4, ema_weight: float = 0.80, decay_rate: float = 0.70) -> None:
        self.persistence_frames = persistence_frames
        self.ema_weight = ema_weight
        self.decay_rate = decay_rate
        self.prev_mask: Optional[torch.Tensor] = None
        self.cooldown: int = 0

    def update(self, current_mask: torch.Tensor, has_valid_detection: bool) -> torch.Tensor:
        """
        Actualiza el estado temporal con la máscara actual.
        Si la detección actual es nula pero hay cooldown activo, persiste con decaimiento.
        """
        device = current_mask.device

        if has_valid_detection:
            self.cooldown = self.persistence_frames
            if self.prev_mask is not None and self.prev_mask.shape == current_mask.shape:
                # EMA: Suavizar transición de bordes entre fotogramas consecutivos
                smoothed = current_mask * self.ema_weight + self.prev_mask * (1.0 - self.ema_weight)
            else:
                smoothed = current_mask
            self.prev_mask = smoothed.clone()
            return smoothed

        # Sin detección válida en el fotograma actual
        if self.cooldown > 0 and self.prev_mask is not None:
            self.cooldown -= 1
            # Decaimiento suave
            decayed = self.prev_mask * self.decay_rate
            self.prev_mask = decayed.clone()
            return decayed

        # Sin detección ni cooldown
        self.prev_mask = None
        self.cooldown = 0
        return torch.zeros_like(current_mask)

    def reset(self) -> None:
        """Reinicia el historial de memoria temporal."""
        self.prev_mask = None
        self.cooldown = 0


class ArmMaskSelector:
    """
    Selector inteligente de máscara de extremidades superiores.
    Combina anclas anatómicas de MediaPipe con detección temprana en bordes
    para lograr entrada instantánea con cero parpadeo y bordes finos reales.
    """

    def __init__(self, config: Optional[ArmSelectorConfig] = None) -> None:
        self.cfg = config or ArmSelectorConfig()
        self.temporal_filter = TemporalMaskFilter(
            persistence_frames=self.cfg.persistence_frames,
            ema_weight=self.cfg.ema_weight,
            decay_rate=self.cfg.decay_rate
        )

    def reset(self) -> None:
        """Reinicia la memoria temporal."""
        self.temporal_filter.reset()

    def select(
        self,
        alpha_matte: torch.Tensor,
        h: int,
        w: int,
        hand_landmarks_list: Optional[List[Any]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Aísla el brazo y mano de la máscara alfa y devuelve un tensor refinado en VRAM.

        Args:
            alpha_matte: Tensor de entrada [1, 1, H, W] con valores en [0.0, 1.0].
            h: Alto del fotograma original (ej: 720).
            w: Ancho del fotograma original (ej: 1280).
            hand_landmarks_list: Lista de landmarks anatómicos de MediaPipe (puede ser vacía o None).
            device: Dispositivo torch destino.

        Returns:
            Tensor refinado [1, 1, H, W] con contornos orgánicos en el mismo dispositivo.
        """
        target_device = device or alpha_matte.device

        # Caso base: tensor vacío o nulo
        if alpha_matte is None or alpha_matte.numel() == 0:
            return torch.zeros((1, 1, h, w), device=target_device)

        # 1. Downsample rápido a resolución reducida para cálculo instantáneo de componentes
        ds = max(1, self.cfg.downsample_factor)
        h_ds = max(1, h // ds)
        w_ds = max(1, w // ds)

        # Interpolación en GPU o CPU según el tensor de entrada
        alpha_small = F.interpolate(alpha_matte, size=(h_ds, w_ds), mode="bilinear", align_corners=False)
        alpha_np_ds = alpha_small.squeeze().detach().cpu().numpy()

        bin_mask_ds = (alpha_np_ds > 0.12).astype(np.uint8)
        total_active_px = int(bin_mask_ds.sum())

        if total_active_px < self.cfg.min_component_area_ds:
            # Máscara demasiado pequeña o ruido
            filtered = self.temporal_filter.update(
                torch.zeros((1, 1, h, w), device=target_device),
                has_valid_detection=False
            )
            return filtered

        # 2. Componentes conectados a 180p (< 0.5 ms en CPU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask_ds, connectivity=8)
        if num_labels <= 1:
            filtered = self.temporal_filter.update(
                torch.zeros((1, 1, h, w), device=target_device),
                has_valid_detection=False
            )
            return filtered

        kept_labels: set[int] = set()
        has_anatomical_anchor = False

        # 3. ESTRATEGIA A: Ancla anatómica MediaPipe (cuando la mano está detectada)
        if hand_landmarks_list:
            for lms in hand_landmarks_list:
                for lm in lms:
                    lx = min(w_ds - 1, max(0, int(lm.x * w_ds)))
                    ly = min(h_ds - 1, max(0, int(lm.y * h_ds)))
                    lbl = labels[ly, lx]
                    if lbl > 0:
                        kept_labels.add(lbl)
                    # Búsqueda vecina 3x3 en escala reducida (equivalente a 12x12 en nativa)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny = min(h_ds - 1, max(0, ly + dy))
                            nx = min(w_ds - 1, max(0, lx + dx))
                            lbl_n = labels[ny, nx]
                            if lbl_n > 0:
                                kept_labels.add(lbl_n)
            if kept_labels:
                has_anatomical_anchor = True

        # 4. ESTRATEGIA B: Heurística de Entrada por Borde (Cero Latencia / Sin esperar palma)
        # Si no hay landmarks (brazo entrando parcialmente por el marco), buscamos componentes
        # que toquen el borde exterior de la pantalla y tengan área significativa.
        if not has_anatomical_anchor:
            edge_margin = self.cfg.edge_margin_ds
            for lbl_idx in range(1, num_labels):
                area = stats[lbl_idx, cv2.CC_STAT_AREA]
                if area < self.cfg.min_component_area_ds:
                    continue

                left = stats[lbl_idx, cv2.CC_STAT_LEFT]
                top = stats[lbl_idx, cv2.CC_STAT_TOP]
                width = stats[lbl_idx, cv2.CC_STAT_WIDTH]
                height = stats[lbl_idx, cv2.CC_STAT_HEIGHT]
                right = left + width
                bottom = top + height

                # ¿Toca alguno de los cuatro bordes del fotograma?
                touches_left = left <= edge_margin
                touches_top = top <= edge_margin
                touches_right = right >= (w_ds - edge_margin)
                touches_bottom = bottom >= (h_ds - edge_margin)

                if touches_left or touches_top or touches_right or touches_bottom:
                    # El componente proviene de fuera del encuadre: aceptarlo como extremidad entrante
                    kept_labels.add(lbl_idx)

        # Si tras ambas estrategias no hay componentes válidos, pasar a histéresis temporal
        if not kept_labels:
            filtered = self.temporal_filter.update(
                torch.zeros((1, 1, h, w), device=target_device),
                has_valid_detection=False
            )
            return filtered

        # 5. Construcción de máscara de selección binaria combinada
        selection_mask_ds = np.isin(labels, list(kept_labels)).astype(np.float32)

        # 6. Atenuación vectorial del torso (si hay landmarks para orientar el antebrazo)
        if hand_landmarks_list and has_anatomical_anchor:
            for lms in hand_landmarks_list:
                pts = [(int(lm.x * w_ds), int(lm.y * h_ds)) for lm in lms]
                wrist = pts[0]
                palm = pts[9]
                dx = wrist[0] - palm[0]
                dy = wrist[1] - palm[1]
                dist = max(1.0, math.hypot(dx, dy))
                nx, ny = dx / dist, dy / dist

                fade_start = dist * self.cfg.fade_start_dist_ratio
                fade_end = dist * self.cfg.fade_end_dist_ratio

                Y, X = np.ogrid[:h_ds, :w_ds]
                proj = (X - wrist[0]) * nx + (Y - wrist[1]) * ny
                attenuation = np.clip(1.0 - (proj - fade_start) / max(1.0, (fade_end - fade_start)), 0.0, 1.0)
                attenuation[proj < fade_start] = 1.0
                selection_mask_ds = selection_mask_ds * attenuation.astype(np.float32)

        # 7. Upsampling de la máscara de selección a resolución nativa y multiplicación en VRAM
        selection_tensor_ds = torch.from_numpy(selection_mask_ds).to(target_device).unsqueeze(0).unsqueeze(0)
        selection_tensor = F.interpolate(selection_tensor_ds, size=(h, w), mode="bilinear", align_corners=False)

        # Multiplicación por la máscara alfa continua para preservar contornos y dedos intactos
        arm_alpha = alpha_matte * selection_tensor

        # 8. Filtro temporal con histéresis y EMA
        filtered_arm_alpha = self.temporal_filter.update(arm_alpha, has_valid_detection=True)

        return filtered_arm_alpha
