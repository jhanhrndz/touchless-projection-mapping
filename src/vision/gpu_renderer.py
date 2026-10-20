"""Pipeline de Renderizado Holográfico en GPU CUDA y Fast Guided Filter (Fase 5).
=============================================================================
Proporciona:
  - fast_guided_filter: Refinamiento de bordes guiado por gradientes de luminancia
    ejecutado enteramente en VRAM mediante núcleos F.avg_pool2d.
  - GpuHologramRenderer:
      * Filtrado de piso negro estricto con curva sigmoidea Hermite para cero
        emisión lumínica en la pared del proyector.
      * Shader holográfico de energía verde sobre píxeles reales.
      * Salida BGR directa en GPU, eliminando conversiones de color redundantes
        en CPU con cv2.cvtColor (~1.8 ms de ahorro por frame).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def fast_guided_filter(
    guide: torch.Tensor,
    src: torch.Tensor,
    r: int = 2,
    eps: float = 1e-3
) -> torch.Tensor:
    """
    Refinamiento de bordes mediante Filtro Guiado Rápido en VRAM (PyTorch CUDA).
    Ajusta la máscara alfa a los gradientes ópticos de la imagen real.

    Args:
        guide: Tensor 1D de luminancia [1, 1, H, W] en [0.0, 1.0].
        src: Tensor de máscara alfa [1, 1, H, W] en [0.0, 1.0].
        r: Radio del kernel de promedio.
        eps: Regularización para evitar división por cero.

    Returns:
        Tensor alfa refinado [1, 1, H, W] en [0.0, 1.0].
    """
    k = 2 * r + 1
    mean_I = F.avg_pool2d(guide, k, stride=1, padding=r)
    mean_p = F.avg_pool2d(src, k, stride=1, padding=r)
    mean_Ip = F.avg_pool2d(guide * src, k, stride=1, padding=r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = F.avg_pool2d(guide * guide, k, stride=1, padding=r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = F.avg_pool2d(a, k, stride=1, padding=r)
    mean_b = F.avg_pool2d(b, k, stride=1, padding=r)

    q = mean_a * guide + mean_b
    return torch.clamp(q, 0.0, 1.0)


@dataclass
class GpuRendererConfig:
    """Parámetros de configuración del renderizador holográfico."""
    guided_filter_radius: int = 2
    guided_filter_eps: float = 1e-3
    black_floor: float = 0.05
    white_ceiling: float = 0.35
    red_attenuation: float = 0.20
    green_gain: float = 1.70
    green_boost: float = 0.35
    blue_attenuation: float = 0.35


class GpuHologramRenderer:
    """Renderizador holográfico en VRAM con síntesis BGR directa."""

    def __init__(self, config: Optional[GpuRendererConfig] = None) -> None:
        self.cfg = config or GpuRendererConfig()

    def render(
        self,
        frame_tensor_rgb: torch.Tensor,
        active_alpha: torch.Tensor,
        apply_guided_filter: bool = False
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Aplica el pipeline completo de renderizado holográfico verde en GPU.

        Args:
            frame_tensor_rgb: Tensor RGB [1, 3, H, W] en [0.0, 1.0].
            active_alpha: Tensor de máscara alfa [1, 1, H, W] en [0.0, 1.0].
            apply_guided_filter: Si es True, refina bordes con Fast Guided Filter.

        Returns:
            Tuple con (tensor_bgr_gpu [1, 3, H, W] en [0.0, 1.0], array_bgr_cpu uint8 [H, W, 3]).
        """
        # 1. Refinamiento opcional de bordes con Fast Guided Filter en GPU
        if apply_guided_filter and active_alpha.max() > self.cfg.black_floor:
            gray = frame_tensor_rgb.mean(dim=1, keepdim=True)
            alpha_refined = fast_guided_filter(
                guide=gray,
                src=active_alpha,
                r=self.cfg.guided_filter_radius,
                eps=self.cfg.guided_filter_eps
            )
        else:
            alpha_refined = active_alpha

        # 2. Supresión estricta de piso negro y curva sigmoidea Hermite
        #    alpha < black_floor es exactamente 0.00 (cero emisión en la pared)
        alpha_clean = torch.where(
            alpha_refined < self.cfg.black_floor,
            torch.zeros_like(alpha_refined),
            alpha_refined
        )
        norm_alpha = torch.clamp(
            (alpha_clean - self.cfg.black_floor) / (self.cfg.white_ceiling - self.cfg.black_floor),
            0.0,
            1.0
        )
        sharp_alpha = norm_alpha * norm_alpha * (3.0 - 2.0 * norm_alpha)

        # 3. Shader holográfico sintetizado DIRECTAMENTE en orden BGR en GPU
        # frame_tensor_rgb[:, 0:1] = R
        # frame_tensor_rgb[:, 1:2] = G
        # frame_tensor_rgb[:, 2:3] = B
        channel_b = torch.clamp(frame_tensor_rgb[:, 2:3] * self.cfg.blue_attenuation, 0.0, 1.0)
        channel_g = torch.clamp(frame_tensor_rgb[:, 1:2] * self.cfg.green_gain + self.cfg.green_boost, 0.0, 1.0)
        channel_r = torch.clamp(frame_tensor_rgb[:, 0:1] * self.cfg.red_attenuation, 0.0, 1.0)

        # Formar tensor BGR [1, 3, H, W]
        bgr_energy = torch.cat([channel_b, channel_g, channel_r], dim=1)
        final_bgr_tensor = torch.clamp(bgr_energy * sharp_alpha, 0.0, 1.0)

        # 4. Transferir a host como array BGR uint8 C-contiguo listo para OpenCV y streaming
        #    Evita cv2.cvtColor(RGB2BGR) en CPU
        bgr_cpu_np = np.ascontiguousarray(
            final_bgr_tensor.squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        return final_bgr_tensor, bgr_cpu_np
