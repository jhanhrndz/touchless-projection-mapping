"""Pruebas unitarias para fast_guided_filter y GpuHologramRenderer (Fase 5).
=============================================================================
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
VISION_ENGINE_DIR = ROOT_DIR / "vision_engine"
if str(VISION_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_ENGINE_DIR))

from gpu_renderer import GpuHologramRenderer, GpuRendererConfig, fast_guided_filter


def test_fast_guided_filter_output_contract():
    """Verifica el contrato de forma y rango del Filtro Guiado Rápido."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guide = torch.rand((1, 1, 120, 160), device=device)
    src = torch.rand((1, 1, 120, 160), device=device)

    out = fast_guided_filter(guide, src, r=2, eps=1e-3)

    assert out.shape == (1, 1, 120, 160)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_fast_guided_filter_edge_alignment():
    """Verifica que el filtro alinee la máscara con el gradiente de borde de la guía."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = 100, 100
    guide = torch.zeros((1, 1, h, w), device=device)
    guide[:, :, :, 50:] = 1.0  # Borde vertical nítido en columna 50

    # Máscara ligeramente desalineada (columna 48)
    src = torch.zeros((1, 1, h, w), device=device)
    src[:, :, :, 48:] = 1.0

    refined = fast_guided_filter(guide, src, r=2, eps=1e-3)

    # El filtro debe regularizar la transición siguiendo el escalón de la guía
    assert refined[:, :, :, 50].mean().item() > refined[:, :, :, 48].mean().item()


def test_gpu_hologram_renderer_black_floor_suppression():
    """Verifica que cualquier valor de alfa residual bajo (< 0.05) sea estrictamente negro cero."""
    renderer = GpuHologramRenderer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fotograma gris uniforme
    frame_rgb = torch.full((1, 3, 100, 100), 0.5, device=device)
    # Máscara con ruido residual de fondo de 0.04 (< 0.05)
    alpha_noise = torch.full((1, 1, 100, 100), 0.04, device=device)

    tensor_bgr, np_bgr = renderer.render(frame_rgb, alpha_noise, apply_guided_filter=False)

    # El tensor y el array de salida deben ser estrictamente 0 (cero emisión en la pared)
    assert tensor_bgr.max().item() == 0.0
    assert np_bgr.max() == 0


def test_gpu_hologram_renderer_green_dominance():
    """Verifica que en las regiones de brazo digital el canal verde sea dominante."""
    renderer = GpuHologramRenderer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fotograma con piel simulada (R=0.7, G=0.5, B=0.4)
    frame_rgb = torch.tensor([[[[0.7]], [[0.5]], [[0.4]]]], device=device)
    alpha = torch.tensor([[[[1.0]]]], device=device)

    tensor_bgr, np_bgr = renderer.render(frame_rgb, alpha, apply_guided_filter=False)

    # En orden BGR: canal 0 = B, canal 1 = G, canal 2 = R
    b_val = tensor_bgr[0, 0, 0, 0].item()
    g_val = tensor_bgr[0, 1, 0, 0].item()
    r_val = tensor_bgr[0, 2, 0, 0].item()

    assert g_val > b_val * 2.0
    assert g_val > r_val * 2.0
    assert np_bgr[0, 0, 1] > np_bgr[0, 0, 0]  # G > B en numpy uint8
    assert np_bgr[0, 0, 1] > np_bgr[0, 0, 2]  # G > R en numpy uint8


def test_gpu_hologram_renderer_direct_bgr_output_shape():
    """Verifica que el array numpy entregado tenga dimensiones espaciales y orden de canales BGR uint8."""
    renderer = GpuHologramRenderer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_rgb = torch.rand((1, 3, 180, 320), device=device)
    alpha = torch.rand((1, 1, 180, 320), device=device)

    _, np_bgr = renderer.render(frame_rgb, alpha, apply_guided_filter=True)

    assert np_bgr.shape == (180, 320, 3)
    assert np_bgr.dtype == np.uint8
