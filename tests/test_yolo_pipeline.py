"""Pruebas unitarias para YoloArmDetector y FastSamArmDetector (Fase 2).
=============================================================================
Valida inicialización, retorno de tensores normalizados, retina_masks y
manejo orgánico sin bounding boxes rectangulares.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINES_DIR = PROJECT_ROOT / "vision_engine" / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))

from yolo_pipeline import FastSamArmDetector, YoloArmDetector, YoloPipelineConfig


def test_yolo_detector_initialization() -> None:
    """Verifica que el detector de YOLO inicialice con configuración tipada."""
    cfg = YoloPipelineConfig(retina_masks=True, use_half=False)
    detector = YoloArmDetector(config=cfg, device=torch.device("cpu"))
    assert detector is not None
    assert detector.cfg.retina_masks is True
    assert detector.cfg.conf_base == 0.22


def test_fastsam_detector_initialization() -> None:
    """Verifica que el detector de FastSAM inicialice correctamente."""
    cfg = YoloPipelineConfig(fastsam_conf=0.20)
    detector = FastSamArmDetector(config=cfg, device=torch.device("cpu"))
    assert detector is not None
    assert detector.cfg.fastsam_conf == 0.20


def test_fastsam_no_landmarks_handling() -> None:
    """
    Verifica que FastSAM sin landmarks devuelva una máscara de ceros
    en lugar de intentar segmentar a ciegas o pintar bounding boxes rectangulares.
    """
    detector = FastSamArmDetector(device=torch.device("cpu"))
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    mask, valid = detector.predict_mask(dummy_frame, hand_landmarks_list=None)
    assert valid is False
    assert mask.shape == (1, 1, 720, 1280)
    assert float(mask.sum()) == 0.0


def test_yolo_predict_output_contract() -> None:
    """
    Verifica que la salida de YOLO cumpla estrictamente con el contrato
    tensorial [1, 1, H, W] acotado en [0.0, 1.0].
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = YoloArmDetector(device=device)
    if detector.model is None:
        pytest.skip("Modelo YOLO no disponible en el entorno")

    dummy_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    mask, valid = detector.predict_mask(dummy_frame, hand_landmarks_list=None, device=device)

    assert mask.shape == (1, 1, 720, 1280)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
    assert mask.device.type == device.type
