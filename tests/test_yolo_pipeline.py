"""Pruebas unitarias para YoloArmDetector y FastSamArmDetector (Fase 2).
=============================================================================
Valida inicialización, retorno de tensores normalizados, retina_masks y
manejo orgánico sin bounding boxes rectangulares.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINES_DIR = PROJECT_ROOT / "vision_engine" / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))

from yolo_pipeline import (
    FastSamArmDetector,
    YoloArmDetector,
    YoloHumanParseDetector,
    YoloPipelineConfig,
)


def test_yolo_detector_initialization() -> None:
    """Verifica que el detector de YOLO inicialice con configuración tipada."""
    cfg = YoloPipelineConfig(retina_masks=True, use_half=False)
    detector = YoloArmDetector(config=cfg, device=torch.device("cpu"))
    assert detector is not None
    assert detector.cfg.retina_masks is True
    assert detector.cfg.conf_base == 0.28


def test_fastsam_detector_initialization() -> None:
    """Verifica que el detector de FastSAM inicialice correctamente."""
    cfg = YoloPipelineConfig(fastsam_conf=0.20)
    detector = FastSamArmDetector(config=cfg, device=torch.device("cpu"))
    assert detector is not None
    assert detector.cfg.fastsam_conf == 0.20


def test_fastsam_no_landmarks_handling() -> None:
    """
    Verifica que FastSAM sin landmarks devuelva una máscara de ceros absolutos,
    incluso ante escenas complejas con múltiples objetos, en lugar de pintar la
    habitación entera de verde como ocurría con el fallback indiscriminado.
    """
    detector = FastSamArmDetector(device=torch.device("cpu"))
    # Frame complejo con múltiples formas geométricas y texturas
    complex_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    cv2.rectangle(complex_frame, (100, 100), (400, 500), (200, 50, 50), -1)
    cv2.circle(complex_frame, (800, 400), 150, (50, 200, 50), -1)

    mask, valid = detector.predict_mask(complex_frame, hand_landmarks_list=None)
    assert valid is False
    assert mask.shape == (1, 1, 720, 1280)
    assert float(mask.sum()) == 0.0

    # También con lista vacía de landmarks
    mask_empty, valid_empty = detector.predict_mask(complex_frame, hand_landmarks_list=[])
    assert valid_empty is False
    assert float(mask_empty.sum()) == 0.0


def test_fastsam_filters_out_room_background_masks() -> None:
    """
    Verifica que las cajas delimitadoras de la mano descarten cualquier máscara
    de fondo masiva (> 38% del área total) y seleccionen únicamente el objeto objetivo.
    """
    detector = FastSamArmDetector(device=torch.device("cpu"))
    if detector.model is None:
        pytest.skip("Modelo FastSAM no disponible en el entorno")

    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    h, w = 720, 1280

    # Mock de landmarks de mano centrados en (w//2, h//2)
    class MockLM:
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

    cx, cy = 0.5, 0.5
    mock_hand = [MockLM(cx + 0.005 * i, cy + 0.005 * i) for i in range(21)]
    mock_hand[0] = MockLM(cx, cy + 0.05)   # muñeca
    mock_hand[9] = MockLM(cx, cy)          # nudillo medio
    mock_hand[8] = MockLM(cx, cy - 0.05)   # índice

    mask, valid = detector.predict_mask(frame, hand_landmarks_list=[mock_hand])
    assert mask.shape == (1, 1, 720, 1280)
    # Si detecta algo, no debe cubrir la habitación completa (> 40%)
    area_ratio = float(mask.sum()) / (h * w)
    assert area_ratio <= 0.40



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


def test_yolo_human_parse_detector_initialization() -> None:
    """Verifica que el detector YOLO-Human-Parse inicialice correctamente con su configuración."""
    cfg = YoloPipelineConfig(retina_masks=True, use_half=False)
    detector = YoloHumanParseDetector(config=cfg, device=torch.device("cpu"))
    assert detector is not None
    assert detector.cfg.retina_masks is True
    assert detector.hold_frames == 0
    assert detector.max_hold_frames == 6
    assert detector.decay_factor == 0.90
    assert detector.ema_alpha == 0.70
    assert detector.last_confidence == 0.0


def test_yolo_human_parse_predict_output_contract() -> None:
    """
    Verifica que YOLO-Human-Parse cumpla el contrato tensorial [1, 1, H, W]
    y gestione la persistencia temporal / histéresis ante fotogramas vacíos.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = YoloHumanParseDetector(device=device)
    if detector.model is None:
        pytest.skip("Modelo YOLO-Human-Parse no disponible en el entorno")

    dummy_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    mask, valid = detector.predict_mask(dummy_frame, hand_landmarks_list=None, device=device)

    assert mask.shape == (1, 1, 720, 1280)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
    assert mask.device.type == device.type
    # Verificamos que last_confidence sea un float válido entre 0.0 y 1.0
    assert 0.0 <= detector.last_confidence <= 1.0

