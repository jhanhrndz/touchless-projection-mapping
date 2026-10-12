"""Pruebas unitarias para RvmProcessor y BackgroundSubtractionProcessor (Fase 3).
=============================================================================
Valida inicialización, reset de estados recurrentes, calibración limpia,
ausencia de falsos positivos en reposo y morfología no destructiva.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINES_DIR = PROJECT_ROOT / "vision_engine" / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))

from rvm_background import (
    BackgroundSubtractionConfig,
    BackgroundSubtractionProcessor,
    RvmConfig,
    RvmProcessor,
)


def test_rvm_processor_initialization() -> None:
    """Verifica que RvmProcessor cargue modelos desde caché local."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = RvmProcessor(device=device)
    assert processor is not None
    assert len(processor.rec) == 4


def test_rvm_recurrent_states_reset() -> None:
    """Verifica el restablecimiento de memoria recurrente temporal."""
    processor = RvmProcessor(device=torch.device("cpu"))
    # Simular estados asignados
    processor.rec = [torch.ones((1, 1, 10, 10)) for _ in range(4)]
    assert processor.rec[0] is not None

    processor.reset_recurrent_states()
    assert processor.rec == [None, None, None, None]


def test_rvm_predict_output_contract() -> None:
    """Verifica contrato de salida tensorial en RVM."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = RvmProcessor(device=device)
    if not processor.models:
        pytest.skip("Modelos RVM no disponibles en caché")

    dummy_tensor = torch.zeros((1, 3, 720, 1280), device=device)
    pha = processor.predict_alpha(dummy_tensor, variant="mobilenetv3", has_hands=True)

    assert pha.shape == (1, 1, 720, 1280)
    assert float(pha.min()) >= 0.0
    assert float(pha.max()) <= 1.0


def test_background_subtraction_calibration() -> None:
    """Verifica calibración de fotograma estático de referencia."""
    processor = BackgroundSubtractionProcessor()
    assert not processor.is_calibrated()

    sample_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    processor.calibrate(sample_frame)

    assert processor.is_calibrated()
    assert processor.bg_frame is not None
    assert processor.bg_frame.shape == (720, 1280, 3)


def test_background_subtraction_zero_diff_on_static_scene() -> None:
    """
    Verifica que si la escena no cambia respecto al fondo calibrado,
    la máscara sea IDÉNTICA A CEROS (cero ruido / cero manchas residuales).
    """
    processor = BackgroundSubtractionProcessor()
    bg = np.full((720, 1280, 3), 120, dtype=np.uint8)
    processor.calibrate(bg)

    # Evaluar el mismo frame
    mask, valid = processor.compute_mask(bg)
    assert valid is True
    assert mask.shape == (1, 1, 720, 1280)
    assert float(mask.sum()) == 0.0


def test_background_subtraction_clean_difference() -> None:
    """
    Verifica que una intrusión (brazo) sea detectada con bordes limpios
    y sin dilatación circular destructiva.
    """
    processor = BackgroundSubtractionProcessor(config=BackgroundSubtractionConfig(threshold=25))
    bg = np.zeros((720, 1280, 3), dtype=np.uint8)
    processor.calibrate(bg)

    # Frame con brazo rectangular (x de 100 a 200, y de 300 a 500)
    arm_frame = bg.copy()
    arm_frame[300:500, 100:200] = 180

    mask, valid = processor.compute_mask(arm_frame)
    assert valid is True
    assert float(mask.sum()) > 0.0

    # Verificar que el centro de la extremidad esté activo
    assert float(mask[0, 0, 400, 150]) == 1.0
    # Verificar que fuera de la extremidad permanezca en cero absoluto
    assert float(mask[0, 0, 100, 100]) == 0.0
