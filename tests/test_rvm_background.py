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


def test_background_subtraction_fills_interior_holes_on_occluding_objects() -> None:
    """
    Verifica que si un objeto estático del fondo coincide en color/brillo con la mano o brazo,
    el algoritmo de relleno de contornos externos cierre el hueco y NO genere manchas negras.
    """
    processor = BackgroundSubtractionProcessor(config=BackgroundSubtractionConfig(threshold=25))
    h, w = 720, 1280
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Objeto estático del fondo (círculo de radio 30 en el centro de la futura trayectoria del brazo)
    cv2.circle(bg, (500, 350), 30, (180, 180, 180), -1)
    processor.calibrate(bg)

    # Frame donde el brazo cubre el área completa (rectángulo de 200x200 sobre el objeto)
    # y tiene el mismo color que el objeto del fondo (180, 180, 180)
    arm_frame = bg.copy()
    cv2.rectangle(arm_frame, (400, 250), (600, 450), (180, 180, 180), -1)

    mask, valid = processor.compute_mask(arm_frame)
    assert valid is True

    # El centro (500, 350), que en la resta cruda daría diferencia 0 (mancha negra),
    # DEBE estar sólidamente activo (1.0) gracias al relleno morfológico de huecos
    assert float(mask[0, 0, 350, 500]) == 1.0, "El centro no debe tener hueco o mancha negra"
    # El perímetro del brazo también debe estar activo
    assert float(mask[0, 0, 260, 410]) == 1.0


def test_camera_shift_auto_recalibration() -> None:
    """
    Verifica que si la cámara se mueve (desplazamiento global masivo > 70% sostenido
    durante 30 fotogramas), el procesador se auto-recalibre automáticamente al nuevo
    fondo y devuelva cero absoluto sin dejar la pantalla eternamente verde.
    """
    processor = BackgroundSubtractionProcessor()
    h, w = 720, 1280
    initial_bg = np.zeros((h, w, 3), dtype=np.uint8)
    processor.calibrate(initial_bg)

    # Simular nuevo fondo tras mover la cámara
    new_camera_view = np.full((h, w, 3), 160, dtype=np.uint8)

    # Alimentar 29 fotogramas: aún no debe haber auto-recalibrado (en espera de estabilización)
    for _ in range(29):
        mask, valid = processor.compute_mask(new_camera_view)
        assert processor.auto_recalibrated is False

    # Fotograma 30: se cumple la condición temporal de cámara movida -> auto-recalibra
    mask, valid = processor.compute_mask(new_camera_view)
    assert processor.auto_recalibrated is True
    assert float(mask.sum()) == 0.0, "Al auto-recalibrar debe limpiar a cero la pantalla"
    assert processor.bg_frame is not None
    assert np.array_equal(processor.bg_frame, new_camera_view)


def test_background_subtraction_blocks_auto_recalibration_when_hands_present() -> None:
    """
    Verifica que la auto-recalibración quede estrictamente congelada si hay manos
    presentes en cámara (has_hands=True), impidiendo quemar la silueta de la mano
    en el fondo y generar fantasmas verdes al moverla.
    """
    processor = BackgroundSubtractionProcessor()
    h, w = 720, 1280
    initial_bg = np.zeros((h, w, 3), dtype=np.uint8)
    processor.calibrate(initial_bg)

    # Simular escena con cambio drástico (e.g. luz o cámara movida)
    changed_view = np.full((h, w, 3), 150, dtype=np.uint8)

    # Alimentar 40 fotogramas continuos pero CON MANOS en el encuadre (has_hands=True)
    for _ in range(40):
        mask, valid = processor.compute_mask(changed_view, has_hands=True)
        assert processor.auto_recalibrated is False
        assert processor.stable_diff_counter == 0
        assert np.array_equal(processor.bg_frame, initial_bg)

    # Una vez que el usuario retira las manos (has_hands=False) y la escena persiste en reposo
    for _ in range(29):
        mask, valid = processor.compute_mask(changed_view, has_hands=False)
        assert processor.auto_recalibrated is False

    # Al fotograma 30 sin manos, recalibra limpiamente
    mask, valid = processor.compute_mask(changed_view, has_hands=False)
    assert processor.auto_recalibrated is True
    assert np.array_equal(processor.bg_frame, changed_view)
    assert float(mask.sum()) == 0.0


