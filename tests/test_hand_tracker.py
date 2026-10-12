"""Pruebas unitarias para HandInteractionTracker, OneEuroFilter y HysteresisPinchDetector (Fase 4).
=============================================================================
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
VISION_ENGINE_DIR = ROOT_DIR / "vision_engine"
if str(VISION_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_ENGINE_DIR))

from hand_tracker import (
    HandInteractionTracker,
    HandTrackerConfig,
    HysteresisPinchDetector,
    OneEuroFilter1D,
    OneEuroFilter2D,
    PinchConfig,
)


def test_one_euro_filter_reduces_static_jitter():
    """Verifica que el filtro One-Euro reduzca drásticamente la varianza de temblor en reposo."""
    f = OneEuroFilter1D(min_cutoff=1.0, beta=0.007)
    base_pos = 500.0
    np.random.seed(42)
    # Ruido gaussiano simulando jitter subpíxel de la mano en reposo
    noise = np.random.normal(0.0, 3.0, size=60)
    samples = base_pos + noise

    filtered_vals = []
    t = 0.0
    for s in samples:
        filtered_vals.append(f.filter(float(s), timestamp=t))
        t += 0.033  # ~30 FPS

    raw_std = float(np.std(samples[10:]))
    filtered_std = float(np.std(filtered_vals[10:]))

    # El filtro debe atenuar al menos un 50% de la varianza en reposo
    assert filtered_std < raw_std * 0.55
    assert abs(np.mean(filtered_vals) - base_pos) < 1.0


def test_one_euro_filter_step_responsiveness():
    """Verifica que el filtro responda rápidamente ante desplazamientos bruscos (alta velocidad)."""
    f = OneEuroFilter2D(min_cutoff=1.0, beta=0.01)
    # Paso brusco de (100, 100) a (800, 800)
    for t_step in range(10):
        f.filter(100.0, 100.0, timestamp=t_step * 0.033)

    # Gran salto
    x_out, y_out = f.filter(800.0, 800.0, timestamp=10 * 0.033)
    # Con beta alto, debe avanzar más del 50% en el primer fotograma tras el salto
    assert x_out > 450.0
    assert y_out > 450.0


def test_hysteresis_pinch_detector_cycle_and_anti_chatter():
    """Verifica el ciclo de activación/liberación de pellizco con doble umbral e histéresis."""
    cfg = PinchConfig(down_threshold_px=55.0, up_threshold_px=75.0, debounce_frames=2)
    detector = HysteresisPinchDetector(cfg)

    # 1. Distancia amplia: no pellizco
    assert not detector.update(120.0)
    assert not detector.update(90.0)

    # 2. Entra en zona intermedia (65 px): todavía NO debe activar porque down=55.0
    assert not detector.update(65.0)

    # 3. Baja por debajo de 55 px: 1er frame no activa por debounce=2
    assert not detector.update(50.0)
    # 2do frame: debe activar
    assert detector.update(48.0)

    # 4. Sube a zona intermedia (65 px): debe PERMANECER activo por histéresis
    assert detector.update(65.0)
    assert detector.update(68.0)
    assert detector.update(60.0)

    # 5. Sube por encima de up_threshold (80 px): 1er frame no libera por debounce
    assert detector.update(80.0)
    # 2do frame por encima: libera
    assert not detector.update(82.0)

    # 6. Oscilación en zona intermedia (70 px): permanece liberado
    assert not detector.update(70.0)
    assert not detector.update(65.0)


def test_hand_interaction_tracker_contract_on_empty_frame():
    """Verifica que el contrato devuelto cumpla exactamente el esquema de telemetría."""
    tracker = HandInteractionTracker()
    empty_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)

    landmarks, cursor = tracker.process_frame(empty_rgb, timestamp_ms=100)

    assert isinstance(landmarks, list)
    assert len(landmarks) == 0
    assert isinstance(cursor, dict)
    assert cursor["detected"] is False
    assert cursor["x"] == 0.0
    assert cursor["y"] == 0.0
    assert cursor["is_pinching"] is False
    assert cursor["pinch_dist"] == 0.0


def test_hand_interaction_tracker_monotonic_timestamp():
    """Verifica que el tracker maneje timestamps no estrictos garantizando monotonicidad."""
    tracker = HandInteractionTracker()
    empty_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    # Enviar timestamps desordenados o repetidos
    tracker.process_frame(empty_rgb, timestamp_ms=500)
    assert tracker.last_timestamp_ms == 500

    # Timestamp igual o menor no debe producir excepción
    tracker.process_frame(empty_rgb, timestamp_ms=500)
    assert tracker.last_timestamp_ms == 501

    tracker.process_frame(empty_rgb, timestamp_ms=200)
    assert tracker.last_timestamp_ms == 502
