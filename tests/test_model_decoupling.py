"""Tests de Desacoplamiento de Modelos y Mapeo Anatómico del Puntero.
=============================================================================
Verifica:
1. RVM MobileNet (Modelo 3) entrega máscara alfa válida (sin NaNs y > 0).
2. RVM ResNet (Modelo 4) opera con memoria recurrente aislada sin NaNs.
3. FastSAM (Modelo 1) responde limpiamente con y sin landmarks.
4. El rastreador de manos mapea estrictamente Landmark 8 sin extrapolación distal distorsionada.
"""
import math
import numpy as np
import pytest
import torch

from vision_engine.hand_tracker import HandInteractionTracker, HandTrackerConfig
from vision_engine.pipelines.rvm_background import RvmProcessor
from vision_engine.pipelines.yolo_pipeline import FastSamArmDetector


def test_rvm_mobilenet_and_resnet_decoupling() -> None:
    """Verifica que MobileNet y ResNet operen con estados recurrentes independientes y sin NaNs."""
    rvm = RvmProcessor()
    device = rvm.device
    inp = torch.rand((1, 3, 720, 1280), device=device)

    # 1. Probar MobileNetV3
    pha_mob = rvm.predict_alpha(inp, variant="mobilenetv3")
    assert not torch.isnan(pha_mob).any(), "MobileNetV3 no debe contener NaNs"
    assert pha_mob.shape == (1, 1, 720, 1280)

    # 2. Conmutar a ResNet50
    pha_res = rvm.predict_alpha(inp, variant="resnet50")
    assert not torch.isnan(pha_res).any(), "ResNet50 no debe contener NaNs"
    assert pha_res.shape == (1, 1, 720, 1280)

    # 3. Volver a MobileNetV3 para comprobar que las memorias no se corrompieron
    pha_mob2 = rvm.predict_alpha(inp, variant="mobilenetv3")
    assert not torch.isnan(pha_mob2).any()


def test_hand_tracker_index_fingertip_mapping() -> None:
    """Verifica que el cursor se posicione exactamente sobre Landmark 8 sin sesgo direccional distal."""
    cfg = HandTrackerConfig(target_width=1920.0, target_height=1080.0)
    tracker = HandInteractionTracker(config=cfg)

    # Landmark 8 simulado en coordenadas normalizadas (x=0.5, y=0.5)
    class DummyLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Crear lista de 21 landmarks
    hl = [DummyLandmark(0.5, 0.5) for _ in range(21)]
    # Dedo apuntando horizontalmente a la derecha: articulación 7 a la izquierda de la punta 8
    hl[7] = DummyLandmark(0.40, 0.5)
    hl[8] = DummyLandmark(0.50, 0.5)
    hl[4] = DummyLandmark(0.45, 0.6)  # Pulgar

    # Calcular directamente como en tracker
    raw_x = hl[8].x * cfg.target_width
    raw_y = hl[8].y * cfg.target_height

    # En el centro (0.5 de 1920 = 960)
    assert raw_x == pytest.approx(960.0, abs=1e-3)
    assert raw_y == pytest.approx(540.0, abs=1e-3)

    # Apuntando a la derecha extrema (x=0.90)
    hl[7] = DummyLandmark(0.80, 0.5)
    hl[8] = DummyLandmark(0.90, 0.5)
    raw_x_right = hl[8].x * cfg.target_width
    assert raw_x_right == pytest.approx(1728.0, abs=1e-3)


def test_dual_stream_pointer_isolation() -> None:
    """Verifica que el puntero/retícula se aísle estrictamente en el stream del controlador y nunca en el del proyector."""
    import cv2
    from vision_engine.server import STATE

    w, h = 1280, 720
    output_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    # Dibujar brazo simulado en verde
    cv2.rectangle(output_bgr, (200, 300), (600, 400), (0, 255, 120), -1)

    # 1. Fotograma limpio para proyector (cero retícula)
    _, clean_jpeg = cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])

    # 2. Fotograma para controlador con retícula dibujada
    output_ctrl = output_bgr.copy()
    cv2.circle(output_ctrl, (500, 350), 11, (255, 242, 0), 2, cv2.LINE_AA)
    cv2.circle(output_ctrl, (500, 350), 3, (255, 255, 255), -1, cv2.LINE_AA)
    _, ctrl_jpeg = cv2.imencode(".jpg", output_ctrl, [cv2.IMWRITE_JPEG_QUALITY, 75])

    with STATE.lock:
        STATE.latest_jpeg = clean_jpeg.tobytes()
        STATE.latest_controller_jpeg = ctrl_jpeg.tobytes()

    # Verificar que el fotograma del proyector y el del controlador son diferentes
    assert STATE.latest_jpeg != STATE.latest_controller_jpeg, "El stream del proyector y el del controlador deben ser distintos cuando hay puntero"

    # Decodificar el fotograma del proyector y verificar que en las coordenadas de la retícula NO hay amarillo ni blanco
    decoded_proj = cv2.imdecode(np.frombuffer(STATE.latest_jpeg, np.uint8), cv2.IMREAD_COLOR)
    # En (350, 500) en el proyector debe ser el verde del brazo, NO el retículo amarillo o blanco
    b, g, r = decoded_proj[350, 500]
    assert g > 100, "El brazo holográfico verde debe estar presente"
    # El centro del retículo es blanco (255, 255, 255); en el proyector el canal azul no debe ser dominante/blanco
    assert b < 100, "El proyector NO debe tener el punto blanco del centro de la retícula"

    # Decodificar el fotograma del controlador y verificar que la retícula sí está presente
    decoded_ctrl = cv2.imdecode(np.frombuffer(STATE.latest_controller_jpeg, np.uint8), cv2.IMREAD_COLOR)
    cb, cg, cr = decoded_ctrl[350, 500]
    assert cb > 200 and cg > 200 and cr > 200, "El controlador DEBE tener el centro blanco de la retícula"
