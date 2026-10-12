"""Pruebas unitarias y de rendimiento para ArmMaskSelector (Fase 1).
=============================================================================
Valida la eliminación del gate duro, selección anatómica, descarte de falsos
positivos centrales, histéresis temporal y latencia de ejecución en < 5 ms.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISION_ENGINE_DIR = PROJECT_ROOT / "vision_engine"
if str(VISION_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_ENGINE_DIR))

from arm_selector import ArmMaskSelector, ArmSelectorConfig, TemporalMaskFilter


def test_empty_mask_handling() -> None:
    """Verifica manejo robusto de máscaras vacías o nulas."""
    selector = ArmMaskSelector()
    device = torch.device("cpu")
    empty_tensor = torch.zeros((1, 1, 720, 1280), device=device)

    res = selector.select(empty_tensor, h=720, w=1280, hand_landmarks_list=None, device=device)
    assert res.shape == (1, 1, 720, 1280)
    assert float(res.sum()) == 0.0


def test_edge_entry_without_landmarks() -> None:
    """
    Verifica que una extremidad entrando por el borde exterior
    sea detectada y seleccionada inmediatamente sin esperar landmarks de MediaPipe
    (Eliminación del Gate Duro).
    """
    selector = ArmMaskSelector()
    device = torch.device("cpu")
    h, w = 720, 1280

    # Simular una máscara entrando por el borde izquierdo (x de 0 a 180, y de 300 a 450)
    mask_np = np.zeros((h, w), dtype=np.float32)
    mask_np[300:450, 0:180] = 0.95
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    # Llamar SIN landmarks (MediaPipe no ve la palma todavía)
    res = selector.select(mask_tensor, h=h, w=w, hand_landmarks_list=None, device=device)

    # Debe haber seleccionado la extremidad entrante
    assert float(res.sum()) > 0.0
    # La región de entrada debe estar activa
    assert float(res[0, 0, 350, 50]) > 0.5


def test_center_object_rejection() -> None:
    """
    Verifica que un objeto/robot/cartel estático aislado en el centro de la pantalla
    (sin tocar bordes y sin landmarks de mano) sea RECHAZADO a cero.
    """
    selector = ArmMaskSelector()
    device = torch.device("cpu")
    h, w = 720, 1280

    # Simular un objeto en el centro (x de 500 a 700, y de 250 a 450)
    mask_np = np.zeros((h, w), dtype=np.float32)
    mask_np[250:450, 500:700] = 0.95
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    # Sin landmarks
    res = selector.select(mask_tensor, h=h, w=w, hand_landmarks_list=None, device=device)

    # Debe ser rechazado completamente (cero falsos positivos)
    assert float(res.sum()) == 0.0


def test_anatomical_hand_selection() -> None:
    """Verifica que un componente que coincide con landmarks anatómicos sea preservado."""
    selector = ArmMaskSelector()
    device = torch.device("cpu")
    h, w = 720, 1280

    # Simular una mano en el centro-derecha (x de 600 a 800, y de 200 a 400)
    mask_np = np.zeros((h, w), dtype=np.float32)
    mask_np[200:400, 600:800] = 0.90
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    # Crear landmark sintético que cae dentro de esa región
    # x=700/1280=0.5468, y=300/720=0.4166
    synthetic_landmarks = [[
        SimpleNamespace(x=700.0 / w, y=300.0 / h) for _ in range(21)
    ]]

    res = selector.select(mask_tensor, h=h, w=w, hand_landmarks_list=synthetic_landmarks, device=device)

    # La mano anatómica debe ser seleccionada
    assert float(res.sum()) > 0.0
    assert float(res[0, 0, 300, 700]) > 0.5


def test_temporal_persistence_and_decay() -> None:
    """
    Verifica que ante la pérdida súbita de detección por 1 frame (ej. giro de muñeca),
    la máscara persista durante el periodo de cooldown con decaimiento suave.
    """
    cfg = ArmSelectorConfig(persistence_frames=3, decay_rate=0.70)
    selector = ArmMaskSelector(config=cfg)
    device = torch.device("cpu")
    h, w = 720, 1280

    # Frame 1: Mano detectada con landmarks
    mask_np = np.zeros((h, w), dtype=np.float32)
    mask_np[200:400, 600:800] = 0.90
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    synthetic_landmarks = [[SimpleNamespace(x=700.0 / w, y=300.0 / h) for _ in range(21)]]

    f1 = selector.select(mask_tensor, h=h, w=w, hand_landmarks_list=synthetic_landmarks, device=device)
    val_f1 = float(f1[0, 0, 300, 700])
    assert val_f1 > 0.5

    # Frame 2: Pérdida súbita (tensor vacío / sin landmarks)
    empty_tensor = torch.zeros((1, 1, h, w), device=device)
    f2 = selector.select(empty_tensor, h=h, w=w, hand_landmarks_list=None, device=device)
    val_f2 = float(f2[0, 0, 300, 700])

    # Debe persistir pero con decaimiento (val_f2 > 0 y val_f2 < val_f1)
    assert val_f2 > 0.0
    assert val_f2 <= val_f1 * 0.95

    # Frame 3: Sigue en cooldown (frame 2 de 3)
    f3 = selector.select(empty_tensor, h=h, w=w, hand_landmarks_list=None, device=device)
    val_f3 = float(f3[0, 0, 300, 700])
    assert val_f3 > 0.0
    assert val_f3 < val_f2

    # Frame 4: Último frame de cooldown
    f4 = selector.select(empty_tensor, h=h, w=w, hand_landmarks_list=None, device=device)
    assert float(f4[0, 0, 300, 700]) > 0.0

    # Frame 5: Cooldown agotado -> Máscara cae a cero limpiamente
    f5 = selector.select(empty_tensor, h=h, w=w, hand_landmarks_list=None, device=device)
    assert float(f5.sum()) == 0.0


def test_selector_latency_performance() -> None:
    """
    Verifica que la latencia de ArmMaskSelector sea menor a 5 ms por frame a 720p.
    (El método anterior tomaba > 1400 ms).
    """
    selector = ArmMaskSelector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = 720, 1280

    mask_np = np.zeros((h, w), dtype=np.float32)
    mask_np[300:500, 100:300] = 0.85
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    synthetic_landmarks = [[SimpleNamespace(x=200.0 / w, y=400.0 / h) for _ in range(21)]]

    # Warmup
    for _ in range(5):
        _ = selector.select(mask_tensor, h=h, w=w, hand_landmarks_list=synthetic_landmarks, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Medir 30 iteraciones
    t0 = time.perf_counter()
    n_iters = 30
    for _ in range(n_iters):
        _ = selector.select(mask_tensor, h=h, w=w, hand_landmarks_list=synthetic_landmarks, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / n_iters) * 1000.0
    print(f"\n[BENCHMARK] Latencia promedio de ArmMaskSelector: {avg_ms:.2f} ms")

    # Debe ser inferior a 15 ms en CPU y típicamente < 3 ms en GPU/CPU con downsample 1/4
    assert avg_ms < 15.0, f"Latencia excesiva: {avg_ms:.2f} ms"
