"""Rastreador Anatómico e Interacción Gestual con MediaPipe (Fase 4).
=============================================================================
Proporciona:
  - Detección continua de manos en modo VIDEO con marcas de tiempo monótonas crecientes.
  - OneEuroFilter2D: Supresión de temblor (jitter) del cursor a baja velocidad
    sin introducir retraso perceptivo (lag) en movimientos rápidos.
  - HysteresisPinchDetector: Detección robusta de pellizco (clic) con doble umbral
    anatómico e histéresis temporal anti-rebote (cero parpadeo de clic).
  - Normalización anatómica por tamaño de palma para robustez frente a distancia
    de cámara.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vision_engine.config import MODELS_DIR

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class SilenceStderr:
    """Suprime advertencias directas a nivel de descriptor de archivo C++ (fd 2)."""
    def __enter__(self):
        try:
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.old_err = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
            self.old_err = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_err is not None:
            try:
                os.dup2(self.old_err, 2)
                os.close(self.null_fd)
                os.close(self.old_err)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  FILTRO ONE-EURO (Eliminación de Jitter y Cero Retardo)
# ─────────────────────────────────────────────────────────────────────────────
class OneEuroFilter1D:
    """Filtro One-Euro 1D adaptativo para señales temporales ruidosas."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        """Reinicia el estado histórico del filtro."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(te: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """Filtra una muestra escalar usando el tiempo actual o provisto."""
        t = timestamp if timestamp is not None else time.perf_counter()

        if self.t_prev is None or self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        te = max(1e-5, t - self.t_prev)

        # 1. Derivada estimada de velocidad
        edx = (x - self.x_prev) / te
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = a_d * edx + (1.0 - a_d) * self.dx_prev

        # 2. Frecuencia de corte adaptativa a la velocidad
        cutoff = self.min_cutoff + self.beta * abs(dx)

        # 3. Suavizado de la posición
        a = self._smoothing_factor(te, cutoff)
        x_filtered = a * x + (1.0 - a) * self.x_prev

        # 4. Actualizar memorias
        self.x_prev = x_filtered
        self.dx_prev = dx
        self.t_prev = t

        return x_filtered


class OneEuroFilter2D:
    """Filtro One-Euro 2D para coordenadas espaciales (X, Y) de cursor."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ) -> None:
        self.fx = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def reset(self) -> None:
        """Reinicia los estados de los ejes X e Y."""
        self.fx.reset()
        self.fy.reset()

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """Filtra un par de coordenadas (X, Y)."""
        t = timestamp if timestamp is not None else time.perf_counter()
        return self.fx.filter(x, t), self.fy.filter(y, t)


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTOR DE PELLIZCO CON HISTÉRESIS (Antirrebote / Cero Flicker)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PinchConfig:
    """Configuración del detector de pellizco."""
    down_threshold_px: float = 58.0   # Umbral inferior para activar clic
    up_threshold_px: float = 78.0     # Umbral superior para soltar clic
    debounce_frames: int = 2          # Cuadros consecutivos de confirmación


class HysteresisPinchDetector:
    """Detector de gesto de pellizco con doble umbral y antirrebote."""

    def __init__(self, config: Optional[PinchConfig] = None) -> None:
        self.cfg = config or PinchConfig()
        self.is_pinching: bool = False
        self._down_count: int = 0
        self._up_count: int = 0

    def reset(self) -> None:
        """Reinicia el estado del detector."""
        self.is_pinching = False
        self._down_count = 0
        self._up_count = 0

    def update(self, dist_px: float) -> bool:
        """
        Actualiza el estado de pellizco a partir de la distancia entre dedos.

        Returns:
            True si el gesto de clic está activo de forma estable, False en caso contrario.
        """
        if not self.is_pinching:
            # Estado actual: NO pellizcando. Buscando activación.
            if dist_px <= self.cfg.down_threshold_px:
                self._down_count += 1
                if self._down_count >= self.cfg.debounce_frames:
                    self.is_pinching = True
                    self._down_count = 0
                    self._up_count = 0
            else:
                self._down_count = 0
        else:
            # Estado actual: PELLIZCANDO. Buscando liberación.
            if dist_px >= self.cfg.up_threshold_px:
                self._up_count += 1
                if self._up_count >= self.cfg.debounce_frames:
                    self.is_pinching = False
                    self._up_count = 0
                    self._down_count = 0
            else:
                self._up_count = 0

        return self.is_pinching


# ─────────────────────────────────────────────────────────────────────────────
#  GESTOR DE RASTREO E INTERACCIÓN DE MANOS (HandInteractionTracker)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HandTrackerConfig:
    """Parámetros de configuración del rastreador de manos."""
    target_width: float = 1920.0
    target_height: float = 1080.0
    min_detection_confidence: float = 0.50
    min_presence_confidence: float = 0.50
    min_tracking_confidence: float = 0.50
    filter_min_cutoff: float = 1.2
    filter_beta: float = 0.008
    pinch_down_px: float = 58.0
    pinch_up_px: float = 78.0


class HandInteractionTracker:
    """Rastreador anatómico e interactivo de manos con filtrado temporal y gestos."""

    def __init__(
        self,
        task_path: Optional[Path] = None,
        config: Optional[HandTrackerConfig] = None
    ) -> None:
        self.cfg = config or HandTrackerConfig()
        self.last_timestamp_ms: int = 0
        self.cursor_filter = OneEuroFilter2D(
            min_cutoff=self.cfg.filter_min_cutoff,
            beta=self.cfg.filter_beta
        )
        self.pinch_detector = HysteresisPinchDetector(
            PinchConfig(
                down_threshold_px=self.cfg.pinch_down_px,
                up_threshold_px=self.cfg.pinch_up_px,
                debounce_frames=2
            )
        )
        self.detector: Optional[Any] = None

        # Resolver modelo MediaPipe
        model_path = task_path
        if model_path is None or not Path(model_path).exists():
            search_paths = [
                MODELS_DIR / "hand_landmarker.task",
                Path(__file__).resolve().parent / "hand_landmarker.task",
                Path(__file__).resolve().parent / "models" / "hand_landmarker.task",
            ]
            for sp in search_paths:
                if sp.exists():
                    model_path = sp
                    break

        if HAS_MEDIAPIPE and model_path and model_path.exists():
            try:
                base_opt = mp_python.BaseOptions(model_asset_path=str(model_path))
                hand_opts = mp_vision.HandLandmarkerOptions(
                    base_options=base_opt,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=self.cfg.min_detection_confidence,
                    min_hand_presence_confidence=self.cfg.min_presence_confidence,
                    min_tracking_confidence=self.cfg.min_tracking_confidence,
                )
                with SilenceStderr():
                    self.detector = mp_vision.HandLandmarker.create_from_options(hand_opts)
            except Exception:
                self.detector = None

    def reset(self) -> None:
        """Reinicia el historial de filtros y el detector de gestos."""
        self.cursor_filter.reset()
        self.pinch_detector.reset()

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Procesa un fotograma RGB y extrae la lista de landmarks y la telemetría del cursor.

        Returns:
            Tuple con (landmarks_list, cursor_dict).
        """
        # 1. Garantizar timestamp estrictamente monótono creciente
        if timestamp_ms is not None:
            ts_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        else:
            cur_perf_ms = int(time.perf_counter() * 1000)
            ts_ms = max(cur_perf_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = ts_ms

        landmarks_list: List[Any] = []

        if self.detector is not None:
            with SilenceStderr():
                try:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    res = self.detector.detect_for_video(mp_img, ts_ms)
                    if res and res.hand_landmarks:
                        valid_hands = []
                        for i, hl in enumerate(res.hand_landmarks):
                            # Validar score de clasificación anatómica de mano
                            if res.handedness and i < len(res.handedness) and len(res.handedness[i]) > 0:
                                score = res.handedness[i][0].score
                                if score < 0.45:
                                    continue

                            # Validación geométrica estricta de la mano para rechazar caras/orejas
                            dist_palm = math.hypot(hl[0].x - hl[9].x, hl[0].y - hl[9].y)
                            if dist_palm < 0.025:  # Clúster menor a 2.5% de pantalla es falso positivo
                                continue

                            # El dedo índice debe proyectarse coherentemente desde la muñeca
                            dist_wrist_index = math.hypot(hl[0].x - hl[8].x, hl[0].y - hl[8].y)
                            if dist_wrist_index < dist_palm * 0.70:
                                continue

                            valid_hands.append(hl)
                        landmarks_list = valid_hands
                except Exception:
                    pass

        # 2. Extraer y filtrar cursor de índice y pellizco
        if landmarks_list:
            hl = landmarks_list[0]
            # Landmark 8 es directamente la punta anatómica del dedo índice (INDEX_FINGER_TIP)
            # No aplicar distorsión direccional distal para evitar deriva horizontal proporcional al orientar la mano
            raw_x = hl[8].x * self.cfg.target_width
            raw_y = hl[8].y * self.cfg.target_height

            # Pulgar (punto 4)
            thm_x = hl[4].x * self.cfg.target_width
            thm_y = hl[4].y * self.cfg.target_height

            # Filtrar coordenadas para eliminar temblor
            smooth_x, smooth_y = self.cursor_filter.filter(raw_x, raw_y, timestamp=ts_ms / 1000.0)

            # Distancia de pellizco entre puntas de índice y pulgar
            dist_pinch = math.hypot(raw_x - thm_x, raw_y - thm_y)
            is_pinching = self.pinch_detector.update(dist_pinch)

            cursor_dict = {
                "detected": True,
                "x": round(smooth_x, 1),
                "y": round(smooth_y, 1),
                "is_pinching": is_pinching,
                "pinch_dist": round(dist_pinch, 1),
            }
        else:
            self.cursor_filter.reset()
            self.pinch_detector.reset()
            cursor_dict = {
                "detected": False,
                "x": 0.0,
                "y": 0.0,
                "is_pinching": False,
                "pinch_dist": 0.0,
            }

        return landmarks_list, cursor_dict

    def close(self) -> None:
        """Libera los recursos del detector MediaPipe."""
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None
