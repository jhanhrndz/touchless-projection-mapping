"""Motor central de inferencia GPU y captura para vision_engine."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from vision_engine.arm_selector import ArmMaskSelector
from vision_engine.config import EngineConfig, MODELS_DIR
from vision_engine.gpu_renderer import GpuHologramRenderer
from vision_engine.hand_tracker import HandInteractionTracker
from vision_engine.pipelines.rvm_background import (
    BackgroundSubtractionConfig,
    BackgroundSubtractionProcessor,
    RvmConfig,
    RvmProcessor,
)
from vision_engine.pipelines.yolo_pipeline import (
    FastSamArmDetector,
    YoloArmDetector,
    YoloPipelineConfig,
)
from vision_engine.server import STATE

logger = logging.getLogger("vision_engine.engine")


def frame_is_valid(frame: Optional[np.ndarray]) -> bool:
    """Verifica si un fotograma de cámara contiene imagen válida (no negro ni nulo)."""
    return frame is not None and frame.size > 0 and float(frame.max()) > 10


def open_camera(preferred: Optional[int] = None) -> Optional[cv2.VideoCapture]:
    """Abre la primera cámara disponible que entregue imagen real."""
    indices = [preferred] if preferred is not None else [0, 1, 2]
    backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    for idx in indices:
        for be_name, be in backends:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            good = False
            for _ in range(30):
                ok, fr = cap.read()
                if ok and frame_is_valid(fr):
                    good = True
                    break
                time.sleep(0.02)
            if good:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info("Cámara #%d (%s) activa: %dx%d", idx, be_name, w, h)
                return cap
            cap.release()
    return None


class VideoFileSource:
    """Reproductor de archivo de video offline en bucle para pruebas deterministas."""

    def __init__(self, path: Union[str, Path], loop: bool = True):
        self.path = str(path)
        self.loop = loop
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el archivo de video: {path}")
        self._is_opened: bool = True
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def is_opened(self) -> bool:
        return self._is_opened and self.cap.isOpened()

    def isOpened(self) -> bool:
        return self._is_opened and self.cap.isOpened()

    def get_dimensions(self) -> Tuple[int, int]:
        return self.w, self.h

    def release(self) -> None:
        self._is_opened = False
        self.cap.release()

    def get(self, prop: int) -> float:
        return self.cap.get(prop)

    def set(self, prop: int, val: float) -> bool:
        return self.cap.set(prop, val)


_ARM_SELECTOR = ArmMaskSelector()


def isolate_arm_and_hand_mask(
    alpha_matte: torch.Tensor,
    h: int,
    w: int,
    hand_landmarks_list: Optional[List[Any]],
    device: torch.device,
) -> torch.Tensor:
    """Aísla manos y brazos eliminando torso y fondo orgánicamente."""
    return _ARM_SELECTOR.select(
        alpha_matte=alpha_matte,
        h=h,
        w=w,
        hand_landmarks_list=hand_landmarks_list,
        device=device,
    )


class VisionEngine:
    """Orquestador principal del pipeline de visión por computadora y proyección."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.cfg = config or EngineConfig.from_env()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Modelos
        self.yolo_detector = YoloArmDetector(
            config=YoloPipelineConfig(
                yolo_model_path=str(self.cfg.yolo_model_path),
                imgsz=self.cfg.yolo_imgsz,
                conf_base=self.cfg.yolo_conf,
            ),
            device=self.device,
        )
        self.fastsam_detector = FastSamArmDetector(
            config=YoloPipelineConfig(
                fastsam_model_path=str(self.cfg.fastsam_model_path),
                imgsz=self.cfg.fastsam_imgsz,
                fastsam_conf=self.cfg.fastsam_conf,
            ),
            device=self.device,
        )
        self.rvm_processor = RvmProcessor(
            config=RvmConfig(
                dsr_mobilenet=self.cfg.rvm_dsr_mobilenet,
                dsr_resnet=self.cfg.rvm_dsr_resnet,
            ),
            device=self.device,
        )
        self.bg_processor = BackgroundSubtractionProcessor(
            config=BackgroundSubtractionConfig(
                threshold=self.cfg.bg_threshold,
            )
        )
        self.hand_tracker = HandInteractionTracker(task_path=self.cfg.hand_landmarker_path)
        self.gpu_renderer = GpuHologramRenderer()

        # Configurar estado inicial
        with STATE.lock:
            STATE.arm_only = self.cfg.arm_only
            STATE.guided_filter_on = self.cfg.guided_filter
            STATE.bg_threshold = self.cfg.bg_threshold
            STATE.backbone_name = self._resolve_backbone_name(self.cfg.default_model)

    def _resolve_backbone_name(self, model_key: str) -> str:
        k = model_key.lower()
        if "fastsam" in k:
            return "FastSAM Point-Prompt (CUDA - 58 FPS)"
        elif "yolo" in k:
            return "YOLO11-seg (CUDA - 59 FPS)"
        elif "mobilenet" in k:
            return "RVM MobileNetV3 (CUDA - 35 FPS)"
        elif "resnet" in k:
            return "RVM ResNet50 (CUDA - 10 FPS)"
        elif "fondo" in k or "resta" in k:
            return "Resta de Fondo (Navegador - Frame Inicial)"
        return "FastSAM Point-Prompt (CUDA - 58 FPS)"

    def run(
        self,
        source: Optional[Union[int, str, Path]] = None,
        native_projector: bool = False,
        show_window: bool = False,
    ) -> None:
        """Bucle principal de inferencia y transmisión."""
        # 1. Abrir fuente de video
        if isinstance(source, (str, Path)) and str(source):
            cap: Any = VideoFileSource(str(source))
        elif self.cfg.source:
            cap = VideoFileSource(self.cfg.source)
        else:
            preferred = int(source) if isinstance(source, int) else None
            cap = open_camera(preferred)

        if cap is None:
            logger.error("No se pudo iniciar la cámara web.")
            return

        window_name = "PROYECCION - BRAZO DIGITAL"
        show_gui = native_projector or show_window
        if show_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if native_projector:
                cv2.moveWindow(window_name, 1920, 0)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(window_name, 1280, 720)

        frame_count = 0
        fps_timer = time.perf_counter()
        measured_fps = 0.0
        black_streak = 0
        active_mode = STATE.backbone_name

        try:
            while STATE.running:
                loop_start = time.perf_counter()
                ret, frame_bgr = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if not frame_is_valid(frame_bgr):
                    black_streak += 1
                    if black_streak == 30:
                        logger.warning("Cámara entregando imagen negra continua.")
                    time.sleep(0.033)
                    continue
                black_streak = 0

                # Espejado horizontal para interacción natural
                frame_bgr = cv2.flip(frame_bgr, 1)
                h, w = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Comprobar solicitudes dinámicas de cambio de modelo o reseteo
                with STATE.lock:
                    if STATE.requested_mode is not None:
                        active_mode = STATE.requested_mode
                        STATE.requested_mode = None
                        STATE.backbone_name = active_mode
                        self.rvm_processor.reset_recurrent_states()
                        _ARM_SELECTOR.reset()
                        self.hand_tracker.reset()
                        logger.info("Motor de IA conmutado a: %s", active_mode)
                    if STATE.reset_requested:
                        self.rvm_processor.reset_recurrent_states()
                        _ARM_SELECTOR.reset()
                        self.hand_tracker.reset()
                        STATE.reset_requested = False
                        logger.info("Memoria temporal reiniciada.")

                    arm_only = STATE.arm_only
                    guided_filter_on = STATE.guided_filter_on
                    bg_thresh = STATE.bg_threshold

                # Actualizar frame crudo
                _, raw_jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with STATE.lock:
                    STATE.latest_raw_jpeg = raw_jpeg.tobytes()
                    STATE.latest_raw_bgr = frame_bgr.copy()

                # Tracking de manos con MediaPipe
                hand_landmarks_list, cur_info = self.hand_tracker.process_frame(frame_rgb)
                detected_hands = len(hand_landmarks_list)

                with STATE.lock:
                    STATE.hands_count = detected_hands
                    STATE.cursor = cur_info

                # Tensor GPU normalizado
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0

                # Inferencia según el modelo activo
                if "YOLO" in active_mode and self.yolo_detector.model is not None:
                    active_alpha, _ = self.yolo_detector.predict_mask(frame_bgr, hand_landmarks_list, device=self.device)
                elif "FastSAM" in active_mode and self.fastsam_detector.model is not None:
                    active_alpha, _ = self.fastsam_detector.predict_mask(frame_bgr, hand_landmarks_list, device=self.device)
                elif "Resta de Fondo" in active_mode or "Fondo" in active_mode:
                    with STATE.lock:
                        if STATE.bg_frame is not None and not self.bg_processor.is_calibrated():
                            self.bg_processor.calibrate(STATE.bg_frame)
                        elif STATE.bg_frame is not None and STATE.bg_frame is not self.bg_processor.bg_frame:
                            self.bg_processor.calibrate(STATE.bg_frame)
                        self.bg_processor.cfg.threshold = bg_thresh
                    active_alpha, was_cal = self.bg_processor.compute_mask(frame_bgr, device=self.device)
                    if not was_cal:
                        with STATE.lock:
                            if self.bg_processor.bg_frame is not None:
                                STATE.bg_frame = self.bg_processor.bg_frame.copy()
                else:
                    variant = "resnet50" if "ResNet50" in active_mode else "mobilenetv3"
                    active_alpha = self.rvm_processor.predict_alpha(
                        frame_tensor,
                        variant=variant,
                        has_hands=(detected_hands > 0),
                        device=self.device,
                    )

                # Filtrar solo brazos si está activo
                if arm_only:
                    active_alpha = isolate_arm_and_hand_mask(active_alpha, h, w, hand_landmarks_list, self.device)

                # Renderizar holograma verde en GPU
                _, output_bgr = self.gpu_renderer.render(
                    frame_tensor_rgb=frame_tensor,
                    active_alpha=active_alpha,
                    apply_guided_filter=guided_filter_on,
                )

                # Medición de FPS y latencia
                loop_time = (time.perf_counter() - loop_start) * 1000.0
                frame_count += 1
                if time.perf_counter() - fps_timer >= 1.0:
                    measured_fps = frame_count / (time.perf_counter() - fps_timer)
                    frame_count = 0
                    fps_timer = time.perf_counter()

                with STATE.lock:
                    STATE.fps = measured_fps
                    STATE.latency_ms = loop_time
                    STATE.backbone_name = active_mode

                # Guardar JPEG para el streaming
                _, jpeg_buf = cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with STATE.lock:
                    STATE.latest_jpeg = jpeg_buf.tobytes()

                if show_gui:
                    cv2.imshow(window_name, output_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:
                        break
        finally:
            cap.release()
            if show_gui:
                cv2.destroyAllWindows()
            with STATE.lock:
                STATE.running = False
