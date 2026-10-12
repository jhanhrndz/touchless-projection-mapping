"""Módulo de compatibilidad hacia atrás para tests y herramientas que importan cuda_engine."""

from vision_engine.config import EngineConfig, OBJECTS_PATH
from vision_engine.engine import (
    VisionEngine,
    VideoFileSource,
    frame_is_valid,
    open_camera,
    isolate_arm_and_hand_mask,
)
from vision_engine.server import (
    STATE,
    ServerState,
    EngineState,
    SERVER_STATE,
    VisionEngineHTTPHandler as EngineHTTPHandler,
    VisionEngineHTTPHandler as ProjectionHTTPHandler,
    MAX_POST_BODY_BYTES,
    start_http_server,
)

state = STATE

__all__ = [
    "EngineConfig",
    "VideoFileSource",
    "frame_is_valid",
    "open_camera",
    "isolate_arm_and_hand_mask",
    "VisionEngine",
    "STATE",
    "state",
    "ServerState",
    "EngineState",
    "SERVER_STATE",
    "EngineHTTPHandler",
    "ProjectionHTTPHandler",
    "MAX_POST_BODY_BYTES",
    "start_http_server",
    "OBJECTS_PATH",
]
