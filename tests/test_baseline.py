"""Pruebas unitarias de línea base (Fase 0).

Verifican importación, VideoFileSource, estado del servidor y endpoints HTTP.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from vision_engine.config import EngineConfig
from vision_engine.engine import VideoFileSource
from vision_engine.server import (
    STATE as state,
    ServerState,
    VisionEngineHTTPHandler as EngineHTTPHandler,
    MAX_POST_BODY_BYTES,
)


def test_imports_and_constants() -> None:
    """Verifica que vision_engine expone constantes y clases base esperadas."""
    assert VideoFileSource is not None
    assert EngineConfig is not None
    assert ServerState is not None
    assert EngineHTTPHandler is not None
    assert MAX_POST_BODY_BYTES == 65536


def test_video_file_source() -> None:
    """Verifica que VideoFileSource lee frames válidos y hace loop correctamente."""
    clip_path = PROJECT_ROOT / "tests" / "clips" / "clip_a_hand_entry_borders.mp4"
    if not clip_path.exists():
        pytest.skip(f"Clip sintético no encontrado en {clip_path}")

    source = VideoFileSource(str(clip_path), loop=True)
    assert source.is_opened()
    width, height = source.get_dimensions()
    assert width > 0
    assert height > 0

    # Leer primeros 5 frames
    for i in range(5):
        ret, frame = source.read()
        assert ret is True
        assert frame is not None
        assert frame.shape[0] == height
        assert frame.shape[1] == width

    source.release()
    assert not source.is_opened()


def test_server_state_defaults() -> None:
    """Verifica valores iniciales del estado compartido del servidor."""
    s = ServerState()
    assert s.arm_only is False
    assert s.guided_filter is False
    assert s.fps == 0.0
    assert s.frame_count == 0
    assert s.is_running is True


class DummyServerRunner:
    """Inicia EngineHTTPHandler en un puerto de prueba local."""

    def __init__(self, port: int = 5991) -> None:
        self.port = port
        self.state = state
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), EngineHTTPHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()
        time.sleep(0.1)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_http_api_endpoints() -> None:
    """Verifica respuestas de /api/status, /api/model y /api/settings."""
    runner = DummyServerRunner(port=5991)
    runner.start()
    base_url = f"http://127.0.0.1:{runner.port}"

    try:
        # 1. GET /api/status
        with urllib.request.urlopen(f"{base_url}/api/status", timeout=2) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["active_model"] == "rvm_mobilenet"
            assert data["arm_only"] is False

        # 2. POST /api/model cambiar a 'yolo'
        req = urllib.request.Request(
            f"{base_url}/api/model",
            data=json.dumps({"model": "yolo"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert runner.state.active_model == "yolo"

        # 2b. POST /api/model cambiar a 'background_subtraction' (Modelo 5)
        req_bg = urllib.request.Request(
            f"{base_url}/api/model",
            data=json.dumps({"model": "background_subtraction"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_bg, timeout=2) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert runner.state.active_model == "background_subtraction"

        # 3. POST /api/settings modificar guided_filter
        req = urllib.request.Request(
            f"{base_url}/api/settings",
            data=json.dumps({"guided_filter": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["guided_filter"] is True
            assert runner.state.guided_filter is True

        # 4. SEC-01: Probar que un payload mayor a MAX_POST_BODY_BYTES es rechazado con 413
        oversized_data = b"x" * (MAX_POST_BODY_BYTES + 1024)
        req_overflow = urllib.request.Request(
            f"{base_url}/api/settings",
            data=oversized_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_overflow, timeout=2)
        assert exc_info.value.code == 413

    finally:
        runner.stop()
