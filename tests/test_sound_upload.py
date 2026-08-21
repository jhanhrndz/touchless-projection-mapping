"""Tests for the per-object hover-sound upload endpoint (POST /api/style/sound).

Same shape as test_image_upload.py — happy paths for each accepted format
and negative paths for bad extension, oversize, non-audio content, malformed
object_id. Negative tests are required by project convention
(feedback_negative_tests.md): every test file gets failure cases.
"""

from __future__ import annotations

import io
import struct

import pytest
from fastapi.testclient import TestClient

from runtime.server import SOUND_MAX_BYTES, create_app


def _wav_bytes() -> bytes:
    """Minimal valid 1-frame mono 8-bit WAV. Enough to pass magic-byte
    sniffing without bringing in a real audio library."""
    sample = b"\x80"  # one silent 8-bit unsigned sample
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 8000, 8000, 1, 8)
    data_chunk = struct.pack("<4sI", b"data", len(sample)) + sample
    body = b"WAVE" + fmt_chunk + data_chunk
    return struct.pack("<4sI", b"RIFF", len(body)) + body


def _mp3_bytes() -> bytes:
    """Minimal-ish MP3 stub: ID3 v2 header + a single byte. The endpoint
    only inspects the first 12 bytes for the sniff check."""
    # ID3v2.3 header: 'ID3' + version(2) + flags(1) + size(4 syncsafe = 0)
    header = b"ID3\x03\x00\x00" + b"\x00\x00\x00\x00"
    return header + b"\x00" * 16


def _ogg_bytes() -> bytes:
    """Minimal OggS-prefixed stub."""
    return b"OggS" + b"\x00" * 28


def _m4a_bytes() -> bytes:
    """Stub with the 'ftyp' marker at bytes 4-7 that M4A files share."""
    return b"\x00\x00\x00 ftypM4A " + b"\x00" * 16


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient anchored at a tmp cwd so uploads land in tmp_path/data/sounds.
    Same hermetic pattern as test_image_upload.py."""
    monkeypatch.chdir(tmp_path)
    app = create_app()
    return TestClient(app)


# ----------------- happy path -----------------


@pytest.mark.parametrize("name,bytes_fn,ext", [
    ("hover.wav", _wav_bytes, ".wav"),
    ("hover.mp3", _mp3_bytes, ".mp3"),
    ("hover.ogg", _ogg_bytes, ".ogg"),
    ("hover.m4a", _m4a_bytes, ".m4a"),
])
def test_upload_each_format(client, tmp_path, name, bytes_fn, ext):
    res = client.post(
        "/api/style/sound",
        files={"file": (name, bytes_fn(), "audio/whatever")},
        data={"object_id": "obj_test"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert body["src"].startswith("data/sounds/obj_test_")
    assert body["src"].endswith(ext)
    assert (tmp_path / body["src"]).exists()


def test_upload_then_served_by_data_route(client):
    """End-to-end: upload, then GET /data/sounds/<name> returns the bytes."""
    up = client.post(
        "/api/style/sound",
        files={"file": ("h.wav", _wav_bytes(), "audio/wav")},
        data={"object_id": "obj_e2e"},
    )
    assert up.status_code == 200
    src = up.json()["src"]
    name = src.removeprefix("data/sounds/")
    served = client.get(f"/data/sounds/{name}")
    assert served.status_code == 200
    assert served.headers["content-type"] == "audio/wav"


# ----------------- negative path -----------------


@pytest.mark.parametrize("ext", [".exe", ".svg", ".midi", ".flac", ""])
def test_upload_rejects_disallowed_extensions(client, ext):
    res = client.post(
        "/api/style/sound",
        files={"file": ("evil" + ext, _wav_bytes(), "application/octet-stream")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400


def test_upload_rejects_non_audio_content_with_audio_extension(client):
    """Renamed binary content masquerading as .wav must be caught by magic-byte
    sniffing."""
    res = client.post(
        "/api/style/sound",
        files={"file": ("malware.wav", b"this is not a wav file at all", "audio/wav")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400
    assert "magic-byte" in res.text.lower() or "look like" in res.text.lower()


def test_upload_rejects_format_mismatch(client):
    """A real WAV uploaded with a .mp3 extension should be rejected — the
    extension determines the format we're sniffing for."""
    res = client.post(
        "/api/style/sound",
        files={"file": ("wrong.mp3", _wav_bytes(), "audio/mpeg")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400


def test_upload_rejects_empty_file(client):
    res = client.post(
        "/api/style/sound",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400


def test_upload_rejects_oversize_file(client):
    too_big = b"\x00" * (SOUND_MAX_BYTES + 1)
    res = client.post(
        "/api/style/sound",
        files={"file": ("huge.wav", too_big, "audio/wav")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400
    assert "too large" in res.text.lower()


@pytest.mark.parametrize("bad_id", [
    "obj_with/slash", "obj with space", "../escape", "/absolute",
    "a" * 65, "obj_$pecial",
])
def test_upload_rejects_malformed_object_id(client, bad_id):
    res = client.post(
        "/api/style/sound",
        files={"file": ("h.wav", _wav_bytes(), "audio/wav")},
        data={"object_id": bad_id},
    )
    assert res.status_code in (400, 422)


def test_data_sounds_sandbox_rejects_traversal(client):
    res = client.get("/data/sounds/..%2F..%2Fetc%2Fpasswd")
    assert res.status_code in (400, 404)


def test_data_sounds_404_when_missing(client):
    res = client.get("/data/sounds/no_such_file.mp3")
    assert res.status_code == 404
