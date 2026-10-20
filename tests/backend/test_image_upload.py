"""Tests for the per-object image upload endpoint (POST /api/style/image).

Happy path: PNG / JPG / WebP upload returns a valid src + dimensions, the
file lands under data/objects/, and Pillow round-trip preserves the content.

Negative path per project convention (feedback_negative_tests.md): bad
extension, oversize file, non-image content, malformed object_id (would let
an attacker write `..` into the filename), and empty upload.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from runtime.server import UPLOAD_MAX_BYTES, create_app


def _png_bytes(w: int = 32, h: int = 32, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """Build a real in-memory PNG of the requested size + color."""
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes(w: int = 32, h: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (0, 200, 100)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _webp_bytes(w: int = 32, h: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (50, 50, 200)).save(buf, format="WEBP")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient anchored at a tmp cwd so uploads land in tmp_path/data/objects.

    The endpoint writes to `data/objects/` relative to CWD — pointing cwd at
    tmp_path keeps each test hermetic and avoids polluting the real
    data/objects/ during CI.
    """
    monkeypatch.chdir(tmp_path)
    app = create_app()
    return TestClient(app)


# ----------------- happy path -----------------


def test_upload_png_returns_src_and_dims(client, tmp_path):
    res = client.post(
        "/api/style/image",
        files={"file": ("cover.png", _png_bytes(40, 60), "image/png")},
        data={"object_id": "obj_1779718692_e80311e7"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert body["src"].startswith("data/objects/obj_1779718692_e80311e7_")
    assert body["src"].endswith(".png")
    assert body["w"] == 40 and body["h"] == 60
    # File actually written to disk under the cwd-relative path.
    assert (tmp_path / body["src"]).exists()


def test_upload_jpg_keeps_jpg_extension(client):
    res = client.post(
        "/api/style/image",
        files={"file": ("photo.jpg", _jpg_bytes(), "image/jpeg")},
        data={"object_id": "obj_test_123"},
    )
    assert res.status_code == 200
    assert res.json()["src"].endswith(".jpg")


def test_upload_jpeg_extension_normalizes_to_jpg(client):
    """`.jpeg` and `.jpg` are the same format — pick one for the filename so
    the served URL is predictable."""
    res = client.post(
        "/api/style/image",
        files={"file": ("photo.jpeg", _jpg_bytes(), "image/jpeg")},
        data={"object_id": "obj_test_456"},
    )
    assert res.status_code == 200
    assert res.json()["src"].endswith(".jpg")


def test_upload_webp(client):
    res = client.post(
        "/api/style/image",
        files={"file": ("cover.webp", _webp_bytes(), "image/webp")},
        data={"object_id": "obj_test_789"},
    )
    assert res.status_code == 200
    assert res.json()["src"].endswith(".webp")


def test_upload_large_image_gets_downscaled(client, tmp_path):
    """Originals above 2048 on the long axis are resized to keep bandwidth down."""
    res = client.post(
        "/api/style/image",
        files={"file": ("big.png", _png_bytes(4000, 2000), "image/png")},
        data={"object_id": "obj_big"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["w"] == 2048
    assert body["h"] == 1024  # aspect preserved


def test_upload_round_trip_is_a_valid_image(client, tmp_path):
    """The on-disk file must still open with Pillow — guards against the
    code path silently writing garbage when format conversion fails."""
    res = client.post(
        "/api/style/image",
        files={"file": ("cover.png", _png_bytes(32, 32, (200, 100, 50)), "image/png")},
        data={"object_id": "obj_rt"},
    )
    assert res.status_code == 200
    saved = tmp_path / res.json()["src"]
    with Image.open(saved) as reopened:
        reopened.verify()


def test_upload_then_served_by_data_route(client, tmp_path):
    """End-to-end: upload, then GET /data/objects/<name> returns the bytes."""
    up = client.post(
        "/api/style/image",
        files={"file": ("c.png", _png_bytes(8, 8), "image/png")},
        data={"object_id": "obj_e2e"},
    )
    assert up.status_code == 200
    src = up.json()["src"]  # "data/objects/obj_e2e_<ts>.png"
    name = src.removeprefix("data/objects/")
    served = client.get(f"/data/objects/{name}")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    # Bytes round-trip into a valid PNG via Pillow.
    with Image.open(io.BytesIO(served.content)) as reopened:
        reopened.verify()


# ----------------- negative path -----------------


@pytest.mark.parametrize("ext", [".exe", ".svg", ".gif", ".bmp", ""])
def test_upload_rejects_disallowed_extensions(client, ext):
    res = client.post(
        "/api/style/image",
        files={"file": ("malware" + ext, _png_bytes(), "application/octet-stream")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400
    assert "unsupported" in res.text.lower() or "file type" in res.text.lower()


def test_upload_rejects_non_image_content_with_image_extension(client):
    """An attacker renames a binary to `cover.png` — Pillow must detect that
    the bytes aren't a valid image."""
    res = client.post(
        "/api/style/image",
        files={"file": ("malware.png", b"this is not a png at all", "image/png")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400
    assert "not a valid image" in res.text


def test_upload_rejects_empty_file(client):
    res = client.post(
        "/api/style/image",
        files={"file": ("empty.png", b"", "image/png")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400
    assert "empty" in res.text.lower()


def test_upload_rejects_oversize_file(client):
    """Anything above UPLOAD_MAX_BYTES is rejected before any image processing."""
    too_big = b"a" * (UPLOAD_MAX_BYTES + 1)
    res = client.post(
        "/api/style/image",
        files={"file": ("huge.png", too_big, "image/png")},
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 400
    assert "too large" in res.text.lower()


@pytest.mark.parametrize("bad_id", [
    "obj_with/slash",
    "obj_with..dots",
    "obj with space",
    "../escape",
    "/absolute",
    "",
    "a" * 65,             # over the 64-char cap
    "obj_$pecial",        # $ is not in the allowed charset
])
def test_upload_rejects_malformed_object_id(client, bad_id):
    """A bad object_id would leak into the saved filename — rejected up front
    so we never end up with a `../etc/passwd_123.png` on disk. Either our
    regex check (400) or FastAPI's required-field validator (422 for empty
    strings) is a valid rejection — both keep the bad ID out."""
    res = client.post(
        "/api/style/image",
        files={"file": ("cover.png", _png_bytes(), "image/png")},
        data={"object_id": bad_id},
    )
    assert res.status_code in (400, 422)


def test_upload_missing_object_id_returns_422(client):
    """FastAPI's Form(...) marks the field required — missing it surfaces as
    422 from the validator, not a hand-rolled 400."""
    res = client.post(
        "/api/style/image",
        files={"file": ("cover.png", _png_bytes(), "image/png")},
    )
    assert res.status_code == 422


def test_upload_missing_file_returns_422(client):
    res = client.post(
        "/api/style/image",
        data={"object_id": "obj_x"},
    )
    assert res.status_code == 422


# ----------------- /data/objects sandbox -----------------


def test_data_objects_sandbox_rejects_traversal(client):
    """The image-serve route must refuse `..` segments so a maliciously-edited
    objects.json can't read /etc/passwd via the projector page."""
    res = client.get("/data/objects/..%2F..%2Fetc%2Fpasswd")
    assert res.status_code in (400, 404)


def test_data_objects_404_when_missing(client):
    res = client.get("/data/objects/nope_does_not_exist.png")
    assert res.status_code == 404
