"""Tests for the editor REST endpoints (/editor, /api/editor/*).

Negative cases: removed calibration routes must 404, malformed save payload,
unconfigured save path, invalid object data on save.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime.server import create_app


# ---------------- /editor page ----------------


def test_editor_page_served():
    app = create_app()
    client = TestClient(app)
    res = client.get("/editor")
    assert res.status_code == 200
    assert "Object editor" in res.text


# ---------------- editor canvas geometry ----------------
#
# There is no camera and no homography: the editor canvas IS the projector
# rectangle, so /api/editor/scene reports a configured size and always says
# there's no background image to draw.


def test_scene_meta_reports_default_projector_size():
    client = TestClient(create_app())
    body = client.get("/api/editor/scene").json()
    assert body["projector"] == [1920, 1080]
    assert body["scene_available"] is False


def test_scene_meta_reports_configured_projector_size():
    client = TestClient(create_app(projector_size=(1280, 800)))
    assert client.get("/api/editor/scene").json()["projector"] == [1280, 800]


def test_scene_meta_never_advertises_a_background_image():
    """The editor keys its background layer off scene_available/image_url.

    Both must stay absent-or-false, or the canvas tries to fetch a capture
    endpoint that no longer exists and renders a broken image over the wall.
    """
    body = TestClient(create_app()).get("/api/editor/scene").json()
    assert body["scene_available"] is False
    assert "image_url" not in body


def test_camera_and_calibration_routes_are_gone():
    """Negative: the removed calibration surface must 404, not linger.

    A stale /calibrate page that still renders would send someone through a
    setup step this build no longer supports.
    """
    client = TestClient(create_app())
    for route in (
        "/calibrate",
        "/calibrate/manual",
        "/probe",
        "/api/calibrate/markers",
        "/api/calibrate/homography",
        "/api/editor/scene.png",
    ):
        assert client.get(route).status_code == 404, route
    assert client.post("/api/editor/scene/refresh").status_code == 404


# ---------------- save endpoint ----------------


def test_save_writes_objects_json(tmp_path: Path):
    objects_path = tmp_path / "objects.json"
    app = create_app(objects_path=objects_path, objects=[])
    client = TestClient(app)

    payload = {
        "objects": [
            {"id": "obj_1", "label": "lamp",
             "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]]}
        ]
    }
    res = client.post("/api/editor/save", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["saved"] == 1

    saved = json.loads(objects_path.read_text())
    assert len(saved) == 1
    assert saved[0]["id"] == "obj_1"

    # /api/objects should reflect the new state immediately, no server restart.
    objs = client.get("/api/objects").json()
    assert len(objs) == 1


def test_save_503_when_path_unconfigured():
    app = create_app()  # no objects_path
    client = TestClient(app)
    res = client.post("/api/editor/save", json={"objects": []})
    assert res.status_code == 503


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        ({}, 400),
        ({"objects": "notalist"}, 400),
        ({"objects": [{"id": "x"}]}, 400),  # missing polygon
    ],
)
def test_save_validation(tmp_path: Path, payload, expected_status):
    app = create_app(objects_path=tmp_path / "objects.json")
    client = TestClient(app)
    res = client.post("/api/editor/save", json=payload)
    assert res.status_code == expected_status


def test_save_drops_objects_with_too_few_points(tmp_path: Path):
    """Editor's runtime-compatible save filters objects below 3 points."""
    objects_path = tmp_path / "objects.json"
    app = create_app(objects_path=objects_path)
    client = TestClient(app)

    payload = {
        "objects": [
            {"id": "good", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]]},
            {"id": "incomplete", "label": "", "polygon": [[0, 0], [1, 0]]},
        ]
    }
    res = client.post("/api/editor/save", json=payload)
    assert res.status_code == 200
    saved = json.loads(objects_path.read_text())
    assert len(saved) == 1
    assert saved[0]["id"] == "good"


def test_save_persists_device_id_binding(tmp_path: Path):
    objects_path = tmp_path / "objects.json"
    app = create_app(objects_path=objects_path)
    client = TestClient(app)

    payload = {
        "objects": [
            {
                "id": "obj_pump",
                "label": "Pump",
                "polygon": [[0, 0], [10, 0], [10, 10]],
                "device_id": "water_pump",
            }
        ]
    }
    res = client.post("/api/editor/save", json=payload)
    assert res.status_code == 200
    saved = json.loads(objects_path.read_text())
    assert saved[0]["device_id"] == "water_pump"
    # /api/objects reflects it too
    assert client.get("/api/objects").json()[0]["device_id"] == "water_pump"


def test_save_omits_device_id_when_not_set(tmp_path: Path):
    """Unbound polygons should not have a device_id key in the saved JSON
    (keeps the file clean and matches the editor's optional-field convention)."""
    objects_path = tmp_path / "objects.json"
    app = create_app(objects_path=objects_path)
    client = TestClient(app)

    payload = {"objects": [
        {"id": "obj_x", "label": "", "polygon": [[0, 0], [10, 0], [10, 10]]}
    ]}
    res = client.post("/api/editor/save", json=payload)
    assert res.status_code == 200
    saved = json.loads(objects_path.read_text())
    assert "device_id" not in saved[0]


def test_save_strips_style_only_fields(tmp_path: Path):
    """The /style inspector's `active_id` and `preview_mode` are ephemeral
    UI state — they MUST NOT leak into the saved objects.json (where the
    main wall would read them and accidentally treat one object as
    permanently "active"). This guards against future bugs where someone
    might naively pass-through extra payload keys."""
    objects_path = tmp_path / "objects.json"
    app = create_app(objects_path=objects_path)
    client = TestClient(app)

    payload = {
        # These belong on /api/style/draft, not on save. The save endpoint
        # should silently drop them since ObjectEditor only reads its known
        # schema fields.
        "active_id": "obj_x",
        "preview_mode": "active_hovered",
        "objects": [
            {
                "id": "obj_x", "label": "book",
                "polygon": [[0, 0], [10, 0], [10, 10]],
                # And per-object pollution too — extra keys must drop.
                "active": True,
                "preview_mode": "all_idle",
            }
        ],
    }
    res = client.post("/api/editor/save", json=payload)
    assert res.status_code == 200
    saved = json.loads(objects_path.read_text())
    assert len(saved) == 1
    assert "active" not in saved[0]
    assert "preview_mode" not in saved[0]
    assert "active_id" not in saved[0]
    # /api/objects (what the main wall sees) must also be free of these.
    served = client.get("/api/objects").json()
    assert "active" not in served[0]
    assert "preview_mode" not in served[0]
