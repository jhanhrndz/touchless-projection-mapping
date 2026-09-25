"""Tests for runtime endpoints — /, /api/objects, /ws, and load_objects.

Negative cases per project convention: invalid JSON, malformed object specs,
unknown object IDs in WS, bad button names, missing message fields.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime.devices import MockDevice
from runtime.devices.registry import DeviceRegistry
from runtime.server import create_app, load_objects


@pytest.fixture
def app_with_mock_switch() -> TestClient:
    registry = DeviceRegistry()
    registry.add(MockDevice("lamp_switch", label="Lamp switch"))
    objects = [
        {
            "id": "lamp_switch",
            "label": "Lamp switch",
            "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]],
        }
    ]
    app = create_app(devices=registry, objects=objects)
    return TestClient(app)


# ---------------- runtime page + /api/objects ----------------


def test_runtime_page_served(app_with_mock_switch: TestClient):
    res = app_with_mock_switch.get("/")
    assert res.status_code == 200
    assert "Interactive Projection" in res.text


def test_get_objects_returns_loaded_list(app_with_mock_switch: TestClient):
    res = app_with_mock_switch.get("/api/objects")
    assert res.status_code == 200
    objects = res.json()
    assert len(objects) == 1
    assert objects[0]["id"] == "lamp_switch"


def test_get_devices_lists_registry(app_with_mock_switch: TestClient):
    res = app_with_mock_switch.get("/api/devices")
    assert res.status_code == 200
    devices = res.json()
    assert len(devices) == 1
    assert devices[0]["object_id"] == "lamp_switch"
    assert devices[0]["label"] == "Lamp switch"
    assert devices[0]["type"] == "MockDevice"


def test_get_devices_empty_when_no_registry(app_with_mock_switch: TestClient):
    """A registry with no entries returns an empty list, not a 500."""
    from runtime.devices.registry import DeviceRegistry
    app_with_mock_switch.app.state.devices = DeviceRegistry()
    res = app_with_mock_switch.get("/api/devices")
    assert res.status_code == 200
    assert res.json() == []


# ---------------- /ws hover/click dispatch ----------------


def test_ws_hover_round_trip(app_with_mock_switch: TestClient):
    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hover", "id": "lamp_switch", "enter": True})
        msg = ws.receive_json()
        assert msg == {"type": "ack", "id": "lamp_switch", "kind": "hover"}


def test_ws_click_round_trip(app_with_mock_switch: TestClient):
    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "click", "id": "lamp_switch", "button": "left"})
        msg = ws.receive_json()
        assert msg == {"type": "ack", "id": "lamp_switch", "kind": "click", "button": "left"}


def test_ws_dispatches_to_registered_device(app_with_mock_switch: TestClient):
    """Each WS event should reach the underlying MockDevice and get recorded."""
    registry: DeviceRegistry = app_with_mock_switch.app.state.devices
    switch = registry.get("lamp_switch")
    assert isinstance(switch, MockDevice)

    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hover", "id": "lamp_switch", "enter": True})
        ws.receive_json()
        ws.send_json({"type": "click", "id": "lamp_switch", "button": "left"})
        ws.receive_json()

    kinds = [c.kind for c in switch.calls]
    assert kinds == ["hover", "click"]


def test_ws_unknown_object_id(app_with_mock_switch: TestClient):
    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hover", "id": "nonexistent", "enter": True})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "unknown object" in msg["error"]


def test_ws_unknown_event_type(app_with_mock_switch: TestClient):
    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "explode", "id": "lamp_switch"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "unknown event type" in msg["error"]


def test_ws_bad_button(app_with_mock_switch: TestClient):
    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "click", "id": "lamp_switch", "button": "weird"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "bad button" in msg["error"]


def test_ws_missing_required_fields(app_with_mock_switch: TestClient):
    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hover"})  # no id
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_ws_menu_returns_items(app_with_mock_switch: TestClient):
    with app_with_mock_switch.websocket_connect("/ws") as ws:
        ws.send_json({"type": "menu", "id": "lamp_switch"})
        msg = ws.receive_json()
        assert msg["type"] == "menu"
        assert isinstance(msg["items"], list)


# ---------------- load_objects ----------------


def test_load_objects_missing_file_returns_empty(tmp_path: Path):
    assert load_objects(tmp_path / "nope.json") == []


def test_load_objects_happy(tmp_path: Path):
    p = tmp_path / "objects.json"
    p.write_text(json.dumps([
        {"id": "a", "label": "A", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
    ]))
    objects = load_objects(p)
    assert len(objects) == 1
    assert objects[0]["id"] == "a"


@pytest.mark.parametrize(
    "raw,match",
    [
        # A top-level scalar is neither a list nor the {scene, objects} mapping.
        ('42', "expected a list of objects or a mapping"),
        # Mapping form is valid, but `objects` must be a list.
        ('{"objects": 42}', "'objects' must be a list"),
        ('[42]', "expected mapping"),
        ('[{"id": "a"}]', "invalid object"),  # missing polygon
        ('[{"polygon": [[0,0],[1,0],[1,1]]}]', "invalid object"),  # missing id
    ],
)
def test_load_objects_validation(tmp_path: Path, raw: str, match: str):
    p = tmp_path / "objects.json"
    p.write_text(raw)
    with pytest.raises(ValueError, match=match):
        load_objects(p)
