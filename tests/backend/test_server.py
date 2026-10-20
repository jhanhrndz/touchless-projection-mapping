"""Tests for the FastAPI runtime server.

Negative cases: malformed draft payloads, bad cursor values, config paths that
escape the data dir, unparseable JSON, missing objects files.
"""

import json

import pytest
from fastapi.testclient import TestClient

from runtime.server import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


# ---------- editor draft (split-window editor) ----------

def test_editor_projection_page_served(client: TestClient):
    res = client.get("/editor/projection")
    assert res.status_code == 200
    assert "Editor projection" in res.text


def test_tune_page_served(client: TestClient):
    res = client.get("/tune")
    assert res.status_code == 200
    assert "Animation tuner" in res.text


def test_editor_draft_starts_empty(client: TestClient):
    res = client.get("/api/editor/draft")
    assert res.status_code == 200
    body = res.json()
    assert body["objects"] == []
    assert body["active_id"] is None
    assert body["cursor"] is None
    # New fields added when the projection page became a peer editor — both
    # /editor and /editor/projection POST changes, and poll for the other side's
    # writes. client_id lets each peer ignore its own echoes; version is a
    # monotonic counter so polls can detect newer state.
    assert body["client_id"] is None
    assert body["version"] == 0


def test_editor_draft_round_trip(client: TestClient):
    payload = {
        "active_id": "obj_a",
        "objects": [
            {"id": "obj_a", "label": "Lamp", "points": [[10, 10], [20, 10], [20, 20]],
             "device_id": None, "style": "normal"},
            {"id": "obj_b", "label": "Mask", "points": [[30, 30], [40, 30], [40, 40]],
             "style": "occluder"},
        ],
    }
    post = client.post("/api/editor/draft", json=payload)
    assert post.status_code == 200
    body = post.json()
    assert body["ok"] is True
    assert body["count"] == 2
    assert isinstance(body["version"], int)  # bumped on every write

    got = client.get("/api/editor/draft").json()
    assert got["active_id"] == "obj_a"
    assert len(got["objects"]) == 2
    assert got["objects"][0]["id"] == "obj_a"
    assert got["objects"][1]["style"] == "occluder"


def test_editor_draft_replaces_previous(client: TestClient):
    """Draft is whole-state replacement, not merge — deleting an object on the
    laptop must clear it from the projector mirror immediately."""
    client.post("/api/editor/draft", json={"objects": [{"id": "x"}], "active_id": "x"})
    client.post("/api/editor/draft", json={"objects": [], "active_id": None})
    got = client.get("/api/editor/draft").json()
    assert got["objects"] == []
    assert got["active_id"] is None
    assert got["cursor"] is None


@pytest.mark.parametrize(
    "bad,reason",
    [
        ({}, "must include 'objects'"),
        ({"objects": "not a list"}, "must include 'objects'"),
        ({"objects": [42]}, "must be a mapping"),
        ({"objects": [None]}, "must be a mapping"),
    ],
)
def test_editor_draft_rejects_malformed(client: TestClient, bad, reason):
    res = client.post("/api/editor/draft", json=bad)
    assert res.status_code == 400
    assert reason in res.text


def test_editor_draft_accepts_partial_polygons(client: TestClient):
    """Drafts are intentionally permissive — 1-vertex shapes mid-draw must save
    so the projection can render the polyline trace."""
    res = client.post("/api/editor/draft", json={
        "objects": [{"id": "obj_x", "label": "", "points": [[5, 5]]}],
        "active_id": "obj_x",
    })
    assert res.status_code == 200


def test_editor_draft_round_trips_cursor(client: TestClient):
    """Held-spacebar cursor preview must survive POST→GET so the projection
    can render it. Both visible=true and visible=false must pass through."""
    payload = {"objects": [], "active_id": None,
               "cursor": {"x": 123.5, "y": 456, "visible": True}}
    client.post("/api/editor/draft", json=payload)
    got = client.get("/api/editor/draft").json()
    assert got["cursor"] == {"x": 123.5, "y": 456, "visible": True}

    payload["cursor"]["visible"] = False
    client.post("/api/editor/draft", json=payload)
    assert client.get("/api/editor/draft").json()["cursor"]["visible"] is False


def test_editor_draft_accepts_null_cursor(client: TestClient):
    """Server must allow drafts without a cursor — same as before this feature."""
    res = client.post("/api/editor/draft", json={"objects": [], "active_id": None})
    assert res.status_code == 200
    assert client.get("/api/editor/draft").json()["cursor"] is None


@pytest.mark.parametrize(
    "bad_cursor,reason",
    [
        ({"x": "not-a-number", "y": 0, "visible": True}, "cursor.x"),
        ({"x": 0, "y": None,    "visible": True}, "cursor.y"),
        ({"x": 0, "y": 0,       "visible": "yes"}, "cursor.visible"),
        ("a string",                                "cursor must be an object"),
    ],
)
def test_editor_draft_rejects_malformed_cursor(client: TestClient, bad_cursor, reason):
    """Bad cursor data should error explicitly — easier to debug than a silent drop."""
    res = client.post("/api/editor/draft", json={
        "objects": [], "active_id": None, "cursor": bad_cursor,
    })
    assert res.status_code == 400
    assert reason in res.text


def test_editor_draft_round_trips_client_id(client: TestClient):
    """Both /editor and /editor/projection now POST changes; each tags its push
    with a client_id so the other side's poll can ignore its own echoes."""
    res = client.post("/api/editor/draft", json={
        "objects": [], "active_id": None, "client_id": "abc-123",
    })
    assert res.status_code == 200
    got = client.get("/api/editor/draft").json()
    assert got["client_id"] == "abc-123"


def test_editor_draft_version_increments_on_post(client: TestClient):
    """version is a monotonic counter so polling clients can detect new state
    without comparing the full object payload."""
    before = client.get("/api/editor/draft").json()["version"]
    client.post("/api/editor/draft", json={"objects": [], "active_id": None})
    after_1 = client.get("/api/editor/draft").json()["version"]
    assert after_1 == before + 1
    client.post("/api/editor/draft", json={"objects": [], "active_id": None})
    after_2 = client.get("/api/editor/draft").json()["version"]
    assert after_2 == after_1 + 1


def test_editor_draft_rejects_bad_client_id(client: TestClient):
    """Non-string client_id is a programming error — fail loud."""
    res = client.post("/api/editor/draft", json={
        "objects": [], "active_id": None, "client_id": 42,
    })
    assert res.status_code == 400
    assert "client_id" in res.text


def test_editor_draft_cursor_patch_bumps_version(client: TestClient):
    """The cursor PATCH endpoint must also bump version so projection polls
    see the update without needing a full POST."""
    before = client.get("/api/editor/draft").json()["version"]
    res = client.patch("/api/editor/draft/cursor",
                       json={"x": 1, "y": 2, "visible": True})
    assert res.status_code == 200
    after = client.get("/api/editor/draft").json()["version"]
    assert after == before + 1


# ---------- PATCH /api/editor/draft/cursor ----------

def test_cursor_patch_updates_cursor_only(client: TestClient):
    """PATCH /api/editor/draft/cursor must update only the cursor field,
    leaving any existing objects in the draft untouched."""
    # First seed some objects via the full POST.
    client.post("/api/editor/draft", json={
        "objects": [{"id": "obj_1", "label": "Lamp", "points": [[10, 10], [20, 10], [20, 20]]}],
        "active_id": "obj_1",
    })
    # Now PATCH just the cursor.
    res = client.patch("/api/editor/draft/cursor",
                       json={"x": 123.5, "y": 456.0, "visible": True})
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    draft = client.get("/api/editor/draft").json()
    assert draft["cursor"] == {"x": 123.5, "y": 456.0, "visible": True}
    # Objects must be preserved — PATCH must not clobber them.
    assert len(draft["objects"]) == 1
    assert draft["objects"][0]["id"] == "obj_1"


def test_cursor_patch_visible_false_hides_dot(client: TestClient):
    """visible=false must update the cursor so projection page hides the dot."""
    client.patch("/api/editor/draft/cursor", json={"x": 100, "y": 200, "visible": True})
    client.patch("/api/editor/draft/cursor", json={"x": 0, "y": 0, "visible": False})
    assert client.get("/api/editor/draft").json()["cursor"]["visible"] is False


@pytest.mark.parametrize("bad,reason", [
    ({"x": "str", "y": 0, "visible": True}, "x is not a number"),
    ({"x": 0, "y": None,  "visible": True}, "y is None"),
    ({"x": 0, "y": 0},                     "visible missing"),
])
def test_cursor_patch_rejects_bad_payload(client: TestClient, bad, reason):
    """Malformed cursor patches should return 422 — not silently corrupt state."""
    res = client.patch("/api/editor/draft/cursor", json=bad)
    assert res.status_code == 422, f"expected 422 for {reason}, got {res.status_code}"


def test_cursor_patch_on_empty_draft(client: TestClient):
    """PATCH with no prior POST should work fine — draft starts empty."""
    res = client.patch("/api/editor/draft/cursor",
                       json={"x": 50, "y": 75, "visible": True})
    assert res.status_code == 200
    draft = client.get("/api/editor/draft").json()
    assert draft["cursor"] == {"x": 50, "y": 75, "visible": True}
    assert draft["objects"] == []  # no objects were added


# ---------- configs browser ----------

def test_configs_list_returns_objects_and_devices(tmp_path, monkeypatch):
    """The configs endpoint should enumerate data/objects*.json and data/devices*.yaml
    with per-file counts. Path-aware: it reads from CWD's `data/` dir."""
    import json as _json
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "objects.json").write_text(_json.dumps([{"id": "a"}, {"id": "b"}, {"id": "c"}]))
    (data_dir / "objects.snapshot_2026-05-22.json").write_text(_json.dumps([{"id": "x"}]))
    (data_dir / "devices.yaml").write_text("- object_id: lamp\n  type: mock\n- object_id: pump\n  type: mock\n")
    (data_dir / "devices.dev.yaml").write_text("- object_id: lamp\n  type: mock\n")

    app = create_app(
        objects_path=data_dir / "objects.json",
        devices_path=data_dir / "devices.yaml",
    )
    res = TestClient(app).get("/api/configs/list")
    assert res.status_code == 200
    body = res.json()

    obj_by_name = {f["name"]: f for f in body["objects"]}
    assert obj_by_name["objects.json"]["count"] == 3
    assert obj_by_name["objects.json"]["active"] is True
    assert obj_by_name["objects.snapshot_2026-05-22.json"]["count"] == 1
    assert obj_by_name["objects.snapshot_2026-05-22.json"]["active"] is False

    dev_by_name = {f["name"]: f for f in body["devices"]}
    assert dev_by_name["devices.yaml"]["count"] == 2
    assert dev_by_name["devices.yaml"]["active"] is True
    assert dev_by_name["devices.dev.yaml"]["count"] == 1
    assert dev_by_name["devices.dev.yaml"]["active"] is False


def test_configs_list_marks_malformed_files_with_negative_count(tmp_path, monkeypatch):
    """Parse failures should not crash the endpoint — count=-1 lets the UI flag them."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "objects.broken.json").write_text("{ not valid json")
    (data_dir / "devices.broken.yaml").write_text("- this:\n  not_balanced: [oops")

    app = create_app()
    body = TestClient(app).get("/api/configs/list").json()
    assert any(f["name"] == "objects.broken.json" and f["count"] == -1 for f in body["objects"])
    assert any(f["name"] == "devices.broken.yaml" and f["count"] == -1 for f in body["devices"])


def test_configs_file_reads_json_within_data_dir(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "objects.dev.json").write_text(_json.dumps([{"id": "x"}, {"id": "y"}]))

    app = create_app()
    res = TestClient(app).get("/api/configs/file?path=data/objects.dev.json")
    assert res.status_code == 200
    body = res.json()
    assert body["data"] == [{"id": "x"}, {"id": "y"}]


def test_configs_file_rejects_paths_outside_data_dir(tmp_path, monkeypatch):
    """Sandbox: must refuse anything not under data/ (no /etc/passwd, no ../escape)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "secret.json").write_text("[]")

    app = create_app()
    client = TestClient(app)
    res1 = client.get("/api/configs/file?path=../secret.json")
    assert res1.status_code == 400
    res2 = client.get("/api/configs/file?path=/etc/passwd")
    assert res2.status_code == 400


def test_configs_file_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    app = create_app()
    res = TestClient(app).get("/api/configs/file?path=data/nope.json")
    assert res.status_code == 404


def test_configs_file_400_on_bad_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "broken.json").write_text("{ not valid")
    app = create_app()
    res = TestClient(app).get("/api/configs/file?path=data/broken.json")
    assert res.status_code == 400
    assert "failed to parse" in res.text


# ---------- scene flags: per-scene (objects file) + global (devices.yaml) ----------

def test_load_objects_accepts_mapping_form_with_scene(tmp_path):
    from runtime.server import load_objects, load_object_scene

    p = tmp_path / "objects.json"
    p.write_text(json.dumps({
        "scene": {"flame_carry": True},
        "objects": [{"id": "a", "label": "A", "polygon": [[0, 0], [1, 0], [1, 1]]}],
    }))
    assert [o["id"] for o in load_objects(p)] == ["a"]
    assert load_object_scene(p) == {"flame_carry": True}


def test_load_objects_bare_list_has_empty_scene(tmp_path):
    from runtime.server import load_objects, load_object_scene

    p = tmp_path / "objects.json"
    p.write_text(json.dumps([{"id": "a", "label": "A", "polygon": [[0, 0], [1, 0], [1, 1]]}]))
    assert [o["id"] for o in load_objects(p)] == ["a"]
    assert load_object_scene(p) == {}  # bare list => no scene flags


def test_load_objects_mapping_with_bad_objects_raises(tmp_path):
    from runtime.server import load_objects

    p = tmp_path / "objects.json"
    p.write_text(json.dumps({"scene": {}, "objects": 42}))
    with pytest.raises(ValueError, match="'objects' must be a list"):
        load_objects(p)


def test_editor_save_preserves_scene_block(tmp_path):
    """Saving from the editor must NOT wipe the objects file's scene flags."""
    import json as _json

    from runtime.editor import ObjectEditor
    from runtime.server import load_object_scene, load_objects

    p = tmp_path / "objects.json"
    p.write_text(_json.dumps({
        "scene": {"flame_carry": True},
        "objects": [{"id": "a", "label": "A", "polygon": [[0, 0], [1, 0], [1, 1]]}],
    }))
    ed = ObjectEditor.from_objects_json(load_objects(p))
    ed.save(p, scene=load_object_scene(p) or None)  # mimic /api/editor/save
    assert load_object_scene(p) == {"flame_carry": True}
    assert isinstance(_json.loads(p.read_text()), dict)
    # No scene => stays a bare list
    ed.save(p, scene=None)
    assert isinstance(_json.loads(p.read_text()), list)


def test_api_scene_reports_object_scene_over_devices_global():
    """/api/scene reflects the resolved scene_settings (objects file wins) and
    passes through the 🔥 cursor tuning fields with defaults."""
    app = create_app(scene_settings={"flame_carry": True, "flame_size": 52})
    with TestClient(app) as client:
        body = client.get("/api/scene").json()
        assert body["flame_carry"] is True
        assert body["flame_size"] == 52.0           # from scene
        assert body["flame_offset_y"] == 14.0       # default
    app2 = create_app()  # nothing set => default off + default sizing
    with TestClient(app2) as client:
        assert client.get("/api/scene").json() == {
            "flame_carry": False, "flame_size": 30.0, "flame_offset_y": 14.0,
        }
