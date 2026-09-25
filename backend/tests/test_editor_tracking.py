"""Tests for the per-object `tracking` field — the pointer-follow flag used
by the popcorn-slider project. Mirrors test_editor.py's structure for the
animation/click blocks (round-trip + negatives)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.editor import EditableObject, ObjectEditor


# ---------- happy paths ----------


def test_new_object_starts_with_empty_tracking():
    obj = EditableObject(id="x")
    assert obj.tracking == {}


def test_set_tracking_round_trips_all_fields():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_tracking(obj.id, {
        "axis": "x", "binding": "click_to_bind", "home_x_frac": 0.5,
    })
    assert ed.get(obj.id).tracking == {
        "axis": "x", "binding": "click_to_bind", "home_x_frac": 0.5,
    }


def test_set_tracking_partial_only_keeps_provided_keys():
    """Missing keys fall through to runtime defaults — schema must not invent any."""
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_tracking(obj.id, {"binding": "click_to_bind"})
    assert ed.get(obj.id).tracking == {"binding": "click_to_bind"}


def test_set_tracking_none_clears():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_tracking(obj.id, {"axis": "x", "binding": "click_to_bind"})
    ed.set_tracking(obj.id, None)
    assert ed.get(obj.id).tracking == {}


def test_set_tracking_empty_dict_clears():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_tracking(obj.id, {"axis": "x"})
    ed.set_tracking(obj.id, {})
    assert ed.get(obj.id).tracking == {}


def test_set_tracking_unknown_object_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.set_tracking("nope", {"axis": "x"})


def test_to_objects_json_omits_empty_tracking():
    """Default empty value must not bloat objects.json — files that predate
    the tracking field stay byte-identical on round-trip."""
    ed = ObjectEditor()
    obj = ed.create_object("popcorn")
    for p in [(0, 0), (10, 0), (10, 10)]:
        ed.add_point(obj.id, p)
    out = ed.to_objects_json()
    assert "tracking" not in out[0]


def test_to_objects_json_includes_tracking_when_set():
    ed = ObjectEditor()
    obj = ed.create_object("popcorn")
    for p in [(0, 0), (10, 0), (10, 10)]:
        ed.add_point(obj.id, p)
    ed.set_tracking(obj.id, {
        "axis": "x", "binding": "click_to_bind", "home_x_frac": 0.5,
    })
    out = ed.to_objects_json()
    assert out[0]["tracking"] == {
        "axis": "x", "binding": "click_to_bind", "home_x_frac": 0.5,
    }


def test_round_trip_preserves_tracking(tmp_path: Path):
    ed = ObjectEditor()
    obj = ed.create_object("popcorn")
    for p in [(0, 0), (100, 0), (100, 80), (0, 80)]:
        ed.add_point(obj.id, p)
    ed.set_tracking(obj.id, {
        "axis": "x", "binding": "click_to_bind", "home_x_frac": 0.25,
    })
    path = tmp_path / "objects.json"
    ed.save(path)
    loaded = ObjectEditor.load(path)
    assert loaded.get(obj.id).tracking == {
        "axis": "x", "binding": "click_to_bind", "home_x_frac": 0.25,
    }


def test_from_objects_json_defaults_missing_tracking_to_empty():
    """A pre-tracking objects.json file must load cleanly with no tracking set."""
    data = [{
        "id": "old_obj",
        "polygon": [[0, 0], [1, 0], [1, 1]],
    }]
    ed = ObjectEditor.from_objects_json(data)
    assert ed.get("old_obj").tracking == {}


# ---------- negative cases ----------


@pytest.mark.parametrize("bad,reason", [
    ({"axis": "y"},                   r"axis must be one of"),     # y not supported yet
    ({"axis": "diagonal"},            r"axis must be one of"),
    ({"binding": "always_bound"},     r"binding must be one of"),  # only click_to_bind today
    ({"binding": "drag"},             r"binding must be one of"),
    ({"home_x_frac": 1.5},            r"home_x_frac"),
    ({"home_x_frac": -0.1},           r"home_x_frac"),
    ({"home_x_frac": "half"},         r"home_x_frac"),
    ({"home_x_frac": True},           r"home_x_frac"),             # bool is not a number
    ({"unknown_key": "wat"},          r"unknown key"),
    ({"axis": "x", "extra": 1},       r"unknown key"),
])
def test_set_tracking_validates(bad, reason):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=reason):
        ed.set_tracking(obj.id, bad)


def test_set_tracking_rejects_non_dict():
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match="tracking must be an object"):
        ed.set_tracking(obj.id, "x")  # type: ignore[arg-type]


def test_from_objects_json_surfaces_bad_tracking_with_index():
    """Schema errors on load must include the index path so the editor can
    point at the broken entry instead of failing opaquely."""
    data = [{
        "id": "ok", "polygon": [[0, 0], [1, 0], [1, 1]],
    }, {
        "id": "bad", "polygon": [[0, 0], [1, 0], [1, 1]],
        "tracking": {"axis": "z"},
    }]
    with pytest.raises(ValueError, match=r"objects\[1\].*axis must be one of"):
        ObjectEditor.from_objects_json(data)
