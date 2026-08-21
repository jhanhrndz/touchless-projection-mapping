"""Tests for the headless ObjectEditor — covers the operations the UI will perform.

Negative cases per project convention: unknown IDs, malformed points, missing
fields on load, removing from empty, deleting nonexistent, drop-on-save semantics.
"""

import json
import re
from pathlib import Path

import pytest

from runtime.editor import EditableObject, ObjectEditor, new_object_id


# ---------------- ID generation ----------------


def test_new_object_id_format():
    oid = new_object_id()
    assert re.fullmatch(r"obj_\d+_[0-9a-f]{8}", oid), f"bad id format: {oid!r}"


def test_new_object_id_unique_across_many_calls():
    """8 hex chars = 4-billion space; 1000 calls in the same second won't collide."""
    ids = {new_object_id() for _ in range(1000)}
    assert len(ids) == 1000


# ---------------- create / delete / label ----------------


def test_create_object_returns_obj_with_id():
    ed = ObjectEditor()
    obj = ed.create_object(label="lamp")
    assert isinstance(obj, EditableObject)
    assert obj.id.startswith("obj_")
    assert obj.label == "lamp"
    assert obj.points == []
    assert obj.id in ed
    assert len(ed) == 1


def test_create_multiple_objects_have_distinct_ids():
    ed = ObjectEditor()
    a = ed.create_object("a")
    b = ed.create_object("b")
    assert a.id != b.id


def test_delete_object_removes_from_list():
    ed = ObjectEditor()
    a = ed.create_object("a")
    b = ed.create_object("b")
    ed.delete_object(a.id)
    assert a.id not in ed
    assert b.id in ed
    assert len(ed) == 1


def test_delete_unknown_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError, match="unknown object"):
        ed.delete_object("obj_nope")


def test_set_label_updates_object():
    ed = ObjectEditor()
    obj = ed.create_object("old")
    ed.set_label(obj.id, "new")
    assert ed.get(obj.id).label == "new"


def test_set_label_unknown_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.set_label("obj_nope", "x")


def test_get_unknown_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.get("obj_nope")


# ---------------- add / remove / clear points ----------------


def test_add_point_appends_in_order():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.add_point(obj.id, (10, 20))
    ed.add_point(obj.id, (30, 40))
    assert ed.get(obj.id).points == [[10.0, 20.0], [30.0, 40.0]]


def test_add_point_with_floats():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.add_point(obj.id, (10.5, 20.25))
    assert ed.get(obj.id).points == [[10.5, 20.25]]


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3), ()])
def test_add_point_rejects_non_2d(bad):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=r"\(x, y\)"):
        ed.add_point(obj.id, bad)


def test_add_point_to_unknown_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.add_point("obj_nope", (10, 20))


def test_remove_last_point_returns_popped_value():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.add_point(obj.id, (10, 20))
    ed.add_point(obj.id, (30, 40))
    popped = ed.remove_last_point(obj.id)
    assert popped == (30.0, 40.0)
    assert ed.get(obj.id).points == [[10.0, 20.0]]


def test_remove_last_point_returns_none_on_empty():
    ed = ObjectEditor()
    obj = ed.create_object()
    assert ed.remove_last_point(obj.id) is None


def test_remove_last_point_unknown_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.remove_last_point("obj_nope")


def test_clear_points():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.add_point(obj.id, (1, 2))
    ed.add_point(obj.id, (3, 4))
    ed.clear_points(obj.id)
    assert ed.get(obj.id).points == []


# ---------------- serialization round trips ----------------


def test_to_objects_json_drops_objects_below_min_points():
    """An incomplete polygon (1-2 points) shouldn't ship to runtime."""
    ed = ObjectEditor()
    incomplete = ed.create_object("partial")
    ed.add_point(incomplete.id, (0, 0))
    ed.add_point(incomplete.id, (10, 0))

    full = ed.create_object("full")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(full.id, p)

    out = ed.to_objects_json()
    assert len(out) == 1
    assert out[0]["id"] == full.id
    assert len(out[0]["polygon"]) == 4


def test_from_objects_json_round_trip():
    ed = ObjectEditor()
    a = ed.create_object("lamp")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(a.id, p)
    json_data = ed.to_objects_json()

    ed2 = ObjectEditor.from_objects_json(json_data)
    assert len(ed2) == 1
    obj = ed2.get(a.id)
    assert obj.label == "lamp"
    assert obj.points == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


@pytest.mark.parametrize(
    "bad,match",
    [
        ([{"label": "x", "polygon": []}], "missing field 'id'"),
        ([{"id": "x"}], "missing field 'polygon'"),
    ],
)
def test_from_objects_json_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        ObjectEditor.from_objects_json(bad)


def test_save_and_load_disk_round_trip(tmp_path: Path):
    ed = ObjectEditor()
    a = ed.create_object("lamp")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(a.id, p)
    p = tmp_path / "objects.json"
    ed.save(p)

    ed2 = ObjectEditor.load(p)
    assert len(ed2) == 1
    assert ed2.get(a.id).label == "lamp"
    assert ed2.get(a.id).points == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def test_load_missing_file_returns_empty(tmp_path: Path):
    ed = ObjectEditor.load(tmp_path / "nope.json")
    assert len(ed) == 0


def test_save_writes_runtime_compatible_schema(tmp_path: Path):
    ed = ObjectEditor()
    a = ed.create_object("lamp")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(a.id, p)
    p = tmp_path / "objects.json"
    ed.save(p)
    raw = json.loads(p.read_text())
    assert isinstance(raw, list)
    assert raw[0].keys() == {"id", "label", "polygon"}


def test_save_omits_incomplete_objects(tmp_path: Path):
    ed = ObjectEditor()
    incomplete = ed.create_object("partial")
    ed.add_point(incomplete.id, (0, 0))
    p = tmp_path / "objects.json"
    ed.save(p)
    raw = json.loads(p.read_text())
    assert raw == []


# ---------------- device_id binding ----------------


def test_create_object_starts_with_no_device_binding():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    assert obj.device_id is None


def test_set_device_id_updates_binding():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    ed.set_device_id(obj.id, "water_pump")
    assert ed.get(obj.id).device_id == "water_pump"


def test_set_device_id_empty_clears_binding():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    ed.set_device_id(obj.id, "water_pump")
    ed.set_device_id(obj.id, "")
    assert ed.get(obj.id).device_id is None


def test_set_device_id_unknown_object_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.set_device_id("obj_nope", "water_pump")


def test_to_objects_json_includes_device_id_when_set():
    ed = ObjectEditor()
    a = ed.create_object("lamp")
    ed.set_device_id(a.id, "water_pump")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(a.id, p)
    out = ed.to_objects_json()
    assert out[0]["device_id"] == "water_pump"


def test_to_objects_json_omits_device_id_when_unbound():
    ed = ObjectEditor()
    a = ed.create_object("lamp")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(a.id, p)
    out = ed.to_objects_json()
    assert "device_id" not in out[0]


def test_from_objects_json_round_trip_with_device_id():
    raw = [{
        "id": "obj_x", "label": "Pump",
        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "device_id": "water_pump",
    }]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("obj_x").device_id == "water_pump"
    assert ed.to_objects_json()[0]["device_id"] == "water_pump"


def test_from_objects_json_treats_empty_device_id_as_none():
    raw = [{
        "id": "obj_x", "label": "",
        "polygon": [[0, 0], [10, 0], [10, 10]],
        "device_id": "",
    }]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("obj_x").device_id is None


# ---------- occluder (style="occluder") ----------

def test_new_object_default_style_is_normal():
    ed = ObjectEditor()
    obj = ed.create_object()
    assert obj.style == "normal"


def test_set_style_to_occluder():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_style(obj.id, "occluder")
    assert ed.get(obj.id).style == "occluder"


def test_set_style_rejects_unknown():
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match="must be one of"):
        ed.set_style(obj.id, "nope")


def test_set_style_unknown_object_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.set_style("missing", "occluder")


def test_to_objects_json_omits_style_when_normal():
    """Default `style: 'normal'` is implicit — saving normal polygons keeps the
    JSON compact and unchanged from before the feature existed."""
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (10, 0), (10, 10)]:
        ed.add_point(obj.id, p)
    out = ed.to_objects_json()
    assert "style" not in out[0]


def test_to_objects_json_includes_style_when_occluder():
    ed = ObjectEditor()
    obj = ed.create_object("door-frame")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(obj.id, p)
    ed.set_style(obj.id, "occluder")
    out = ed.to_objects_json()
    assert out[0]["style"] == "occluder"


def test_from_objects_json_accepts_occluder_style():
    raw = [{
        "id": "occ_1", "label": "Window",
        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "style": "occluder",
    }]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("occ_1").style == "occluder"


def test_from_objects_json_defaults_missing_style_to_normal():
    raw = [{
        "id": "obj_x", "label": "",
        "polygon": [[0, 0], [10, 0], [10, 10]],
    }]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("obj_x").style == "normal"


def test_from_objects_json_rejects_unknown_style():
    raw = [{
        "id": "obj_x", "label": "",
        "polygon": [[0, 0], [10, 0], [10, 10]],
        "style": "rainbow",
    }]
    with pytest.raises(ValueError, match="style must be one of"):
        ObjectEditor.from_objects_json(raw)


def test_round_trip_preserves_occluder_style(tmp_path: Path):
    ed = ObjectEditor()
    obj = ed.create_object("masking-tape")
    for p in [(0, 0), (5, 0), (5, 5), (0, 5)]:
        ed.add_point(obj.id, p)
    ed.set_style(obj.id, "occluder")
    p = tmp_path / "objects.json"
    ed.save(p)
    loaded = ObjectEditor.load(p)
    assert loaded.get(obj.id).style == "occluder"


# ---------- per-object animation + notes ----------

def test_new_object_has_empty_animation_and_notes():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    assert obj.animation == {}
    assert obj.notes == ""


def test_set_animation_round_trips_all_fields():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_animation(obj.id, {
        "effect": "pulse", "color": "#00aaff",
        "intensity": 0.5, "speed": 1.5, "fill_opacity": 0.9,
    })
    assert ed.get(obj.id).animation == {
        "effect": "pulse", "color": "#00aaff",
        "intensity": 0.5, "speed": 1.5, "fill_opacity": 0.9,
    }


def test_set_animation_partial_only_keeps_provided_keys():
    """Only `effect` set → other defaults take over at runtime. The store
    must not invent missing keys."""
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_animation(obj.id, {"effect": "glow"})
    assert ed.get(obj.id).animation == {"effect": "glow"}


def test_set_animation_none_clears():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_animation(obj.id, {"effect": "pulse"})
    ed.set_animation(obj.id, None)
    assert ed.get(obj.id).animation == {}


def test_set_notes():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_notes(obj.id, "make this brighter at night")
    assert ed.get(obj.id).notes == "make this brighter at night"


@pytest.mark.parametrize("bad,reason", [
    ({"effect": "glitter"},     "effect must be one of"),
    ({"color":  "ffaa55"},      "hex string"),     # missing leading #
    ({"color":  "#ffaa5"},      "hex string"),     # too short
    ({"color":  "not-a-color"}, "hex string"),
    ({"intensity": 1.5},        "intensity"),
    ({"intensity": -0.1},       "intensity"),
    ({"intensity": "1"},        "intensity"),
    ({"speed": 0},              "speed"),
    ({"speed": 99},             "speed"),
    ({"speed": "fast"},         "speed"),
    ({"fill_opacity": 1.5},     "fill_opacity"),
    ({"fill_opacity": -0.1},    "fill_opacity"),
    ({"fill_opacity": "half"},  "fill_opacity"),
])
def test_set_animation_validates(bad, reason):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=reason):
        ed.set_animation(obj.id, bad)


def test_to_objects_json_omits_empty_animation_and_notes():
    """Default empty values must not bloat objects.json — backwards-compatible
    with files that predate the tuner feature."""
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    out = ed.to_objects_json()
    assert "animation" not in out[0]
    assert "notes" not in out[0]


def test_to_objects_json_includes_animation_and_notes_when_set():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    ed.set_animation(obj.id, {"effect": "pulse", "color": "#00aaff"})
    ed.set_notes(obj.id, "tune me later")
    out = ed.to_objects_json()
    assert out[0]["animation"] == {"effect": "pulse", "color": "#00aaff"}
    assert out[0]["notes"] == "tune me later"


def test_round_trip_preserves_animation_and_notes(tmp_path: Path):
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(obj.id, p)
    ed.set_animation(obj.id, {
        "effect": "ember", "color": "#22cc88", "intensity": 0.7, "speed": 1.2
    })
    ed.set_notes(obj.id, "feels right")
    f = tmp_path / "objects.json"
    ed.save(f)
    loaded = ObjectEditor.load(f)
    got = loaded.get(obj.id)
    assert got.animation == {
        "effect": "ember", "color": "#22cc88", "intensity": 0.7, "speed": 1.2
    }
    assert got.notes == "feels right"


def test_from_objects_json_rejects_bad_animation():
    """File-load path must surface validation errors with the offending index
    so the user knows which object is broken."""
    raw = [{
        "id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]],
        "animation": {"effect": "garbage"},
    }]
    with pytest.raises(ValueError, match=r"objects\[0\]:.*effect must be one of"):
        ObjectEditor.from_objects_json(raw)


def test_from_objects_json_treats_missing_animation_as_empty():
    raw = [{"id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]]}]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("x").animation == {}
    assert ed.get("x").notes == ""


# ---------- per-object appearance (outline / fill / image) ----------
#
# `appearance` lets the new /style inspector tune the IDLE look of each polygon:
# outline (stroke), fill (interior), and an optional image clipped to the
# polygon. Hover effects (`animation`) layer on top — these tests only cover
# the idle-look store, not the hover behavior.


def test_new_object_has_empty_appearance():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    assert obj.appearance == {}


def test_set_appearance_round_trips_full_block():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_appearance(obj.id, {
        "outline": {"color": "#ffffff", "width": 3, "opacity": 0.4, "dash": "dashed"},
        "fill":    {"color": "#332200", "opacity": 0.2},
        "image":   {
            "src": "data/objects/obj_x.png",
            "fit": "cover",
            "opacity": 1.0,
            "crop": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        },
    })
    got = ed.get(obj.id).appearance
    assert got["outline"] == {"color": "#ffffff", "width": 3.0, "opacity": 0.4, "dash": "dashed"}
    assert got["fill"]    == {"color": "#332200", "opacity": 0.2}
    assert got["image"]   == {
        "src": "data/objects/obj_x.png",
        "fit": "cover",
        "opacity": 1.0,
        "crop": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
    }


def test_set_appearance_partial_blocks():
    """A user who only sets the outline shouldn't get phantom fill / image keys."""
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_appearance(obj.id, {"outline": {"color": "#ff0000"}})
    assert ed.get(obj.id).appearance == {"outline": {"color": "#ff0000"}}


def test_set_appearance_drops_empty_subblocks():
    """`{"outline": {}}` means "nothing to set" — must not leak an empty block
    into the JSON since that bloats files and confuses the runtime renderer."""
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_appearance(obj.id, {"outline": {}, "fill": {}})
    assert ed.get(obj.id).appearance == {}


def test_set_appearance_none_clears():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_appearance(obj.id, {"fill": {"color": "#abcdef"}})
    ed.set_appearance(obj.id, None)
    assert ed.get(obj.id).appearance == {}


def test_set_appearance_unknown_object_raises():
    ed = ObjectEditor()
    with pytest.raises(KeyError):
        ed.set_appearance("obj_nope", {"outline": {"color": "#ffffff"}})


@pytest.mark.parametrize("bad,reason", [
    # outline
    ({"outline": {"color": "ffffff"}},          "hex string"),
    ({"outline": {"color": "#ffff"}},           "hex string"),
    ({"outline": {"width": -1}},                "outline.width"),
    ({"outline": {"width": 99}},                "outline.width"),
    ({"outline": {"width": "thick"}},           "outline.width"),
    ({"outline": {"opacity": 1.5}},             "outline.opacity"),
    ({"outline": {"opacity": -0.1}},            "outline.opacity"),
    ({"outline": {"opacity": "half"}},          "outline.opacity"),
    ({"outline": {"dash": "wavy"}},             "outline.dash"),
    ({"outline": "not-an-object"},              "outline must be"),
    # fill
    ({"fill": {"color": "red"}},                "fill.color"),
    ({"fill": {"opacity": 2}},                  "fill.opacity"),
    ({"fill": []},                              "fill must be"),
    # image
    ({"image": {"src": ""}},                    "image.src"),
    ({"image": {"src": 123}},                   "image.src"),
    ({"image": {"src": "/etc/passwd"}},         "image.src"),
    ({"image": {"src": "data/../etc/passwd"}},  "image.src"),
    ({"image": {"src": "media/cover.png"}},     "image.src"),  # outside data/
    ({"image": {"fit": "letterbox"}},           "image.fit"),
    ({"image": {"opacity": "full"}},            "image.opacity"),
    ({"image": {"crop": {"x": 0, "y": 0, "w": 1.0, "h": 1.5}}}, "crop.h"),
    ({"image": {"crop": {"x": 0, "y": 0, "w": 0, "h": 1}}},     "must be > 0"),
    ({"image": {"crop": {"x": 0.6, "y": 0, "w": 0.6, "h": 0.5}}}, "fit within"),
    ({"image": {"crop": {"x": 0, "y": 0, "w": 1}}},             "crop.h is required"),
    # top-level
    ("not-a-dict",                              "appearance must be"),
])
def test_set_appearance_validates(bad, reason):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=reason):
        ed.set_appearance(obj.id, bad)


def test_to_objects_json_omits_empty_appearance():
    """Backwards-compat: files saved before this feature don't gain an empty
    appearance block."""
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    out = ed.to_objects_json()
    assert "appearance" not in out[0]


def test_to_objects_json_includes_appearance_when_set():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    ed.set_appearance(obj.id, {
        "outline": {"color": "#ffffff", "width": 2},
        "fill":    {"color": "#000000", "opacity": 0.5},
    })
    out = ed.to_objects_json()
    assert out[0]["appearance"] == {
        "outline": {"color": "#ffffff", "width": 2.0},
        "fill":    {"color": "#000000", "opacity": 0.5},
    }


def test_round_trip_preserves_appearance(tmp_path: Path):
    ed = ObjectEditor()
    obj = ed.create_object("book")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(obj.id, p)
    ed.set_appearance(obj.id, {
        "outline": {"color": "#ff8800", "width": 3, "dash": "dotted"},
        "image":   {"src": "data/objects/cover.png", "fit": "cover",
                    "crop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}},
    })
    f = tmp_path / "objects.json"
    ed.save(f)
    loaded = ObjectEditor.load(f)
    got = loaded.get(obj.id).appearance
    assert got["outline"] == {"color": "#ff8800", "width": 3.0, "dash": "dotted"}
    assert got["image"]["src"] == "data/objects/cover.png"
    assert got["image"]["crop"] == {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}


def test_from_objects_json_rejects_bad_appearance():
    """File-load path surfaces validation errors with the offending index so the
    user knows which object's appearance is broken."""
    raw = [{
        "id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]],
        "appearance": {"outline": {"color": "not-hex"}},
    }]
    with pytest.raises(ValueError, match=r"objects\[0\]:.*outline\.color"):
        ObjectEditor.from_objects_json(raw)


def test_from_objects_json_treats_missing_appearance_as_empty():
    raw = [{"id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]]}]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("x").appearance == {}


def test_set_appearance_label_round_trips():
    """label block toggles the on-hover text and optionally overrides it."""
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    ed.set_appearance(obj.id, {
        "label": {"hidden": True, "text": "Custom display text"},
    })
    assert ed.get(obj.id).appearance == {
        "label": {"hidden": True, "text": "Custom display text"},
    }


def test_set_appearance_label_partial():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_appearance(obj.id, {"label": {"hidden": True}})
    assert ed.get(obj.id).appearance == {"label": {"hidden": True}}


@pytest.mark.parametrize("bad,reason", [
    ({"label": {"hidden": "yes"}},                 "label.hidden"),
    ({"label": {"hidden": 1}},                     "label.hidden"),
    ({"label": {"text": 123}},                     "label.text"),
    ({"label": {"text": "x" * 201}},               "200 chars"),
    ({"label": "not-an-object"},                   "label must be"),
])
def test_set_appearance_label_validates(bad, reason):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=reason):
        ed.set_appearance(obj.id, bad)


def test_round_trip_preserves_label(tmp_path: Path):
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    ed.set_appearance(obj.id, {"label": {"hidden": True}})
    f = tmp_path / "objects.json"
    ed.save(f)
    loaded = ObjectEditor.load(f)
    assert loaded.get(obj.id).appearance == {"label": {"hidden": True}}


# ---------- per-object hover sound ----------


def test_new_object_has_empty_sound():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    assert obj.sound == {}


def test_set_sound_round_trips():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_sound(obj.id, {"src": "data/sounds/hover.mp3", "gain": 0.7})
    assert ed.get(obj.id).sound == {"src": "data/sounds/hover.mp3", "gain": 0.7}


def test_set_sound_partial():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_sound(obj.id, {"gain": 0.5})
    assert ed.get(obj.id).sound == {"gain": 0.5}


def test_set_sound_none_clears():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_sound(obj.id, {"src": "data/sounds/x.mp3", "gain": 0.5})
    ed.set_sound(obj.id, None)
    assert ed.get(obj.id).sound == {}


@pytest.mark.parametrize("bad,reason", [
    ({"src": "/etc/passwd"},                  "sound.src"),
    ({"src": "data/sounds/../evil.mp3"},      "sound.src"),
    ({"src": "media/x.mp3"},                  "sound.src"),  # outside data/sounds/
    ({"src": ""},                             "sound.src"),
    ({"src": 123},                            "sound.src"),
    ({"gain": 1.5},                           "sound.gain"),
    ({"gain": -0.1},                          "sound.gain"),
    ({"gain": "loud"},                        "sound.gain"),
    ("not-a-dict",                            "sound must be"),
])
def test_set_sound_validates(bad, reason):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=reason):
        ed.set_sound(obj.id, bad)


def test_to_objects_json_omits_empty_sound():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    out = ed.to_objects_json()
    assert "sound" not in out[0]


def test_to_objects_json_includes_sound_when_set():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    ed.set_sound(obj.id, {"src": "data/sounds/click.wav", "gain": 0.3})
    out = ed.to_objects_json()
    assert out[0]["sound"] == {"src": "data/sounds/click.wav", "gain": 0.3}


def test_round_trip_preserves_sound(tmp_path: Path):
    ed = ObjectEditor()
    obj = ed.create_object("book")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(obj.id, p)
    ed.set_sound(obj.id, {"src": "data/sounds/page.mp3", "gain": 0.6})
    f = tmp_path / "objects.json"
    ed.save(f)
    loaded = ObjectEditor.load(f)
    assert loaded.get(obj.id).sound == {"src": "data/sounds/page.mp3", "gain": 0.6}


def test_from_objects_json_rejects_bad_sound():
    raw = [{
        "id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]],
        "sound": {"src": "/etc/passwd"},
    }]
    with pytest.raises(ValueError, match=r"objects\[0\]:.*sound\.src"):
        ObjectEditor.from_objects_json(raw)


def test_from_objects_json_treats_missing_sound_as_empty():
    raw = [{"id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]]}]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("x").sound == {}


# ---------- per-object click animation ----------


def test_new_object_has_empty_click():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    assert obj.click == {}


def test_set_click_round_trips():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_click(obj.id, {"effect": "loading_bar", "color": "#00ff88", "duration_ms": 1200})
    assert ed.get(obj.id).click == {
        "effect": "loading_bar", "color": "#00ff88", "duration_ms": 1200,
    }


def test_set_click_with_position_and_size():
    """The full schema includes position (x,y) and size (width,height) as
    fractions of the polygon's bbox — lets one config look right on different-
    shaped objects."""
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_click(obj.id, {
        "effect": "loading_bar", "color": "#ff00ff", "duration_ms": 900,
        "x": 0.5, "y": 0.85, "width": 0.6, "height": 0.08,
    })
    assert ed.get(obj.id).click == {
        "effect": "loading_bar", "color": "#ff00ff", "duration_ms": 900,
        "x": 0.5, "y": 0.85, "width": 0.6, "height": 0.08,
    }


def test_set_click_with_nested_sound():
    """click.sound nests inside the click block so all click-related
    behavior lives in one place. Same {src, gain} schema as the top-level
    hover sound; shares data/sounds/ on disk."""
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_click(obj.id, {
        "effect": "loading_bar",
        "sound": { "src": "data/sounds/click.wav", "gain": 0.6 },
    })
    assert ed.get(obj.id).click == {
        "effect": "loading_bar",
        "sound": { "src": "data/sounds/click.wav", "gain": 0.6 },
    }


@pytest.mark.parametrize("bad,reason", [
    ({"effect": "loading_bar", "sound": {"src": "/etc/passwd"}},      "click.sound.src"),
    ({"effect": "loading_bar", "sound": {"src": "media/x.wav"}},      "click.sound.src"),
    ({"effect": "loading_bar", "sound": {"src": "data/sounds/../x"}}, "click.sound.src"),
    ({"effect": "loading_bar", "sound": {"src": ""}},                  "click.sound.src"),
    ({"effect": "loading_bar", "sound": {"gain": 1.1}},                "click.sound.gain"),
    ({"effect": "loading_bar", "sound": {"gain": -0.1}},               "click.sound.gain"),
    ({"effect": "loading_bar", "sound": "not-a-dict"},                 "click.sound must be"),
])
def test_set_click_validates_nested_sound(bad, reason):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=reason):
        ed.set_click(obj.id, bad)


def test_click_position_allows_overshoot():
    """x/y allow -0.5..1.5 so the bar can sit just outside the polygon
    (e.g., below it). Edge values must validate."""
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_click(obj.id, {"effect": "loading_bar", "x": -0.5, "y": 1.5})
    ed.set_click(obj.id, {"effect": "loading_bar", "x": 1.5,  "y": -0.5})


def test_set_click_partial():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_click(obj.id, {"effect": "loading_bar"})
    assert ed.get(obj.id).click == {"effect": "loading_bar"}


def test_set_click_none_clears():
    ed = ObjectEditor()
    obj = ed.create_object()
    ed.set_click(obj.id, {"effect": "loading_bar"})
    ed.set_click(obj.id, None)
    assert ed.get(obj.id).click == {}


@pytest.mark.parametrize("bad,reason", [
    ({"effect": "fireworks"},          "click.effect"),
    ({"effect": ""},                    "click.effect"),
    ({"color": "not-hex"},              "click.color"),
    ({"color": "#ff"},                  "click.color"),
    ({"duration_ms": 0},                "duration_ms"),
    ({"duration_ms": 49},               "duration_ms"),
    ({"duration_ms": 30001},            "duration_ms"),
    ({"duration_ms": "fast"},           "duration_ms"),
    ({"duration_ms": True},             "duration_ms"),   # bool is a number subtype in py
    # x/y allow [-0.5, 1.5] — half a bbox overshoot in any direction
    ({"x": -0.51},                      "click.x"),
    ({"x": 1.51},                       "click.x"),
    ({"x": "center"},                   "click.x"),
    ({"y": -0.51},                      "click.y"),
    ({"y": 1.51},                       "click.y"),
    # width/height allow [0, 2] — up to 2x bbox
    ({"width": -0.01},                  "click.width"),
    ({"width": 2.01},                   "click.width"),
    ({"width": False},                  "click.width"),   # bool guard
    ({"height": -0.01},                 "click.height"),
    ({"height": 2.01},                  "click.height"),
    ("not-a-dict",                      "click must be"),
])
def test_set_click_validates(bad, reason):
    ed = ObjectEditor()
    obj = ed.create_object()
    with pytest.raises(ValueError, match=reason):
        ed.set_click(obj.id, bad)


def test_to_objects_json_omits_empty_click():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    out = ed.to_objects_json()
    assert "click" not in out[0]


def test_to_objects_json_includes_click_when_set():
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    ed.set_click(obj.id, {"effect": "loading_bar", "color": "#ff8800", "duration_ms": 800})
    out = ed.to_objects_json()
    assert out[0]["click"] == {"effect": "loading_bar", "color": "#ff8800", "duration_ms": 800}


def test_round_trip_preserves_click(tmp_path: Path):
    ed = ObjectEditor()
    obj = ed.create_object("book")
    for p in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        ed.add_point(obj.id, p)
    ed.set_click(obj.id, {"effect": "loading_bar", "color": "#00ffff", "duration_ms": 2000})
    f = tmp_path / "objects.json"
    ed.save(f)
    loaded = ObjectEditor.load(f)
    assert loaded.get(obj.id).click == {"effect": "loading_bar", "color": "#00ffff", "duration_ms": 2000}


def test_from_objects_json_rejects_bad_click():
    raw = [{
        "id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]],
        "click": {"effect": "fireworks"},
    }]
    with pytest.raises(ValueError, match=r"objects\[0\]:.*click\.effect"):
        ObjectEditor.from_objects_json(raw)


def test_from_objects_json_treats_missing_click_as_empty():
    raw = [{"id": "x", "label": "", "polygon": [[0, 0], [1, 0], [1, 1]]}]
    ed = ObjectEditor.from_objects_json(raw)
    assert ed.get("x").click == {}


def test_appearance_and_animation_coexist():
    """Both blocks should round-trip independently — they describe disjoint
    states (idle vs hover) and the renderer reads them separately."""
    ed = ObjectEditor()
    obj = ed.create_object("lamp")
    for p in [(0, 0), (1, 0), (1, 1)]:
        ed.add_point(obj.id, p)
    ed.set_animation(obj.id, {"effect": "glow", "color": "#ffaa55"})
    ed.set_appearance(obj.id, {"outline": {"color": "#ffffff", "width": 2}})
    out = ed.to_objects_json()
    assert out[0]["animation"] == {"effect": "glow", "color": "#ffaa55"}
    assert out[0]["appearance"] == {"outline": {"color": "#ffffff", "width": 2.0}}
