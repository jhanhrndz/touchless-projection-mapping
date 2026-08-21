"""In-memory editing model for interactive objects.

Headless: no UI imports. The web layer (server.py + editor.html) is a thin
wrapper that calls these methods via REST + holds the page-side equivalent.
Tests exercise this module directly without spinning up a browser.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Point = tuple[float, float]


def new_object_id() -> str:
    """Stable, unique-enough ID derived from a unix timestamp and 8 random hex chars."""
    return f"obj_{int(time.time())}_{secrets.token_hex(4)}"


VALID_STYLES = ("normal", "occluder")
VALID_EFFECTS = ("glow", "pulse", "aurora", "bloom", "ants", "ember")
VALID_DASHES = ("solid", "dashed", "dotted")
VALID_FITS = ("cover", "contain", "stretch")
VALID_CLICK_EFFECTS = ("loading_bar",)
VALID_TRACK_AXES = ("x",)
VALID_TRACK_BINDINGS = ("click_to_bind",)


@dataclass
class EditableObject:
    id: str
    label: str = ""
    points: list[list[float]] = field(default_factory=list)
    device_id: str | None = None
    # "normal" = interactive polygon (hover/click); "occluder" = solid black,
    # no border, no events — used to mask projector light off areas where you
    # don't want any UI illumination AND want the cursor hidden.
    style: str = "normal"
    # Per-object animation overrides. Any subset of keys may be present; missing
    # keys fall back to the global defaults at runtime. Schema:
    #   {"effect": <one of VALID_EFFECTS>,
    #    "color":  <hex string like "#ffaa55">,
    #    "intensity": <number 0..1>,
    #    "speed":  <number 0.5..2.0>}
    animation: dict[str, Any] = field(default_factory=dict)
    # Per-object visual styling: outline (idle stroke), fill (idle fill),
    # image (texture clipped to the polygon). Any subset may be present; missing
    # blocks fall back to the global defaults in runtime.html. See
    # _validate_appearance for the full schema.
    appearance: dict[str, Any] = field(default_factory=dict)
    # Audio cue played on hover. Sibling to animation (the visual hover effect).
    # Schema: {"src": "data/sounds/...", "gain": 0..1}. Both keys optional.
    # src is path-sandboxed under data/sounds/ (no traversal, no absolute paths);
    # the upload endpoint also magic-byte-sniffs the bytes before saving.
    sound: dict[str, Any] = field(default_factory=dict)
    # Visual feedback played on click — a quick animation that overlays the
    # polygon. Currently supports one effect ("loading_bar"); structured this
    # way so adding "ripple" / "flash" later is just a new enum value.
    # Schema: {"effect": <enum>, "color": "#rrggbb", "duration_ms": int}
    click: dict[str, Any] = field(default_factory=dict)
    # Pointer-tracking behavior. When present, the runtime page binds this
    # polygon to the mouse on click and translates it along the configured
    # axis so the projected outline follows the cursor (used to drive a
    # physical stepper carriage via /ws/paddle). Schema:
    #   {"axis": "x", "binding": "click_to_bind", "home_x_frac": 0..1}
    # All keys optional inside the dict, but if the dict is non-empty it must
    # validate. Empty = not a tracked polygon.
    tracking: dict[str, Any] = field(default_factory=dict)
    # Free-text notes per object. Used by the /tune page for capturing ideas
    # in real time ("this one needs a green that's brighter at low brightness").
    notes: str = ""


class ObjectEditor:
    def __init__(self, objects: list[EditableObject] | None = None) -> None:
        self._objects: list[EditableObject] = list(objects or [])

    @property
    def objects(self) -> list[EditableObject]:
        return self._objects

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, object_id: str) -> bool:
        return any(o.id == object_id for o in self._objects)

    def get(self, object_id: str) -> EditableObject:
        for o in self._objects:
            if o.id == object_id:
                return o
        raise KeyError(f"unknown object id {object_id!r}")

    # ----- mutations -----

    def create_object(self, label: str = "") -> EditableObject:
        obj = EditableObject(id=new_object_id(), label=label, points=[])
        self._objects.append(obj)
        return obj

    def delete_object(self, object_id: str) -> None:
        self.get(object_id)  # raises if missing
        self._objects = [o for o in self._objects if o.id != object_id]

    def set_label(self, object_id: str, label: str) -> None:
        self.get(object_id).label = label

    def set_device_id(self, object_id: str, device_id: str | None) -> None:
        self.get(object_id).device_id = device_id or None

    def set_style(self, object_id: str, style: str) -> None:
        if style not in VALID_STYLES:
            raise ValueError(f"style must be one of {VALID_STYLES}; got {style!r}")
        self.get(object_id).style = style

    def set_animation(self, object_id: str, animation: dict[str, Any] | None) -> None:
        """Replace the animation override dict. Pass None or {} to clear (back to
        global defaults at runtime). Each present field is validated."""
        obj = self.get(object_id)
        if not animation:
            obj.animation = {}
            return
        validated = _validate_animation(animation)
        obj.animation = validated

    def set_appearance(self, object_id: str, appearance: dict[str, Any] | None) -> None:
        """Replace the appearance override dict. Pass None or {} to clear (back
        to runtime defaults). Each present sub-block is validated."""
        obj = self.get(object_id)
        if not appearance:
            obj.appearance = {}
            return
        obj.appearance = _validate_appearance(appearance)

    def set_sound(self, object_id: str, sound: dict[str, Any] | None) -> None:
        """Replace the hover-sound dict. Pass None or {} to clear."""
        obj = self.get(object_id)
        if not sound:
            obj.sound = {}
            return
        obj.sound = _validate_sound(sound)

    def set_click(self, object_id: str, click: dict[str, Any] | None) -> None:
        """Replace the click-animation dict. Pass None or {} to clear."""
        obj = self.get(object_id)
        if not click:
            obj.click = {}
            return
        obj.click = _validate_click(click)

    def set_tracking(self, object_id: str, tracking: dict[str, Any] | None) -> None:
        """Replace the pointer-tracking dict. Pass None or {} to clear (back
        to a static polygon)."""
        obj = self.get(object_id)
        if not tracking:
            obj.tracking = {}
            return
        obj.tracking = _validate_tracking(tracking)

    def set_notes(self, object_id: str, notes: str) -> None:
        self.get(object_id).notes = str(notes)

    def add_point(self, object_id: str, point: Point) -> None:
        if len(point) != 2:
            raise ValueError(f"point must be (x, y); got {point!r}")
        self.get(object_id).points.append([float(point[0]), float(point[1])])

    def remove_last_point(self, object_id: str) -> Point | None:
        obj = self.get(object_id)
        if not obj.points:
            return None
        x, y = obj.points.pop()
        return (x, y)

    def clear_points(self, object_id: str) -> None:
        self.get(object_id).points = []

    # ----- serialization (matches the runtime objects.json schema) -----

    def to_objects_json(self, *, min_points: int = 3) -> list[dict[str, Any]]:
        """Serialize for runtime consumption. Drops objects with too few points."""
        out = []
        for o in self._objects:
            if len(o.points) < min_points:
                continue
            entry: dict[str, Any] = {
                "id": o.id,
                "label": o.label,
                "polygon": list(o.points),
            }
            if o.device_id:
                entry["device_id"] = o.device_id
            if o.style and o.style != "normal":
                # Omit the default to keep existing files unchanged.
                entry["style"] = o.style
            if o.animation:
                entry["animation"] = dict(o.animation)
            if o.appearance:
                entry["appearance"] = dict(o.appearance)
            if o.sound:
                entry["sound"] = dict(o.sound)
            if o.click:
                entry["click"] = dict(o.click)
            if o.tracking:
                entry["tracking"] = dict(o.tracking)
            if o.notes:
                entry["notes"] = o.notes
            out.append(entry)
        return out

    @classmethod
    def from_objects_json(cls, data: list[dict[str, Any]]) -> "ObjectEditor":
        objs: list[EditableObject] = []
        for i, item in enumerate(data):
            try:
                style = item.get("style", "normal")
                if style not in VALID_STYLES:
                    raise ValueError(
                        f"objects[{i}]: style must be one of {VALID_STYLES}; got {style!r}"
                    )
                animation = item.get("animation") or {}
                if animation:
                    try:
                        animation = _validate_animation(animation)
                    except ValueError as e:
                        raise ValueError(f"objects[{i}]: {e}") from None
                appearance = item.get("appearance") or {}
                if appearance:
                    try:
                        appearance = _validate_appearance(appearance)
                    except ValueError as e:
                        raise ValueError(f"objects[{i}]: {e}") from None
                sound = item.get("sound") or {}
                if sound:
                    try:
                        sound = _validate_sound(sound)
                    except ValueError as e:
                        raise ValueError(f"objects[{i}]: {e}") from None
                click = item.get("click") or {}
                if click:
                    try:
                        click = _validate_click(click)
                    except ValueError as e:
                        raise ValueError(f"objects[{i}]: {e}") from None
                tracking = item.get("tracking") or {}
                if tracking:
                    try:
                        tracking = _validate_tracking(tracking)
                    except ValueError as e:
                        raise ValueError(f"objects[{i}]: {e}") from None
                objs.append(
                    EditableObject(
                        id=item["id"],
                        label=item.get("label", ""),
                        points=[list(p) for p in item["polygon"]],
                        device_id=item.get("device_id") or None,
                        style=style,
                        animation=animation,
                        appearance=appearance,
                        sound=sound,
                        click=click,
                        tracking=tracking,
                        notes=str(item.get("notes", "") or ""),
                    )
                )
            except KeyError as missing:
                raise ValueError(f"objects[{i}]: missing field {missing.args[0]!r}") from None
        return cls(objs)

    def save(
        self, path: Path, *, min_points: int = 3, scene: dict[str, Any] | None = None
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        objects = self.to_objects_json(min_points=min_points)
        # Preserve the per-scene flags block (e.g. flame_carry) across editor
        # saves: write the mapping form when there are flags, else the bare list
        # so files without scene flags stay byte-for-byte as before.
        payload: Any = {"scene": dict(scene), "objects": objects} if scene else objects
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ObjectEditor":
        if not path.exists():
            return cls()
        return cls.from_objects_json(json.loads(path.read_text()))


_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _validate_animation(animation: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied dict with only known keys, each validated. Bad
    inputs raise ValueError so the editor save endpoint surfaces a 400 instead
    of silently storing junk that runtime.html can't parse."""
    out: dict[str, Any] = {}
    if "effect" in animation:
        eff = animation["effect"]
        if eff not in VALID_EFFECTS:
            raise ValueError(
                f"animation.effect must be one of {VALID_EFFECTS}; got {eff!r}"
            )
        out["effect"] = eff
    if "color" in animation:
        col = animation["color"]
        if not isinstance(col, str) or not _HEX_RE.match(col):
            raise ValueError(
                f"animation.color must be a #rrggbb hex string; got {col!r}"
            )
        out["color"] = col
    if "intensity" in animation:
        v = animation["intensity"]
        if not isinstance(v, (int, float)) or not (0 <= v <= 1):
            raise ValueError(
                f"animation.intensity must be a number in [0, 1]; got {v!r}"
            )
        out["intensity"] = float(v)
    if "speed" in animation:
        v = animation["speed"]
        if not isinstance(v, (int, float)) or not (0.1 <= v <= 5.0):
            raise ValueError(
                f"animation.speed must be a number in [0.1, 5.0]; got {v!r}"
            )
        out["speed"] = float(v)
    if "fill_opacity" in animation:
        # Direct override of the hover fill-opacity. Each effect has a baked-in
        # default (fx-glow=0.18, fx-aurora=0.10 etc.); intensity multiplies it
        # but can never exceed 1×. fill_opacity lets the user bypass that
        # ceiling — e.g. set 0.9 to get a fill nearly as solid as the border.
        v = animation["fill_opacity"]
        if not isinstance(v, (int, float)) or not (0 <= v <= 1):
            raise ValueError(
                f"animation.fill_opacity must be a number in [0, 1]; got {v!r}"
            )
        out["fill_opacity"] = float(v)
    return out


def _check_opacity(path: str, v: Any) -> float:
    if not isinstance(v, (int, float)) or not (0 <= v <= 1):
        raise ValueError(f"{path} must be a number in [0, 1]; got {v!r}")
    return float(v)


def _check_hex_color(path: str, v: Any) -> str:
    if not isinstance(v, str) or not _HEX_RE.match(v):
        raise ValueError(f"{path} must be a #rrggbb hex string; got {v!r}")
    return v


def _validate_outline(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ValueError(f"appearance.outline must be an object; got {type(block).__name__}")
    out: dict[str, Any] = {}
    if "color" in block:
        out["color"] = _check_hex_color("appearance.outline.color", block["color"])
    if "width" in block:
        v = block["width"]
        if not isinstance(v, (int, float)) or not (0 <= v <= 12):
            raise ValueError(
                f"appearance.outline.width must be a number in [0, 12]; got {v!r}"
            )
        out["width"] = float(v)
    if "opacity" in block:
        out["opacity"] = _check_opacity("appearance.outline.opacity", block["opacity"])
    if "dash" in block:
        d = block["dash"]
        if d not in VALID_DASHES:
            raise ValueError(
                f"appearance.outline.dash must be one of {VALID_DASHES}; got {d!r}"
            )
        out["dash"] = d
    return out


def _validate_fill(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ValueError(f"appearance.fill must be an object; got {type(block).__name__}")
    out: dict[str, Any] = {}
    if "color" in block:
        out["color"] = _check_hex_color("appearance.fill.color", block["color"])
    if "opacity" in block:
        out["opacity"] = _check_opacity("appearance.fill.opacity", block["opacity"])
    return out


def _validate_label(block: Any) -> dict[str, Any]:
    """Validate the label-override block. `hidden=True` suppresses the hover
    text on the wall; `text` overrides what's shown (falls back to obj.label
    when omitted). Capped at 200 chars to keep stray pastes from blowing
    out the projector page."""
    if not isinstance(block, dict):
        raise ValueError(f"appearance.label must be an object; got {type(block).__name__}")
    out: dict[str, Any] = {}
    if "hidden" in block:
        v = block["hidden"]
        if not isinstance(v, bool):
            raise ValueError(f"appearance.label.hidden must be a bool; got {v!r}")
        out["hidden"] = v
    if "text" in block:
        v = block["text"]
        if not isinstance(v, str):
            raise ValueError(f"appearance.label.text must be a string; got {type(v).__name__}")
        if len(v) > 200:
            raise ValueError(
                f"appearance.label.text must be 200 chars or fewer; got {len(v)}"
            )
        out["text"] = v
    return out


def _validate_image(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ValueError(f"appearance.image must be an object; got {type(block).__name__}")
    out: dict[str, Any] = {}
    if "src" in block:
        s = block["src"]
        if not isinstance(s, str) or not s:
            raise ValueError(f"appearance.image.src must be a non-empty string; got {s!r}")
        # Path-sandbox: image references must live under data/. Reject anything
        # with .. segments or absolute paths so a malicious/typo'd objects.json
        # can't point the projector at /etc/passwd or escape via symlinks.
        if s.startswith("/") or ".." in s.replace("\\", "/").split("/"):
            raise ValueError(
                f"appearance.image.src must be a relative path under data/ "
                f"with no .. segments; got {s!r}"
            )
        if not s.startswith("data/"):
            raise ValueError(
                f"appearance.image.src must start with 'data/'; got {s!r}"
            )
        out["src"] = s
    if "fit" in block:
        f = block["fit"]
        if f not in VALID_FITS:
            raise ValueError(
                f"appearance.image.fit must be one of {VALID_FITS}; got {f!r}"
            )
        out["fit"] = f
    if "opacity" in block:
        out["opacity"] = _check_opacity("appearance.image.opacity", block["opacity"])
    if "crop" in block:
        c = block["crop"]
        if not isinstance(c, dict):
            raise ValueError(
                f"appearance.image.crop must be an object; got {type(c).__name__}"
            )
        crop_out: dict[str, float] = {}
        for k in ("x", "y", "w", "h"):
            if k not in c:
                raise ValueError(f"appearance.image.crop.{k} is required")
            v = c[k]
            if not isinstance(v, (int, float)) or not (0 <= v <= 1):
                raise ValueError(
                    f"appearance.image.crop.{k} must be a number in [0, 1]; got {v!r}"
                )
            crop_out[k] = float(v)
        # Sanity: w and h must be positive — a zero-size crop is a misuse, and
        # x+w / y+h > 1 would crop past the source bounds (which renders as
        # transparent edges, not what the user wants).
        if crop_out["w"] <= 0 or crop_out["h"] <= 0:
            raise ValueError("appearance.image.crop.w and .h must be > 0")
        if crop_out["x"] + crop_out["w"] > 1 or crop_out["y"] + crop_out["h"] > 1:
            raise ValueError(
                "appearance.image.crop must fit within [0, 1] "
                f"(got x+w={crop_out['x'] + crop_out['w']}, "
                f"y+h={crop_out['y'] + crop_out['h']})"
            )
        out["crop"] = crop_out
    return out


def _check_click_frac(path: str, v: Any, lo: float, hi: float) -> float:
    """Range-checked float helper used by the click position/size validators.
    `lo` and `hi` are the allowed bounds for the specific axis."""
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= v <= hi):
        raise ValueError(f"{path} must be a number in [{lo}, {hi}]; got {v!r}")
    return float(v)


def _validate_click(click: dict[str, Any]) -> dict[str, Any]:
    """Validate the on-click animation block. effect is the only required key
    when present — it picks which animation runs. color, duration_ms, and the
    x/y/width/height position+size fields are all optional knobs.

    Position (x, y) and size (width, height) are FRACTIONS of the polygon's
    bbox so a single config looks consistent across different-shaped objects:
      - x/y in [-0.5, 1.5] — half a bbox of overshoot in any direction
      - width/height in [0, 2] — up to 2× bbox (lets bars spill past edges)
    """
    if not isinstance(click, dict):
        raise ValueError(f"click must be an object; got {type(click).__name__}")
    out: dict[str, Any] = {}
    if "effect" in click:
        e = click["effect"]
        if e not in VALID_CLICK_EFFECTS:
            raise ValueError(
                f"click.effect must be one of {VALID_CLICK_EFFECTS}; got {e!r}"
            )
        out["effect"] = e
    if "color" in click:
        out["color"] = _check_hex_color("click.color", click["color"])
    if "duration_ms" in click:
        v = click["duration_ms"]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (50 <= v <= 30000):
            raise ValueError(
                f"click.duration_ms must be a number in [50, 30000]; got {v!r}"
            )
        out["duration_ms"] = int(v)
    if "x" in click:
        out["x"] = _check_click_frac("click.x", click["x"], -0.5, 1.5)
    if "y" in click:
        out["y"] = _check_click_frac("click.y", click["y"], -0.5, 1.5)
    if "width" in click:
        out["width"] = _check_click_frac("click.width", click["width"], 0.0, 2.0)
    if "height" in click:
        out["height"] = _check_click_frac("click.height", click["height"], 0.0, 2.0)
    if "sound" in click:
        # Click-sound block — same {src, gain} schema as the top-level hover
        # sound; shares the same data/sounds/ directory and upload endpoint.
        block = _validate_sound_block(click["sound"], "click.sound")
        if block:
            out["sound"] = block
    return out


def _validate_sound_block(block: Any, path: str) -> dict[str, Any]:
    """Shared sound-block validator. `path` is the error-message prefix —
    'sound' for the top-level hover sound, 'click.sound' for the on-click
    sound nested inside the click block. Both reference the same on-disk
    layout (`data/sounds/<file>`), differ only in where they hang off the
    object."""
    if not isinstance(block, dict):
        raise ValueError(
            f"{path} must be an object; got {type(block).__name__}"
        )
    out: dict[str, Any] = {}
    if "src" in block:
        s = block["src"]
        if not isinstance(s, str) or not s:
            raise ValueError(f"{path}.src must be a non-empty string; got {s!r}")
        if s.startswith("/") or ".." in s.replace("\\", "/").split("/"):
            raise ValueError(
                f"{path}.src must be a relative path under data/sounds/ "
                f"with no .. segments; got {s!r}"
            )
        if not s.startswith("data/sounds/"):
            raise ValueError(
                f"{path}.src must start with 'data/sounds/'; got {s!r}"
            )
        out["src"] = s
    if "gain" in block:
        v = block["gain"]
        if not isinstance(v, (int, float)) or not (0 <= v <= 1):
            raise ValueError(f"{path}.gain must be a number in [0, 1]; got {v!r}")
        out["gain"] = float(v)
    return out


def _validate_sound(sound: dict[str, Any]) -> dict[str, Any]:
    """Hover sound — alias for the top-level path so error messages read right."""
    return _validate_sound_block(sound, "sound")


def _validate_tracking(tracking: dict[str, Any]) -> dict[str, Any]:
    """Validate the pointer-tracking block. Only X-axis click-to-bind is
    supported today; the enums are here so adding Y-axis or always-bound
    later is just a new value, not a schema migration. home_x_frac is where
    the polygon snaps back to when unbound (defaults to 0.5 = center of the
    viewport)."""
    if not isinstance(tracking, dict):
        raise ValueError(
            f"tracking must be an object; got {type(tracking).__name__}"
        )
    out: dict[str, Any] = {}
    known = {"axis", "binding", "home_x_frac"}
    extra = set(tracking) - known
    if extra:
        raise ValueError(
            f"tracking has unknown key(s) {sorted(extra)!r}; allowed: {sorted(known)!r}"
        )
    if "axis" in tracking:
        a = tracking["axis"]
        if a not in VALID_TRACK_AXES:
            raise ValueError(
                f"tracking.axis must be one of {VALID_TRACK_AXES}; got {a!r}"
            )
        out["axis"] = a
    if "binding" in tracking:
        b = tracking["binding"]
        if b not in VALID_TRACK_BINDINGS:
            raise ValueError(
                f"tracking.binding must be one of {VALID_TRACK_BINDINGS}; got {b!r}"
            )
        out["binding"] = b
    if "home_x_frac" in tracking:
        v = tracking["home_x_frac"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            raise ValueError(
                f"tracking.home_x_frac must be a number in [0, 1]; got {v!r}"
            )
        out["home_x_frac"] = float(v)
    return out


def _validate_appearance(appearance: dict[str, Any]) -> dict[str, Any]:
    """Validate the per-object appearance block. Returns a shallow-copied dict
    containing only the keys the user explicitly set; missing blocks mean "use
    the runtime default" so old objects.json files keep rendering identically.

    Bad inputs raise ValueError so /api/editor/save surfaces a 400 rather than
    silently storing junk that runtime.html can't parse.
    """
    if not isinstance(appearance, dict):
        raise ValueError(
            f"appearance must be an object; got {type(appearance).__name__}"
        )
    out: dict[str, Any] = {}
    if "outline" in appearance:
        block = _validate_outline(appearance["outline"])
        if block:
            out["outline"] = block
    if "fill" in appearance:
        block = _validate_fill(appearance["fill"])
        if block:
            out["fill"] = block
    if "image" in appearance:
        block = _validate_image(appearance["image"])
        if block:
            out["image"] = block
    if "label" in appearance:
        block = _validate_label(appearance["label"])
        if block:
            out["label"] = block
    return out
