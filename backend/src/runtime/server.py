from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .devices.registry import DeviceRegistry
from .editor import ObjectEditor
from .paddle_bus import PaddleBus, PaddleState

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("runtime.server")

# The projector's native resolution. The editor canvas is exactly this
# rectangle, and what you draw at (x, y) lands at (x, y) on the wall — there is
# no camera and no homography in between. Override with --projector-size.
DEFAULT_PROJECTOR_SIZE = (1920, 1080)


class CursorPatch(BaseModel):
    x: float
    y: float
    visible: bool


class PaddlePush(BaseModel):
    """Telemetry frame from runtime.html for a polygon currently bound to the
    mouse. Mirrors brick_breaker_projection's wire format so the existing
    stepper bridge subscribes without changes."""

    x_frac: float
    x_px: float
    width_px: float
    vx: float = 0.0
    t_ms: int = 0


class ObjectSpec(BaseModel):
    id: str
    label: str = ""
    polygon: list[list[float]]
    device_id: str | None = None
    style: str = "normal"  # "normal" | "occluder"
    # Per-object animation overrides — see EditableObject in runtime/editor.py
    # for schema details. Loose typing here so old objects.json files without
    # these fields still validate (animation defaults to empty dict).
    animation: dict[str, Any] = {}
    # Per-object idle appearance (outline / fill / image). Same loose typing
    # as animation — full validation lives in editor._validate_appearance and
    # runs on save.
    appearance: dict[str, Any] = {}
    # Per-object hover sound — {src, gain}. Validation in editor._validate_sound.
    sound: dict[str, Any] = {}
    # Per-object click animation — {effect, color, duration_ms}.
    # Validation in editor._validate_click.
    click: dict[str, Any] = {}
    # Per-object pointer-tracking — {axis, binding, home_x_frac}.
    # Validation in editor._validate_tracking. When non-empty, runtime.html
    # binds this polygon to the mouse on click and publishes its X position
    # over /ws/paddle for a stepper bridge to consume.
    tracking: dict[str, Any] = {}
    notes: str = ""


async def _prewarm_devices(registry: DeviceRegistry) -> None:
    """Boot-time connectivity check. Runs on the uvicorn event loop so any
    connections that devices cache (e.g., Kasa TCP streams) stay valid for
    the rest of the server's lifetime. Concurrent across all devices.

    Failures are logged but never raise — the server still boots if one
    switch is offline, so the editor + other devices stay usable."""
    devices = list(registry)
    if not devices:
        return

    async def _one(dev):
        try:
            return dev, await dev.prewarm()
        except Exception as e:
            return dev, f"prewarm crashed ({type(e).__name__}: {e})"

    results = await asyncio.gather(*(_one(d) for d in devices))
    lines = [(dev, status) for dev, status in results if status is not None]
    if not lines:
        return  # all-mock registry: nothing to warm, so no bare header
    print("Device prewarm:")
    for dev, status in lines:
        mark = "✓" if "reachable at" in status else "✗"
        print(f"  {mark} {dev.object_id:20s} {status}")


def create_app(
    devices: DeviceRegistry | None = None,
    objects: list[dict[str, Any]] | None = None,
    *,
    objects_path: Path | None = None,
    devices_path: Path | None = None,
    projector_size: tuple[int, int] = DEFAULT_PROJECTOR_SIZE,
    stepper_calibration: dict[str, Any] | None = None,
    paddle_state_path: Path | None = None,
    projection_align_path: Path | None = None,
    scene_settings: dict[str, Any] | None = None,
    object_scene: dict[str, Any] | None = None,
    run_prewarm: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Run prewarm on uvicorn's event loop so kasa connections cached
        # by device.prewarm() are bound to the right loop. Default off so
        # unit tests don't try to reach real hardware; __main__.py flips it
        # on when starting the real server (and leaves it off in --mock).
        if run_prewarm:
            await _prewarm_devices(_app.state.devices)
        yield

    app = FastAPI(title="Interactive Projection", lifespan=lifespan)
    app.state.devices = devices or DeviceRegistry()
    app.state.objects = list(objects or [])
    app.state.objects_path = objects_path
    app.state.devices_path = devices_path
    app.state.projector_size = tuple(projector_size)
    # Scene-wide behavior flags surfaced to runtime.html via GET /api/scene.
    # `scene_settings` is the resolved/merged view (devices.yaml global,
    # overridden by the objects file). `object_scene` is just the objects
    # file's own block — kept separately so editor saves write it back to the
    # objects file without leaking the devices.yaml global into it.
    app.state.scene_settings = dict(scene_settings or {})
    app.state.object_scene = dict(object_scene or {})
    # Paddle state hub — runtime.html publishes the X position of the
    # currently-bound (mouse-tracking) polygon here at ~60Hz. External
    # consumers read via GET /api/paddle (one-shot) or WS /ws/paddle
    # (fan-out). The shape matches brick_breaker's so the existing
    # stepper_paddle bridge can subscribe without modification.
    #
    # Persisted across server restarts via paddle_state_path (optional).
    # Without persistence, every server start would mean (a) the bridge
    # reconnects and gets the default x_px=0 snapshot, driving the carriage
    # to the leftmost limit, and (b) any browser reload would snap the
    # polygon back to its drawn home — both of which lie about where the
    # physical carriage actually is.
    app.state.paddle_bus = PaddleBus(persist_path=paddle_state_path)
    if app.state.paddle_bus.bootstrap_from_disk():
        log.info("paddle bus restored state from %s", paddle_state_path)

    # Projection alignment — a pure-visual offset + scale applied to the
    # tracked polygon's DRAWN position only (the published x_px / what the
    # stepper bridge sees is unaffected). Used to nudge the projected outline
    # into visual coincidence with the physical carriage when the projector
    # and rail aren't perfectly co-located. Identity (0, 1) by default.
    app.state.projection_align_path = projection_align_path
    app.state.projection_align = {"x_offset": 0.0, "x_scale": 1.0}
    if projection_align_path and projection_align_path.exists():
        try:
            loaded = json.loads(projection_align_path.read_text())
            x_off = float(loaded.get("x_offset", 0.0))
            x_sca = float(loaded.get("x_scale", 1.0))
            app.state.projection_align = {"x_offset": x_off, "x_scale": x_sca}
            log.info("loaded projection alignment from %s: offset=%.2f scale=%.4f",
                     projection_align_path, x_off, x_sca)
        except (OSError, ValueError, TypeError) as e:
            log.warning("failed to load projection alignment %s: %s — using identity",
                        projection_align_path, e)
    # Draft is the live-tunable copy /align pushes to. Runtime / polls it so
    # changes preview instantly without a save. Seeded from the current
    # saved value so first-time openers don't see identity-by-default and
    # blow away the last save.
    app.state.projection_align_draft = dict(app.state.projection_align)
    app.state.projection_align_draft["version"] = 0
    # Stepper calibration (optional). When set, the runtime page fetches the
    # derived x_px limits from /api/paddle/config and clamps the bound
    # polygon so the user can't drag past the physical rail's reach. Stored
    # as the raw dict — the endpoint does the per-fetch resolution so a
    # config swap doesn't require a server restart (if we ever add reload).
    app.state.stepper_calibration = stepper_calibration
    # Live editor draft — shared state between the laptop /editor page and the
    # projector /editor/projection page. Both pages POST changes here and poll
    # for changes pushed by the other. `client_id` identifies the last writer
    # so each client can ignore its own echoes; `version` is a monotonic
    # counter so clients can tell whether a poll result is newer than what
    # they've already applied. Seeded from saved objects so the projection
    # page shows existing polygons even before /editor is open.
    app.state.editor_draft: dict[str, Any] = {
        "objects": [_object_to_draft_entry(o) for o in app.state.objects],
        "active_id": None,
        "cursor": None,
        "client_id": None,
        "version": 0,
    }
    # Style-inspector draft — one-way pipe from the /style laptop inspector to
    # the /style/preview sandbox projector. Unlike editor_draft, the wall (/)
    # ignores this channel: previewing an appearance change shouldn't disturb
    # the real projection until the user explicitly hits Save (which writes
    # objects.json via /api/editor/save). Seeded with the saved objects so the
    # preview page shows the current state even when the inspector isn't open.
    #
    # preview_mode controls how /style/preview renders hover state:
    #   - "active_hovered" (default): the active_id object renders with the
    #     hover effect baked in, the rest are idle. Lets you tune the
    #     Highlight tab without mousing over.
    #   - "all_idle": no object renders the hover effect; pointer events are
    #     suppressed so you see exactly what the wall looks like before any
    #     interaction.
    #   - "manual": browser-native :hover. Behaves like the real wall.
    app.state.style_draft: dict[str, Any] = {
        "objects": [_object_to_draft_entry(o) for o in app.state.objects],
        "active_id": None,
        "preview_mode": "active_hovered",
        "version": 0,
    }
    # Text projector draft — shared between /text/editor (laptop control) and
    # /text (projector view). x/y are fractions of viewport (0..1) so the two
    # pages stay aligned even when their resolutions differ. POSTs merge so a
    # control can push a single field without re-sending the whole state.
    app.state.text_draft: dict[str, Any] = {
        "text": "",
        "x": 0.5,
        "y": 0.5,
        "size": "8vh",
        "color": "#ffffff",
        "bg": "#000000",
        "weight": "normal",
        "align": "center",
        "line_height": 1.15,
        # Fire transition (one-off): /text/editor sets before/after text and bumps
        # fire_trigger to a new nonce; /text watches the nonce and plays a retro
        # Doom-fire wipe from fire_before → fire_after. fire_trigger starts at 0 so
        # opening /text fresh never auto-fires.
        "fire_before": "5",
        "fire_after": "6",
        "fire_trigger": 0,
    }

    @app.patch("/api/editor/draft/cursor")
    async def editor_draft_cursor_patch(cursor: CursorPatch) -> dict[str, Any]:
        """Update only the cursor without replacing the objects list."""
        prev = app.state.editor_draft
        app.state.editor_draft = {
            **prev,
            "cursor": cursor.model_dump(),
            "version": int(prev.get("version", 0)) + 1,
        }
        return {"ok": True}

    # ---------- runtime page (Phase 5) ----------

    @app.get("/")
    async def runtime_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "runtime.html", media_type="text/html")

    # Browsers request this on every page; without a route it 404s into the
    # access log on each visit. A dark dot matches the black-page aesthetic.
    _FAVICON_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<rect width="16" height="16" fill="#000"/>'
        '<polygon points="8,2 13,7 11,13 5,13 3,7" fill="none" '
        'stroke="#ffc864" stroke-width="1.5"/></svg>'
    )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(content=_FAVICON_SVG, media_type="image/svg+xml")

    # ---------- per-object image overlays (appearance.image.src) ----------
    #
    # Image overlays for the /style editor live under data/objects/. This
    # route serves them to runtime.html / style_preview.html. Path-sandboxed:
    # the requested name must resolve inside data/objects/ — no traversal,
    # no symlinks pointing elsewhere. Content-Type sniffed from the file
    # extension since we only accept a small allowlist on upload.
    @app.get("/data/objects/{name:path}")
    async def serve_object_image(name: str) -> FileResponse:
        safe_root = Path("data/objects").resolve()
        try:
            requested = (safe_root / name).resolve()
            requested.relative_to(safe_root)
        except (ValueError, OSError):
            raise HTTPException(400, "image path must be inside data/objects/")
        if not requested.exists() or not requested.is_file():
            raise HTTPException(404, f"image not found: {name}")
        suffix = requested.suffix.lower()
        media = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix)
        if media is None:
            raise HTTPException(400, f"unsupported image type {suffix!r}")
        return FileResponse(
            requested,
            media_type=media,
            headers={"Cache-Control": "no-store"},
        )

    # ---------- per-object hover sounds (sound.src) ----------
    #
    # Mirrors the /data/objects/ route. data/sounds/ holds user-uploaded
    # MP3/WAV/OGG/M4A/AAC clips played on hover at runtime. Same path
    # sandbox, same Cache-Control: no-store so a re-uploaded file is
    # picked up immediately by the projector page.
    @app.get("/data/sounds/{name:path}")
    async def serve_object_sound(name: str) -> FileResponse:
        safe_root = Path("data/sounds").resolve()
        try:
            requested = (safe_root / name).resolve()
            requested.relative_to(safe_root)
        except (ValueError, OSError):
            raise HTTPException(400, "sound path must be inside data/sounds/")
        if not requested.exists() or not requested.is_file():
            raise HTTPException(404, f"sound not found: {name}")
        suffix = requested.suffix.lower()
        media = {
            ".mp3":  "audio/mpeg",
            ".wav":  "audio/wav",
            ".ogg":  "audio/ogg",
            ".m4a":  "audio/mp4",
            ".aac":  "audio/aac",
        }.get(suffix)
        if media is None:
            raise HTTPException(400, f"unsupported audio type {suffix!r}")
        return FileResponse(
            requested,
            media_type=media,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/objects")
    async def get_objects() -> list[dict[str, Any]]:
        return app.state.objects

    @app.get("/api/devices")
    async def get_devices() -> list[dict[str, Any]]:
        """List the device entries available for binding (used by the editor dropdown)."""
        registry: DeviceRegistry = app.state.devices
        return [
            {"object_id": dev.object_id, "label": dev.label, "type": type(dev).__name__}
            for dev in registry
        ]

    @app.get("/api/scene")
    async def get_scene() -> dict[str, Any]:
        """Scene-wide behavior flags (from the `scene:` block of the objects file,
        falling back to devices.yaml). runtime.html reads these at load; a ?flame=
        URL param still overrides flame_carry."""
        s = app.state.scene_settings
        return {
            "flame_carry": bool(s.get("flame_carry", False)),
            # 🔥 cursor tuning (carry mechanic). Defaults match runtime.html.
            "flame_size": float(s.get("flame_size", 30)),
            "flame_offset_y": float(s.get("flame_offset_y", 14)),
        }

    # ---------- editor page (Phase 4a) ----------

    @app.get("/editor")
    async def editor_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "editor.html", media_type="text/html")

    # ---------- per-object animation tuner ----------

    @app.get("/tune")
    async def tune_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "tune.html", media_type="text/html")

    # ---------- /style — per-object appearance inspector ----------
    # /style          — laptop inspector (outline / fill / image / highlight)
    # /style/preview  — sandbox projector that mirrors the wall's rendering
    #                   but logs clicks instead of firing devices. Polls
    #                   /api/style/draft for live updates.
    @app.get("/style")
    async def style_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "style_editor.html", media_type="text/html")

    @app.get("/style/preview")
    async def style_preview_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "style_preview.html", media_type="text/html")

    @app.get("/api/style/draft")
    async def style_draft_get() -> dict[str, Any]:
        return app.state.style_draft

    @app.post("/api/style/draft")
    async def style_draft_set(payload: dict[str, Any]) -> dict[str, Any]:
        """Replace the style draft. Same shape as /api/editor/draft minus the
        cursor/client_id fields — the style inspector is a one-way pipe to the
        preview sandbox, no peer sync needed."""
        objects = payload.get("objects")
        active_id = payload.get("active_id")
        preview_mode = payload.get("preview_mode")
        if not isinstance(objects, list):
            raise HTTPException(400, "payload must include 'objects': [...]")
        for i, item in enumerate(objects):
            if not isinstance(item, dict):
                raise HTTPException(400, f"objects[{i}] must be a mapping")
        # preview_mode is optional; if omitted, keep the previous setting so
        # the inspector can push objects updates without resending the mode
        # on every keystroke. When supplied, must be one of the known enum
        # values — invalid values would silently break the preview page.
        valid_modes = ("active_hovered", "all_idle", "manual")
        if preview_mode is not None and preview_mode not in valid_modes:
            raise HTTPException(
                400, f"preview_mode must be one of {valid_modes}; got {preview_mode!r}"
            )
        prev_version = int(app.state.style_draft.get("version", 0))
        prev_mode = app.state.style_draft.get("preview_mode", "active_hovered")
        app.state.style_draft = {
            "objects": objects,
            "active_id": active_id,
            "preview_mode": preview_mode if preview_mode is not None else prev_mode,
            "version": prev_version + 1,
        }
        return {"ok": True, "count": len(objects), "version": prev_version + 1}

    @app.post("/api/style/sound")
    async def style_sound_upload(
        file: UploadFile = File(...),
        object_id: str = Form(...),
    ) -> dict[str, Any]:
        """Upload an audio cue for `obj.sound.src`. Magic-byte sniffs the
        first ~16 bytes against known headers (WAV/MP3/OGG/M4A/AAC) so a
        renamed binary can't end up in data/sounds/. Files are saved verbatim
        — no re-encoding — to preserve any silence-trimming / loudness work
        the user did externally."""
        return await _save_uploaded_sound(file, object_id)

    @app.post("/api/style/image")
    async def style_image_upload(
        file: UploadFile = File(...),
        object_id: str = Form(...),
    ) -> dict[str, Any]:
        """Upload an image for use as `appearance.image.src`. The file is
        re-encoded via Pillow to:
          - verify it's actually an image (not a renamed .exe)
          - strip EXIF metadata (privacy + size)
          - apply EXIF orientation so portraits land upright
          - downscale very large originals (>2048 on either axis) so we don't
            ship multi-MB textures to the projector page.
        Saves under data/objects/<object_id>_<unix_ts>.<ext> and returns the
        relative path the schema expects in `appearance.image.src`.

        object_id is the polygon's id from objects.json; we don't require it
        to match a known object here (the schema validates that on save) but
        we sanity-check the format to keep the filename safe.
        """
        return await _save_uploaded_image(file, object_id)

    # ---------- text projector (typing → fullscreen white text on black) ----
    # /text         — projection view (polls /api/text/draft, renders text at
    #                 the current state). URL params still override individual
    #                 fields for one-off standalone use.
    # /text/editor  — laptop editor (controls + draggable preview canvas).
    @app.get("/text")
    async def text_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "text.html", media_type="text/html")

    @app.get("/text/editor")
    async def text_editor_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "text_editor.html", media_type="text/html")

    @app.get("/api/text/draft")
    async def text_draft_get() -> dict[str, Any]:
        return app.state.text_draft

    @app.post("/api/text/draft")
    async def text_draft_set(payload: dict[str, Any]) -> dict[str, Any]:
        """Merge-style update. Send any subset of the fields and the rest are
        left alone — lets the editor push a single field on every input event
        without re-sending the whole state on the wire."""
        allowed_str   = {"text", "size", "color", "bg", "weight", "align",
                         "fire_before", "fire_after"}
        allowed_num   = {"x", "y", "line_height", "fire_trigger"}
        merged = dict(app.state.text_draft)
        for k, v in payload.items():
            if k in allowed_str:
                if not isinstance(v, str):
                    raise HTTPException(400, f"{k} must be a string")
                merged[k] = v
            elif k in allowed_num:
                if not isinstance(v, (int, float)):
                    raise HTTPException(400, f"{k} must be a number")
                merged[k] = float(v)
            # Silently ignore unknown keys — same forgiving stance as
            # /api/editor/draft on missing optional fields.
        app.state.text_draft = merged
        return merged

    # ---------- text presets (named, persisted text configs) ----------
    #
    # The text-projector editor lets the user save the current state (text,
    # position, typography, colors) under a name and re-load it later. Storage
    # is a flat JSON list at data/text_presets.json — gitignored because it's
    # per-user state. Same field allowlist as the draft endpoint so the two
    # stay in lockstep.

    PRESET_STR_FIELDS = {"text", "size", "color", "bg", "weight", "align"}
    PRESET_NUM_FIELDS = {"x", "y", "line_height"}

    def _presets_path() -> Path:
        return Path("data/text_presets.json")

    def _load_presets() -> list[dict[str, Any]]:
        p = _presets_path()
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_presets(presets: list[dict[str, Any]]) -> None:
        p = _presets_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(presets, indent=2))

    def _validate_preset_config(cfg: Any) -> dict[str, Any]:
        if not isinstance(cfg, dict):
            raise HTTPException(400, "config must be an object")
        out: dict[str, Any] = {}
        for k, v in cfg.items():
            if k in PRESET_STR_FIELDS:
                if not isinstance(v, str):
                    raise HTTPException(400, f"config.{k} must be a string")
                out[k] = v
            elif k in PRESET_NUM_FIELDS:
                if not isinstance(v, (int, float)):
                    raise HTTPException(400, f"config.{k} must be a number")
                out[k] = float(v)
            # unknown keys silently dropped
        return out

    @app.get("/api/text/presets")
    async def text_presets_list() -> list[dict[str, Any]]:
        return _load_presets()

    @app.post("/api/text/presets")
    async def text_presets_save(payload: dict[str, Any]) -> dict[str, Any]:
        """Save the supplied config under `name`. If a preset already exists with
        that name, it's replaced (updated timestamp). Returns the saved entry."""
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(400, "name must be a non-empty string")
        name = name.strip()
        if len(name) > 80:
            raise HTTPException(400, "name must be 80 chars or fewer")
        config = _validate_preset_config(payload.get("config"))

        presets = _load_presets()
        entry = {"name": name, "updated": int(time.time()), "config": config}
        existing_idx = next(
            (i for i, p in enumerate(presets) if p.get("name") == name), None
        )
        if existing_idx is not None:
            presets[existing_idx] = entry
        else:
            presets.append(entry)
        _save_presets(presets)
        return {"ok": True, "saved": entry, "count": len(presets)}

    @app.delete("/api/text/presets/{name}")
    async def text_presets_delete(name: str) -> dict[str, Any]:
        presets = _load_presets()
        new = [p for p in presets if p.get("name") != name]
        if len(new) == len(presets):
            raise HTTPException(404, f"preset {name!r} not found")
        _save_presets(new)
        return {"ok": True, "deleted": name, "count": len(new)}

    @app.get("/api/editor/scene")
    async def editor_scene_meta() -> dict[str, Any]:
        """Canvas geometry for the editor.

        There's no camera and no homography here: the editor canvas simply *is*
        the projector rectangle, so this reports its size and nothing else.
        `scene_available: false` keeps the editor's background layer switched
        off, which is what makes it a plain black canvas.
        """
        return {
            "scene_available": False,
            "projector": list(app.state.projector_size),
        }

    # ---------- live editor draft (split-window editor) ----------

    @app.get("/editor/projection")
    async def editor_projection_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "editor_projection.html", media_type="text/html")

    @app.get("/api/editor/draft")
    async def editor_draft_get() -> dict[str, Any]:
        return app.state.editor_draft

    @app.post("/api/editor/draft")
    async def editor_draft_set(payload: dict[str, Any]) -> dict[str, Any]:
        objects = payload.get("objects")
        active_id = payload.get("active_id")
        cursor = payload.get("cursor")
        client_id = payload.get("client_id")
        if not isinstance(objects, list):
            raise HTTPException(400, "payload must include 'objects': [...]")
        # Light validation: each entry must be a mapping; no further shape
        # checks since drafts are intentionally permissive (incomplete polygons,
        # missing labels, etc.) — that's the whole point of "draft".
        for i, item in enumerate(objects):
            if not isinstance(item, dict):
                raise HTTPException(400, f"objects[{i}] must be a mapping")
        # cursor is optional. When present it must have x/y numbers + visible bool
        # so the projection page can safely render a live preview dot. Reject
        # malformed cursors instead of silently dropping them — easier to debug.
        if cursor is not None:
            if not isinstance(cursor, dict):
                raise HTTPException(400, "cursor must be an object or null")
            for k in ("x", "y"):
                if not isinstance(cursor.get(k), (int, float)):
                    raise HTTPException(400, f"cursor.{k} must be a number")
            if not isinstance(cursor.get("visible"), bool):
                raise HTTPException(400, "cursor.visible must be a bool")
        # client_id is optional metadata used by multi-client polling to
        # ignore one's own echoes. Empty string treated as None.
        if client_id is not None and not isinstance(client_id, str):
            raise HTTPException(400, "client_id must be a string or null")
        prev_version = int(app.state.editor_draft.get("version", 0))
        app.state.editor_draft = {
            "objects": objects,
            "active_id": active_id,
            "cursor": cursor,
            "client_id": client_id or None,
            "version": prev_version + 1,
        }
        return {
            "ok": True,
            "count": len(objects),
            "version": prev_version + 1,
        }

    # ---------- config-file browser (sidebar "Configs" panel) ----------
    #
    # Lists every objects*.json and devices*.yaml file under data/ with a
    # quick-glance count so the user can see snapshots, dev variants, etc.
    # at the same time as their live config. Objects files are loadable
    # into the editor (in-memory swap); devices files are read-only because
    # hot-swapping a live device registry needs more care than a UI button.

    @app.get("/api/configs/list")
    async def configs_list() -> dict[str, Any]:
        data_dir = Path("data")
        # Resolve to absolute for stable comparison against the configured
        # active paths (which may be passed as relative or absolute).
        def _resolved_str(p: Path | None) -> str | None:
            try:
                return str(p.resolve()) if p else None
            except OSError:
                return str(p) if p else None
        active_objects = _resolved_str(app.state.objects_path)
        active_devices = _resolved_str(app.state.devices_path)

        def _count_json(p: Path) -> int:
            try:
                data = json.loads(p.read_text())
                return len(data) if isinstance(data, list) else -1
            except Exception:
                return -1

        def _count_yaml(p: Path) -> int:
            try:
                data = yaml.safe_load(p.read_text())
                return len(data) if isinstance(data, list) else -1
            except Exception:
                return -1

        objects_files: list[dict[str, Any]] = []
        if data_dir.exists():
            for p in sorted(data_dir.glob("objects*.json")):
                objects_files.append({
                    "path": str(p),
                    "name": p.name,
                    "count": _count_json(p),
                    "active": _resolved_str(p) == active_objects,
                    "size_bytes": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                })

        devices_files: list[dict[str, Any]] = []
        if data_dir.exists():
            for p in sorted(data_dir.glob("devices*.yaml")):
                devices_files.append({
                    "path": str(p),
                    "name": p.name,
                    "count": _count_yaml(p),
                    "active": _resolved_str(p) == active_devices,
                    "size_bytes": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                })

        return {"objects": objects_files, "devices": devices_files}

    @app.get("/api/configs/file")
    async def configs_get_file(path: str) -> dict[str, Any]:
        """Return the contents of one config file. Path is sandboxed to data/."""
        data_dir = Path("data").resolve()
        try:
            requested = Path(path).resolve()
            requested.relative_to(data_dir)
        except (ValueError, OSError):
            raise HTTPException(400, "path must be within data/")
        if not requested.exists():
            raise HTTPException(404, f"file not found: {path}")
        suffix = requested.suffix.lower()
        try:
            if suffix == ".json":
                data = json.loads(requested.read_text())
            elif suffix in (".yaml", ".yml"):
                data = yaml.safe_load(requested.read_text())
            else:
                raise HTTPException(400, f"unsupported extension {suffix!r}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise HTTPException(400, f"failed to parse {requested.name}: {e}")
        return {"path": str(requested.relative_to(data_dir.parent)), "data": data}

    @app.post("/api/editor/save")
    async def editor_save(payload: dict[str, Any]) -> dict[str, Any]:
        if app.state.objects_path is None:
            raise HTTPException(503, "editor save path not configured")
        objects = payload.get("objects")
        if not isinstance(objects, list):
            raise HTTPException(400, "payload must be {'objects': [...]}")
        try:
            editor = ObjectEditor.from_objects_json(objects)
        except ValueError as e:
            raise HTTPException(400, str(e))
        editor.save(app.state.objects_path, scene=app.state.object_scene or None)
        app.state.objects = editor.to_objects_json()
        return {"ok": True, "saved": len(app.state.objects), "path": str(app.state.objects_path)}

    # ---------- projection alignment (visual nudge for the tracked polygon) ----

    @app.get("/align")
    async def align_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "align.html", media_type="text/html")

    @app.get("/api/projection_align")
    async def projection_align_get() -> dict[str, Any]:
        return dict(app.state.projection_align)

    def _validate_align_payload(payload: Any) -> tuple[float, float]:
        if not isinstance(payload, dict):
            raise HTTPException(400, "payload must be an object")
        try:
            x_off = float(payload.get("x_offset", 0.0))
            x_sca = float(payload.get("x_scale", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "x_offset and x_scale must be numbers")
        # Generous but bounded ranges. Tighter than free-for-all so a typo
        # can't drive the polygon kilometers off-screen.
        if not (-2000.0 <= x_off <= 2000.0):
            raise HTTPException(400, f"x_offset out of range [-2000, 2000]: {x_off}")
        if not (0.1 <= x_sca <= 10.0):
            raise HTTPException(400, f"x_scale out of range [0.1, 10.0]: {x_sca}")
        return x_off, x_sca

    @app.get("/api/projection_align/draft")
    async def projection_align_draft_get() -> dict[str, Any]:
        return dict(app.state.projection_align_draft)

    @app.post("/api/projection_align/draft")
    async def projection_align_draft_set(payload: dict[str, Any]) -> dict[str, Any]:
        """Live preview channel — /align pushes here on every slider tick,
        runtime / polls here to reflect changes in real time. Doesn't write
        to disk; only the explicit /save commits."""
        x_off, x_sca = _validate_align_payload(payload)
        prev_v = int(app.state.projection_align_draft.get("version", 0))
        app.state.projection_align_draft = {
            "x_offset": x_off, "x_scale": x_sca, "version": prev_v + 1,
        }
        return {"ok": True, "version": prev_v + 1}

    @app.post("/api/projection_align/save")
    async def projection_align_save() -> dict[str, Any]:
        """Persist the current draft to disk + apply as the new saved value.
        Reads from the draft (so the /align page can iterate live and then
        commit the final values), atomic write to projection_align_path."""
        x_off = float(app.state.projection_align_draft["x_offset"])
        x_sca = float(app.state.projection_align_draft["x_scale"])
        app.state.projection_align = {"x_offset": x_off, "x_scale": x_sca}
        path: Path | None = app.state.projection_align_path
        if path is not None:
            import os as _os, tempfile as _tf
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = _tf.mkstemp(
                    dir=str(path.parent),
                    prefix=".projection_align.", suffix=".json.tmp",
                )
                try:
                    with _os.fdopen(fd, "w") as f:
                        json.dump({"x_offset": x_off, "x_scale": x_sca}, f, indent=2)
                    _os.replace(tmp, path)
                except Exception:
                    try: _os.unlink(tmp)
                    except FileNotFoundError: pass
                    raise
            except OSError as e:
                raise HTTPException(500, f"failed to persist alignment: {e}")
        return {"ok": True, "x_offset": x_off, "x_scale": x_sca, "persisted": path is not None}

    @app.post("/api/projection_align/reset")
    async def projection_align_reset() -> dict[str, Any]:
        """Reset draft to identity (offset=0, scale=1). Doesn't save."""
        prev_v = int(app.state.projection_align_draft.get("version", 0))
        app.state.projection_align_draft = {
            "x_offset": 0.0, "x_scale": 1.0, "version": prev_v + 1,
        }
        return {"ok": True, "version": prev_v + 1}

    # ---------- paddle telemetry (stepper bridge consumer) -----------------

    @app.get("/api/paddle/config")
    async def paddle_config() -> dict[str, Any]:
        """Return the physical-rail X-pixel limits the runtime page uses to
        clamp the bound polygon. Computed from the loaded stepper calibration
        as zero_px + (min_steps | max_steps) / steps_per_px. Returns nulls if
        --stepper-calibration wasn't passed at startup (no clamping then)."""
        cal = app.state.stepper_calibration
        if not cal:
            return {"x_px_min": None, "x_px_max": None, "zero_px": None}
        try:
            zero_px = float(cal["zero_px"])
            steps_per_px = float(cal["steps_per_px"])
            min_steps = float(cal["min_steps"])
            max_steps = float(cal["max_steps"])
        except (KeyError, TypeError, ValueError):
            return {"x_px_min": None, "x_px_max": None, "zero_px": None}
        if steps_per_px == 0:
            return {"x_px_min": None, "x_px_max": None, "zero_px": zero_px}
        # px-space limits depend on the sign of steps_per_px. With a flipped
        # rail (negative scale), min_steps maps to the larger px and vice
        # versa — so always min/max the two derived values before returning,
        # the runtime just wants "leftmost px / rightmost px on screen".
        a = zero_px + min_steps / steps_per_px
        b = zero_px + max_steps / steps_per_px
        return {"x_px_min": min(a, b), "x_px_max": max(a, b), "zero_px": zero_px}

    @app.get("/api/paddle")
    async def paddle_get() -> dict[str, Any]:
        # Include `published` so the runtime page can distinguish "fresh
        # server with no data — keep polygon at drawn home" from "real last
        # position exists — translate polygon there". Without this the
        # browser couldn't tell the two cases apart.
        out = app.state.paddle_bus.snapshot().to_json()
        out["published"] = app.state.paddle_bus.has_published
        return out

    @app.post("/api/paddle/push")
    async def paddle_push(p: PaddlePush) -> dict[str, Any]:
        # Clamp x_frac defensively — a runaway client shouldn't be able to
        # drive the downstream stepper bridge outside [0, 1]. The bridge has
        # its own min/max-steps clamp too (calibration soft limits), this is
        # defense in depth.
        state = PaddleState(
            x_frac=max(0.0, min(1.0, p.x_frac)),
            x_px=p.x_px,
            width_px=p.width_px,
            vx=p.vx,
            t_ms=p.t_ms,
            paused=False,
        )
        await app.state.paddle_bus.publish(state)
        return {"ok": True}

    @app.websocket("/ws/paddle")
    async def paddle_ws(ws: WebSocket) -> None:
        await ws.accept()
        await app.state.paddle_bus.add(ws)
        try:
            while True:
                # We don't expect inbound messages; this keeps the socket
                # alive so disconnects raise WebSocketDisconnect.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.info("paddle ws closed: %s", e)
        finally:
            await app.state.paddle_bus.remove(ws)

    @app.websocket("/ws")
    async def runtime_ws(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_json()
                await _handle_runtime_event(app.state.devices, msg, ws)
        except WebSocketDisconnect:
            return
        except Exception as e:
            log.exception("ws handler crashed")
            try:
                await ws.send_json({"type": "error", "error": str(e)})
            except Exception:
                pass

    return app


async def _handle_runtime_event(
    registry: DeviceRegistry, msg: dict[str, Any], ws: WebSocket
) -> None:
    """Dispatch a single client event to the device registry. Replies with ack/error."""
    kind = msg.get("type")
    obj_id = msg.get("id")
    if not isinstance(kind, str) or not isinstance(obj_id, str):
        await ws.send_json({"type": "error", "error": "missing or invalid type/id"})
        return
    if obj_id not in registry:
        await ws.send_json({"type": "error", "error": f"unknown object {obj_id!r}"})
        return

    try:
        if kind == "hover":
            await registry.dispatch_hover(obj_id, bool(msg.get("enter", False)))
            await ws.send_json({"type": "ack", "id": obj_id, "kind": "hover"})
        elif kind == "click":
            button = msg.get("button", "left")
            if button not in ("left", "right", "middle"):
                await ws.send_json({"type": "error", "error": f"bad button {button!r}"})
                return
            await registry.dispatch_click(obj_id, button)
            await ws.send_json({"type": "ack", "id": obj_id, "kind": "click", "button": button})
            # ----- HARDCODED CHAIN: book_pusher click → lamp_switch on, +3s -----
            # Temporary scripted scene. Should move to a config-driven
            # chains/scenes system later (see TODO note in plan). Runs
            # fire-and-forget so the WS handler doesn't block on the delay.
            if obj_id == "book_pusher" and button == "left" and "lamp_switch" in registry:
                asyncio.create_task(_delayed_action(registry, "lamp_switch", "on", 3.0))
        elif kind == "menu":
            items = await registry.dispatch_context_menu(obj_id)
            await ws.send_json({
                "type": "menu",
                "id": obj_id,
                "items": [{"label": i.label, "action_id": i.action_id} for i in items],
            })
        elif kind == "menu_action":
            action_id = msg.get("action_id")
            value = msg.get("value")
            if not isinstance(action_id, str):
                await ws.send_json({"type": "error", "error": "missing action_id"})
                return
            await registry.dispatch_menu_action(obj_id, action_id, value)
            await ws.send_json({"type": "ack", "id": obj_id, "kind": "menu_action"})
        else:
            await ws.send_json({"type": "error", "error": f"unknown event type {kind!r}"})
    except Exception as e:
        log.exception("dispatch failed")
        await ws.send_json({"type": "error", "error": str(e), "id": obj_id})


async def _delayed_action(
    registry: DeviceRegistry,
    obj_id: str,
    action_id: str,
    delay_s: float,
) -> None:
    """Fire-and-forget helper for scripted chains (e.g., 'turn the lamp on
    3s after the book pusher clicks'). Sleeps, then dispatches the action to
    the registry. Swallows + logs exceptions so a delayed failure doesn't
    take down the WS handler that scheduled it.

    Currently used by the hardcoded book_pusher → lamp_switch chain in
    _handle_runtime_event. Refactor target: a config-driven chains/scenes
    system that reads from data/chains.yaml or similar."""
    try:
        await asyncio.sleep(delay_s)
        if obj_id not in registry:
            log.warning("delayed_action: %r not in registry, skipping", obj_id)
            return
        await registry.dispatch_menu_action(obj_id, action_id, None)
    except Exception:
        log.exception("delayed_action failed: %s.%s after %ss", obj_id, action_id, delay_s)


def _object_to_draft_entry(obj: dict[str, Any]) -> dict[str, Any]:
    """Convert one objects.json entry to the editor-draft schema. The editor
    uses `points` while objects.json uses `polygon` — we send both keys so the
    projection page renders correctly regardless of where the entry came from."""
    return {
        "id": obj["id"],
        "label": obj.get("label", ""),
        "points": list(obj.get("polygon", [])),
        "device_id": obj.get("device_id") or None,
        "style": obj.get("style", "normal"),
        "animation": dict(obj.get("animation") or {}),
        "appearance": dict(obj.get("appearance") or {}),
        "sound": dict(obj.get("sound") or {}),
        "click": dict(obj.get("click") or {}),
        "tracking": dict(obj.get("tracking") or {}),
        "notes": obj.get("notes", ""),
    }


UPLOAD_ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".webp")
UPLOAD_MAX_BYTES = 8 * 1024 * 1024     # 8 MB raw upload cap
UPLOAD_MAX_DIM = 2048                   # downscale anything larger than this on the long axis
_OBJ_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

SOUND_ALLOWED_EXTS = (".mp3", ".wav", ".ogg", ".m4a", ".aac")
SOUND_MAX_BYTES = 8 * 1024 * 1024


def _looks_like_audio(data: bytes, suffix: str) -> bool:
    """Magic-byte sniff for the formats we accept. Catches the common
    rename-the-extension attack — a renamed .exe won't have any of these
    headers. Not perfect (an attacker who crafts the header could still
    sneak content past) but good enough alongside the extension allowlist
    and same-origin upload."""
    if len(data) < 12:
        return False
    # WAV: 'RIFF....WAVE'
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return suffix == ".wav"
    # OGG: 'OggS'
    if data[:4] == b"OggS":
        return suffix == ".ogg"
    # M4A / AAC-in-MP4: bytes 4-7 == 'ftyp'
    if data[4:8] == b"ftyp":
        return suffix in (".m4a", ".aac")
    # MP3: 'ID3' tag at start, OR MPEG frame sync (0xFF, second byte 0xE0..0xFF)
    if data[:3] == b"ID3":
        return suffix == ".mp3"
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        # MPEG sync — could be MP3 or raw AAC ADTS (0xFF F1 / 0xFF F9 for AAC)
        return suffix in (".mp3", ".aac")
    return False


async def _save_uploaded_sound(file: UploadFile, object_id: str) -> dict[str, Any]:
    """Validate and persist a user-uploaded audio cue. Saves to
    data/sounds/<object_id>_<unix_ts>.<ext> with the original bytes
    untouched (no re-encoding, no normalization)."""
    if not _OBJ_ID_RE.match(object_id):
        raise HTTPException(400, "object_id must match [A-Za-z0-9_-]{1,64}")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SOUND_ALLOWED_EXTS:
        raise HTTPException(
            400, f"unsupported file type {suffix!r}; allowed: {SOUND_ALLOWED_EXTS}"
        )

    raw = await file.read()
    if len(raw) > SOUND_MAX_BYTES:
        raise HTTPException(
            400, f"file too large: {len(raw)} bytes (max {SOUND_MAX_BYTES})"
        )
    if len(raw) == 0:
        raise HTTPException(400, "empty upload")

    if not _looks_like_audio(raw, suffix):
        raise HTTPException(
            400, f"file does not look like {suffix} audio (magic-byte check failed)"
        )

    out_dir = Path("data/sounds")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{object_id}_{int(time.time())}{suffix}"
    out_path = out_dir / name
    out_path.write_bytes(raw)

    return {
        "ok": True,
        "src": f"data/sounds/{name}",
        "bytes": len(raw),
    }


async def _save_uploaded_image(file: UploadFile, object_id: str) -> dict[str, Any]:
    """Validate, normalize, and persist a user-uploaded image. Returns the
    payload the /style inspector writes into appearance.image.src. Pulled out
    of the route so tests can call it without an HTTP client. Hot path:
      1. sanity-check object_id format (used in the filename, so no traversal)
      2. read bytes, enforce size cap before doing any work
      3. open with Pillow — raises UnidentifiedImageError on non-image content
      4. apply EXIF rotation, drop EXIF, downscale to UPLOAD_MAX_DIM
      5. write to data/objects/<object_id>_<unix_ts>.<ext>
    """
    if not _OBJ_ID_RE.match(object_id):
        # Reject anything weirder than the new_object_id() format. Keeps the
        # uploaded filename safe even if validate_appearance later catches a
        # mismatch — defense in depth.
        raise HTTPException(400, "object_id must match [A-Za-z0-9_-]{1,64}")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_ALLOWED_EXTS:
        raise HTTPException(
            400, f"unsupported file type {suffix!r}; allowed: {UPLOAD_ALLOWED_EXTS}"
        )

    raw = await file.read()
    if len(raw) > UPLOAD_MAX_BYTES:
        raise HTTPException(
            400, f"file too large: {len(raw)} bytes (max {UPLOAD_MAX_BYTES})"
        )
    if len(raw) == 0:
        raise HTTPException(400, "empty upload")

    # Verify it's actually an image (not a renamed binary). Pillow's `verify()`
    # is cheap but leaves the file in an unusable state — we re-open after.
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"not a valid image: {e}")

    # EXIF auto-orientation (portrait photos taken on phones land upright).
    img = ImageOps.exif_transpose(img)

    # Force a mode the runtime can render. WebP / JPEG / PNG all round-trip
    # to RGB / RGBA cleanly; PA / P-mode palette images get converted to
    # something SVG <image> can definitely display.
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")

    # Downscale large originals. Preserves aspect ratio. The runtime renders
    # at the polygon's bbox so 4K source is wasted bandwidth.
    w, h = img.size
    if max(w, h) > UPLOAD_MAX_DIM:
        scale = UPLOAD_MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Pick the on-disk format. .jpg/.jpeg must save as JPEG (no alpha); other
    # formats keep their identity. RGBA→JPEG would error, so coerce to RGB.
    out_ext = ".jpg" if suffix in (".jpg", ".jpeg") else suffix
    if out_ext == ".jpg" and img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        img = bg

    out_dir = Path("data/objects")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{object_id}_{int(time.time())}{out_ext}"
    out_path = out_dir / name
    save_kwargs: dict[str, Any] = {}
    if out_ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 88
        save_kwargs["optimize"] = True
    elif out_ext == ".png":
        save_kwargs["optimize"] = True
    img.save(out_path, **save_kwargs)

    return {
        "ok": True,
        "src": f"data/objects/{name}",
        "w": img.size[0],
        "h": img.size[1],
    }


def _projector_dims(app: FastAPI) -> list[int]:
    """The projector rectangle, in pixels. Configured, never measured."""
    return list(app.state.projector_size)


def _objects_doc(path: Path) -> Any:
    return json.loads(path.read_text())


def _objects_list(path: Path, data: Any) -> list:
    """Pull the object list out of an objects file, accepting either the bare
    list form or the mapping form `{scene: {...}, objects: [...]}`."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        objs = data.get("objects", [])
        if not isinstance(objs, list):
            raise ValueError(
                f"{path}: 'objects' must be a list, got {type(objs).__name__}"
            )
        return objs
    raise ValueError(
        f"{path}: expected a list of objects or a mapping with 'objects', "
        f"got {type(data).__name__}"
    )


def load_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for i, item in enumerate(_objects_list(path, _objects_doc(path))):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{i}]: expected mapping, got {type(item).__name__}")
        try:
            ObjectSpec(**item)
        except Exception as e:
            raise ValueError(f"{path}[{i}]: invalid object — {e}") from None
        out.append(item)
    return out


def load_object_scene(path: Path) -> dict[str, Any]:
    """Per-scene behavior flags from the `scene:` block of an objects file in
    the mapping form. Empty dict for the bare-list form or a missing file. These
    OVERRIDE the global devices.yaml `scene:` flags for this objects file."""
    if not path.exists():
        return {}
    data = _objects_doc(path)
    if not isinstance(data, dict):
        return {}
    scene = data.get("scene") or {}
    if not isinstance(scene, dict):
        raise ValueError(f"{path}: 'scene' must be a mapping")
    return scene
