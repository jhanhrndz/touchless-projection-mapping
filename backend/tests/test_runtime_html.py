"""Source-level checks on runtime.html — confirms the effect gallery,
cursor hiding, and right-click menu wiring are present in the static
file. These are cheap text assertions; the real visual verification
is in scripts/screenshot_animations.py (Playwright).

Negative cases per project convention: invalid effect must fall back
to the default; the picker must be hidden by default.
"""

from pathlib import Path

import pytest

RUNTIME_HTML = (
    Path(__file__).parents[1] / "src" / "runtime" / "static" / "runtime.html"
)


@pytest.fixture(scope="module")
def html() -> str:
    return RUNTIME_HTML.read_text()


def test_html_file_exists():
    assert RUNTIME_HTML.exists(), f"missing {RUNTIME_HTML}"


@pytest.mark.parametrize("effect", ["glow", "pulse", "aurora", "bloom", "ants", "ember"])
def test_each_effect_has_a_style_block(html: str, effect: str):
    """Every advertised effect must have a `polygon.obj.fx-{effect}` rule,
    otherwise selecting it via ?effect=X yields a polygon with no hover styling
    at all. (Refactored from body.fx-* on 2026-05-08 so per-object overrides
    win specificity-wise.)"""
    assert f"polygon.obj.fx-{effect}" in html, f"no CSS rules for effect {effect!r}"


def test_effect_allowlist_in_js(html: str):
    """The JS effect allowlist must mention every effect with a CSS rule —
    otherwise ?effect=ember silently falls back to glow."""
    for effect in ["glow", "pulse", "aurora", "bloom", "ants", "ember"]:
        assert f"'{effect}'" in html, f"effect {effect!r} missing from JS EFFECTS array"


def test_invalid_effect_falls_back(html: str):
    """The JS guard must use an allowlist (.includes), not raw assignment."""
    assert "EFFECTS.includes(params.get('effect'))" in html


def test_cursor_hidden_over_polygons(html: str):
    """The whole point of the projection illusion is that no OS cursor sits on
    top of the projected object. polygon.obj must explicitly hide the cursor."""
    assert "polygon.obj" in html
    assert "cursor: none" in html


def test_body_cursor_default_is_regular_pointer(html: str):
    """Default body cursor should be the OS default arrow, not crosshair —
    crosshair stayed available as an opt-in via ?cursor=crosshair."""
    assert "body.cursor-default" in html
    assert "cursor: default" in html


@pytest.mark.parametrize("name", ["default", "crosshair", "none"])
def test_each_cursor_option_has_a_class(html: str, name: str):
    assert f"body.cursor-{name}" in html
    assert f"'{name}'" in html  # in the JS CURSORS allowlist


def test_invalid_cursor_falls_back(html: str):
    assert "CURSORS.includes(params.get('cursor'))" in html


def test_picker_hidden_by_default(html: str):
    """?picker=1 reveals it; without that param, it must be display: none."""
    assert "#picker" in html
    assert "display: none;" in html
    assert "body[data-picker=\"1\"] #picker" in html


def test_context_menu_uses_menu_event_not_click(html: str):
    """Right-click must request the menu (so server can return MenuItems),
    not just send a click with button=right (the previous behaviour)."""
    assert "type:'menu'" in html or "type: 'menu'" in html
    # And the response handler must render items returned by the server.
    assert "msg.type === 'menu'" in html


def test_context_menu_clamped_to_viewport(html: str):
    """Menus near the bottom-right of a 1024x768 frame would otherwise be
    clipped off-screen and unclickable on the projector."""
    assert "projectorW" in html and "projectorH" in html
    assert "Math.min" in html  # the clamp


def test_pulse_effect_uses_class_toggle_not_sibling_selector(html: str):
    """Earlier draft used `polygon.obj:hover ~ polygon.pulse-ring` which would
    activate ALL later sibling rings, not just the matching one. Confirm the
    code now uses an explicit class toggle keyed by data-for=targetId."""
    assert "polygon.pulse-ring.active" in html
    assert "data-for=" in html


def test_aurora_gradient_defined(html: str):
    """The aurora effect references stroke: url(#aurora-gradient) — make sure
    the gradient itself is in <defs> or hovering crashes silently."""
    assert 'id="aurora-gradient"' in html
    assert 'id="aurora-stop-1"' in html


def test_bloom_radial_gradient_defined(html: str):
    assert 'id="bloom-gradient"' in html
    assert "<radialGradient" in html


# ---------- occluder polygons ----------

def test_occluder_class_renders_solid_black(html: str):
    """Occluder polygons must be solid black with no border, beating any fx-* hover rule."""
    assert "polygon.obj.occluder" in html
    assert "fill: #000 !important" in html
    assert "stroke: none !important" in html


def test_occluder_skips_hover_label(html: str):
    """A bright label over a black hole defeats the 'no UI here' intent."""
    assert "polygon.obj.occluder + text.label { display: none; }" in html


def test_occluder_skips_websocket_events_in_js(html: str):
    """The hover/click/menu listeners are inside `if (!isOccluder)` — confirm
    that gate exists so occluders never produce server-side events."""
    assert "isOccluder" in html
    assert "if (!isOccluder)" in html


EDITOR_HTML = (
    Path(__file__).parents[1] / "src" / "runtime" / "static" / "editor.html"
)


@pytest.fixture(scope="module")
def editor_html() -> str:
    return EDITOR_HTML.read_text()


def test_editor_has_occluder_checkbox(editor_html: str):
    """The editor must expose the occluder choice — otherwise users can only
    set style by hand-editing objects.json."""
    assert 'id="occluder"' in editor_html
    assert 'type="checkbox"' in editor_html


def test_editor_save_payload_includes_style(editor_html: str):
    """When checked, save payload must propagate `style: "occluder"` to the server."""
    assert "out.style = o.style" in editor_html


def test_editor_save_strips_device_for_occluders(editor_html: str):
    """Occluders ignore device bindings — make sure save doesn't smuggle one in."""
    assert "o.style !== 'occluder'" in editor_html


# ---------- editor projection (split-window) ----------

PROJECTION_HTML = (
    Path(__file__).parents[1] / "src" / "runtime" / "static" / "editor_projection.html"
)


@pytest.fixture(scope="module")
def projection_html() -> str:
    return PROJECTION_HTML.read_text()


def test_projection_page_polls_draft_endpoint(projection_html: str):
    """The whole point: this page must poll the draft state continuously
    so polygons appear on the projector as the editor draws."""
    assert "/api/editor/draft" in projection_html
    assert "setTimeout(tick" in projection_html  # the polling loop


def test_projection_page_renders_active_and_saved_distinctly(projection_html: str):
    """User needs to see WHICH polygon is currently being drawn."""
    assert "polygon.active" in projection_html
    assert "polygon.saved" in projection_html


def test_projection_page_renders_occluders_visibly(projection_html: str):
    """Solid-black occluders against a black wall would be invisible while
    editing — must have a dashed outline so user can see where they are."""
    assert "polygon.occluder" in projection_html
    assert "stroke-dasharray" in projection_html


def test_projection_page_shows_in_progress_polylines(projection_html: str):
    """1-2 point objects render as polyline so the trace is visible."""
    assert "polyline.draft" in projection_html
    assert "createElementNS(SVG_NS, 'polyline')" in projection_html


def test_projection_page_renders_live_cursor_preview(projection_html: str):
    """Held-spacebar in the editor must surface as a visible dot here."""
    assert "circle.cursor-preview" in projection_html
    assert "draft.cursor" in projection_html
    # Visibility gate — don't render when the editor isn't holding space.
    assert "cursor.visible" in projection_html


def test_editor_pushes_draft_after_changes(editor_html: str):
    """The split editor only works if the laptop side actually pushes drafts
    after every state mutation. Verify the call sites exist."""
    assert "pushDraft()" in editor_html
    # render() ends with pushDraft so add-vertex / new / undo / clear / delete all push.
    assert "renderList();\n  pushDraft();" in editor_html
    # Label/device handlers also push (they call renderList, not render). The
    # 2026-05-22 undo refactor moved their calls onto separate lines so look
    # for both on the same line OR consecutive lines.
    assert "renderList();" in editor_html
    label_block = editor_html[editor_html.index("labelInput.addEventListener"):]
    assert "pushDraft" in label_block[:600]  # within the label handler body


def test_editor_draft_push_is_throttled(editor_html: str):
    """Otherwise drag/typing hammers the server."""
    assert "_draftTimer" in editor_html
    assert "setTimeout" in editor_html


def test_editor_links_to_projection_mirror(editor_html: str):
    """Discoverability — without this hint the user wouldn't know the projection
    page exists."""
    assert "/editor/projection" in editor_html


def test_editor_has_single_canvas_layout(editor_html: str):
    """One black canvas plus the controls sidebar — no camera pane.

    The canvas IS the projector rectangle, so there's nothing to reconcile it
    against and no second pane to draw.
    """
    assert 'id="editor-pane"' in editor_html
    assert 'id="canvas"' in editor_html
    edit_idx = editor_html.index('id="editor-pane"')
    canvas_idx = editor_html.index('id="canvas"')
    assert edit_idx < canvas_idx, "the click-target SVG must live inside the editor pane"


def test_editor_canvas_has_no_backdrop_image(editor_html: str):
    """The editor pane is fullscreen black to match the projector — render()
    must never append an <image> backdrop to the SVG."""
    assert "createElementNS(SVG_NS, 'image')" not in editor_html
    assert "background: #000;" in editor_html


def test_editor_has_no_camera_surface(editor_html: str):
    """Negative: every camera affordance must be gone, not merely hidden.

    A leftover Recapture button or live-camera toggle would call an endpoint
    that no longer exists, and the failure would look like a broken editor
    rather than a removed feature.
    """
    for gone in (
        'id="camera-pane"',
        'id="camera-img"',
        'id="live-toggle"',
        'id="recapture"',
        "/api/calibrate/camera_frame.jpg",
        "/api/editor/scene.png",
        "/api/editor/scene/refresh",
        "liveCamera",
        "refreshCameraImg",
    ):
        assert gone not in editor_html, f"stale camera reference: {gone}"


def test_editor_viewbox_is_projector_pixels(editor_html: str):
    """The SVG viewBox must be driven by the reported projector size.

    That's what makes editor coordinates equal projector pixels 1:1 — the
    property that replaces calibration entirely.
    """
    assert "svg.setAttribute('viewBox', `0 0 ${projector[0]} ${projector[1]}`)" in editor_html
    assert "/api/editor/scene" in editor_html, "canvas size still comes from the scene endpoint"


def test_editor_spacebar_toggles_cursor_visibility(editor_html: str):
    """Spacebar TOGGLES the cursor preview on/off (changed from hold-to-show on
    2026-05-23 because toggle is better when aligning polygons to real objects
    — you can let go of the keyboard while you reposition). The toggle path
    is `cursorVisible = !cursorVisible` inside the keydown handler; there's no
    keyup handler anymore."""
    assert "cursorVisible = !cursorVisible" in editor_html
    assert "e.code === 'Space'" in editor_html
    assert "!e.repeat" in editor_html, "key-repeat suppression keeps a held key from ping-ponging the toggle"


def test_editor_ctrl_z_calls_undo(editor_html: str):
    """Cmd/ctrl-z must call the undo() function (action-level history, replaced
    the old undoLastPoint vertex-only undo on 2026-05-23). The undo button
    in the sidebar wires to the same function."""
    assert "function undo(" in editor_html, "undo() function must be defined"
    assert "history.pop()" in editor_html, "undo must actually pop from the history stack"
    # Both metaKey (mac) and ctrlKey (linux/windows) need to work.
    assert "e.ctrlKey || e.metaKey" in editor_html
    # Must not fight browser undo while typing in the label input.
    assert "isTypingTarget" in editor_html


def test_editor_cursor_in_draft_payload(editor_html: str):
    """Without this, the projection page can't render the dot — the whole
    feature dies silently if the payload doesn't carry the cursor."""
    assert "cursor: {x: cursorPos.x, y: cursorPos.y, visible: cursorVisible}" in editor_html


# ---------- per-object animation in runtime ----------

def test_runtime_supports_hovered_class_for_simulation(html: str):
    """Tuner page applies `.hovered` to fake :hover. Every fx-* hover rule
    must accept the class — otherwise tuner preview shows a blank polygon."""
    for fx in ("glow", "pulse", "aurora", "bloom", "ants", "ember"):
        assert f"polygon.obj.fx-{fx}.hovered" in html, f"missing .hovered for fx-{fx}"


def test_runtime_polygon_picks_per_object_effect(html: str):
    """JS must read obj.animation.effect with fallback to global. Without this
    the per-object override is a dead schema field."""
    assert "obj.animation || {}" in html
    assert "polyEffect" in html
    assert "EFFECTS.includes(a.effect)" in html


def test_runtime_polygon_emits_css_vars_for_overrides(html: str):
    """Color/intensity/speed must reach CSS via inline custom properties
    on the polygon — the fx-* rules read them via var(--obj-color, ...)."""
    assert "--obj-color" in html
    assert "--obj-intensity" in html
    assert "--obj-speed" in html
    # And the JS must actually write these onto the polygon.
    assert "styleParts.push(`--obj-color:" in html


def test_runtime_effect_rules_use_intensity_var(html: str):
    """fill-opacity / stroke-opacity / drop-shadow blur should multiply by
    --obj-intensity. Otherwise the intensity slider does nothing."""
    assert "calc(0.18 * var(--obj-intensity))" in html  # glow fill-opacity
    assert "calc(12px * var(--obj-intensity))" in html  # glow drop-shadow blur


def test_runtime_effect_rules_use_speed_var(html: str):
    """Animation durations must divide by --obj-speed so the speed slider
    actually accelerates/decelerates per-polygon. Pulse/bloom/ants are the
    animated effects; aurora's gradient stops are global on purpose."""
    assert "1.4s / var(--obj-speed" in html  # pulse-ring
    assert "0.8s / var(--obj-speed" in html  # ants
    assert "2.4s / var(--obj-speed" in html  # bloom-breath


# ---------- /tune page ----------

TUNE_HTML = (
    Path(__file__).parents[1] / "src" / "runtime" / "static" / "tune.html"
)


@pytest.fixture(scope="module")
def tune_html() -> str:
    return TUNE_HTML.read_text()


def test_tune_page_has_all_form_fields(tune_html: str):
    """Without these IDs the wiring breaks silently — check every knob the
    user picked (effect/color/intensity/speed) plus notes."""
    for fid in ("f-effect", "f-color", "f-intensity", "f-speed", "f-notes"):
        assert f'id="{fid}"' in tune_html, f"tune.html missing #{fid}"


def test_tune_page_lists_all_effects_in_dropdown(tune_html: str):
    for fx in ("glow", "pulse", "aurora", "bloom", "ants", "ember"):
        assert f'value="{fx}"' in tune_html, f"effect {fx!r} missing from dropdown"


def test_tune_page_save_uses_existing_endpoint(tune_html: str):
    """Reuse /api/editor/save so animation/notes go through the same validation
    path as polygon edits — single source of truth on the wire."""
    assert "'/api/editor/save'" in tune_html


def test_tune_page_omits_color_when_unset(tune_html: str):
    """Saving must not emit a color field unless the user actively picked one
    — otherwise every saved object would lock to the form's default #ffaa55."""
    # The dataset.set sentinel gates color emission.
    assert "fColor.dataset.set" in tune_html


def test_tune_page_warns_on_unsaved_changes(tune_html: str):
    """beforeunload guard so accidentally closing the tab doesn't lose tuning."""
    assert "beforeunload" in tune_html


def test_tune_page_color_picker_disabled_for_gradient_effects(tune_html: str):
    """Aurora and bloom use gradients — color override has no visible effect.
    Surface that explicitly in the dropdown labels so users know."""
    assert "color override N/A" in tune_html


# ---------- text projector (/text) ----------

TEXT_HTML = (
    Path(__file__).parents[1] / "src" / "runtime" / "static" / "text.html"
)


@pytest.fixture(scope="module")
def text_html() -> str:
    return TEXT_HTML.read_text()


def test_text_page_uses_futura_first(text_html: str):
    """Futura is the requested face. Fallbacks must follow but Futura must be
    the first name in the stack — without it, macOS would skip straight to
    Trebuchet/Avenir even when Futura is installed."""
    # The font-family stack appears once on textarea#text.
    block = text_html[text_html.index("font-family:"):]
    # First quoted name must be Futura.
    first_quote = block.index("'")
    end_quote = block.index("'", first_quote + 1)
    first_font = block[first_quote + 1:end_quote]
    assert first_font == 'Futura', f"first font must be Futura, was {first_font!r}"


def test_text_page_has_black_bg_and_white_default(text_html: str):
    """Defaults: black background, white text. Overridable via URL params, but
    the bare /text URL has to look right."""
    assert "--bg: #000" in text_html
    assert "--fg: #fff" in text_html


def test_text_page_supports_url_param_overrides(text_html: str):
    """Each tunable maps to a CSS var so ?size=12vh etc. takes effect without
    a page reload of the source. Verify all the documented params are wired."""
    for css_var in ['--bg', '--fg', '--size', '--weight', '--align', '--lineheight']:
        assert f"'{css_var}'" in text_html, f"missing wiring for CSS var {css_var}"
    # URL param names referenced in JS:
    for param in ('bg', 'color', 'size', 'weight', 'align', 'lineheight',
                  'placeholder', 'text', 'cursor', 'x', 'y'):
        assert f"params.get('{param}')" in text_html or f"params.has('{param}')" in text_html, \
            f"missing URL param {param!r}"


def test_text_page_cursor_visible_by_default(text_html: str):
    """Reverted 2026-05-23 per user feedback: the OS cursor stays visible on
    /text by default. Pass ?cursor=hide for a pure projection. The cursor-hidden
    state lives on body.cursor-hidden so the CSS rule must reference that class
    (not the body element directly)."""
    assert "body.cursor-hidden { cursor: none" in text_html
    assert "params.get('cursor') === 'hide'" in text_html


def test_text_page_polls_draft_endpoint(text_html: str):
    """The projection view must poll /api/text/draft so anything typed in
    /text/editor (the laptop control) renders here live."""
    assert "/api/text/draft" in text_html
    assert "setInterval" in text_html


# ---------- text editor (/text/editor) ----------

TEXT_EDITOR_HTML = (
    Path(__file__).parents[1] / "src" / "runtime" / "static" / "text_editor.html"
)


@pytest.fixture(scope="module")
def text_editor_html() -> str:
    return TEXT_EDITOR_HTML.read_text()


def test_text_editor_has_two_pane_layout(text_editor_html: str):
    """Preview (left) + controls (right) — matches the polygon editor's pattern
    so muscle memory transfers."""
    assert 'id="preview"' in text_editor_html
    assert 'id="panel"' in text_editor_html
    preview_idx = text_editor_html.index('id="preview"')
    panel_idx = text_editor_html.index('id="panel"')
    assert preview_idx < panel_idx, "preview must come before panel (left side)"


def test_text_editor_pushes_draft_on_input(text_editor_html: str):
    """The whole sync model breaks if the laptop side doesn't push changes."""
    assert "/api/text/draft" in text_editor_html
    assert "pushPatch" in text_editor_html


def test_text_editor_text_block_is_draggable(text_editor_html: str):
    """The user explicitly asked for drag-to-position. Verify the pointer
    event wiring exists and that the dragging class toggles for cursor feedback."""
    assert "addEventListener('pointerdown'" in text_editor_html
    assert "addEventListener('pointermove'" in text_editor_html
    assert "addEventListener('pointerup'" in text_editor_html
    assert "classList.add('dragging')" in text_editor_html


def test_text_editor_drag_converts_to_fractional_coords(text_editor_html: str):
    """x/y must be fractions of the preview rect so editor + projector with
    different resolutions stay aligned. The drag math divides by rect.width/height."""
    assert "rect.width" in text_editor_html
    assert "rect.height" in text_editor_html


def test_text_editor_has_quick_presets(text_editor_html: str):
    """Sanity check on the preset buttons — covers the discoverability hint
    in the help bullet (title-card / lower-third / overlay)."""
    for preset in ("title-card", "lower-third", "subtle overlay"):
        assert preset in text_editor_html


# ---------- /api/text/draft endpoint ----------

def test_text_draft_get_returns_defaults(tmp_path):
    """Out-of-the-box GET returns the documented defaults so /text renders
    even if no one has ever opened /text/editor."""
    from runtime.server import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    body = TestClient(app).get("/api/text/draft").json()
    assert body["text"] == ""
    assert body["x"] == 0.5
    assert body["y"] == 0.5
    assert body["color"] == "#ffffff"
    assert body["bg"] == "#000000"


def test_text_draft_post_merges_partial_payload(tmp_path):
    """POST is merge-style — the editor pushes a single field per input event,
    untouched fields must keep their prior values."""
    from runtime.server import create_app
    from fastapi.testclient import TestClient
    client = TestClient(create_app())
    client.post("/api/text/draft", json={"text": "hello"})
    client.post("/api/text/draft", json={"x": 0.2, "y": 0.8})
    body = client.get("/api/text/draft").json()
    assert body["text"] == "hello"        # preserved
    assert body["x"] == 0.2 and body["y"] == 0.8
    assert body["color"] == "#ffffff"     # untouched defaults preserved
    assert body["align"] == "center"


def test_text_draft_post_rejects_bad_types(tmp_path):
    """Strings vs numbers must be enforced — without this a stray null or number
    in `text` would make /text crash trying to call .length."""
    from runtime.server import create_app
    from fastapi.testclient import TestClient
    client = TestClient(create_app())
    res1 = client.post("/api/text/draft", json={"text": 42})
    assert res1.status_code == 400
    res2 = client.post("/api/text/draft", json={"x": "left"})
    assert res2.status_code == 400


def test_text_draft_post_ignores_unknown_fields(tmp_path):
    """Forgiving on unknown keys — older clients pushing new fields shouldn't
    error. Same stance as /api/editor/draft."""
    from runtime.server import create_app
    from fastapi.testclient import TestClient
    client = TestClient(create_app())
    res = client.post("/api/text/draft", json={"text": "hi", "future_field": True})
    assert res.status_code == 200
    assert res.json()["text"] == "hi"


def test_text_editor_route_serves_html(tmp_path):
    from runtime.server import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    res = TestClient(app).get("/text/editor")
    assert res.status_code == 200
    assert "Text projector — editor" in res.text


# ---------- saved text presets (/api/text/presets) ----------

def _client_in(tmp_path, monkeypatch):
    """Make a fresh TestClient in an isolated CWD so each test gets a clean
    data/text_presets.json file (no cross-contamination, no risk of touching
    the dev environment's real file)."""
    from runtime.server import create_app
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def test_text_presets_empty_when_no_file(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    body = client.get("/api/text/presets").json()
    assert body == []


def test_text_presets_save_creates_file_and_returns_entry(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    config = {"text": "hi", "x": 0.5, "y": 0.5, "size": "8vh",
              "color": "#ffffff", "bg": "#000000",
              "weight": "normal", "align": "center", "line_height": 1.15}
    res = client.post("/api/text/presets", json={"name": "intro", "config": config})
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 1
    assert body["saved"]["name"] == "intro"
    assert body["saved"]["config"]["text"] == "hi"
    assert "updated" in body["saved"]
    # File on disk
    import json as _json
    p = tmp_path / "data" / "text_presets.json"
    assert p.exists()
    on_disk = _json.loads(p.read_text())
    assert on_disk[0]["name"] == "intro"


def test_text_presets_save_with_same_name_overwrites(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    client.post("/api/text/presets", json={"name": "A", "config": {"text": "v1"}})
    client.post("/api/text/presets", json={"name": "A", "config": {"text": "v2"}})
    body = client.get("/api/text/presets").json()
    assert len(body) == 1, "same-name save must replace, not append"
    assert body[0]["config"]["text"] == "v2"


def test_text_presets_save_multiple_names_appends(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    client.post("/api/text/presets", json={"name": "A", "config": {"text": "a"}})
    client.post("/api/text/presets", json={"name": "B", "config": {"text": "b"}})
    client.post("/api/text/presets", json={"name": "C", "config": {"text": "c"}})
    body = client.get("/api/text/presets").json()
    assert len(body) == 3
    names = {p["name"] for p in body}
    assert names == {"A", "B", "C"}


def test_text_presets_save_rejects_empty_name(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    res1 = client.post("/api/text/presets", json={"name": "", "config": {}})
    assert res1.status_code == 400
    res2 = client.post("/api/text/presets", json={"name": "   ", "config": {}})
    assert res2.status_code == 400
    res3 = client.post("/api/text/presets", json={"config": {}})
    assert res3.status_code == 400


def test_text_presets_save_rejects_overlong_name(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    res = client.post("/api/text/presets", json={"name": "x" * 200, "config": {}})
    assert res.status_code == 400
    assert "80 chars" in res.text


def test_text_presets_save_rejects_bad_config_types(tmp_path, monkeypatch):
    """Type validation must match the draft endpoint so loading a preset can't
    crash the projector view downstream."""
    client = _client_in(tmp_path, monkeypatch)
    res1 = client.post("/api/text/presets", json={"name": "ok", "config": {"text": 123}})
    assert res1.status_code == 400
    res2 = client.post("/api/text/presets", json={"name": "ok", "config": {"x": "center"}})
    assert res2.status_code == 400


def test_text_presets_save_ignores_unknown_config_fields(tmp_path, monkeypatch):
    """Forgiving on unknown fields, same as draft endpoint."""
    client = _client_in(tmp_path, monkeypatch)
    res = client.post("/api/text/presets", json={
        "name": "ok",
        "config": {"text": "hi", "secret_field": "ignore-me"},
    })
    assert res.status_code == 200
    saved = res.json()["saved"]["config"]
    assert "secret_field" not in saved
    assert saved["text"] == "hi"


def test_text_presets_delete_removes_one(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    client.post("/api/text/presets", json={"name": "A", "config": {"text": "a"}})
    client.post("/api/text/presets", json={"name": "B", "config": {"text": "b"}})
    res = client.delete("/api/text/presets/A")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "deleted": "A", "count": 1}
    remaining = {p["name"] for p in client.get("/api/text/presets").json()}
    assert remaining == {"B"}


def test_text_presets_delete_404_when_missing(tmp_path, monkeypatch):
    client = _client_in(tmp_path, monkeypatch)
    res = client.delete("/api/text/presets/nope")
    assert res.status_code == 404


def test_text_presets_survive_corrupt_file(tmp_path, monkeypatch):
    """If the file is malformed (manual edit / partial write), GET returns []
    instead of 500, so the user can still proceed and save a fresh preset
    on top. Otherwise a single bad keystroke breaks the feature permanently."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "text_presets.json").write_text("{ not valid json")

    from runtime.server import create_app
    from fastapi.testclient import TestClient
    client = TestClient(create_app())
    assert client.get("/api/text/presets").json() == []
    # And a save still works.
    res = client.post("/api/text/presets", json={"name": "ok", "config": {"text": "x"}})
    assert res.status_code == 200


# ---------- text editor UI: saved-presets wiring ----------

def test_text_editor_has_saved_presets_section(text_editor_html: str):
    """The Saved presets UI exists with a name input, save button, and list."""
    assert 'id="saved-name"' in text_editor_html
    assert 'id="saved-save"' in text_editor_html
    assert 'id="saved-list"' in text_editor_html


def test_text_editor_save_button_posts_to_presets_endpoint(text_editor_html: str):
    """Saving must hit POST /api/text/presets — verify the URL and the
    `config` payload key are present."""
    assert "/api/text/presets" in text_editor_html
    assert "method: 'POST'" in text_editor_html
    assert "saveCurrentAsPreset" in text_editor_html


def test_text_editor_delete_confirms_before_removing(text_editor_html: str):
    """One-click deletion is hostile — must confirm first."""
    assert "confirm(" in text_editor_html
    assert "method: 'DELETE'" in text_editor_html


def test_text_route_serves_html(tmp_path):
    """The /text route should return the text projector HTML."""
    from runtime.server import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    res = TestClient(app).get("/text")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Text projector" in res.text
    assert "Futura" in res.text
