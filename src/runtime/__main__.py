"""Runtime entry point — serves the interactive projection UX with hover/click → device.

Usage:
    uv run python -m runtime [--host 127.0.0.1] [--port 8000]
                             [--objects data/objects.json]
                             [--devices data/devices.yaml]
                             [--projector-size 1920x1080] [--mock]

Advanced flags for stepper-rail rigs (inert when unset):
    [--stepper-calibration PATH] [--paddle-state PATH] [--projection-align PATH]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .config import load_registry, load_scene_settings
from .server import create_app, load_object_scene, load_objects


def parse_projector_size(text: str) -> tuple[int, int]:
    """'1920x1080' -> (1920, 1080). Raises ValueError on anything else.

    Fails loudly rather than falling back to a default: a silently-wrong canvas
    size means every polygon you draw lands in the wrong place on the wall, and
    you'd chase that for an hour before suspecting the flag.
    """
    parts = text.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError("expected WxH, e.g. 1920x1080")
    try:
        w, h = (int(p) for p in parts)
    except ValueError:
        raise ValueError("width and height must be whole numbers") from None
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")
    return w, h


class _QuietPollAccessFilter(logging.Filter):
    """Drop access-log lines for the sub-second draft/telemetry polls.

    The whole dev feedback loop is "click the polygon, watch the terminal" —
    but one open page polls its draft endpoint every 250ms, which is enough
    log traffic to scroll a [mock] click line out of sight within seconds.
    Every other request still logs normally.
    """

    _POLL_PATHS = (
        "/api/editor/draft",
        "/api/style/draft",
        "/api/text/draft",
        "/api/projection_align/draft",
        "/api/paddle",           # also covers /config and /push
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(path in msg for path in self._POLL_PATHS):
            return False
        # In the container, the Docker healthcheck curls /api/editor/scene
        # from loopback every 10s — same terminal-noise problem at 1/10th the
        # rate. Gated on the env flag the Dockerfile sets, so a host-side
        # `curl localhost:8000/api/editor/scene` (the AGENTS.md smoke check)
        # still logs when running outside Docker.
        if (
            os.environ.get("RUNTIME_IN_CONTAINER")
            and "127.0.0.1" in msg
            and "/api/editor/scene" in msg
        ):
            return False
        return True


def main(argv: list[str] | None = None) -> int:
    # .env is loaded before the parser is built so APP_HOST / APP_PORT can
    # seed the flag defaults (kasa creds are read later, at connect time).
    # The explicit path stops python-dotenv's default parent-directory walk,
    # which crashes outright on an unreadable ancestor .env — and, worse,
    # silently inherits another project's SMART_SWITCH_HOST from one it can
    # read. Only ./yourrepo/.env is ever consulted; missing is a no-op.
    load_dotenv(".env")

    parser = argparse.ArgumentParser(prog="python -m runtime")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("APP_PORT", "8000")),
        help="TCP port to serve on (default: $APP_PORT or 8000)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("APP_HOST", "127.0.0.1"),
        help="interface to bind (default: $APP_HOST or 127.0.0.1 — loopback "
             "only). Use 0.0.0.0 to reach the server from another machine on "
             "the LAN, or from outside a container.",
    )
    parser.add_argument(
        "--objects",
        default="data/objects.json",
        help="polygon scene file, written by /editor (default: %(default)s)",
    )
    parser.add_argument(
        "--devices",
        default="data/devices.yaml",
        help="object_id → driver bindings; use data/devices.dev.yaml for the "
             "all-mock hardware-free setup (default: %(default)s)",
    )
    parser.add_argument(
        "--projector-size",
        default="1920x1080",
        metavar="WxH",
        help="your projector's native resolution (default 1920x1080). The "
             "editor canvas is exactly this rectangle and what you draw at "
             "(x, y) lands at (x, y) on the wall.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="rewrite every device entry's type to 'mock' — same object_ids, "
             "no real hardware fired (clicks/menu actions go to MockDevice).",
    )
    parser.add_argument(
        "--stepper-calibration",
        default=None,
        metavar="PATH",
        help="path to a stepper calibration.json from your rail rig. When "
             "provided, the runtime page clamps the bound polygon's X to the "
             "physical rail's reachable range so you can't drag past where "
             "the carriage can actually go. Off by default — no clamping "
             "when not set.",
    )
    parser.add_argument(
        "--paddle-state",
        default=None,
        metavar="PATH",
        help="path where the paddle bus persists the last-known X position "
             "(e.g. data/paddle_state.json). Without this, every server "
             "restart resets to default x_px=0 — a stepper bridge would then "
             "drive the carriage to the leftmost rail limit on every "
             "reconnect, and browser reloads would snap the polygon back to "
             "its drawn home.",
    )
    parser.add_argument(
        "--projection-align",
        default=None,
        metavar="PATH",
        help="path to projection_align.json — a pure-visual x_offset + x_scale "
             "applied to where tracked polygons are DRAWN on the projector. "
             "Lets you nudge the projected outline into alignment with the "
             "physical carriage without touching stepper calibration. Tune "
             "via the /align page; saved here on commit. Identity (0, 1) "
             "when missing.",
    )
    args = parser.parse_args(argv)

    try:
        projector_size = parse_projector_size(args.projector_size)
    except ValueError as e:
        print(f"error: --projector-size {args.projector_size!r}: {e}", file=sys.stderr)
        return 2

    devices_path = Path(args.devices)
    if not devices_path.exists():
        # A typo'd path would otherwise boot a server with zero devices and
        # no clue why every click does nothing. Fail loudly instead, the way
        # a malformed --projector-size does.
        candidates = (
            sorted(str(p) for p in devices_path.parent.glob("devices*.y*ml"))
            if devices_path.parent.exists()
            else []
        )
        hint = f" — available: {', '.join(candidates)}" if candidates else ""
        print(f"error: --devices {args.devices!r} not found{hint}", file=sys.stderr)
        return 2

    objects_path = Path(args.objects)
    objects = load_objects(objects_path)
    registry = load_registry(devices_path, mock=args.mock)
    # Scene flags: devices.yaml is the global default; the objects file's own
    # `scene:` block overrides it per-scene (purely visual — e.g. flame_carry).
    object_scene = load_object_scene(objects_path)
    scene_settings = {**load_scene_settings(Path(args.devices)), **object_scene}

    if args.mock:
        print("** MOCK MODE — no real hardware will be triggered **")
    print(f"Loaded {len(objects)} object(s), {len(registry)} device(s).")
    if scene_settings:
        print(f"Scene flags: {scene_settings}")
    for obj in objects:
        # The runtime page dispatches on `device_id` when set, falling back to
        # the polygon's own id (runtime.html: `obj.device_id || obj.id`).
        # Resolve the same way here or the banner reports "no device" for every
        # polygon authored in the editor, which assigns a generated id.
        bound = (obj.get("device_id") or obj["id"]) in registry
        marker = "✓" if bound else "✗ (no device)"
        print(f"  {marker} {obj['id']:20s} '{obj.get('label', '')}'")
    if not objects:
        print(
            "No objects yet — draw one in /editor, or seed the worked example:\n"
            "  cp data/examples/objects.lamp.json data/objects.json"
        )

    # The three-window setup, printed every boot: this is the part of the
    # README people skip, and without it the wall shows a URL bar.
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    base = f"http://{display_host}:{args.port}"
    print("Open these three windows:")
    print(f"  {base}/                   final scene — fullscreen on the projector (F11 / ctrl-cmd-F)")
    print(f"  {base}/editor             your monitor — create polygons, bind devices + logic")
    print(f"  {base}/editor/projection  projector — editable 1:1 twin; keep as a tab next to /")

    # Prewarm runs in the FastAPI lifespan (server.py), not here — kasa
    # connections must be cached on the uvicorn event loop, not a throwaway
    # asyncio.run() loop. Skipped entirely in mock mode.

    # Load the stepper calibration once at startup if provided. The runtime page
    # fetches the resolved pixel limits at /api/paddle/config and clamps the
    # bound polygon so users can't drag past the rail's reachable range.
    stepper_calibration = None
    if args.stepper_calibration:
        import json
        cal_path = Path(args.stepper_calibration)
        try:
            stepper_calibration = json.loads(cal_path.read_text())
            print(f"loaded stepper calibration from {cal_path}: "
                  f"zero={stepper_calibration.get('zero_px')} "
                  f"scale={stepper_calibration.get('steps_per_px')} "
                  f"limits=[{stepper_calibration.get('min_steps')}, "
                  f"{stepper_calibration.get('max_steps')}]")
        except (OSError, ValueError) as e:
            # Soft failure — server still boots, runtime page just won't clamp.
            # Avoids "can't run anything without calibrating first" lock-out.
            print(f"WARN: failed to load --stepper-calibration {cal_path}: {e}")
            print("WARN: runtime page will run without rail-limit clamping")

    paddle_state_path = Path(args.paddle_state) if args.paddle_state else None
    if paddle_state_path:
        print(f"paddle bus persisting to {paddle_state_path}")
    projection_align_path = Path(args.projection_align) if args.projection_align else None
    if projection_align_path:
        print(f"projection alignment file: {projection_align_path}")

    app = create_app(
        devices=registry,
        objects=objects,
        objects_path=objects_path,
        devices_path=Path(args.devices),
        projector_size=projector_size,
        stepper_calibration=stepper_calibration,
        paddle_state_path=paddle_state_path,
        projection_align_path=projection_align_path,
        scene_settings=scene_settings,
        object_scene=object_scene,
        # Prewarm only for real devices — in --mock there's nothing to warm up.
        run_prewarm=not args.mock,
    )
    logging.getLogger("uvicorn.access").addFilter(_QuietPollAccessFilter())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
