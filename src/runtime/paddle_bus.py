"""In-memory paddle-state hub.

The runtime page POSTs the tracked polygon's current x position at ~60Hz to
/api/paddle/push, which calls PaddleBus.publish. External consumers either:
  - GET /api/paddle for a one-shot read of the latest state, or
  - WebSocket /ws/paddle to receive every update as it lands.

Subscribers that fall behind get dropped on a 50ms send timeout — paddle data
is current-state, not a log. Losing a frame is fine; blocking the publisher is not.

Ported verbatim from brick_breaker_projection/src/brick_breaker/paddle_bus.py
so the existing stepper_paddle/host/bridge.py works against this server with
zero changes (it subscribes to /ws/paddle and reads payload['x_px']).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PaddleState:
    x_frac: float = 0.5          # 0.0–1.0, paddle center as fraction of viewport width
    x_px: float = 0.0            # paddle center in design-space pixels
    width_px: float = 0.0        # paddle width in design-space pixels
    vx: float = 0.0              # px/s estimate over last ~100ms
    t_ms: int = 0                # ms timestamp from the browser's performance.now() (monotonic-ish)
    paused: bool = False         # True when no polygon is currently bound to the mouse

    def to_json(self) -> dict:
        return asdict(self)


class PaddleBus:
    def __init__(self, persist_path: Path | None = None) -> None:
        self._state = PaddleState()
        self._subs: set = set()
        self._lock = asyncio.Lock()
        # `_published` distinguishes "real data exists" from "default state".
        # Without it, new subscribers (e.g. the stepper bridge starting up
        # after the server) would receive the default `x_px=0.0` snapshot and
        # interpret it as a legitimate target — driving the carriage to the
        # leftmost rail limit before the user has done anything. Flipped
        # whenever publish() runs OR bootstrap_from_disk() loads real data.
        self._published = False
        self._persist_path = persist_path

    @property
    def has_published(self) -> bool:
        return self._published

    def bootstrap_from_disk(self) -> bool:
        """Seed state from `persist_path` if it exists and parses. Returns
        True if real data was loaded (callers can log it). Sets _published
        so the next subscriber sees the restored state on connect."""
        if not self._persist_path or not self._persist_path.exists():
            return False
        try:
            data = json.loads(self._persist_path.read_text())
        except (OSError, ValueError):
            return False
        # Only the fields PaddleState defines; ignore anything extra so a
        # forward-compatible file (newer server wrote extra keys) still loads.
        keep = {k: v for k, v in data.items() if k in PaddleState.__dataclass_fields__}
        try:
            self._state = PaddleState(**keep)
        except TypeError:
            return False
        self._published = True
        return True

    def _write_atomic(self) -> None:
        """Persist current state. Atomic via tempfile + os.replace so a crash
        mid-write doesn't leave a half-JSON file the next start can't parse.
        Failures are swallowed — losing one frame of persistence is fine, but
        breaking the publish hot path isn't."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._persist_path.parent),
                prefix=".paddle_state.", suffix=".json.tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._state.to_json(), f)
                os.replace(tmp, self._persist_path)
            except Exception:
                try: os.unlink(tmp)
                except FileNotFoundError: pass
                raise
        except OSError:
            pass  # disk full, permission denied, etc. — don't break publish

    def snapshot(self) -> PaddleState:
        return PaddleState(**self._state.to_json())

    async def publish(self, state: PaddleState) -> None:
        self._state = state
        self._published = True
        self._write_atomic()
        async with self._lock:
            subs = list(self._subs)
        dead = []
        for ws in subs:
            try:
                await asyncio.wait_for(ws.send_json(state.to_json()), timeout=0.05)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subs.discard(ws)

    async def add(self, ws) -> None:
        async with self._lock:
            self._subs.add(ws)
        # Only send the snapshot if we actually have real data — otherwise
        # the subscriber would treat `x_px=0` as "drive to step coordinate
        # equivalent to x=0px", which clamps to min_steps and drives the
        # carriage to the left limit. Subscribers that connect with no data
        # available just wait for the first real publish.
        if self._published:
            try:
                await ws.send_json(self._state.to_json())
            except Exception:
                pass

    async def remove(self, ws) -> None:
        async with self._lock:
            self._subs.discard(ws)
