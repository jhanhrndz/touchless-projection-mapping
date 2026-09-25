"""Tests for the paddle telemetry bus + HTTP/WS endpoints.

These are the hooks the popcorn-slider project uses to drive the stepper
bridge. The wire format (PaddleState fields) matches brick_breaker so the
existing stepper_paddle/host/bridge.py works without changes — keep the field
list in sync with brick_breaker/src/brick_breaker/paddle_bus.py.

Negative cases: missing fields → 422, non-numeric fields → 422, out-of-range
x_frac is clamped (not rejected) so a glitched browser doesn't break the loop.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from runtime.paddle_bus import PaddleBus, PaddleState
from runtime.server import create_app


# ---------- PaddleBus unit tests ----------


def test_paddle_state_to_json_shape_matches_bridge_contract():
    """The stepper bridge reads `x_px` off each frame; the dataclass must
    serialize that key (and the other documented ones) verbatim."""
    s = PaddleState(x_frac=0.5, x_px=960.0, width_px=200.0, vx=10.0, t_ms=1234, paused=False)
    assert s.to_json() == {
        "x_frac": 0.5,
        "x_px": 960.0,
        "width_px": 200.0,
        "vx": 10.0,
        "t_ms": 1234,
        "paused": False,
    }


def test_paddle_state_defaults_are_safe():
    """A freshly-constructed PaddleState must serialize without raising and
    have x_frac centered so a consumer that reads before any publish sees
    a sensible value (not NaN, not negative)."""
    s = PaddleState()
    j = s.to_json()
    assert j["x_frac"] == 0.5
    assert j["x_px"] == 0.0
    assert j["paused"] is False


def test_paddle_bus_starts_unpublished():
    """has_published starts False so /ws/paddle correctly distinguishes
    'never had data' from 'has zero data'. The bridge reconnecting after a
    server restart must not be told 'paddle is at x_px=0' (which would drive
    the carriage to the leftmost rail limit)."""
    bus = PaddleBus()
    assert bus.has_published is False


@pytest.mark.asyncio
async def test_paddle_bus_publish_flips_has_published_flag():
    bus = PaddleBus()
    await bus.publish(PaddleState(x_px=100.0))
    assert bus.has_published is True


def test_paddle_bus_bootstrap_from_disk_round_trip(tmp_path):
    """Persistence path: write a state via publish(), construct a new bus
    pointed at the same file, bootstrap_from_disk should restore the state
    and flip has_published. This is the path that makes the bridge see the
    real last position on its first reconnect after a server restart."""
    persist = tmp_path / "paddle_state.json"
    bus1 = PaddleBus(persist_path=persist)
    import asyncio
    asyncio.run(bus1.publish(PaddleState(x_frac=0.7, x_px=1344.0, width_px=120.0, t_ms=42)))

    bus2 = PaddleBus(persist_path=persist)
    assert bus2.has_published is False
    assert bus2.bootstrap_from_disk() is True
    assert bus2.has_published is True
    snap = bus2.snapshot()
    assert snap.x_px == 1344.0
    assert snap.x_frac == 0.7
    assert snap.t_ms == 42


def test_paddle_bus_bootstrap_returns_false_when_file_missing(tmp_path):
    bus = PaddleBus(persist_path=tmp_path / "nope.json")
    assert bus.bootstrap_from_disk() is False
    assert bus.has_published is False


def test_paddle_bus_bootstrap_returns_false_on_corrupt_file(tmp_path):
    """A garbage paddle_state.json must not crash startup — bus stays
    unpublished and the system runs as if no persistence existed."""
    p = tmp_path / "paddle_state.json"
    p.write_text("{not json")
    bus = PaddleBus(persist_path=p)
    assert bus.bootstrap_from_disk() is False
    assert bus.has_published is False


def test_paddle_bus_snapshot_returns_independent_copy():
    """Mutating a returned snapshot must not mutate the bus's internal state —
    otherwise a buggy consumer could clobber the canonical value."""
    bus = PaddleBus()
    snap = bus.snapshot()
    snap.x_px = 9999.0
    # A re-snapshot still reads the original.
    assert bus.snapshot().x_px == 0.0


@pytest.mark.asyncio
async def test_paddle_bus_publish_updates_snapshot():
    bus = PaddleBus()
    await bus.publish(PaddleState(x_frac=0.75, x_px=1440.0, width_px=200.0))
    snap = bus.snapshot()
    assert snap.x_frac == 0.75
    assert snap.x_px == 1440.0
    assert snap.width_px == 200.0


# ---------- HTTP endpoint tests ----------


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_get_paddle_returns_default_state(client: TestClient):
    res = client.get("/api/paddle")
    assert res.status_code == 200
    body = res.json()
    # Same field set as brick_breaker — the bridge keys off `x_px`. Plus
    # `published` (added so the runtime page can tell "real data exists"
    # from "default state"); bridge ignores it.
    assert set(body.keys()) == {"x_frac", "x_px", "width_px", "vx", "t_ms", "paused", "published"}
    assert body["x_frac"] == 0.5
    assert body["paused"] is False


def test_post_paddle_push_round_trips_through_get(client: TestClient):
    payload = {
        "x_frac": 0.42,
        "x_px": 800.0,
        "width_px": 160.0,
        "vx": 25.0,
        "t_ms": 1700,
    }
    res = client.post("/api/paddle/push", json=payload)
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    snap = client.get("/api/paddle").json()
    assert snap["x_frac"] == 0.42
    assert snap["x_px"] == 800.0
    assert snap["vx"] == 25.0
    assert snap["t_ms"] == 1700


def test_post_paddle_push_clamps_x_frac_above_one(client: TestClient):
    """Defense in depth — a glitched browser must not drive the downstream
    bridge with x_frac > 1. The push succeeds but the value is clamped."""
    res = client.post("/api/paddle/push", json={
        "x_frac": 2.5, "x_px": 9999.0, "width_px": 100.0,
    })
    assert res.status_code == 200
    assert client.get("/api/paddle").json()["x_frac"] == 1.0


def test_post_paddle_push_clamps_x_frac_below_zero(client: TestClient):
    res = client.post("/api/paddle/push", json={
        "x_frac": -0.3, "x_px": -10.0, "width_px": 100.0,
    })
    assert res.status_code == 200
    assert client.get("/api/paddle").json()["x_frac"] == 0.0


@pytest.mark.parametrize("bad", [
    {},                                                # missing all required
    {"x_frac": 0.5, "x_px": 100.0},                    # missing width_px
    {"x_frac": "half", "x_px": 100.0, "width_px": 1},  # x_frac wrong type
    {"x_frac": 0.5, "x_px": "wat", "width_px": 1},     # x_px wrong type
])
def test_post_paddle_push_validation(client: TestClient, bad):
    res = client.post("/api/paddle/push", json=bad)
    assert res.status_code == 422, res.text


# ---------- /api/paddle/config (stepper-derived clamp limits) ----------


def test_paddle_config_returns_nulls_when_no_calibration(client: TestClient):
    """No --stepper-calibration → endpoint returns nulls so the runtime
    page falls back to "no clamping" (its pre-clamp behavior)."""
    body = client.get("/api/paddle/config").json()
    assert body == {"x_px_min": None, "x_px_max": None, "zero_px": None}


def test_paddle_config_resolves_limits_from_calibration():
    cal = {"zero_px": 960.0, "steps_per_px": 5.0, "min_steps": -1000, "max_steps": 2000}
    app = create_app(stepper_calibration=cal)
    body = TestClient(app).get("/api/paddle/config").json()
    # zero_px + min_steps / scale = 960 - 200 = 760
    # zero_px + max_steps / scale = 960 + 400 = 1360
    assert body == {"x_px_min": 760.0, "x_px_max": 1360.0, "zero_px": 960.0}


def test_paddle_config_handles_flipped_rail_negative_scale():
    """Negative steps_per_px (flipped rail) — min/max in px space invert
    relative to the step-space limits. The endpoint must still return
    x_px_min < x_px_max so the runtime clamp logic is sign-agnostic."""
    cal = {"zero_px": 960.0, "steps_per_px": -5.0, "min_steps": -1000, "max_steps": 2000}
    body = TestClient(app := create_app(stepper_calibration=cal)).get("/api/paddle/config").json()
    # zero + min/scale = 960 - (-200) = 1160; zero + max/scale = 960 - 400 = 560
    # min should be the smaller of the two:
    assert body["x_px_min"] == 560.0
    assert body["x_px_max"] == 1160.0


@pytest.mark.parametrize("bad_cal", [
    {},                                                                # all missing
    {"zero_px": 0, "steps_per_px": 1, "min_steps": 0},                 # max_steps missing
    {"zero_px": 0, "steps_per_px": "fast", "min_steps": 0, "max_steps": 1},
])
def test_paddle_config_returns_nulls_on_malformed_calibration(bad_cal):
    """A garbage calibration shouldn't crash boot or 500 the route — the
    runtime just runs without the clamp."""
    body = TestClient(create_app(stepper_calibration=bad_cal)).get("/api/paddle/config").json()
    assert body["x_px_min"] is None
    assert body["x_px_max"] is None


def test_paddle_config_zero_scale_returns_nulls_for_limits():
    """steps_per_px of 0 would divide by zero — must degrade safely."""
    cal = {"zero_px": 100.0, "steps_per_px": 0, "min_steps": -50, "max_steps": 50}
    body = TestClient(create_app(stepper_calibration=cal)).get("/api/paddle/config").json()
    assert body["x_px_min"] is None
    assert body["x_px_max"] is None
    assert body["zero_px"] == 100.0  # still echoes the valid field


def test_ws_paddle_emits_snapshot_on_connect_and_pushes(client: TestClient):
    """After at least one publish, the WS handler sends the current snapshot
    on connect (so a subscriber that joins after the last push isn't stuck
    waiting), and then fans out subsequent publishes."""
    # Prime the bus so has_published flips and the snapshot is sent on connect.
    client.post("/api/paddle/push", json={
        "x_frac": 0.33, "x_px": 633.0, "width_px": 120.0,
    })
    with client.websocket_connect("/ws/paddle") as ws:
        first = ws.receive_json()
        assert first["x_frac"] == 0.33
        assert first["x_px"] == 633.0

        client.post("/api/paddle/push", json={
            "x_frac": 0.77, "x_px": 1478.0, "width_px": 120.0,
        })
        second = ws.receive_json()
        assert second["x_frac"] == 0.77
        assert second["x_px"] == 1478.0


@pytest.mark.asyncio
async def test_paddle_bus_add_skips_snapshot_when_unpublished():
    """Critical: with no prior publish, PaddleBus.add must NOT send a
    snapshot. Without this gate, the stepper bridge reconnecting after a
    server restart would receive x_px=0 and drive the carriage to the
    leftmost rail limit. We exercise the bus directly with a fake WS that
    records its send_json calls — avoids using TestClient.websocket_connect,
    whose receive_json blocks the worker thread (asyncio.wait_for can't
    cancel a blocked sync call → deadlock)."""
    class FakeWS:
        def __init__(self):
            self.sent = []
        async def send_json(self, payload):
            self.sent.append(payload)

    bus = PaddleBus()
    ws = FakeWS()
    await bus.add(ws)
    assert ws.sent == [], "unpublished bus must not snapshot on connect"

    # After a publish, has_published flips and subsequent adds DO get the snapshot.
    await bus.publish(PaddleState(x_frac=0.42, x_px=800.0, width_px=160.0))
    ws2 = FakeWS()
    await bus.add(ws2)
    assert len(ws2.sent) == 1
    assert ws2.sent[0]["x_px"] == 800.0


def test_api_paddle_includes_published_flag(client: TestClient):
    body = client.get("/api/paddle").json()
    assert body["published"] is False
    client.post("/api/paddle/push", json={"x_frac": 0.5, "x_px": 500.0, "width_px": 100.0})
    body = client.get("/api/paddle").json()
    assert body["published"] is True
    assert body["x_px"] == 500.0


def test_server_bootstraps_paddle_state_from_disk(tmp_path):
    """End-to-end: a server started with a populated paddle_state.json
    presents has_published=true on /api/paddle and sends a snapshot to a
    fresh WS subscriber — so the bridge reconnecting after a restart
    immediately sees the carriage's last known X."""
    import json as _json
    persist = tmp_path / "paddle_state.json"
    persist.write_text(_json.dumps({
        "x_frac": 0.6, "x_px": 1152.0, "width_px": 200.0,
        "vx": 0.0, "t_ms": 123, "paused": False,
    }))
    app = create_app(paddle_state_path=persist)
    c = TestClient(app)
    body = c.get("/api/paddle").json()
    assert body["published"] is True
    assert body["x_px"] == 1152.0
    with c.websocket_connect("/ws/paddle") as ws:
        # Snapshot should arrive immediately (bus was bootstrapped).
        snap = ws.receive_json()
        assert snap["x_px"] == 1152.0


# ---------- /api/projection_align (visual alignment overlay) ----------


def test_projection_align_defaults_to_identity(client: TestClient):
    """No --projection-align passed → identity (no draw shift)."""
    body = client.get("/api/projection_align").json()
    assert body == {"x_offset": 0.0, "x_scale": 1.0}


def test_projection_align_draft_round_trip(client: TestClient):
    """/align pushes drafts; runtime / polls drafts. Round-trip the payload
    and confirm the version monotonically increments — clients use it to
    detect new changes without re-applying their own writes."""
    r = client.post("/api/projection_align/draft", json={"x_offset": 15.0, "x_scale": 1.05})
    assert r.status_code == 200
    v1 = r.json()["version"]

    body = client.get("/api/projection_align/draft").json()
    assert body["x_offset"] == 15.0
    assert body["x_scale"] == 1.05
    assert body["version"] == v1

    r = client.post("/api/projection_align/draft", json={"x_offset": -8.5, "x_scale": 0.97})
    assert r.json()["version"] > v1


@pytest.mark.parametrize("bad,reason", [
    ({"x_offset": 99999, "x_scale": 1},   "x_offset out of range"),
    ({"x_offset": -99999, "x_scale": 1},  "x_offset out of range"),
    ({"x_offset": 0, "x_scale": 100},     "x_scale out of range"),
    ({"x_offset": 0, "x_scale": 0.001},   "x_scale out of range"),
    ({"x_offset": "ten", "x_scale": 1},   "must be numbers"),
])
def test_projection_align_draft_validation(client: TestClient, bad, reason):
    r = client.post("/api/projection_align/draft", json=bad)
    assert r.status_code == 400
    assert reason in r.json()["detail"]


def test_projection_align_save_persists_and_updates_saved(tmp_path):
    """Save promotes the draft into the saved value + writes atomically."""
    align_path = tmp_path / "projection_align.json"
    app = create_app(projection_align_path=align_path)
    c = TestClient(app)
    c.post("/api/projection_align/draft", json={"x_offset": 22.5, "x_scale": 1.123})
    r = c.post("/api/projection_align/save")
    assert r.status_code == 200
    body = r.json()
    assert body["x_offset"] == 22.5 and body["x_scale"] == 1.123 and body["persisted"] is True
    # GET reflects the saved value now (not just the draft)
    saved = c.get("/api/projection_align").json()
    assert saved == {"x_offset": 22.5, "x_scale": 1.123}
    # On-disk JSON matches
    import json as _json
    on_disk = _json.loads(align_path.read_text())
    assert on_disk == {"x_offset": 22.5, "x_scale": 1.123}


def test_projection_align_loads_from_disk_on_startup(tmp_path):
    """Persistence loop — a server started against a populated file
    presents that value on /api/projection_align."""
    align_path = tmp_path / "projection_align.json"
    import json as _json
    align_path.write_text(_json.dumps({"x_offset": -12.5, "x_scale": 1.04}))
    app = create_app(projection_align_path=align_path)
    body = TestClient(app).get("/api/projection_align").json()
    assert body == {"x_offset": -12.5, "x_scale": 1.04}


def test_projection_align_reset_returns_to_identity(client: TestClient):
    client.post("/api/projection_align/draft", json={"x_offset": 50.0, "x_scale": 1.1})
    r = client.post("/api/projection_align/reset")
    assert r.status_code == 200
    body = client.get("/api/projection_align/draft").json()
    assert body["x_offset"] == 0.0 and body["x_scale"] == 1.0


def test_align_page_served(client: TestClient):
    r = client.get("/align")
    assert r.status_code == 200
    assert "Projection alignment" in r.text


def test_publish_writes_to_paddle_state_file(tmp_path):
    """Each push triggers an atomic write so the next server start can
    bootstrap from disk. This is the persistence link that makes browser
    reloads + bridge reconnects see the same X."""
    persist = tmp_path / "paddle_state.json"
    app = create_app(paddle_state_path=persist)
    c = TestClient(app)
    c.post("/api/paddle/push", json={"x_frac": 0.7, "x_px": 1344.0, "width_px": 100.0})
    import json as _json
    on_disk = _json.loads(persist.read_text())
    assert on_disk["x_px"] == 1344.0
    assert on_disk["x_frac"] == 0.7
