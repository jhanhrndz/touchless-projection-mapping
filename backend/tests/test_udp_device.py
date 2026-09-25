"""Tests for UdpDevice — binds a real UDP listener on localhost so we
verify the actual byte payloads and the double-send pattern.

Negative cases per project convention: missing host config, unknown menu
action, malformed payload not on wire (state must alternate cleanly).
"""

import asyncio
import socket
import time

import pytest

from runtime.devices.registry import DeviceRegistry
from runtime.devices.udp_device import UdpDevice


@pytest.fixture
def listener():
    """A UDP listener bound to 127.0.0.1 on an OS-assigned port. Yields
    (port, recv_all) where recv_all() drains every datagram currently queued."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    port = sock.getsockname()[1]

    def recv_all() -> list[bytes]:
        msgs: list[bytes] = []
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                msgs.append(data)
            except BlockingIOError:
                break
        return msgs

    yield port, recv_all
    sock.close()


async def _drain_at_least(recv_all, expected: int, timeout_s: float = 0.5) -> list[bytes]:
    """Poll recv_all() until at least `expected` datagrams have arrived or
    timeout elapses. Replaces a fixed `await asyncio.sleep(0.02)` which was
    flaky under load — the OS sometimes hadn't delivered the packets yet."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    msgs: list[bytes] = []
    while len(msgs) < expected and asyncio.get_event_loop().time() < deadline:
        msgs.extend(recv_all())
        if len(msgs) >= expected:
            break
        await asyncio.sleep(0.005)
    # One last drain in case more packets arrived between the check and now.
    msgs.extend(recv_all())
    return msgs


def _make_device(port: int, gap: float = 0.0) -> UdpDevice:
    return UdpDevice(
        object_id="water_pump",
        label="Water pump",
        host="127.0.0.1",
        port=port,
        double_send_gap_s=gap,
    )


@pytest.mark.asyncio
async def test_first_click_sends_on_twice(listener):
    port, recv_all = listener
    dev = _make_device(port)
    await dev.on_click("left")
    await asyncio.sleep(0.02)
    msgs = recv_all()
    assert msgs == [b"ON\n", b"ON\n"]
    await dev.close()


@pytest.mark.asyncio
async def test_second_click_sends_off(listener):
    port, recv_all = listener
    dev = _make_device(port)
    await dev.on_click("left")
    await dev.on_click("left")
    await asyncio.sleep(0.02)
    msgs = recv_all()
    assert msgs == [b"ON\n", b"ON\n", b"OFF\n", b"OFF\n"]
    await dev.close()


@pytest.mark.asyncio
async def test_state_alternates_across_many_clicks(listener):
    port, recv_all = listener
    dev = _make_device(port)
    for _ in range(4):
        await dev.on_click("left")
    await asyncio.sleep(0.02)
    msgs = recv_all()
    expected = [b"ON\n", b"ON\n", b"OFF\n", b"OFF\n"] * 2
    assert msgs == expected
    await dev.close()


@pytest.mark.asyncio
async def test_double_send_actually_waits_the_gap():
    """The whole point of the gap is wake-on-demand reliability. Verify
    the second packet really lands ~gap seconds later, not immediately."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    port = sock.getsockname()[1]

    dev = _make_device(port, gap=0.15)
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    click = asyncio.create_task(dev.on_click("left"))

    await loop.sock_recvfrom(sock, 1024)
    t1 = time.perf_counter() - start
    await loop.sock_recvfrom(sock, 1024)
    t2 = time.perf_counter() - start

    await click
    sock.close()
    await dev.close()

    assert t1 < 0.05, f"first packet should arrive immediately, got {t1*1000:.0f}ms"
    assert 0.13 < t2 < 0.30, f"second packet should arrive ~150ms later, got {t2*1000:.0f}ms"


@pytest.mark.asyncio
async def test_menu_on_forces_state_on(listener):
    port, recv_all = listener
    dev = _make_device(port)
    await dev.invoke_menu_action("on")
    await dev.invoke_menu_action("on")  # idempotent — still ON
    await asyncio.sleep(0.02)
    msgs = recv_all()
    assert msgs == [b"ON\n", b"ON\n", b"ON\n", b"ON\n"]
    # Subsequent click should send OFF (state was forced to True).
    await dev.on_click("left")
    await asyncio.sleep(0.02)
    assert recv_all() == [b"OFF\n", b"OFF\n"]
    await dev.close()


@pytest.mark.asyncio
async def test_menu_off_forces_state_off(listener):
    port, recv_all = listener
    dev = _make_device(port)
    await dev.invoke_menu_action("on")
    # Wait for the ON×2 to fully arrive before draining — otherwise a slow
    # OS handoff can leak ON packets into the OFF assertion.
    await _drain_at_least(recv_all, 2)
    await dev.invoke_menu_action("off")
    msgs = await _drain_at_least(recv_all, 2)
    assert msgs == [b"OFF\n", b"OFF\n"]
    await dev.close()


@pytest.mark.asyncio
async def test_unknown_menu_action_raises(listener):
    port, _ = listener
    dev = _make_device(port)
    with pytest.raises(NotImplementedError, match="unknown action"):
        await dev.invoke_menu_action("explode")
    await dev.close()


@pytest.mark.asyncio
async def test_close_releases_socket(listener):
    port, _ = listener
    dev = _make_device(port)
    await dev.on_click("left")
    assert dev._sock is not None
    await dev.close()
    assert dev._sock is None
    # Reopens lazily on next send.
    await dev.on_click("left")
    assert dev._sock is not None
    await dev.close()


def test_factory_builds_udp_device_from_yaml_spec():
    registry = DeviceRegistry()
    spec = {
        "label": "Water pump",
        "config": {"host": "10.0.0.51", "port": 5006},
    }
    dev = registry.build("water_pump", "udp", spec)
    assert isinstance(dev, UdpDevice)
    assert dev.label == "Water pump"
    assert dev._host == "10.0.0.51"
    assert dev._port == 5006
    assert dev._gap == 0.15  # default


def test_factory_rejects_missing_host():
    registry = DeviceRegistry()
    with pytest.raises(ValueError, match="missing required config.host"):
        registry.build("x", "udp", {"label": "x", "config": {}})


def test_factory_uses_defaults():
    registry = DeviceRegistry()
    spec = {"config": {"host": "127.0.0.1"}}
    dev = registry.build("x", "udp", spec)
    assert isinstance(dev, UdpDevice)
    assert dev._port == 5006
    assert dev._gap == 0.15


def test_factory_accepts_overridden_gap():
    registry = DeviceRegistry()
    spec = {"config": {"host": "127.0.0.1", "double_send_gap_s": 0.3}}
    dev = registry.build("x", "udp", spec)
    assert isinstance(dev, UdpDevice)
    assert dev._gap == 0.3


# ---------- fill_cup action ----------

def _make_fill_device(port: int, fill_s: float = 0.05) -> UdpDevice:
    return UdpDevice(
        object_id="water_pump",
        label="Water pump",
        host="127.0.0.1",
        port=port,
        double_send_gap_s=0.0,
        fill_cup_s=fill_s,
    )


@pytest.mark.asyncio
async def test_menu_omits_fill_when_disabled(listener):
    port, _ = listener
    dev = _make_device(port)  # default fill_cup_s=0
    items = await dev.on_context_menu()
    assert [i.action_id for i in items] == ["on", "off"]
    await dev.close()


@pytest.mark.asyncio
async def test_menu_includes_fill_when_enabled(listener):
    port, _ = listener
    dev = _make_fill_device(port, fill_s=24.0)
    items = await dev.on_context_menu()
    assert [i.action_id for i in items] == ["on", "off", "fill_cup"]
    assert "24" in items[2].label
    await dev.close()


@pytest.mark.asyncio
async def test_fill_cup_when_disabled_raises(listener):
    port, _ = listener
    dev = _make_device(port)  # fill_cup_s=0
    with pytest.raises(NotImplementedError, match="not configured"):
        await dev.invoke_menu_action("fill_cup")
    await dev.close()


@pytest.mark.asyncio
async def test_fill_cup_fires_on_then_off(listener):
    """Happy path: ON×2 immediately, then OFF×2 after the fill duration."""
    port, recv_all = listener
    dev = _make_fill_device(port, fill_s=0.1)
    await dev.invoke_menu_action("fill_cup")
    await asyncio.sleep(0.02)
    early = recv_all()
    assert early == [b"ON\n", b"ON\n"], f"expected ON×2 immediately, got {early}"
    # Now wait past the fill duration and confirm OFF lands.
    await asyncio.sleep(0.18)
    assert recv_all() == [b"OFF\n", b"OFF\n"]
    await dev.close()


@pytest.mark.asyncio
async def test_fill_cup_cancellation_still_fires_off(listener):
    """Hardware safety: if the fill is cancelled mid-sleep, OFF MUST still fire.
    Otherwise the pump runs until something else stops it — water everywhere."""
    port, recv_all = listener
    dev = _make_fill_device(port, fill_s=10.0)  # long fill we'll cancel
    await dev.invoke_menu_action("fill_cup")
    await asyncio.sleep(0.02)
    assert recv_all() == [b"ON\n", b"ON\n"]

    # Cancel the running fill task directly and wait for it to wind down.
    assert dev._fill_task is not None
    dev._fill_task.cancel()
    try:
        await dev._fill_task
    except (asyncio.CancelledError, Exception):
        pass

    await asyncio.sleep(0.02)
    assert recv_all() == [b"OFF\n", b"OFF\n"], "OFF must fire even when fill is cancelled"
    await dev.close()


@pytest.mark.asyncio
async def test_close_during_fill_fires_off(listener):
    """close() while a fill is running must also flush an OFF."""
    port, recv_all = listener
    dev = _make_fill_device(port, fill_s=10.0)
    await dev.invoke_menu_action("fill_cup")
    await asyncio.sleep(0.02)
    recv_all()  # drain the ON×2

    await dev.close()
    await asyncio.sleep(0.02)
    # close cancels the fill task; the task's finally block fires OFF×2.
    msgs = recv_all()
    assert msgs == [b"OFF\n", b"OFF\n"], f"close should flush OFF, got {msgs}"


@pytest.mark.asyncio
async def test_second_fill_cancels_first(listener):
    port, recv_all = listener
    dev = _make_fill_device(port, fill_s=10.0)

    await dev.invoke_menu_action("fill_cup")
    await asyncio.sleep(0.02)
    assert recv_all() == [b"ON\n", b"ON\n"]

    # Second fill: cancel the first, which fires OFF×2, then start fresh ON×2.
    await dev.invoke_menu_action("fill_cup")
    await asyncio.sleep(0.05)
    msgs = recv_all()
    assert msgs == [b"OFF\n", b"OFF\n", b"ON\n", b"ON\n"], (
        f"second fill should cancel the first (OFF) before starting fresh (ON), got {msgs}"
    )
    await dev.close()


def test_factory_reads_fill_cup_s():
    registry = DeviceRegistry()
    spec = {"config": {"host": "127.0.0.1", "fill_cup_s": 24.0}}
    dev = registry.build("x", "udp", spec)
    assert isinstance(dev, UdpDevice)
    assert dev._fill_s == 24.0


def test_factory_fill_cup_s_defaults_to_zero():
    registry = DeviceRegistry()
    dev = registry.build("y", "udp", {"config": {"host": "127.0.0.1"}})
    assert isinstance(dev, UdpDevice)
    assert dev._fill_s == 0.0
