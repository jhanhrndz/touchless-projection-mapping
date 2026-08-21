"""Tests for BookPusherDevice — binds a real UDP listener on localhost and
verifies the actual byte payloads. Negative cases per project convention:
unknown menu action, missing host config, factory rejects bad type spec.
"""

import asyncio
import socket

import pytest

from runtime.devices.book_pusher_device import BookPusherDevice
from runtime.devices.registry import DeviceRegistry


@pytest.fixture
def listener():
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


def _make_device(port: int) -> BookPusherDevice:
    return BookPusherDevice(
        object_id="book_pusher",
        label="Book pusher",
        host="127.0.0.1",
        port=port,
    )


@pytest.mark.asyncio
async def test_click_sends_single_push(listener):
    port, recv_all = listener
    dev = _make_device(port)
    await dev.on_click("left")
    await asyncio.sleep(0.02)
    assert recv_all() == [b"PUSH"], "click should send a single PUSH (no double-send)"
    await dev.close()


@pytest.mark.asyncio
async def test_repeated_clicks_send_distinct_pushes(listener):
    """Each click is independent — there's no toggle state. Three clicks
    send three identical PUSH datagrams."""
    port, recv_all = listener
    dev = _make_device(port)
    for _ in range(3):
        await dev.on_click("left")
    await asyncio.sleep(0.02)
    assert recv_all() == [b"PUSH", b"PUSH", b"PUSH"]
    await dev.close()


@pytest.mark.asyncio
async def test_context_menu_items(listener):
    port, _ = listener
    dev = _make_device(port)
    items = await dev.on_context_menu()
    assert [i.action_id for i in items] == ["push", "zero", "setgoal"]
    await dev.close()


@pytest.mark.asyncio
async def test_menu_actions_send_expected_commands(listener):
    port, recv_all = listener
    dev = _make_device(port)
    await dev.invoke_menu_action("push")
    await dev.invoke_menu_action("zero")
    await dev.invoke_menu_action("setgoal")
    await asyncio.sleep(0.02)
    assert recv_all() == [b"PUSH", b"ZERO", b"SETGOAL"]
    await dev.close()


@pytest.mark.asyncio
async def test_unknown_menu_action_raises(listener):
    port, _ = listener
    dev = _make_device(port)
    with pytest.raises(NotImplementedError, match="unknown action"):
        await dev.invoke_menu_action("frobnicate")
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


def test_factory_builds_book_pusher_from_yaml_spec():
    registry = DeviceRegistry()
    spec = {
        "label": "Book pusher",
        "config": {"host": "bookshelf-pi.local", "port": 5007},
    }
    dev = registry.build("book_pusher", "book_pusher", spec)
    assert isinstance(dev, BookPusherDevice)
    assert dev.label == "Book pusher"
    assert dev._host == "bookshelf-pi.local"
    assert dev._port == 5007


def test_factory_default_port():
    registry = DeviceRegistry()
    dev = registry.build("x", "book_pusher", {"config": {"host": "127.0.0.1"}})
    assert isinstance(dev, BookPusherDevice)
    assert dev._port == 5007


def test_factory_rejects_missing_host():
    registry = DeviceRegistry()
    with pytest.raises(ValueError, match="missing required config.host"):
        registry.build("x", "book_pusher", {"label": "x", "config": {}})


def test_factory_label_falls_back_to_object_id():
    registry = DeviceRegistry()
    dev = registry.build("knocker", "book_pusher", {"config": {"host": "127.0.0.1"}})
    assert isinstance(dev, BookPusherDevice)
    assert dev.label == "knocker"
