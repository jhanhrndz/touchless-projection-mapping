"""Tests for WifiSwitchDevice — universal WiFi switch driver.

Uses httpx.MockTransport so no physical network or switch is touched.
Tests cover:
- http_toggle mode
- tasmota protocol generation
- shelly protocol generation
- http_stateful on/off alternation
- menu actions (turn_on, turn_off, toggle)
- DeviceRegistry integration
- close cleanup
"""

import httpx
import pytest

from runtime.devices.registry import DeviceRegistry
from runtime.devices.wifi_switch_device import WifiSwitchDevice


class _StubTransport(httpx.MockTransport):
    """Captures every request and returns 200 by default."""

    def __init__(self, status_code: int = 200) -> None:
        self.calls: list[httpx.Request] = []
        self._status = status_code

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            return httpx.Response(self._status, json={"POWER": "TOGGLE", "state": "ok"})

        super().__init__(handler)


def _make_wifi_switch(**kwargs) -> tuple[WifiSwitchDevice, _StubTransport]:
    transport = _StubTransport(status_code=kwargs.pop("status_code", 200))
    dev = WifiSwitchDevice(
        object_id=kwargs.pop("object_id", "lamp_switch"),
        label=kwargs.pop("label", "Lámpara"),
        **kwargs,
    )
    dev._client = httpx.AsyncClient(transport=transport)
    return dev, transport


@pytest.mark.asyncio
async def test_wifi_switch_http_toggle():
    dev, transport = _make_wifi_switch(mode="http_toggle", url="http://192.168.1.150/toggle", method="GET")
    await dev.on_click("left")

    assert len(transport.calls) == 1
    assert str(transport.calls[0].url) == "http://192.168.1.150/toggle"
    assert transport.calls[0].method == "GET"
    assert dev.is_on is True


@pytest.mark.asyncio
async def test_wifi_switch_tasmota_mode():
    dev, transport = _make_wifi_switch(mode="tasmota", host="192.168.1.200")
    await dev.on_click("left")

    assert len(transport.calls) == 1
    assert str(transport.calls[0].url) == "http://192.168.1.200/cm?cmnd=Power%20TOGGLE"
    assert transport.calls[0].method == "GET"


@pytest.mark.asyncio
async def test_wifi_switch_shelly_mode():
    dev, transport = _make_wifi_switch(mode="shelly", host="192.168.1.201", relay_index=0)
    await dev.on_click("left")

    assert len(transport.calls) == 1
    assert str(transport.calls[0].url) == "http://192.168.1.201/relay/0?turn=toggle"
    assert transport.calls[0].method == "GET"


@pytest.mark.asyncio
async def test_wifi_switch_stateful_mode():
    dev, transport = _make_wifi_switch(
        mode="http_stateful",
        on_url="http://192.168.1.150/on",
        off_url="http://192.168.1.150/off",
        method="POST",
    )
    # First toggle -> turns ON
    await dev.toggle()
    assert len(transport.calls) == 1
    assert str(transport.calls[0].url) == "http://192.168.1.150/on"
    assert dev.is_on is True

    # Second toggle -> turns OFF
    await dev.toggle()
    assert len(transport.calls) == 2
    assert str(transport.calls[1].url) == "http://192.168.1.150/off"
    assert dev.is_on is False


@pytest.mark.asyncio
async def test_wifi_switch_menu_actions():
    dev, transport = _make_wifi_switch(
        mode="http_stateful",
        on_url="http://192.168.1.150/on",
        off_url="http://192.168.1.150/off",
    )
    items = await dev.on_context_menu()
    assert len(items) == 3
    action_ids = [item.action_id for item in items]
    assert "toggle" in action_ids
    assert "turn_on" in action_ids
    assert "turn_off" in action_ids

    await dev.invoke_menu_action("turn_on")
    assert str(transport.calls[-1].url) == "http://192.168.1.150/on"
    assert dev.is_on is True

    await dev.invoke_menu_action("turn_off")
    assert str(transport.calls[-1].url) == "http://192.168.1.150/off"
    assert dev.is_on is False


@pytest.mark.asyncio
async def test_registry_builds_wifi_switch():
    registry = DeviceRegistry()
    spec = {
        "label": "Lámpara Test",
        "config": {
            "mode": "http_toggle",
            "url": "http://192.168.1.150/toggle",
        },
    }
    dev = registry.build("lamp_switch", "wifi_switch", spec)
    assert isinstance(dev, WifiSwitchDevice)
    assert dev.object_id == "lamp_switch"
    assert dev.label == "Lámpara Test"


@pytest.mark.asyncio
async def test_wifi_switch_close_shuts_down_client():
    dev, _ = _make_wifi_switch(url="http://test/toggle")
    assert dev._client is not None
    await dev.close()
    assert dev._client is None
