"""Tests for HttpDevice — uses httpx.MockTransport so no network is touched.

Negative cases per project convention: missing base_url config, non-2xx responses,
unknown menu action, default vs custom paths/methods.
"""

import httpx
import pytest

from runtime.devices.http_device import HttpDevice
from runtime.devices.registry import DeviceRegistry


class _StubTransport(httpx.MockTransport):
    """Captures every request and returns 200 by default."""

    def __init__(self, status_code: int = 200) -> None:
        self.calls: list[httpx.Request] = []
        self._status = status_code

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            return httpx.Response(self._status, json={"ok": True})

        super().__init__(handler)


def _make_device(**kwargs) -> tuple[HttpDevice, _StubTransport]:
    """Construct an HttpDevice and pre-inject a stub transport."""
    transport = _StubTransport(status_code=kwargs.pop("status_code", 200))
    dev = HttpDevice(
        kwargs.pop("object_id", "x"),
        kwargs.pop("label", ""),
        base_url=kwargs.pop("base_url", "http://test"),
        click_path=kwargs.pop("click_path", "/click"),
        click_method=kwargs.pop("click_method", "POST"),
        timeout_s=kwargs.pop("timeout_s", 1.0),
    )
    dev._client = httpx.AsyncClient(transport=transport)
    return dev, transport


@pytest.mark.asyncio
async def test_click_sends_post_to_base_plus_path():
    dev, transport = _make_device(base_url="http://pi1:8001", click_path="/tap")
    await dev.on_click("left")
    assert len(transport.calls) == 1
    req = transport.calls[0]
    assert str(req.url) == "http://pi1:8001/tap"
    assert req.method == "POST"


@pytest.mark.asyncio
async def test_click_uses_configured_method():
    dev, transport = _make_device(click_method="GET")
    await dev.on_click("left")
    assert transport.calls[0].method == "GET"


@pytest.mark.asyncio
async def test_path_normalized_with_leading_slash():
    """Config might omit leading slash; we must add it before joining."""
    dev, transport = _make_device(click_path="tap")
    await dev.on_click("left")
    assert str(transport.calls[0].url).endswith("/tap")


@pytest.mark.asyncio
async def test_base_url_trailing_slash_stripped():
    dev, transport = _make_device(base_url="http://pi1:8001/", click_path="/tap")
    await dev.on_click("left")
    assert str(transport.calls[0].url) == "http://pi1:8001/tap"


@pytest.mark.asyncio
async def test_non_2xx_response_raises():
    dev, _ = _make_device(status_code=500)
    with pytest.raises(httpx.HTTPStatusError):
        await dev.on_click("left")


@pytest.mark.asyncio
async def test_context_menu_triggers_via_action():
    dev, transport = _make_device()
    items = await dev.on_context_menu()
    assert items[0].action_id == "trigger"
    await dev.invoke_menu_action("trigger")
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_unknown_menu_action_raises():
    dev, _ = _make_device()
    with pytest.raises(NotImplementedError, match="unknown action"):
        await dev.invoke_menu_action("explode")


@pytest.mark.asyncio
async def test_close_aclosed_client():
    dev, _ = _make_device()
    await dev.on_click("left")
    await dev.close()
    assert dev._client is None


def test_factory_builds_http_device_from_yaml_spec():
    registry = DeviceRegistry()
    spec = {
        "label": "Water pump",
        "config": {"base_url": "http://pi1:8001", "click_path": "/tap"},
    }
    dev = registry.build("water_pump", "http", spec)
    assert isinstance(dev, HttpDevice)
    assert dev.label == "Water pump"
    assert dev._base == "http://pi1:8001"
    assert dev._click_path == "/tap"


def test_factory_rejects_missing_base_url():
    registry = DeviceRegistry()
    with pytest.raises(ValueError, match="missing required config.base_url"):
        registry.build("x", "http", {"label": "x", "config": {}})


def test_factory_uses_defaults_for_method_and_path():
    registry = DeviceRegistry()
    spec = {"config": {"base_url": "http://pi1"}}
    dev = registry.build("x", "http", spec)
    assert isinstance(dev, HttpDevice)
    assert dev._click_path == "/click"
    assert dev._click_method == "POST"
