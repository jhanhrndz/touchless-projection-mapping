from __future__ import annotations

from typing import Any, Callable

from .base import Device, MenuItem, MouseButton
from .book_pusher_device import BookPusherDevice
from .http_device import HttpDevice
from .kasa_device import KasaDevice
from .mock_device import MockDevice
from .udp_device import UdpDevice

DeviceFactory = Callable[[str, dict[str, Any]], Device]


class DeviceRegistry:
    def __init__(self) -> None:
        self._devices: dict[str, Device] = {}
        self._factories: dict[str, DeviceFactory] = {}
        self.register_factory("mock", _build_mock)
        self.register_factory("kasa", _build_kasa)
        self.register_factory("http", _build_http)
        self.register_factory("udp", _build_udp)
        self.register_factory("book_pusher", _build_book_pusher)

    def register_factory(self, type_name: str, factory: DeviceFactory) -> None:
        self._factories[type_name] = factory

    def add(self, device: Device) -> None:
        if device.object_id in self._devices:
            raise ValueError(f"duplicate object_id: {device.object_id!r}")
        self._devices[device.object_id] = device

    def build(self, object_id: str, type_name: str, spec: dict[str, Any]) -> Device:
        if type_name not in self._factories:
            raise ValueError(
                f"unknown device type {type_name!r} for object {object_id!r}; "
                f"known: {sorted(self._factories)}"
            )
        device = self._factories[type_name](object_id, spec)
        self.add(device)
        return device

    def get(self, object_id: str) -> Device:
        if object_id not in self._devices:
            raise KeyError(f"no device registered for object_id {object_id!r}")
        return self._devices[object_id]

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._devices

    def __iter__(self):
        return iter(self._devices.values())

    def __len__(self) -> int:
        return len(self._devices)

    async def dispatch_hover(self, object_id: str, enter: bool) -> None:
        await self.get(object_id).on_hover(enter)

    async def dispatch_click(self, object_id: str, button: MouseButton) -> None:
        await self.get(object_id).on_click(button)

    async def dispatch_context_menu(self, object_id: str) -> list[MenuItem]:
        return await self.get(object_id).on_context_menu()

    async def dispatch_menu_action(
        self, object_id: str, action_id: str, value: Any | None = None
    ) -> None:
        await self.get(object_id).invoke_menu_action(action_id, value)

    async def close_all(self) -> None:
        for device in self._devices.values():
            await device.close()


def _build_mock(object_id: str, spec: dict[str, Any]) -> Device:
    return MockDevice(object_id=object_id, label=spec.get("label", ""))


def _build_kasa(object_id: str, spec: dict[str, Any]) -> Device:
    from pathlib import Path
    cfg = spec.get("config", {})
    cache_path = cfg.get("cache_path")
    return KasaDevice(
        object_id=object_id,
        label=spec.get("label", ""),
        host_env=cfg.get("host_env", "SMART_SWITCH_HOST"),
        username_env=cfg.get("username_env", "KASA_USERNAME"),
        password_env=cfg.get("password_env", "KASA_PASSWORD"),
        mac=cfg.get("mac"),
        cache_path=Path(cache_path) if cache_path else None,
    )


def _build_http(object_id: str, spec: dict[str, Any]) -> Device:
    cfg = spec.get("config", {})
    if "base_url" not in cfg:
        raise ValueError(f"http device {object_id!r} missing required config.base_url")
    return HttpDevice(
        object_id=object_id,
        label=spec.get("label", ""),
        base_url=cfg["base_url"],
        click_path=cfg.get("click_path", "/click"),
        click_method=cfg.get("click_method", "POST"),
        timeout_s=float(cfg.get("timeout_s", 2.0)),
    )


def _build_udp(object_id: str, spec: dict[str, Any]) -> Device:
    cfg = spec.get("config", {})
    if "host" not in cfg:
        raise ValueError(f"udp device {object_id!r} missing required config.host")
    return UdpDevice(
        object_id=object_id,
        label=spec.get("label", ""),
        host=cfg["host"],
        port=int(cfg.get("port", 5006)),
        double_send_gap_s=float(cfg.get("double_send_gap_s", 0.15)),
        fill_cup_s=float(cfg.get("fill_cup_s", 0.0)),
        hostname=cfg.get("hostname"),
        probe_timeout_s=float(cfg.get("probe_timeout_s", 0.4)),
        max_on_s=float(cfg.get("max_on_s", 0.0)),
    )


def _build_book_pusher(object_id: str, spec: dict[str, Any]) -> Device:
    cfg = spec.get("config", {})
    if "host" not in cfg:
        raise ValueError(
            f"book_pusher device {object_id!r} missing required config.host"
        )
    return BookPusherDevice(
        object_id=object_id,
        label=spec.get("label", ""),
        host=cfg["host"],
        port=int(cfg.get("port", 5007)),
    )
