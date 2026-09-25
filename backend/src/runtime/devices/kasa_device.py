from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .base import Device, MenuItem, MouseButton


def _normalize_mac(mac: str) -> str:
    """Uppercase, strip separators. 'aa:bb:cc:DD:ee:ff' → 'AABBCCDDEEFF'."""
    return mac.upper().replace(":", "").replace("-", "").replace(".", "")


class KasaDevice(Device):
    """Any TP-Link Kasa-protocol device with an on/off state.

    The Smart Lamp demo drives a wall switch; a smart plug or a bulb works
    identically, because the only device API touched here is
    update / is_on / turn_on / turn_off.

    Click toggles on/off. Connection is lazy — first click discovers the device.
    Credentials and host are read from environment variables (so they don't
    live in YAML or git).

    If `mac` is configured, a stale/wrong IP triggers a broadcast discovery to
    find the switch by MAC and update a small cache file. Lets DHCP leases drift
    without manually editing .env / devices.yaml.
    """

    def __init__(
        self,
        object_id: str,
        label: str = "",
        host_env: str = "SMART_SWITCH_HOST",
        username_env: str = "KASA_USERNAME",
        password_env: str = "KASA_PASSWORD",
        mac: str | None = None,
        cache_path: Path | None = None,
        discover_timeout_s: float = 5.0,
    ) -> None:
        super().__init__(object_id, label)
        self._host_env = host_env
        self._username_env = username_env
        self._password_env = password_env
        self._mac = _normalize_mac(mac) if mac else None
        self._cache_path = cache_path or Path("data/.kasa_cache.json")
        self._discover_timeout_s = discover_timeout_s
        self._dev: Any = None
        self._lock = asyncio.Lock()

    def _read_cache(self) -> dict[str, str]:
        if not self._cache_path.exists():
            return {}
        try:
            data = json.loads(self._cache_path.read_text())
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_cache(self, cache: dict[str, str]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(cache, indent=2))

    async def _try_connect(
        self, host: str, username: str | None, password: str | None
    ) -> Any | None:
        """Unicast connect to `host`. Returns the device on success, else None.

        If a MAC is configured, the discovered device's MAC must match — protects
        against connecting to a different Kasa device that landed on the cached
        IP after a DHCP shuffle."""
        from kasa import Discover

        try:
            dev = await Discover.discover_single(
                host, username=username, password=password
            )
            await dev.update()
        except Exception as e:
            print(f"  kasa: connect to {host} failed ({type(e).__name__}: {e})")
            return None
        if self._mac:
            found_mac = _normalize_mac(getattr(dev, "mac", "") or "")
            if found_mac != self._mac:
                print(
                    f"  kasa: {host} responded but MAC mismatch "
                    f"({found_mac} != {self._mac})"
                )
                return None
        return dev

    async def _discover_by_mac(
        self, username: str | None, password: str | None
    ) -> tuple[str, Any] | None:
        """Broadcast Kasa discovery. Returns (ip, device) for the matching MAC, or None."""
        if not self._mac:
            return None
        from kasa import Discover

        print(f"  kasa: broadcasting to find MAC {self._mac} on LAN…")
        devices = await Discover.discover(
            username=username, password=password, timeout=self._discover_timeout_s
        )
        for ip, dev in devices.items():
            found_mac = _normalize_mac(getattr(dev, "mac", "") or "")
            if found_mac == self._mac:
                print(f"  kasa: found {self._mac} at {ip}")
                return ip, dev
        seen = [
            (ip, _normalize_mac(getattr(d, "mac", "") or ""))
            for ip, d in devices.items()
        ]
        print(f"  kasa: MAC {self._mac} not found. Devices discovered: {seen}")
        return None

    async def _ensure_connected(self) -> Any:
        async with self._lock:
            if self._dev is not None:
                return self._dev

            username = os.environ.get(self._username_env)
            password = os.environ.get(self._password_env)

            # Candidate IPs in priority order: cache (per-MAC), then host_env.
            candidates: list[str] = []
            cache = self._read_cache()
            cached_ip = cache.get(self._mac) if self._mac else None
            if cached_ip:
                candidates.append(cached_ip)
            host_env = os.environ.get(self._host_env)
            if host_env and host_env not in candidates:
                candidates.append(host_env)

            for ip in candidates:
                dev = await self._try_connect(ip, username, password)
                if dev is not None:
                    self._dev = dev
                    return dev

            if self._mac:
                result = await self._discover_by_mac(username, password)
                if result is not None:
                    ip, dev = result
                    cache[self._mac] = ip
                    self._write_cache(cache)
                    self._dev = dev
                    return dev
                raise RuntimeError(
                    f"Kasa device with MAC {self._mac} not found on the LAN "
                    f"(tried {candidates or 'no static IPs'}, then broadcast-discover)"
                )

            if not host_env:
                raise RuntimeError(
                    f"{self._host_env} not set — populate .env (see .env.example)"
                )
            raise RuntimeError(
                f"could not connect to {host_env}. Add a `mac:` field in "
                f"devices.yaml to enable MAC→IP fallback when DHCP shuffles "
                f"the switch's address."
            )

    async def on_click(self, button: MouseButton) -> None:
        dev = await self._ensure_connected()
        await dev.update()
        if dev.is_on:
            await dev.turn_off()
        else:
            await dev.turn_on()

    async def prewarm(self) -> str | None:
        """Boot-time connectivity check.

        Tries the cached/env IP first (fast). If that fails and a MAC is
        configured, broadcasts to find the switch on the LAN. Either way, the
        first user click won't pay any discovery cost. On failure: log and
        return the error string instead of raising — the server still boots
        so other devices/the editor remain usable.

        Must run on the same asyncio event loop that subsequent uses run on
        — kasa connection streams are loop-bound. See server.py lifespan,
        which calls this on the uvicorn loop."""
        try:
            dev = await self._ensure_connected()
        except Exception as e:
            return f"unreachable ({type(e).__name__}: {e})"
        host = getattr(dev, "host", None) or "?"
        mac = _normalize_mac(getattr(dev, "mac", "") or "")
        return f"reachable at {host}" + (f" (MAC {mac})" if mac else "")

    async def on_context_menu(self) -> list[MenuItem]:
        return [
            MenuItem(label="Turn on", action_id="on"),
            MenuItem(label="Turn off", action_id="off"),
        ]

    async def invoke_menu_action(
        self, action_id: str, value: Any | None = None
    ) -> None:
        dev = await self._ensure_connected()
        if action_id == "on":
            await dev.turn_on()
        elif action_id == "off":
            await dev.turn_off()
        else:
            raise NotImplementedError(f"unknown action {action_id!r} for KasaDevice")

    async def close(self) -> None:
        if self._dev is not None:
            try:
                await self._dev.disconnect()
            except Exception:
                pass
            self._dev = None
