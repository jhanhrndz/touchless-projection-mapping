from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any

from .base import Device, MenuItem, MouseButton

log = logging.getLogger(__name__)


class UdpDevice(Device):
    """Device controlled via UDP datagrams to a battery-powered ESP32 in light sleep.

    Click flips a locally-tracked on/off state and sends the new-state command
    ("ON\\n" or "OFF\\n") twice with a short gap. The double-send works around
    bimodal latency in WiFi light-sleep wake: the AP buffers unicast UDP
    between DTIM beacons; a single packet sometimes drops while the chip is
    mid-sleep, but by the time the second arrives ~150 ms later the chip is
    in an awake DTIM window. Idempotent commands means double-application is
    harmless — sending ON twice is the same as sending it once.

    Optional `fill_cup_s` enables a "Fill cup" menu action that fires ON,
    waits N seconds, then fires OFF. The sleep runs in a background task so
    the WS handler doesn't block for the duration. If a second fill starts
    while one is in progress, the old task is cancelled cleanly — but OFF
    still fires in the cancellation path so the pump never runs forever.

    Optional `hostname` (an mDNS name like "candle-1.local") enables IP-first
    resolution with a hostname fallback — same idea as the Kasa MAC→IP cache.
    macOS `.local` resolution via the system resolver costs ~5 s, so we never
    want it on the click path. Instead we trust the configured `host` IP, probe
    it once with a STATUS round-trip, and only if that fails do we resolve the
    mDNS `hostname` to pick up a drifted DHCP lease. The result is cached for
    the process lifetime, so steady-state clicks are a pure IP send (~0 ms).
    Without `hostname`, behavior is unchanged: trust `host`, never probe.

    YAML:

        - object_id: water_pump
          type: udp
          label: "Water pump"
          config:
            host: 10.0.0.51
            port: 5006
            double_send_gap_s: 0.15   # default
            fill_cup_s: 24.0          # 0 disables the Fill action; default 0
            hostname: candle-1.local  # optional mDNS fallback if host IP fails
    """

    def __init__(
        self,
        object_id: str,
        label: str = "",
        *,
        host: str,
        port: int = 5006,
        double_send_gap_s: float = 0.15,
        fill_cup_s: float = 0.0,
        hostname: str | None = None,
        probe_timeout_s: float = 0.4,
        max_on_s: float = 0.0,
    ) -> None:
        super().__init__(object_id, label)
        self._host = host
        self._port = port
        self._gap = double_send_gap_s
        self._fill_s = fill_cup_s
        self._hostname = hostname
        self._probe_timeout = probe_timeout_s
        # If the device self-disables its output after `max_on_s` (the candle's
        # 5 s hardware safety auto-off), mirror that here: after sending ON we
        # flip our tracked state back to False on a timer (no packet sent — the
        # device already turned itself off). Without this the host would think
        # the candle is still ON, so the next click would send OFF (a no-op on
        # an already-off candle) instead of re-lighting it. 0 disables.
        self._max_on_s = max_on_s
        self._auto_off_task: asyncio.Task[None] | None = None
        self._state = False
        self._sock: socket.socket | None = None
        self._fill_task: asyncio.Task[None] | None = None
        # Cached resolved IP (the actual sendto target). None until first resolve.
        self._target: str | None = None

    def _ensure_sock(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._sock

    def _probe(self, ip: str) -> bool:
        """Send STATUS to `ip` and wait briefly for an on/off reply. Blocking —
        call via asyncio.to_thread. Returns True iff the device answered."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(self._probe_timeout)
            s.sendto(b"STATUS\n", (ip, self._port))
            data, _ = s.recvfrom(64)
            return data.strip().lower() in (b"on", b"off")
        except OSError:
            return False
        finally:
            s.close()

    async def _resolve_target(self) -> str:
        """Pick the IP to send to, cached after first success.

        1. No `hostname` configured → trust `host` verbatim (legacy behavior,
           e.g. the water pump). No probe, no DNS.
        2. `hostname` set → STATUS-probe the configured `host` IP; if it answers,
           use it. Otherwise resolve the mDNS `hostname` to its current IP and
           probe that. If everything fails, fall back to `host` (blind send) so
           a transient probe miss never wedges the device permanently.
        """
        if self._target is not None:
            return self._target
        if not self._hostname:
            self._target = self._host
            return self._target

        if await asyncio.to_thread(self._probe, self._host):
            self._target = self._host
            return self._target

        log.warning(
            "udp %s: host %s did not answer STATUS; resolving %s via mDNS",
            self.object_id, self._host, self._hostname,
        )
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo, self._hostname, self._port,
                socket.AF_INET, socket.SOCK_DGRAM,
            )
            ip = infos[0][4][0]
        except OSError as e:
            log.warning("udp %s: mDNS resolve of %s failed: %s; using %s blind",
                        self.object_id, self._hostname, e, self._host)
            self._target = self._host
            return self._target

        if await asyncio.to_thread(self._probe, ip):
            log.info("udp %s: resolved %s -> %s", self.object_id, self._hostname, ip)
            self._target = ip
            return self._target

        log.warning("udp %s: resolved %s -> %s but it did not answer; using it anyway",
                    self.object_id, self._hostname, ip)
        self._target = ip
        return self._target

    async def _send_twice(self, payload: bytes) -> None:
        target = await self._resolve_target()
        sock = self._ensure_sock()
        sock.sendto(payload, (target, self._port))
        await asyncio.sleep(self._gap)
        sock.sendto(payload, (target, self._port))

    async def on_click(self, button: MouseButton) -> None:
        self._state = not self._state
        if self._state:
            await self._send_twice(b"ON\n")
            self._arm_auto_off()
        else:
            self._cancel_auto_off()
            await self._send_twice(b"OFF\n")

    def _arm_auto_off(self) -> None:
        """Schedule the local state to revert to OFF after max_on_s, matching the
        device's hardware safety auto-off. Re-arming cancels any prior timer."""
        self._cancel_auto_off()
        if self._max_on_s > 0:
            self._auto_off_task = asyncio.create_task(self._auto_off_run())

    def _cancel_auto_off(self) -> None:
        if self._auto_off_task is not None and not self._auto_off_task.done():
            self._auto_off_task.cancel()
        self._auto_off_task = None

    async def _auto_off_run(self) -> None:
        try:
            await asyncio.sleep(self._max_on_s)
            # Mirror the device — no OFF packet, it already auto-disabled.
            self._state = False
            log.info("udp %s: local state auto-reverted to OFF (max_on %.1fs)",
                     self.object_id, self._max_on_s)
        except asyncio.CancelledError:
            pass

    async def on_context_menu(self) -> list[MenuItem]:
        items = [
            MenuItem(label="Turn on", action_id="on"),
            MenuItem(label="Turn off", action_id="off"),
        ]
        if self._fill_s > 0:
            items.append(
                MenuItem(label=f"Fill cup ({self._fill_s:g}s)", action_id="fill_cup")
            )
        return items

    async def invoke_menu_action(self, action_id: str, value: Any | None = None) -> None:
        if action_id == "on":
            self._state = True
            await self._send_twice(b"ON\n")
            self._arm_auto_off()
        elif action_id == "off":
            self._state = False
            self._cancel_auto_off()
            await self._send_twice(b"OFF\n")
        elif action_id == "fill_cup":
            await self._start_fill()
        else:
            raise NotImplementedError(f"unknown action {action_id!r} for UdpDevice")

    async def _start_fill(self) -> None:
        if self._fill_s <= 0:
            raise NotImplementedError("fill_cup not configured (fill_cup_s <= 0)")
        # If a previous fill is still running, cancel it; the cancelled task
        # fires OFF in its finally block before exiting.
        if self._fill_task is not None and not self._fill_task.done():
            self._fill_task.cancel()
            try:
                await self._fill_task
            except (asyncio.CancelledError, Exception):
                pass
        self._fill_task = asyncio.create_task(self._fill_run(self._fill_s))

    async def _fill_run(self, seconds: float) -> None:
        """ON, sleep, OFF. If cancelled mid-sleep, still fire OFF — never leave the pump on."""
        self._state = True
        await self._send_twice(b"ON\n")
        try:
            await asyncio.sleep(seconds)
        finally:
            self._state = False
            await self._send_twice(b"OFF\n")

    async def close(self) -> None:
        self._cancel_auto_off()
        if self._fill_task is not None and not self._fill_task.done():
            self._fill_task.cancel()
            try:
                await self._fill_task
            except (asyncio.CancelledError, Exception):
                pass
            self._fill_task = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
