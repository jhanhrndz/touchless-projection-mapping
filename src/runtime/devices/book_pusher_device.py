from __future__ import annotations

import socket
from typing import Any

from .base import Device, MenuItem, MouseButton


class BookPusherDevice(Device):
    """Device controlled via UDP datagrams to the book-pusher service on a Pi.

    Click → "PUSH" (the service runs forward-to-goal then back-to-home as one
    motion). Right-click menu also exposes Zero (set encoder origin) and
    Set goal (capture current position as the persistent goal target).

    Fire-and-forget: we send the command and don't block on the service's
    reply. The pi-side service serializes commands behind a lock and handles
    its own state (encoder count, homed flag, persisted goal). After a
    service restart the rod must be physically positioned at home and the
    Zero action invoked once before PUSH is accepted.

    Unlike the UdpDevice for the water-pump ESP32, this peer is a Pi4 that's
    not in light sleep, so a single send is sufficient.

    YAML:

        - object_id: book_pusher
          type: book_pusher
          label: "Book pusher"
          config:
            host: bookshelf-pi.local
            port: 5007                 # default
    """

    def __init__(
        self,
        object_id: str,
        label: str = "",
        *,
        host: str,
        port: int = 5007,
    ) -> None:
        super().__init__(object_id, label)
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None

    def _ensure_sock(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._sock

    def _send(self, payload: bytes) -> None:
        self._ensure_sock().sendto(payload, (self._host, self._port))

    async def on_click(self, button: MouseButton) -> None:
        self._send(b"PUSH")

    async def on_context_menu(self) -> list[MenuItem]:
        return [
            MenuItem(label="Push", action_id="push"),
            MenuItem(label="Zero (set home)", action_id="zero"),
            MenuItem(label="Set goal here", action_id="setgoal"),
        ]

    async def invoke_menu_action(self, action_id: str, value: Any | None = None) -> None:
        if action_id == "push":
            self._send(b"PUSH")
        elif action_id == "zero":
            self._send(b"ZERO")
        elif action_id == "setgoal":
            self._send(b"SETGOAL")
        else:
            raise NotImplementedError(
                f"unknown action {action_id!r} for BookPusherDevice"
            )

    async def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
