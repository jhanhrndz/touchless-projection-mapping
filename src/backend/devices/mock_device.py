from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Device, MenuItem, MouseButton


@dataclass
class Call:
    kind: str
    args: tuple[Any, ...] = ()


class MockDevice(Device):
    def __init__(
        self,
        object_id: str,
        label: str = "",
        menu: list[MenuItem] | None = None,
    ) -> None:
        super().__init__(object_id, label)
        self._menu = menu or [MenuItem(label="Ping", action_id="ping")]
        self.calls: list[Call] = []

    def _log(self, what: str) -> None:
        # The terminal is the mock's whole feedback channel — the hardware-free
        # quickstart has nothing else to show that a click landed. self.calls
        # alone satisfies the tests but leaves a human watching silence.
        print(f"[mock] {self.object_id}: {what}", flush=True)

    async def on_hover(self, enter: bool) -> None:
        self.calls.append(Call("hover", (enter,)))
        self._log(f"hover {'enter' if enter else 'leave'}")

    async def on_click(self, button: MouseButton) -> None:
        self.calls.append(Call("click", (button,)))
        self._log(f"{button}-click -> would fire the real device here")

    async def on_context_menu(self) -> list[MenuItem]:
        self.calls.append(Call("context_menu"))
        self._log("context menu opened")
        return list(self._menu)

    async def invoke_menu_action(self, action_id: str, value: Any | None = None) -> None:
        self.calls.append(Call("menu", (action_id, value)))
        self._log(f"menu action {action_id!r} value={value!r}")
