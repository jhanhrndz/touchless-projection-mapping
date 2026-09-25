from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

MouseButton = Literal["left", "right", "middle"]


@dataclass(frozen=True)
class MenuItem:
    label: str
    action_id: str
    input: dict[str, Any] | None = None


@runtime_checkable
class Transport(Protocol):
    async def send(self, payload: Any) -> Any: ...
    async def close(self) -> None: ...


class Device(ABC):
    def __init__(self, object_id: str, label: str = "") -> None:
        self.object_id = object_id
        self.label = label or object_id

    async def on_hover(self, enter: bool) -> None:
        return None

    @abstractmethod
    async def on_click(self, button: MouseButton) -> None: ...

    async def on_context_menu(self) -> list[MenuItem]:
        return []

    async def invoke_menu_action(self, action_id: str, value: Any | None = None) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}({self.object_id!r}) has no menu action {action_id!r}"
        )

    async def close(self) -> None:
        return None

    async def prewarm(self) -> str | None:
        """Optional startup connectivity check. Default: no-op.

        Devices that benefit from eagerly establishing a connection (e.g.
        Kasa devices over DHCP) override this to ensure they're reachable
        before the first user click, so a click never pays the discovery cost.

        Return a short status string for logging (e.g. "10.0.0.42
        (cached)") or None if the device opts out of prewarm logging."""
        return None
