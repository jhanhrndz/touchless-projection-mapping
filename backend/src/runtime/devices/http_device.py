from __future__ import annotations

from typing import Any

import httpx

from .base import Device, MenuItem, MouseButton


class HttpDevice(Device):
    """A device that issues an HTTP request on click.

    Configurable via YAML:

        - object_id: water_pump
          type: http
          label: "Water pump"
          config:
            base_url: http://pi1:8001
            click_path: /tap
            click_method: POST     # default
            timeout_s: 2.0         # default

    Click hits {base_url}{click_path} with the configured method. A non-2xx
    response raises, which is surfaced to the WebSocket client as an error toast.
    """

    def __init__(
        self,
        object_id: str,
        label: str = "",
        *,
        base_url: str,
        click_path: str = "/click",
        click_method: str = "POST",
        timeout_s: float = 2.0,
    ) -> None:
        super().__init__(object_id, label)
        self._base = base_url.rstrip("/")
        self._click_path = click_path if click_path.startswith("/") else "/" + click_path
        self._click_method = click_method.upper()
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def on_click(self, button: MouseButton) -> None:
        client = await self._ensure_client()
        response = await client.request(
            self._click_method, self._base + self._click_path
        )
        response.raise_for_status()

    async def on_context_menu(self) -> list[MenuItem]:
        return [MenuItem(label="Trigger", action_id="trigger")]

    async def invoke_menu_action(self, action_id: str, value: Any | None = None) -> None:
        if action_id == "trigger":
            await self.on_click("left")
        else:
            raise NotImplementedError(f"unknown action {action_id!r} for HttpDevice")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
