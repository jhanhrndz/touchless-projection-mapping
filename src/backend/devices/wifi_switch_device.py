from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal
import httpx

from .base import Device, MenuItem, MouseButton

logger = logging.getLogger("runtime.devices.wifi_switch")

WifiSwitchMode = Literal["http_toggle", "http_stateful", "tasmota", "shelly"]


class WifiSwitchDevice(Device):
    """Controlador universal de interruptor WiFi para interacción táctil o touchless.

    Diseñado para accionar interruptores físicos (relés ESP32/ESP8266, Tasmota,
    Shelly o servidores HTTP embebidos) vinculados a polígonos interactivos (ej. lámpara).

    Modos soportados mediante configuración YAML:
      1. `http_toggle` (por defecto):
         Envía una solicitud HTTP GET/POST a una URL que conmuta el estado del relé.
         Ej: url: "http://192.168.1.150/toggle"

      2. `http_stateful`:
         Mantiene seguimiento del estado local y llama a `on_url` o `off_url`.
         Ej: on_url: "http://192.168.1.150/on", off_url: "http://192.168.1.150/off"

      3. `tasmota`:
         Utiliza el comando estándar HTTP de Tasmota (/cm?cmnd=Power%20TOGGLE).
         Ej: host: "192.168.1.150"

      4. `shelly`:
         Utiliza el endpoint estándar de relé de Shelly (/relay/0?turn=toggle).
         Ej: host: "192.168.1.150"
    """

    def __init__(
        self,
        object_id: str,
        label: str = "",
        *,
        mode: WifiSwitchMode = "http_toggle",
        host: str = "",
        url: str = "",
        on_url: str = "",
        off_url: str = "",
        status_url: str = "",
        method: str = "GET",
        timeout_s: float = 2.0,
        relay_index: int = 0,
    ) -> None:
        super().__init__(object_id, label)
        self.mode = mode
        self.host = host.strip().rstrip("/")
        self.url = url.strip()
        self.on_url = on_url.strip()
        self.off_url = off_url.strip()
        self.status_url = status_url.strip()
        self.method = method.upper()
        self.timeout_s = timeout_s
        self.relay_index = relay_index

        self._is_on: bool = False
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

        # Configurar URLs según el modo si se especificó solo el host
        if self.mode == "tasmota":
            if not self.host and self.url:
                from urllib.parse import urlparse
                self.host = urlparse(self.url).netloc
            base = f"http://{self.host}" if not self.host.startswith("http") else self.host
            pwr_cmd = f"Power{self.relay_index + 1}" if self.relay_index > 0 else "Power"
            self.url = f"{base}/cm?cmnd={pwr_cmd}%20TOGGLE"
            self.on_url = f"{base}/cm?cmnd={pwr_cmd}%20ON"
            self.off_url = f"{base}/cm?cmnd={pwr_cmd}%20OFF"
            self.status_url = f"{base}/cm?cmnd={pwr_cmd}"
            self.method = "GET"

        elif self.mode == "shelly":
            base = f"http://{self.host}" if not self.host.startswith("http") else self.host
            self.url = f"{base}/relay/{self.relay_index}?turn=toggle"
            self.on_url = f"{base}/relay/{self.relay_index}?turn=on"
            self.off_url = f"{base}/relay/{self.relay_index}?turn=off"
            self.status_url = f"{base}/relay/{self.relay_index}"
            self.method = "GET"

        elif self.mode == "http_toggle":
            if not self.url and self.host:
                base = f"http://{self.host}" if not self.host.startswith("http") else self.host
                self.url = f"{base}/toggle"

        elif self.mode == "http_stateful":
            if self.host:
                base = f"http://{self.host}" if not self.host.startswith("http") else self.host
                if not self.on_url:
                    self.on_url = f"{base}/on"
                if not self.off_url:
                    self.off_url = f"{base}/off"

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    @property
    def is_on(self) -> bool:
        return self._is_on

    async def turn_on(self) -> None:
        async with self._lock:
            client = await self._ensure_client()
            target_url = self.on_url or self.url
            if not target_url:
                raise ValueError(f"WifiSwitchDevice({self.object_id!r}) sin URL configurada para encender")
            res = await client.request(self.method, target_url)
            res.raise_for_status()
            self._is_on = True
            logger.info("WifiSwitch '%s' encendido exitosamente (URL: %s)", self.object_id, target_url)

    async def turn_off(self) -> None:
        async with self._lock:
            client = await self._ensure_client()
            target_url = self.off_url or self.url
            if not target_url:
                raise ValueError(f"WifiSwitchDevice({self.object_id!r}) sin URL configurada para apagar")
            res = await client.request(self.method, target_url)
            res.raise_for_status()
            self._is_on = False
            logger.info("WifiSwitch '%s' apagado exitosamente (URL: %s)", self.object_id, target_url)

    async def toggle(self) -> None:
        async with self._lock:
            client = await self._ensure_client()
            if self.mode == "http_stateful":
                target_url = self.off_url if self._is_on else self.on_url
                res = await client.request(self.method, target_url)
                res.raise_for_status()
                self._is_on = not self._is_on
            else:
                target_url = self.url
                if not target_url:
                    raise ValueError(f"WifiSwitchDevice({self.object_id!r}) sin URL configurada para conmutar")
                res = await client.request(self.method, target_url)
                res.raise_for_status()
                self._is_on = not self._is_on
            logger.info("WifiSwitch '%s' conmutado -> estado: %s", self.object_id, "ON" if self._is_on else "OFF")

    async def on_click(self, button: MouseButton) -> None:
        """Acción por defecto al hacer clic/pellizco sobre el polígono."""
        await self.toggle()

    async def on_context_menu(self) -> list[MenuItem]:
        return [
            MenuItem(label="Conmutar Lámpara", action_id="toggle"),
            MenuItem(label="Encender Lámpara", action_id="turn_on"),
            MenuItem(label="Apagar Lámpara", action_id="turn_off"),
        ]

    async def invoke_menu_action(self, action_id: str, value: Any | None = None) -> None:
        if action_id == "toggle":
            await self.toggle()
        elif action_id == "turn_on":
            await self.turn_on()
        elif action_id == "turn_off":
            await self.turn_off()
        else:
            raise NotImplementedError(f"Acción {action_id!r} desconocida para WifiSwitchDevice")

    async def prewarm(self) -> str | None:
        """Verificación no bloqueante de conectividad en el arranque."""
        probe_target = self.status_url or self.url or self.on_url
        if not probe_target:
            return None
        try:
            client = await self._ensure_client()
            res = await client.get(probe_target, timeout=min(self.timeout_s, 1.0))
            return f"{probe_target} (código {res.status_code})"
        except Exception as e:
            logger.warning("WifiSwitch '%s' prewarm offline: %s", self.object_id, e)
            return None

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
