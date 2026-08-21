from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .devices.registry import DeviceRegistry


def load_registry(devices_yaml: Path | None, *, mock: bool = False) -> DeviceRegistry:
    """
    Build a DeviceRegistry from a YAML file. If `devices_yaml` is None or the
    file is missing, returns an empty registry — callers can register devices
    programmatically (useful for tests and dev mode).

    If `mock=True`, every entry's `type` is rewritten to `"mock"` before
    building, so the resulting registry has the same `object_id`s as the file
    declares but every device is a `MockDevice`. Polygons in objects.json
    keep resolving (no rebinding needed); clicks just record into a buffer
    instead of reaching real hardware.

    Two equivalent YAML schemas are accepted:

        # (1) legacy — a bare list of device entries
        - object_id: switch_left
          type: mock          # or http, udp, kasa (added in Phase 6)
          label: "Left switch"
          # type-specific fields go here, opaque to this loader

        # (2) mapping form — lets you add scene-wide flags alongside devices
        scene:
          flame_carry: true   # see load_scene_settings()
        devices:
          - object_id: switch_left
            type: mock
    """
    registry = DeviceRegistry()
    if devices_yaml is None or not devices_yaml.exists():
        return registry

    entries = _device_entries(devices_yaml, yaml.safe_load(devices_yaml.read_text()))

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{devices_yaml}[{i}]: entry must be a mapping")
        spec = dict(entry)
        try:
            object_id = spec.pop("object_id")
            type_name = spec.pop("type")
        except KeyError as missing:
            raise ValueError(
                f"{devices_yaml}[{i}]: missing required field {missing.args[0]!r}"
            ) from None
        if mock:
            # Discard type-specific config (host/url/etc) — MockDevice ignores it,
            # and leaving it in would let typos slip past the original factory's checks.
            type_name = "mock"
            spec = {"label": spec.get("label", "")}
        registry.build(object_id, type_name, spec)

    return registry


def _device_entries(devices_yaml: Path, raw: Any) -> list:
    """Pull the list of device entries out of a parsed devices.yaml doc,
    accepting either the legacy bare-list form or the mapping form
    (`{scene: {...}, devices: [...]}`)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        devices = raw.get("devices", [])
        if not isinstance(devices, list):
            raise ValueError(
                f"{devices_yaml}: 'devices' must be a list, got {type(devices).__name__}"
            )
        return devices
    raise ValueError(
        f"{devices_yaml}: expected a list of device entries or a mapping with "
        f"'devices', got {type(raw).__name__}"
    )


def load_scene_settings(devices_yaml: Path | None) -> dict[str, Any]:
    """Scene-wide behavior flags from a devices.yaml in the mapping form.

    Returns the `scene:` block (a mapping) when present, else an empty dict —
    so the legacy bare-list form and a missing file both mean "no flags / all
    defaults". Currently recognized by the runtime page:

        scene:
          flame_carry: true   # enable the click-and-carry flame mechanic
    """
    if devices_yaml is None or not devices_yaml.exists():
        return {}
    raw = yaml.safe_load(devices_yaml.read_text())
    if not isinstance(raw, dict):
        return {}
    scene = raw.get("scene") or {}
    if not isinstance(scene, dict):
        raise ValueError(f"{devices_yaml}: 'scene' must be a mapping")
    return scene
