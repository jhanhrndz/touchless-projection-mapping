"""Run a single click through KasaDevice — the same path the runtime uses.

Useful for isolating whether a 'click not working' bug is in the device
layer or higher up (websocket / object binding / etc).

The device is built from data/devices.yaml through the same loader the
server uses, so what gets tested is exactly what you configured — the MAC
fallback, the env-var names, all of it.

Run:
    uv run python scripts/probe_kasa_toggle.py [--devices data/devices.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from runtime.config import load_registry

PLACEHOLDER_MAC = "AA:BB:CC:DD:EE:FF"


def _first_kasa_entry(devices_path: Path) -> dict:
    raw = yaml.safe_load(devices_path.read_text())
    entries = raw.get("devices", []) if isinstance(raw, dict) else (raw or [])
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "kasa":
            return entry
    sys.exit(f"error: no `type: kasa` entry in {devices_path}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="data/devices.yaml")
    args = parser.parse_args()

    load_dotenv(".env")  # explicit path: no parent-directory walk
    devices_path = Path(args.devices)
    entry = _first_kasa_entry(devices_path)
    mac = entry.get("config", {}).get("mac")
    print(f"testing {entry['object_id']!r} from {devices_path} (mac: {mac or 'not set'})")
    if mac == PLACEHOLDER_MAC:
        print(
            "  note: that's still the placeholder MAC shipped in devices.yaml —\n"
            "  put your switch's real MAC there to test the broadcast fallback."
        )

    switch = load_registry(devices_path).get(entry["object_id"])
    try:
        print("first click:")
        await switch.on_click("left")
        print("  done. waiting 1.5s…")
        await asyncio.sleep(1.5)
        print("second click:")
        await switch.on_click("left")
        print("  done.")
    except Exception as e:
        # A missing .env or an unreachable switch is the expected first-run
        # failure — answer it in plain language first, exception detail
        # second. All on stdout: the driver's own progress lines go there,
        # and a stdout/stderr split would interleave them out of order on a
        # terminal.
        print("\nNothing toggled — couldn't reach the switch.")
        print(f"  (detail: {e})")
        print(
            "\nChecklist:\n"
            "  1. cp .env.example .env, then set SMART_SWITCH_HOST to the "
            "switch's LAN IP\n"
            "  2. most modern Kasa hardware also needs KASA_USERNAME / "
            "KASA_PASSWORD (your TP-Link account)\n"
            "  3. can't find the IP? uv run python scripts/probe_kasa_discover.py\n"
            "  4. put the switch's real MAC in data/devices.yaml so DHCP "
            "moves can't break it"
        )
        return 1
    finally:
        await switch.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
