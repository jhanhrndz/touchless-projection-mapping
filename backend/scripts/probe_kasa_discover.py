"""Broadcast-discover Kasa devices on the LAN, print MAC + IP + alias.

Useful for confirming the switch's MAC matches what's in data/devices.yaml,
or finding the right MAC for a new switch to add.

Run:
    uv run python scripts/probe_kasa_discover.py
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv
from kasa import Discover


async def main() -> None:
    load_dotenv(".env")  # explicit path: no parent-directory walk
    username = os.environ.get("KASA_USERNAME")
    password = os.environ.get("KASA_PASSWORD")
    print(f"Broadcasting on LAN (timeout 5s)…")
    t0 = time.time()
    devices = await Discover.discover(username=username, password=password, timeout=5)
    elapsed_ms = (time.time() - t0) * 1000

    if not devices:
        print(f"  → no Kasa devices found ({elapsed_ms:.0f}ms)")
        print("  Check: switch powered? laptop on same subnet?")
        return

    print(f"  → found {len(devices)} device(s) in {elapsed_ms:.0f}ms\n")
    for ip, dev in devices.items():
        try:
            await dev.update()
        except Exception as e:
            print(f"  {ip:16s}  update failed: {type(e).__name__}: {e}")
            continue
        mac = getattr(dev, "mac", "?")
        alias = getattr(dev, "alias", "?") or "?"
        model = getattr(dev, "model", "?") or "?"
        on = "ON " if getattr(dev, "is_on", False) else "off"
        print(f"  {ip:16s}  {mac}  {on}  '{alias}'  ({model})")


if __name__ == "__main__":
    asyncio.run(main())
