"""Toggle the lamp switch with python-kasa directly — no repo code involved.

The bottom rung of the debugging ladder:

    scripts/control_lamp.py        raw python-kasa            (this file)
    scripts/probe_kasa_toggle.py   this repo's KasaDevice driver
    the runtime itself             browser → WebSocket → KasaDevice

If this works but probe_kasa_toggle.py doesn't, the bug is in this repo.
If this fails too, look at the switch, the LAN, or your .env credentials.

Run:
    uv run python scripts/control_lamp.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from kasa import Discover


async def main() -> None:
    load_dotenv(".env")  # explicit path: no parent-directory walk
    host = os.getenv("SMART_SWITCH_HOST")
    if not host:
        raise SystemExit("SMART_SWITCH_HOST not set — populate .env (see .env.example)")
    dev = await Discover.discover_single(
        host,
        username=os.getenv("KASA_USERNAME"),
        password=os.getenv("KASA_PASSWORD"),
    )
    await dev.update()
    print(f"{dev.alias or host}: {'on' if dev.is_on else 'off'} — toggling off, then on")
    await dev.turn_off()
    await asyncio.sleep(0.5)
    await dev.turn_on()
    await dev.update()
    print(f"{dev.alias or host}: {'on' if dev.is_on else 'off'}")


if __name__ == "__main__":
    asyncio.run(main())
