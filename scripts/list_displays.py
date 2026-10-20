#!/usr/bin/env python3
"""Print the displays this machine can see, so you know what to pass to
--projector-size and which screen to drag the projection window onto.

    uv run python scripts/list_displays.py
    uv run python scripts/list_displays.py --self-test

Shells out to the OS rather than taking a dependency — there is no camera and
no OpenCV in this project, and enumerating monitors isn't worth adding one.

If your platform isn't handled, read the resolution off your OS display
settings; it's the only thing this script tells you that the server needs.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Display:
    name: str
    width: int | None
    height: int | None
    is_main: bool

    def describe(self) -> str:
        res = f"{self.width}x{self.height}" if self.width and self.height else "unknown resolution"
        tag = "  ← main (your laptop)" if self.is_main else "  ← candidate projector"
        return f"{self.name}: {res}{tag}"


# --------------------------------------------------------------------------
# Parsers. Kept separate from the subprocess calls so they're testable against
# captured fixture text — see run_self_test().
# --------------------------------------------------------------------------

def parse_macos(text: str) -> list[Display]:
    """Parse `system_profiler SPDisplaysDataType` output.

    Shape (indentation is meaningful — the display name is the shallowest key):

        Color LCD:
          Resolution: 3024 x 1964 Retina
          Main Display: Yes
    """
    displays: list[Display] = []
    current: Display | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # A display header is "Name:" with nothing after the colon.
        if line.endswith(":") and ":" not in line[:-1]:
            name = line[:-1].strip()
            # Section headers we don't want as displays.
            if name in {"Displays", "Graphics/Displays"} or name.endswith("Chipset Model"):
                current = None
                continue
            current = Display(name=name, width=None, height=None, is_main=False)
            displays.append(current)
            continue
        if current is None:
            continue
        m = re.match(r"Resolution:\s*(\d+)\s*[x×]\s*(\d+)", line)
        if m:
            current.width, current.height = int(m.group(1)), int(m.group(2))
        elif re.match(r"Main Display:\s*Yes", line):
            current.is_main = True
    return [d for d in displays if d.width or d.is_main]


def parse_xrandr(text: str) -> list[Display]:
    """Parse `xrandr --listmonitors` output.

        Monitors: 2
         0: +*eDP-1 1920/344x1080/193+0+0  eDP-1
         1: +HDMI-1 1920/509x1080/286+1920+0  HDMI-1

    The '*' marks the primary monitor.
    """
    displays: list[Display] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"\d+:\s*\+(\*?)(\S+)\s+(\d+)(?:/\d+)?x(\d+)(?:/\d+)?", line)
        if not m:
            continue
        primary, name, w, h = m.groups()
        displays.append(
            Display(name=name, width=int(w), height=int(h), is_main=bool(primary))
        )
    return displays


def parse_windows(text: str) -> list[Display]:
    """Parse `wmic path Win32_VideoController get ...` CSV-ish output.

        CurrentHorizontalResolution,CurrentVerticalResolution,Name
        1920,1080,Intel(R) UHD Graphics
    """
    displays: list[Display] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("currenthorizontal"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            w, h = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        displays.append(Display(name=",".join(parts[2:]), width=w, height=h, is_main=not displays))
    return displays


# --------------------------------------------------------------------------


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def list_displays() -> list[Display]:
    system = platform.system()
    if system == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"])
        return parse_macos(out) if out else []
    if system == "Linux":
        out = _run(["xrandr", "--listmonitors"])
        return parse_xrandr(out) if out else []
    if system == "Windows":
        out = _run([
            "wmic", "path", "Win32_VideoController", "get",
            "CurrentHorizontalResolution,CurrentVerticalResolution,Name", "/format:csv",
        ])
        return parse_windows(out) if out else []
    return []


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        return run_self_test()
    if len(argv) != 1:
        raise SystemExit(__doc__)

    displays = list_displays()
    if not displays:
        print(
            f"Could not enumerate displays on {platform.system()!r}.\n"
            "Read the projector's resolution off your OS display settings and pass\n"
            "it directly, e.g.:  uv run python -m runtime --projector-size 1920x1080",
            file=sys.stderr,
        )
        return 1

    print(f"{len(displays)} display(s):\n")
    for d in displays:
        print(f"  {d.describe()}")

    externals = [d for d in displays if not d.is_main and d.width and d.height]
    print()
    if externals:
        d = externals[0]
        print(f"Projector is probably {d.name!r}. Start the server with:\n")
        print(f"    uv run python -m runtime --projector-size {d.width}x{d.height}\n")
        print("Then drag the browser window onto that display and press F11.")
    else:
        print("No external display detected — plug the projector in and re-run.")
        print("You can still author polygons on the laptop; they just won't be")
        print("on a wall until a second display exists.")
    return 0


def run_self_test() -> int:
    """Parsers are tested against captured fixture text.

    Negative cases matter most here: a parser that invents a display, or picks
    up a GPU chipset line as a monitor, sends someone chasing a projector that
    isn't there.
    """
    # ---- macOS: real capture from a MacBook with no projector attached ----
    mac_one = """Graphics/Displays:

    Apple M3 Max:

      Chipset Model: Apple M3 Max
      Displays:
        Color LCD:
          Display Type: Built-in Liquid Retina XDR Display
          Resolution: 3024 x 1964 Retina
          Main Display: Yes
          Mirror: Off
"""
    got = parse_macos(mac_one)
    assert len(got) == 1, got
    assert got[0].name == "Color LCD"
    assert (got[0].width, got[0].height) == (3024, 1964)
    assert got[0].is_main is True

    # positive: projector attached shows up as a second, non-main display
    mac_two = mac_one + """        Optoma HD:
          Resolution: 1920 x 1080
          Main Display: No
          Mirror: Off
"""
    got = parse_macos(mac_two)
    assert len(got) == 2, got
    assert got[1].name == "Optoma HD"
    assert (got[1].width, got[1].height) == (1920, 1080)
    assert got[1].is_main is False
    assert sum(d.is_main for d in got) == 1, "exactly one main display"

    # negative: the GPU chipset line must not be mistaken for a monitor
    assert all(d.name != "Apple M3 Max" for d in got), got
    assert all("Chipset" not in d.name for d in got)

    # negative: empty / junk input yields nothing, not a phantom display
    assert parse_macos("") == []
    assert parse_macos("no displays here\njust text\n") == []

    # ---- Linux ----
    xr = """Monitors: 2
 0: +*eDP-1 1920/344x1080/193+0+0  eDP-1
 1: +HDMI-1 1280/509x800/286+1920+0  HDMI-1
"""
    got = parse_xrandr(xr)
    assert len(got) == 2, got
    assert got[0].name == "eDP-1" and got[0].is_main is True
    assert got[1].name == "HDMI-1" and got[1].is_main is False
    assert (got[1].width, got[1].height) == (1280, 800)

    # negative: the "Monitors: 2" header line is not itself a monitor
    assert all(d.name != "Monitors" for d in got)
    assert parse_xrandr("Monitors: 0\n") == []
    assert parse_xrandr("") == []

    # ---- Windows ----
    win = """CurrentHorizontalResolution,CurrentVerticalResolution,Name
1920,1080,Intel(R) UHD Graphics
1280,800,NVIDIA GeForce
"""
    got = parse_windows(win)
    assert len(got) == 2, got
    assert (got[0].width, got[0].height) == (1920, 1080)
    assert got[0].is_main is True and got[1].is_main is False

    # negative: header row and blank/short rows are skipped
    assert parse_windows("CurrentHorizontalResolution,CurrentVerticalResolution,Name\n") == []
    assert parse_windows("") == []
    assert parse_windows("garbage,row\n") == []
    # negative: non-numeric resolutions (a card reporting nothing) are skipped
    assert parse_windows("CurrentHorizontalResolution,CurrentVerticalResolution,Name\n,,Headless\n") == []

    # ---- describe() ----
    assert "1920x1080" in Display("X", 1920, 1080, False).describe()
    assert "unknown resolution" in Display("X", None, None, False).describe()
    assert "main" in Display("X", 1, 1, True).describe()

    print("self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
