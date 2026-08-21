# Interactive projection mapping

Point a projector at your wall. Draw an outline around a real object — a lamp,
a book on a shelf. Click the outline. **The real thing happens.**

Two builds are documented here end to end:

| Demo | You click | The world does |
|---|---|---|
| **Smart Lamp** | an outline projected around your actual lamp | the lamp turns on |
| **Bookshelf** | one book's outline on your actual shelf | a motorised rod pushes that book out |

### ▶ [Watch it working](https://www.instagram.com/buildwbrendan/reel/DYuwZPcSyr2/)

**No camera. No calibration. No computer vision.** The editor canvas *is* the
projector rectangle, 1:1 — a polygon you draw at (x, y) lands at (x, y) on the
wall. You line it up by looking at the wall, which takes about thirty seconds
and is the part you'd have to do anyway.

Build the Smart Lamp first: a projector and a $15 smart switch is the whole bill
of materials.

> **Working on this with an AI agent?** Point it at
> [`AGENTS.md`](AGENTS.md) — setup on an unfamiliar machine, the invariants
> not to break, and an honest list of what an agent can verify itself versus
> what only a human standing in front of the wall can.

---

## How it works

**The whole premise:** a projector and a computer with a monitor. The
projector is just a second display, and what it throws at the wall is a
**fullscreen browser window**. Everything on the wall is a webpage.

A browser was a deliberate choice, not a shortcut: hover, click, and
fullscreen already mean exactly what you want them to mean — and webpages are
the thing coding agents are best at building, so the ceiling for extending
this is very high.

```
   1. AUTHOR  (once per object)
      /editor shows a black canvas the size of your projector.
      Trace a polygon around where the lamp is on the wall.
                        └─> saved to data/objects.json

   2. RUN  (forever)
      A browser, fullscreen on the projector, paints those polygons.
      Your mouse hovers/clicks one ──WebSocket──> Device driver ──> real hardware
```

That's the whole system. Because step 2 is just a webpage, "hover" and "click"
mean exactly what they mean in a browser — you move your laptop's mouse, the
cursor lands on the wall, on the real object.

The **two-window model** is the one thing to internalise: every authoring page
has a projector twin, live-synced.

```
  monitor (you look at)     projector (the wall shows)
  ────────────────────      ──────────────────────────
  /editor              →    /editor/projection
  /style               →    /style/preview      (clicks log on the page, nothing fires)
                            /                   ← the actual show
```

How I actually run it (the URLs below say `:8000` — swap in whatever `--port`
you chose):

- **<http://127.0.0.1:8000/> — the final scene window.** Fullscreen on the
  projector display (F11 / ⌃⌘F), so there's no URL bar or tab strip
  projected on your wall.
- **<http://127.0.0.1:8000/editor> — keep this up on your computer monitor.**
  It's where you create new polygons, link them to the objects and code
  logic behind them, and see the whole scene at once.
- **<http://127.0.0.1:8000/editor/projection> — pairs with `/editor`, lives
  on the projector.** It's 1:1 with the final scene window — except you can
  edit it directly: drag vertices or whole polygons, ⌘-Z to undo. In my
  opinion it's the best way to make shapes match the real world, because
  you're editing at wall scale, on the wall itself. Since you want it
  matching the final scene window anyway, I keep it as a second tab in the
  same fullscreen browser window as `/` — switching between "editing the
  wall" and "the show" is one tab switch on the projector display.

### Why there's no calibration

The usual way to do this is to photograph projected markers with a camera and
solve a homography that maps camera pixels to projector pixels. That earns its
complexity when software has to *find* something in the room.

Here it doesn't. You're the one who knows where the lamp is. Draw the polygon,
look at the wall, nudge it until it fits. The projector's own edges are the
frame — set them where you want them by aiming the projector.

---

## Bill of materials

**Both demos**
- A projector. Set it to its native resolution and tell the server with
  `--projector-size` (defaults to `1920x1080`).
- A laptop. The server is plain Python + FastAPI; nothing here is OS-specific.

**Smart Lamp**
- A **TP-Link Kasa smart switch** on the same LAN, wired to the lamp — that's
  what the demo actually uses. A Kasa smart plug or smart bulb drops in
  unchanged; the driver only ever calls on/off.
- Any other brand works at the cost of a ~40-line driver — see
  [Adding your own device](#adding-your-own-device).

> **Using a switch? Leave the lamp's own switch ON.** The smart switch cuts
> power upstream, so a lamp switched off at the lamp stays dark no matter what
> the projection does. Costs an embarrassing amount of debugging to rediscover.

**Bookshelf**
- A Raspberry Pi on the LAN.
- A stepper or linear actuator behind the shelf that can push one book forward
  and pull back. Mine is a 3D-printed rod on a lead screw.
- **The Pi-side service is not in this repo** — it's ~100 lines and it's
  specific to your motor. [`docs/BOOK_PUSHER.md`](docs/BOOK_PUSHER.md) specifies
  the three-command UDP protocol so you can write it against your own hardware.

---

## Quickstart — no hardware at all

The whole authoring experience runs on a laptop alone. Every click goes to a
mock device that logs to the terminal:

```
[mock] lamp_switch: left-click -> would fire the real device here
```

```bash
git clone https://github.com/buildwbrendan/interactive-projection-mapping
cd interactive-projection-mapping
uv sync
uv run python -m runtime --devices data/devices.dev.yaml
```

Open <http://127.0.0.1:8000/editor>, draw a polygon, bind it to `lamp_switch`,
save. Open <http://127.0.0.1:8000/> and click it. The terminal logs the click
that *would* have flipped the switch.

Want something to click before you've drawn anything? Seed the worked example
first (the coordinates are my wall, but it clicks fine):

```bash
cp data/examples/objects.lamp.json data/objects.json
```

> Uses [uv](https://docs.astral.sh/uv/). `pip install -e .` works too for the
> runtime itself; the test tooling lives in a `dev` dependency group that uv
> installs by default.

### Or: one command in Docker

```bash
docker compose up
```

Builds the image, seeds the lamp example automatically, and serves
<http://localhost:8000/> with mock devices — nothing installed on the host.
`docker compose run --rm tests` runs the test suite in the same image (the
`tests` service sits behind a compose profile, so a plain `docker compose up`
never starts it).

The container is for the hardware-free demo. The real demos want the server
on the **host**: Kasa discovery rides LAN broadcasts that Docker's NAT
doesn't carry.

> **Linux note.** The container runs as root, so files it writes into the
> bind-mounted `./data` (the seeded `objects.json`, `/style` uploads) end up
> root-owned on the host. `sudo chown -R $USER data` reclaims them, or run
> the service as yourself with `user: "1000:1000"` in a compose override.
> Docker Desktop on macOS/Windows maps ownership automatically.

---

## Build 1 — Smart Lamp

### 1. Wire up the switch

```bash
cp .env.example .env
```

Fill in `SMART_SWITCH_HOST` with the switch's LAN IP. Don't know it?

```bash
uv run python scripts/probe_kasa_discover.py
```

Most modern Kasa hardware speaks KLAP and wants your TP-Link cloud credentials
even for purely local control — put them in `KASA_USERNAME` / `KASA_PASSWORD`.

Then set the switch's MAC in [`data/devices.yaml`](data/devices.yaml). Optional,
but you want it: if DHCP moves the switch, the runtime broadcasts to find it by
MAC and caches the new IP instead of just failing.

Confirm it works before you involve the projector:

```bash
uv run python scripts/probe_kasa_toggle.py     # toggles twice, via the runtime code path
```

### 2. Aim the projector, and find its resolution

Point it at the wall and get the lamp inside the projected rectangle. The edge
of the projected image is the edge of your canvas — there's nothing to
configure but the aim.

Then find out what resolution it's actually running at, because that's the one
number the server needs:

```bash
uv run python scripts/list_displays.py
```

```
2 display(s):

  Color LCD: 3024x1964  ← main (your laptop)
  Optoma HD: 1920x1080  ← candidate projector

Projector is probably 'Optoma HD'. Start the server with:

    uv run python -m runtime --projector-size 1920x1080
```

No camera and no extra dependency — it just asks the OS. If your platform isn't
handled it says so; read the number off your display settings instead.

### 3. Draw the lamp

```bash
uv run python -m runtime --projector-size 1920x1080
```

Open <http://127.0.0.1:8000/editor> on your monitor and
<http://127.0.0.1:8000/editor/projection> on the projector, fullscreen. The two
are live-synced: what you draw on the left appears on the wall immediately.
And the projection page isn't just a mirror — you can drag vertices or whole
polygons right there on the wall, with ⌘-Z to undo. Fine-tuning a shape onto
the real object is usually easiest done at wall scale.

1. Click points on the black canvas to trace where the lamp is. The faint
   outline around the canvas is the edge of the projected image.
2. Drag vertices to nudge, drag the body to move the whole polygon, click a
   midpoint dot to insert a vertex, shift-click a vertex to delete. ⌘-Z undoes.
   Press `?` for the cheat sheet.
3. Bind it to `lamp_switch` in the dropdown.
4. **Save** → `data/objects.json`.

> **Aligning without seeing your cursor.** Press **spacebar** in the laptop
> editor and an orange dot follows your mouse on the wall. That's how you find
> the lamp: move the mouse until the dot is on it, and draw there.

A worked example is in [`data/examples/objects.lamp.json`](data/examples/objects.lamp.json) —
copy it to `data/objects.json` to see the file's shape. The coordinates are for
my wall, so you'll redraw, but the structure is the point.

### 4. Run the show

```bash
uv run python -m runtime
```

Open <http://127.0.0.1:8000/> on the projector, fullscreen, and click the lamp.

On boot you'll see a device prewarm line:

```
Device prewarm:
  ✓ lamp_switch            reachable at 10.0.0.42 (MAC AABBCCDDEEFF)
```

If a device is offline the server still boots and logs `✗ … unreachable`.

---

## Build 2 — Bookshelf

Same runtime, same editor, one extra polygon per book and a motor behind the
shelf.

### 1. Get the Pi service running

Write a UDP listener on port 5007 that understands `PUSH`, `ZERO`, and
`SETGOAL`. The contract — including the homing rule that will otherwise bite
you — is in [`docs/BOOK_PUSHER.md`](docs/BOOK_PUSHER.md).

Then point [`data/devices.yaml`](data/devices.yaml) at it:

```yaml
- object_id: book_pusher
  type: book_pusher
  label: "Book pusher"
  config:
    host: bookshelf-pi.local
    port: 5007
```

### 2. Outline every book

Draw one polygon per book spine. Bind **only** the book you want to move to
`book_pusher`; leave the others unbound. Unbound polygons are still drawn and
still highlight on hover — which is the whole visual effect. One live book among
a shelf of glowing outlines reads much better on camera than a single lonely
rectangle.

See [`data/examples/objects.bookshelf.json`](data/examples/objects.bookshelf.json):
eight book outlines, exactly one bound to `book_pusher`.

### 3. Style them

```bash
uv run python -m runtime
```

Open <http://127.0.0.1:8000/style> — a per-object inspector with tabs for
outline, fill, image, label, sound, highlight, and click behaviour. Give each
book its own colour, width, and dash style; give the live one a brighter
highlight.

Everything previews to `/style/preview`, which renders identically to the real
show but **logs clicks instead of firing devices**. The wall keeps showing the
last-saved state until you hit **Save to objects.json** and reload `/`. That's
deliberate — you don't want to strobe a real lamp while dragging a slider.

### 4. The lamp chain (surprise behaviour, on purpose)

There's one hardcoded coupling in `server.py`: **left-clicking `book_pusher`
also turns on `lamp_switch` three seconds later**, if a switch is registered.

That's the beat from the original video — the book slides out, then the lamp
comes up behind it. If you don't want it, delete the `HARDCODED CHAIN` block in
[`src/runtime/server.py`](src/runtime/server.py); it's a couple of lines and
nothing else depends on it.

### 5. Hide what shouldn't be lit

Glossy shelf edge bouncing projector light back at you? Draw a polygon over it
and tick **Black-hole (occluder)**. It renders solid black, hides the cursor,
and emits no events.

---

## Gotchas that cost me a session each

- **Display sleep blanks the projector while the server still reports
  healthy.** You get a "no signal" badge on the projector and no error
  anywhere. Run `caffeinate -d` (macOS) for the length of any session.
- **`--projector-size` must match the projector's actual output.** Get it wrong
  and every polygon lands somewhere else on the wall, with nothing in the logs
  to tell you why. (In a non-fullscreen window, `/` scales the whole scene to
  fit — fine for previewing on the laptop; the 1:1 mapping that puts light on
  the real object only holds fullscreen on the projector at native resolution.)
- **Browsers block audio until a real user gesture.** The first hover on a fresh
  page load plays nothing. Any click unlocks it for the session.
- **Idle outlines are deliberately faint** — a 10%-white hairline, tuned for a
  dark room. On a bright laptop screen `/` can read as empty; the count in the
  status corner tells you what actually loaded, and hovering lights things up.
- **Save before testing on the wall.** `/` reads `/api/objects` once on page
  open. A change isn't on the wall until you save *and* reload `/`.
- **Moved the projector? Redraw the polygons.** There's no calibration to
  re-run — but there's also nothing that compensates for a new aim.

---

## Adding your own device

A device is one class with up to four methods. Drop it in
`src/runtime/devices/`, register it in `registry.py`, reference the type name
from `data/devices.yaml`.

```python
from .base import Device, MenuItem, MouseButton

class MyThingDevice(Device):
    def __init__(self, object_id: str, label: str = "", *, host: str) -> None:
        super().__init__(object_id, label)
        self._host = host

    async def on_click(self, button: MouseButton) -> None:
        ...                      # make the real thing happen

    async def on_hover(self, enter: bool) -> None:
        ...                      # optional

    async def on_context_menu(self) -> list[MenuItem]:
        return [MenuItem(label="Reset", action_id="reset")]

    async def close(self) -> None:
        ...
```

Shipped drivers: `kasa` (any TP-Link Kasa switch, plug or bulb — it only uses
on/off), `book_pusher`, `udp` (fire-and-forget datagrams, with a double-send
mode for ESP32s in WiFi light sleep), `http`, and `mock`.

---

## Every page

| URL | What it is |
|---|---|
| `/` | the final scene — fullscreen on the projector, nothing else on screen |
| `/editor` | create polygons, bind devices and logic — keep on your monitor |
| `/editor/projection` | editable 1:1 twin of the scene — drag and ⌘-Z shapes at wall scale |
| `/style` | per-object appearance inspector (laptop side) |
| `/style/preview` | renders like `/` but clicks only log (on the page + console) |
| `/tune` | quick per-object effect / color / speed tuner |
| `/text` | fullscreen text projector |
| `/text/editor` | laptop control for `/text` |
| `/align` | nudge tracked-polygon rendering (stepper rigs only) |

**Server flags.** `--host` (default `127.0.0.1`, loopback only — use
`0.0.0.0` to reach the server from another machine), `--port`, `--objects`,
`--devices`, `--projector-size`, and `--mock`, which rewrites every entry in
the given devices file to a mock — same object ids, nothing real fired.
(`data/devices.dev.yaml` is the same idea as a checked-in file; either is
safe.) `APP_HOST` / `APP_PORT` in `.env` set the host/port defaults. The
stepper-rig flags (`--stepper-calibration`, `--paddle-state`,
`--projection-align`) are documented in `--help` and inert unless set.

## Repo layout

```
AGENTS.md               orientation for AI agents and new contributors
src/runtime/
  __main__.py           CLI — --projector-size lives here
  server.py             FastAPI app, WebSocket hub, object API
  editor.py             polygon authoring + validation
  devices/              one file per device driver
  static/               every page: editor, style, runtime, text
data/devices.yaml       object_id → driver bindings (edit this)
data/devices.dev.yaml   the same ids, all mocks — develop against this
data/examples/          worked objects.json files for both demos
scripts/list_displays.py   which monitor is the projector, and its resolution
scripts/probe_kasa_*.py    find and toggle the switch on the LAN
scripts/control_lamp.py    raw python-kasa toggle — bypasses this repo's code
tests/                  pytest, no hardware or network needed
docs/BOOK_PUSHER.md     the UDP protocol for the shelf motor
Dockerfile              the one-command mock sandbox (docker compose up)
```

```bash
uv run pytest
```

---

## Where this came from

This is an extract of a larger private repo of ten projector installations —
brick breaker you play with your body, a magnifying glass that reveals wiring
inside the wall, googly eyes tracked onto a pair of sunglasses. Those need
cameras and calibration. These two don't, which is why they're the ones worth
handing to someone else.

More at [@buildwbrendan](https://www.instagram.com/buildwbrendan/).

## Forks welcome

I don't plan to actively maintain this repo — it's a working snapshot, not a
product. Fork it and build on it freely under the GPL: personal projects,
art installations, running it anywhere, modifying everything. Just tag
[@buildwbrendan](https://www.instagram.com/buildwbrendan/) when you ship
something — I want to see what you build.

If you contribute code back to this repo, you agree your contribution is
licensed under GPL-3.0 and that I may also offer it under commercial
licenses (that's what keeps the dual-licensing option below workable).

## License

GPL-3.0 — see [LICENSE](LICENSE). Copyright (c) 2026 Brendan
([@buildwbrendan](https://www.instagram.com/buildwbrendan/)).

In short: use it, run it, change it, fork it — freely. If you distribute
something built on it (shipping a product, deploying installs to client
hardware), your derivative must be open source under the GPL too.

**Building it into a closed or commercial product?** Commercial licenses are
available instead — DM [@buildwbrendan](https://www.instagram.com/buildwbrendan/)
and we'll sort it out.
