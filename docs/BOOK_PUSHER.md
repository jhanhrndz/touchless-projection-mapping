# The book-pusher protocol

The Bookshelf demo needs a small service on a Raspberry Pi that drives whatever
motor sits behind your shelf. **That service is not in this repo** — it's a
hundred lines and every line of it depends on your motor, driver board, and
mechanics. This document is the contract it has to satisfy, so you can write it
against your own hardware and the runtime will drive it unchanged.

The client side *is* here: [`src/runtime/devices/book_pusher_device.py`](../src/runtime/devices/book_pusher_device.py).

## Transport

Plain UDP datagrams, default port **5007**. ASCII payload, no framing, no
trailing newline required.

The runtime is **fire-and-forget**: it sends and does not wait for a reply. All
state lives on the Pi. That means a dropped packet is a dropped click, which in
practice is fine on a wired-or-decent LAN and keeps the click path off the
event loop.

> Unlike the ESP32 drivers in this project, the Pi isn't in WiFi light sleep, so
> a single send is enough — no double-send needed.

## Commands

| Payload | Meaning |
|---|---|
| `PUSH` | Run the full gesture: drive forward to the goal position, then return to home. One motion, one command. |
| `ZERO` | Declare the current physical position to be home (encoder origin). |
| `SETGOAL` | Declare the current physical position to be the goal — how far forward a `PUSH` travels. |

`PUSH` fires on left-click of the bound polygon. All three are also on the
polygon's right-click menu in the runtime.

## Behaviour your service must implement

**Serialize everything.** Hold a lock around motion. A second `PUSH` arriving
mid-push must be ignored or queued, never interleaved — two threads stepping the
same motor will lose position.

**Refuse to move until homed.** After a service restart the encoder count is
meaningless. Physically place the rod at home, then send `ZERO` once. Until
that happens, `PUSH` must be rejected — reply `ERR not_homed` and stay still.

> This is the failure mode that will confuse you, because the runtime never
> reads the reply. The symptom is: you click, nothing moves, no error anywhere.
> If the shelf goes dead after a reboot, this is why. Check the Pi's log.

**Persist the goal.** `SETGOAL` should survive a restart — write it to disk.
Re-teaching the throw every reboot gets old fast.

**Bound the travel.** Clamp to a maximum step count in software. A runaway
lead screw will happily destroy the mechanism, and the runtime has no idea
where your rod is.

## Reference sketch

Not tested against your hardware — the motor calls are where your driver goes.

```python
import json, socket, threading
from pathlib import Path

HOME, GOAL_FILE = 0, Path("goal.json")
lock = threading.Lock()
position, homed = 0, False
goal = json.loads(GOAL_FILE.read_text())["goal"] if GOAL_FILE.exists() else 2000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5007))

while True:
    data, addr = sock.recvfrom(64)
    cmd = data.decode("ascii", "replace").strip().upper()

    with lock:
        if cmd == "ZERO":
            position, homed = HOME, True
            sock.sendto(b"OK homed", addr)

        elif cmd == "SETGOAL":
            goal = position
            GOAL_FILE.write_text(json.dumps({"goal": goal}))
            sock.sendto(f"OK goal={goal}".encode(), addr)

        elif cmd == "PUSH":
            if not homed:
                sock.sendto(b"ERR not_homed", addr)   # the runtime won't read
                continue                              # this — check your logs
            move_to(goal)                             # <- your driver
            move_to(HOME)                             # <- your driver
            position = HOME
            sock.sendto(b"OK", addr)

        else:
            sock.sendto(b"ERR unknown", addr)
```

Run it under systemd with `Restart=always`, and remember that every restart
un-homes it.

## Testing without the runtime

```bash
echo -n ZERO | nc -u -w1 bookshelf-pi.local 5007
echo -n PUSH | nc -u -w1 bookshelf-pi.local 5007
```

Then point `data/devices.yaml` at the Pi and click the polygon.
