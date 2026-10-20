#!/usr/bin/env bash
# Container entrypoint: seed a demo scene when none exists, print where to go,
# then exec the documented start command. Nothing here changes app behaviour.
set -euo pipefail

DEVICES="${RUNTIME_DEVICES:-data/devices.dev.yaml}"
OBJECTS="${RUNTIME_OBJECTS:-data/objects.json}"
HOST="${RUNTIME_HOST:-0.0.0.0}"
PORT="${RUNTIME_PORT:-8000}"
PROJECTOR_SIZE="${RUNTIME_PROJECTOR_SIZE:-1920x1080}"
SEED_EXAMPLE="${SEED_EXAMPLE:-lamp}"

# data/objects.json is per-wall and gitignored, so a fresh clone has no
# polygons and / would be an empty black canvas. Seed a worked example —
# but never overwrite an existing objects.json (AGENTS.md invariant).
if [ "$SEED_EXAMPLE" != "none" ] && [ ! -f "$OBJECTS" ]; then
  SRC="data/examples/objects.${SEED_EXAMPLE}.json"
  if [ -f "$SRC" ]; then
    mkdir -p "$(dirname "$OBJECTS")"
    cp "$SRC" "$OBJECTS"
    echo "[sandbox] seeded $OBJECTS from $SRC."
    echo "[sandbox] its coordinates are the author's wall — redraw in /editor for yours."
  else
    echo "[sandbox] WARN: SEED_EXAMPLE=$SEED_EXAMPLE but $SRC does not exist; starting empty."
  fi
elif [ -f "$OBJECTS" ]; then
  echo "[sandbox] using existing $OBJECTS (left untouched)."
fi

cat <<BANNER
[sandbox] -----------------------------------------------------------------
[sandbox]  interactive-projection-mapping — mock / no-hardware sandbox
[sandbox]
[sandbox]  Open:  http://localhost:${PORT}/                   final scene (fullscreen on the projector)
[sandbox]         http://localhost:${PORT}/editor             your monitor — create + bind polygons
[sandbox]         http://localhost:${PORT}/editor/projection  projector — editable 1:1 twin of the scene
[sandbox]
[sandbox]  Devices: ${DEVICES} — every click lands below as "[mock] ...".
[sandbox]  No real switch, motor or projector is involved. Real hardware
[sandbox]  needs the server on the host — see the README.
[sandbox] -----------------------------------------------------------------
BANNER

# `python -m runtime` is the documented entry point; --host 0.0.0.0 is the one
# container-specific override (loopback would never reach the published port).
# The venv python is exec'd directly — NOT via `uv run` — so Python is PID 1
# and receives docker stop's SIGTERM. uv doesn't forward signals, which would
# skip uvicorn's graceful shutdown and every Device.close().
exec /app/.venv/bin/python -m runtime \
  --host "$HOST" \
  --port "$PORT" \
  --devices "$DEVICES" \
  --objects "$OBJECTS" \
  --projector-size "$PROJECTOR_SIZE" \
  "$@"
