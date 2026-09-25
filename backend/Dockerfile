# One-command, no-hardware sandbox:
#
#     docker compose up          →  http://localhost:8000/
#
# Runs the mock demo (devices.dev.yaml) with a seeded example scene. Nothing
# is installed on the host — uv, Python and every dependency live in the image.
# Real hardware wants the server on the HOST instead: Kasa discovery uses LAN
# broadcasts that Docker's NAT doesn't carry.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# curl: AGENTS.md's smoke checks (and the healthcheck below) use it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Unbuffered stdout so the boot banner and the [mock] click lines reach
# `docker compose logs` as they happen, not when the buffer flushes.
# RUNTIME_IN_CONTAINER lets the server's access-log filter drop the
# healthcheck's loopback requests without hiding host-side smoke checks.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    RUNTIME_IN_CONTAINER=1

WORKDIR /app

# Dependency layer first, so source edits don't re-resolve the tree.
COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p src/runtime && touch src/runtime/__init__.py \
 && uv sync --frozen

COPY . .
RUN uv sync --frozen

EXPOSE 8000

# /api/editor/scene is AGENTS.md's own canvas-geometry smoke check, so a green
# healthcheck means exactly what the documented check means.
HEALTHCHECK --interval=10s --timeout=4s --start-period=5s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/editor/scene > /dev/null || exit 1

ENTRYPOINT ["./docker/entrypoint.sh"]
