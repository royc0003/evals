#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: scripts/deploy-vllm-service.sh ubuntu@HOST" >&2
}

if [[ $# -ne 1 || -z ${1:-} || $1 == -* ]]; then
  usage
  exit 2
fi

TARGET=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERVICE_PATH="$SCRIPT_DIR/vllm.service"
REMOTE_UNIT="/tmp/evals-vllm.service.$$.service"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

require_command ssh
require_command scp

if [[ ! -f $SERVICE_PATH ]]; then
  echo "Service file not found: $SERVICE_PATH" >&2
  exit 1
fi

copied=false

cleanup_remote_unit() {
  if [[ $copied == true ]]; then
    ssh "$TARGET" rm -f -- "$REMOTE_UNIT" >/dev/null 2>&1 || true
  fi
}

trap cleanup_remote_unit EXIT

scp "$SERVICE_PATH" "$TARGET:$REMOTE_UNIT"
copied=true

ssh "$TARGET" bash -s -- "$REMOTE_UNIT" <<'REMOTE_SCRIPT'
set -euo pipefail

VLLM_VERSION=0.24.0
HEALTH_ATTEMPTS=180
HEALTH_INTERVAL_SECONDS=5
REMOTE_UNIT=$1
VENV="$HOME/vllm-env"

cleanup() {
  rm -f -- "$REMOTE_UNIT"
}

trap cleanup EXIT

if [[ $(id -un) != ubuntu ]]; then
  echo "Run this deployment as the remote ubuntu user." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Required remote command not found: curl" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv installation did not produce an executable." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$VENV"
fi

installed_version=$(
  "$VENV/bin/python" -c \
    'from importlib.metadata import version; print(version("vllm"))' \
    2>/dev/null || true
)

if [[ $installed_version != "$VLLM_VERSION" ]]; then
  VIRTUAL_ENV="$VENV" "$UV_BIN" pip install --upgrade \
    "vllm==${VLLM_VERSION}"
fi

installed_version=$(
  "$VENV/bin/python" -c \
    'from importlib.metadata import version; print(version("vllm"))'
)

if [[ $installed_version != "$VLLM_VERSION" ]]; then
  echo "Expected vLLM $VLLM_VERSION, found $installed_version." >&2
  exit 1
fi

sudo install -m 0644 "$REMOTE_UNIT" /etc/systemd/system/vllm.service
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl restart vllm

healthy=false
for ((attempt = 1; attempt <= HEALTH_ATTEMPTS; attempt++)); do
  if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
    healthy=true
    break
  fi

  if ((attempt < HEALTH_ATTEMPTS)); then
    sleep "$HEALTH_INTERVAL_SECONDS"
  fi
done

if [[ $healthy != true ]]; then
  echo "vLLM did not become healthy within 15 minutes." >&2
  sudo systemctl status vllm --no-pager || true
  sudo journalctl -u vllm -n 100 --no-pager || true
  exit 1
fi

printf 'vLLM %s is healthy.\n' "$installed_version"
sudo systemctl is-active vllm
REMOTE_SCRIPT

copied=false
trap - EXIT

echo "vLLM service deployed successfully to $TARGET."
