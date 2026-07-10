#!/usr/bin/env bash
# Provision a fresh Lambda instance for the eval pipeline.
# Idempotent: safe to re-run. See docs/lambda-hosting.md.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STORAGE=~/evals-storage

# Redirect caches to persistent storage so model downloads and Docker
# images survive instance termination.
mkdir -p "$STORAGE/hf-cache" "$STORAGE/results" "$STORAGE/uv-cache"
grep -q HF_HOME ~/.bashrc || cat >> ~/.bashrc <<EOF
export HF_HOME=$STORAGE/hf-cache
export UV_CACHE_DIR=$STORAGE/uv-cache
EOF

# uv + vLLM
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/vllm-env
VIRTUAL_ENV=~/vllm-env uv pip install "vllm==0.24.0"

sudo install -m 0644 "$SCRIPT_DIR/vllm.service" \
  /etc/systemd/system/vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now vllm

# Docker is preinstalled on Lambda images; move its data dir onto
# persistent storage so SWE-bench / terminal-bench images survive.
sudo mkdir -p "$STORAGE/docker"
echo '{"data-root": "'"$STORAGE"'/docker"}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
sudo usermod -aG docker ubuntu

echo "provisioning done: log out and back in for the docker group"
