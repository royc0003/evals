# Hosting the Eval Stack on Lambda GPU Cloud

How to host everything on Lambda: the vLLM server for Qwen3.5 9B, and the eval harnesses that run against it. Companion to Phase 1 of [eval-setup-plan.md](eval-setup-plan.md).

Reference docs: [Lambda on-demand instances](https://docs.lambda.ai/public-cloud/on-demand/), [filesystems](https://docs.lambda.ai/public-cloud/filesystems/), [firewalls](https://docs.lambda.ai/public-cloud/firewalls/), [connecting](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/).

## The one Lambda gotcha to know first

Lambda instances cannot be stopped and resumed; they can only be **terminated**, which wipes the root disk. Anything you want to survive between eval sessions (model weights cache, Docker images, results) must live on a **persistent filesystem**, which you can only attach **at launch time** and which must be in the same region as the instance. So the order of operations matters: create the filesystem first, always launch instances with it attached, and treat the root disk as disposable.

## 1. One-time account setup

1. In the [Lambda console](https://cloud.lambda.ai): add your SSH public key (SSH keys page), and generate an API key if you want to script launches (API keys page).
2. Create a persistent filesystem (Filesystems page), e.g. `evals-storage`, in the region where you'll run instances. Size for: model weights (~20GB per 9B model in bf16), SWE-bench Docker images (can reach 50-100GB+), and results. 200GB is a comfortable start.
3. Leave the firewall at its default (inbound SSH only). Do **not** add a rule opening port 8000. The vLLM endpoint runs without authentication (no API keys), so the network is the entire security boundary: the server binds to loopback only, the firewall admits nothing but SSH, and remote access goes through an SSH tunnel.

## 2. Launching an instance

For a 9B model, a single-GPU instance is plenty:

- `gpu_1x_a100_sxm4` (40GB) - cheapest adequate option
- `gpu_1x_h100_pcie` or `gpu_1x_gh200` - faster generation, worth it for the long-generation reasoning evals and overnight agentic runs

**Console:** Instances → Launch instance → pick GPU type and region (must match the filesystem's region) → attach `evals-storage` → select your SSH key.

**API (scriptable relaunch):**

```bash
curl -s https://cloud.lambda.ai/api/v1/instance-operations/launch \
  -H "Authorization: Bearer $LAMBDA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "region_name": "us-east-1",
    "instance_type_name": "gpu_1x_a100_sxm4",
    "ssh_key_names": ["my-key"],
    "file_system_names": ["evals-storage"],
    "name": "evals-qwen35-9b"
  }'
```

List running instances (to get the IP) with `GET https://cloud.lambda.ai/api/v1/instances`, and terminate with `POST .../instance-operations/terminate`. Check the [API browser](https://docs.lambda.ai/public-cloud/on-demand/) for current instance type names (`GET /api/v1/instance-types` also shows live availability per region).

Connect: `ssh ubuntu@<instance-ip>`. Instances run Ubuntu with Lambda Stack preinstalled (NVIDIA drivers, CUDA); verify with `nvidia-smi`. The filesystem mounts at `/home/ubuntu/<filesystem-name>` (e.g. `~/evals-storage`).

## 3. Instance provisioning (repeatable after every relaunch)

Because the root disk is wiped on terminate, keep provisioning as one idempotent script, checked into this repo as `scripts/provision-lambda.sh`, and point all caches at the persistent filesystem:

```bash
#!/usr/bin/env bash
set -euo pipefail
STORAGE=~/evals-storage

# Redirect Hugging Face cache to persistent storage (model downloads survive relaunch)
mkdir -p "$STORAGE/hf-cache" "$STORAGE/results" "$STORAGE/uv-cache"
grep -q HF_HOME ~/.bashrc || cat >> ~/.bashrc <<EOF
export HF_HOME=$STORAGE/hf-cache
export UV_CACHE_DIR=$STORAGE/uv-cache
EOF

# uv + vLLM
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/vllm-env
VIRTUAL_ENV=~/vllm-env uv pip install vllm

# Docker is preinstalled on Lambda images; move its data dir to persistent storage
# so SWE-bench / terminal-bench images survive relaunch
sudo mkdir -p "$STORAGE/docker"
echo '{"data-root": "'"$STORAGE"'/docker"}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
sudo usermod -aG docker ubuntu
```

First launch downloads the model into `$HF_HOME`; every relaunch after that serves from cache in seconds.

## 4. Running the vLLM server as a service

Use a systemd unit rather than a bare terminal so the server survives SSH disconnects and restarts on crash (keep this in the repo as `scripts/vllm.service`):

```ini
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM OpenAI-compatible server (Qwen3.5 9B)
After=network.target

[Service]
User=ubuntu
Environment=HF_HOME=/home/ubuntu/evals-storage/hf-cache
ExecStart=/home/ubuntu/vllm-env/bin/vllm serve Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp scripts/vllm.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now vllm
journalctl -u vllm -f        # watch startup + request logs
```

(`tmux` + `vllm serve ...` works too for quick sessions; systemd is for overnight runs.)

## 5. Where the harnesses run

- **Tier 1 (lm-eval-harness):** either place. From your laptop, tunnel first: `ssh -N -L 8000:localhost:8000 ubuntu@<instance-ip>`, then use `base_url=http://localhost:8000/v1/...`. On-instance, no tunnel needed.
- **Tier 2 (terminal-bench, mini-swe-agent, OpenHands):** run **on the instance**, inside `tmux`. These make thousands of requests and need Docker; the instance has both the endpoint and Docker locally, and the run keeps going after you disconnect. Write outputs to `~/evals-storage/results/` and `rsync` them back to this repo's `results/` when done:

```bash
rsync -avz ubuntu@<instance-ip>:evals-storage/results/ ./results/
```

## 6. Cost hygiene

- GPU billing is per-hour while the instance exists, idle or not. Since there is no stop/resume, the workflow is: launch → provision (scripted, ~10 min) → run eval campaign → `rsync` results off → **terminate**. The persistent filesystem (billed cheaply per GB/month) carries all state between campaigns.
- Batch work so the GPU is always doing something: kick off overnight Tier 2 runs at the end of a session rather than leaving the box idle.
- Rough rates (check the console for current pricing): 1x A100 ≈ $1.3/hr, 1x H100 ≈ $2.5-3.3/hr. A full eval campaign (Tier 1 + one overnight Tier 2 run) is roughly 12-24 GPU-hours.
