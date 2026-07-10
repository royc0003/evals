# evals

Benchmark pipeline for models behind OpenAI-compatible endpoints.
First target: Qwen3.5 9B served by vLLM on a Lambda GPU node, measured
against the reproducible subset of the GLM-5.2 release benchmarks.

To run an eval right now: [docs/running-evals.md](docs/running-evals.md).

## Folder structure

```
evals/
├── README.md            you are here
├── pyproject.toml       uv project; ruff + mypy strict config
├── evals/               tracked vLLM adapter + report generation
├── configs/             one YAML per benchmark + the endpoint config
│   ├── endpoint.yaml    where the model lives (base_url + model name)
│   ├── endpoint-finetuned.example.yaml copyable fine-tuned model profile
│   ├── gpqa-diamond.yaml
│   ├── aime.yaml
│   └── terminal-bench.yaml
├── scripts/
│   ├── run_eval.py           THE entry point: runs an eval from its config
│   ├── check_endpoint.py     smoke test: is the endpoint answering?
│   ├── deploy-vllm-service.sh update vLLM on an existing node over SSH
│   ├── provision-lambda.sh   set up a fresh Lambda GPU node
│   └── vllm.service          systemd unit that runs the vLLM server
├── tasks/               repo-local task definitions the harness lacks
│   └── aime26/          AIME 2026 (used via --include_path)
├── tests/               parser, adapter, runner, and reporting contracts
├── results/
│   ├── summary.md       the scoreboard (tracked in git)
│   └── raw/             full harness outputs (gitignored, bulky)
└── docs/
    ├── README.md                     index of all docs
    ├── running-evals.md              the runbook - start here
    ├── eval-setup-plan.md            the phased plan for the pipeline
    ├── lambda-hosting.md             GPU node setup and hosting
    └── glm-5.2-benchmark-research.md what GLM-5.2 reported, and how
```

## Running an eval

From a local checkout, deploy or update the checked-in service on an existing
GPU node with:

```bash
./scripts/deploy-vllm-service.sh ubuntu@192.222.52.206
```

The command verifies or installs vLLM 0.24.0, copies the systemd unit over
SSH, restarts it, and waits up to 15 minutes for the endpoint to become
healthy. Put non-default keys, ports, or jump hosts in your SSH configuration.
Use `scripts/provision-lambda.sh` on the GPU node only for fresh-node setup.

Every eval runs from its config file - no hand-assembled commands:

```bash
uv run scripts/run_eval.py configs/aime.yaml --limit 3   # pilot (first 3)
uv run scripts/run_eval.py configs/aime.yaml             # full run
uv run scripts/run_eval.py configs/aime.yaml --dry-run   # show the command
```

The runner reads the selected endpoint YAML plus the benchmark YAML, builds
the lm-eval command, and writes output to
`results/raw/<benchmark>-<timestamp>/` automatically (pilots get
`-pilot-` in the name). You never specify an output path.

While a run is in flight the runner shows a spinner with elapsed time. Every
completed canonical AIME run directory contains the harness output, all raw
attempts, `run.log`, `resolved-config.yaml`, `manifest.json`, and `report.md`.

The runner can execute on a laptop, a CPU node, or the GPU node. It does not
need SSH. The checked-in vLLM service binds to `127.0.0.1`, so a client on a
different machine needs an SSH tunnel (or another explicitly secured network
path). A runner on the GPU node uses the endpoint directly.

## How evals are configured

Two kinds of YAML under `configs/`; the runner consumes both.

**`endpoint.yaml` - where the model is.** It is the default endpoint. For the
canonical AIME run these values stay pinned; fine-tuned models use a separate
endpoint YAML and must be labeled non-canonical:

```yaml
evaluation:
  label: qwen3.5-9b
  canonical: true
base_url: http://localhost:8000/v1
model: qwen3.5-9b
source_model: Qwen/Qwen3.5-9B
model_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

**One file per benchmark - how that eval runs.** Field by field:

```yaml
benchmark: aime            # names the run and its results directory
harness: lm-eval-harness   # must be lm-eval-harness for the runner;
                           # agentic harnesses run on the GPU node instead
task: aime26               # exact task name inside lm-eval
include_path: tasks/aime26 # optional: repo-local task definition dir
system_instruction: |-     # the system prompt sent to every problem
  Respond using exactly these fields:
  Explanation: <your reasoning>
  Exact Answer: <your final integer>
  Confidence: <a percentage from 0% through 100%>
generation:
  temperature: 1.0
  top_p: 0.95
  max_gen_toks: 163840
attempts_per_problem: 16
num_concurrent: 8
base_seed: 2026
max_retries: 3             # after the initial HTTP request
grading: rule_based_integer
reference:
  glm_5_2_reported: 99.2   # the number we compare against (not used by
                           # the runner; provenance for the scoreboard)
notes: >-
  No LLM judge; deterministic integer grading reports avg@16 and pass@16.
```

To add a new benchmark: copy the closest existing YAML, change
`benchmark`, `task`, and `reference`, and run it. If lm-eval doesn't
ship the task, define it under `tasks/<name>/` and point
`include_path` at it (see `tasks/aime26/` for the worked example).
The config is the single source of truth: change settings there, never
in ad-hoc commands, so every number in `results/summary.md` is
reproducible from the YAML that produced it.

## Evaluating a fine-tuned model

You can evaluate any fine-tuned chat model that vLLM can expose through its
OpenAI-compatible API. The model may be complete merged weights or a LoRA
adapter. Keep `configs/aime.yaml` unchanged so the base and fine-tuned model
use the same questions, prompt, 16 attempts, seeds, sampling, token cap, and
rule-based grader.

### 1. Serve the fine-tuned model

Run vLLM on the GPU node. Port 8000 must be free; stop the canonical service
first if it is currently running:

```bash
sudo systemctl stop vllm
```

For complete or merged model weights:

```bash
/home/ubuntu/vllm-env/bin/vllm serve \
  /home/ubuntu/models/my-merged-model \
  --served-model-name my-finetuned-model \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --max-model-len 262144 \
  --port 8000
```

For a LoRA adapter:

```bash
/home/ubuntu/vllm-env/bin/vllm serve Qwen/Qwen3.5-9B \
  --revision BASE_MODEL_REVISION \
  --enable-lora \
  --lora-modules \
    my-finetuned-model=/home/ubuntu/adapters/my-adapter \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --max-model-len 262144 \
  --port 8000
```

These examples use Qwen's `qwen3` reasoning parser. For another model family,
use that family's supported vLLM parser or omit `--reasoning-parser`. Keep the
vLLM process in the foreground for an initial test; use your own systemd unit
or service manager after the command works. The checked-in
`deploy-vllm-service.sh` remains specific to the canonical base Qwen service.

On the GPU node, verify the server and find the exact model ID:

```bash
curl --fail http://127.0.0.1:8000/version
curl --fail http://127.0.0.1:8000/v1/models
```

The name in the endpoint YAML's `model` field must exactly match an `id`
returned by `/v1/models`.

### 2. Connect from the evaluation machine

If the evaluator is not running on the GPU node, open the private endpoint as
local port 8000:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -L 8000:127.0.0.1:8000 \
  ubuntu@GPU_NODE
```

Keep that terminal open. In a second terminal, confirm that the endpoint is
reachable through the tunnel:

```bash
curl --fail http://127.0.0.1:8000/version
curl --fail http://127.0.0.1:8000/v1/models
```

### 3. Create one endpoint YAML

Copy the example; do not edit the canonical endpoint:

```bash
cp configs/endpoint-finetuned.example.yaml \
  configs/endpoint-my-model.yaml
```

Edit the copy:

```yaml
evaluation:
  # Used in result directory names; lowercase and filesystem-safe.
  label: my-finetuned-model
  canonical: false

base_url: http://localhost:8000/v1

# Exact ID returned by GET /v1/models.
model: my-finetuned-model

# Immutable identity of the final model or adapter.
source_model: my-org/my-finetuned-model
model_revision: CHECKPOINT_COMMIT_OR_SHA256

fine_tuning:
  type: lora  # Use merged for complete model weights.
  base_model: Qwen/Qwen3.5-9B
  base_revision: BASE_MODEL_COMMIT
  artifact: /home/ubuntu/adapters/my-adapter
  training_data_disclosure: >-
    Describe the training sources and whether AIME 2026 questions or
    solutions were included.

serving:
  expected_vllm_version: 0.24.0
  gpu: H100 80GB
  gpus_used: 1
  dtype: bfloat16
  tensor_parallel_size: 1
  max_model_len: 262144
```

Normally you change `evaluation.label`, `model`, `source_model`,
`model_revision`, every field under `fine_tuning`, and any serving fields that
differ from your deployment. The runner fails before generation if required
provenance is missing, the label is unsafe, or `type` is not `lora` or
`merged`.

### 4. Dry-run, pilot, then evaluate

First inspect the generated harness command without contacting the server:

```bash
uv run scripts/run_eval.py configs/aime.yaml \
  --endpoint-config configs/endpoint-my-model.yaml \
  --limit 3 \
  --dry-run
```

Run the three-question, 48-completion pilot:

```bash
uv run scripts/run_eval.py configs/aime.yaml \
  --endpoint-config configs/endpoint-my-model.yaml \
  --limit 3
```

Inspect `attempts.jsonl`, `manifest.json`, and `report.md` in:

```text
results/raw/aime-my-finetuned-model-pilot-<timestamp>/
```

Only after the pilot has 16 attempts per question, no unexplained failures,
and response variation, run all 30 questions and 480 completions:

```bash
uv run scripts/run_eval.py configs/aime.yaml \
  --endpoint-config configs/endpoint-my-model.yaml
```

The full result is stored in:

```text
results/raw/aime-my-finetuned-model-<timestamp>/
```

Its report is labeled `NON-CANONICAL MODEL COMPARISON`. Compare its primary
`avg@16` and secondary `pass@16` against a base-model run made with the same
benchmark configuration, vLLM version, hardware, and serving precision.

Always disclose the base revision, fine-tuning type, artifact revision or
hash, hardware, training sources, and any AIME 2026 overlap. If AIME 2026
questions or solutions were used during training, do not present the result as
an independent generalization score.

Never serve fine-tuned weights under the canonical `qwen3.5-9b` identity while
leaving canonical metadata unchanged. That would incorrectly attribute the
fine-tuned model's output to the base checkpoint.

## Where the AIME harness and prompts live

- [`scripts/run_eval.py`](scripts/run_eval.py) is the single runner.
- [`scripts/lm_eval_entrypoint.py`](scripts/lm_eval_entrypoint.py) registers
  the repository adapter and delegates to pinned lm-eval 0.4.12.
- [`evals/vllm_adapter.py`](evals/vllm_adapter.py) assigns distinct seeds and
  retains vLLM `finish_reason` and completion-token usage.
- [`configs/aime.yaml`](configs/aime.yaml) contains the system prompt and
  request settings.
- [`tasks/aime26/aime26.yaml`](tasks/aime26/aime26.yaml) supplies the AIME
  problem as the user message. It never puts the answer in the prompt.
- [`tasks/aime26/utils.py`](tasks/aime26/utils.py) performs rule-based grading
  and explicitly reports `avg_at_16`, `pass_at_16`, and the attempt count.

`avg@16` is the fraction correct across all 480 completions and is the primary
score. `pass@16` is the fraction of questions solved by at least one of 16
attempts; it is an oracle best-of-16 diagnostic, not single-attempt accuracy.

## Everything else

Docs index: [docs/README.md](docs/README.md). Scores and comparison
against GLM-5.2: [results/summary.md](results/summary.md).
