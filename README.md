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

The runner reads `configs/endpoint.yaml` plus the benchmark YAML,
builds the lm-eval command, and writes output to
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

**`endpoint.yaml` - where the model is.** For the canonical AIME run these
values stay pinned; exploratory runs must be labeled non-canonical:

```yaml
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
