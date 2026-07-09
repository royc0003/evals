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
├── configs/             one YAML per benchmark + the endpoint config
│   ├── endpoint.yaml    where the model lives (base_url + model name)
│   ├── gpqa-diamond.yaml
│   ├── aime.yaml
│   └── terminal-bench.yaml
├── scripts/
│   ├── run_eval.py           THE entry point: runs an eval from its config
│   ├── check_endpoint.py     smoke test: is the endpoint answering?
│   ├── provision-lambda.sh   set up a fresh Lambda GPU node
│   └── vllm.service          systemd unit that runs the vLLM server
├── tasks/               repo-local task definitions the harness lacks
│   └── aime26/          AIME 2026 (used via --include_path)
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

## How evals are configured

Two kinds of YAML under `configs/`; the runner consumes both.

**`endpoint.yaml` - where the model is.** The only file you touch to
evaluate a different model:

```yaml
base_url: http://localhost:8000/v1   # the endpoint (via SSH tunnel)
model: qwen3.5-9b                    # served model name (or your LoRA name)
```

**One file per benchmark - how that eval runs.** Field by field:

```yaml
benchmark: aime            # names the run and its results directory
harness: lm-eval-harness   # must be lm-eval-harness for the runner;
                           # agentic harnesses run on the GPU node instead
task: aime26               # exact task name inside lm-eval
include_path: tasks/aime26 # optional: repo-local task definition dir
generation:                # passed through as --gen_kwargs, verbatim
  temperature: 0.6
  top_p: 0.95
  max_gen_toks: 32768
num_concurrent: 8          # parallel requests (optional, default 4)
reference:
  glm_5_2_reported: 99.2   # the number we compare against (not used by
                           # the runner; provenance for the scoreboard)
notes: >-                  # methodology deviations, gotchas
  No LLM judge; boxed-answer prompt added vs the built-in aime25 task.
```

To add a new benchmark: copy the closest existing YAML, change
`benchmark`, `task`, and `reference`, and run it. If lm-eval doesn't
ship the task, define it under `tasks/<name>/` and point
`include_path` at it (see `tasks/aime26/` for the worked example).
The config is the single source of truth: change settings there, never
in ad-hoc commands, so every number in `results/summary.md` is
reproducible from the YAML that produced it.

## Everything else

Docs index: [docs/README.md](docs/README.md). Scores and comparison
against GLM-5.2: [results/summary.md](results/summary.md).
