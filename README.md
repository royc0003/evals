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
│   ├── check_endpoint.py     smoke test: is the endpoint answering?
│   ├── provision-lambda.sh   set up a fresh Lambda GPU node
│   └── vllm.service          systemd unit that runs the vLLM server
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

## How evals are configured

Two kinds of YAML under `configs/`, both plain values with no logic:

**`endpoint.yaml` - where the model is.** The only file you touch to
evaluate a different model:

```yaml
base_url: http://localhost:8000/v1   # the endpoint (via SSH tunnel)
model: qwen3.5-9b                    # served model name (or your LoRA name)
```

**One file per benchmark - how that eval runs.** Each records
everything needed to reproduce and compare runs fairly:

```yaml
benchmark: gpqa-diamond
harness: lm-eval-harness         # which tool runs it (+ version, at first run)
task: gpqa_diamond_cot_zeroshot  # exact task name inside the harness
generation:                      # sampling settings passed via --gen_kwargs
  temperature: 0.6
  top_p: 0.95
  max_gen_toks: 32768
num_concurrent: 8                # parallel requests against the endpoint
reference:
  glm_5_2_reported: 91.2         # the number we compare against
notes: >-                        # methodology deviations, gotchas
  Rule-based answer extraction; gated HF dataset needs accepted terms.
```

To add a new benchmark: copy the closest existing YAML, change
`benchmark`, `task`, and `reference`, and note anything unusual in
`notes`. The values map straight onto harness flags (`--tasks`,
`--gen_kwargs`, `num_concurrent` in `--model_args`) as shown in the
runbook. The YAMLs are the source of truth; the commands quote them.

## Everything else

Docs index: [docs/README.md](docs/README.md). Scores and comparison
against GLM-5.2: [results/summary.md](results/summary.md).
