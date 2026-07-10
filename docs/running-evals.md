# Running Evals

Follow these steps in order. Every command is copy-paste ready.

## One-time setup (skip if done)

1. Get a free Hugging Face token (huggingface.co/settings/tokens), then:

   ```bash
   uvx --from "huggingface_hub[cli]" hf auth login --token hf_YOUR_TOKEN
   ```

   (AIME is ungated; only some datasets additionally require accepting
   terms on their HF page while logged in.)

2. Clone this repo and run `uv sync` inside it.
3. From your local checkout, deploy the checked-in systemd unit to an existing
   GPU node:

   ```bash
   ./scripts/deploy-vllm-service.sh ubuntu@192.222.52.206
   ```

   The command verifies or installs vLLM 0.24.0 on the remote node, copies and
   restarts the service over SSH, and waits up to 15 minutes for health. Put
   non-default keys, ports, or jump hosts in your SSH configuration. The unit
   starts Qwen3.5-9B at the pinned revision with BF16, tensor parallel size 1,
   and a 262,144-token context limit.

   For a truly fresh node that also needs cache and Docker setup, clone the
   repository there and run `bash scripts/provision-lambda.sh` on that node
   instead.

4. The deployment command performs the health check. To inspect the pinned
   context independently, run:

   ```bash
   ssh ubuntu@192.222.52.206 \
     "journalctl -u vllm -b --no-pager | grep 'Using max model len 262144'"
   ```

## Where the runner can execute

The harness can run on your laptop, a Lambda CPU node, or the GPU node. It
only sends HTTP requests and grades the returned text; it does not need a GPU.

The checked-in service intentionally listens only on the GPU node's loopback
interface. Therefore:

- on the GPU node, run the evaluator directly against `localhost:8000`;
- on another machine, use an SSH tunnel or an equivalently secured network
  path.

SSH is a transport choice for the current private endpoint, not a requirement
of lm-eval or of this repository.

## Every session

**1. Open the tunnel when running somewhere other than the GPU node:**

```bash
ssh -N -f -L 8000:localhost:8000 ubuntu@192.222.52.206
```

"Address already in use" usually means a tunnel is already open. Skip this
step when the repository is running directly on the GPU node.

**2. Smoke test:**

```bash
uv run scripts/check_endpoint.py
```

Expect: `models endpoint OK` and a `pong` reply. The request deliberately sets
`max_tokens=163840`; the short response proves that the server accepts the
cap, not that it will generate that many tokens.

**3. Pilot run** (first 3 problems × 16 attempts; always do this first):

```bash
uv run scripts/run_eval.py configs/aime.yaml --limit 3
```

Inspect `attempts.jsonl`, `manifest.json`, and `report.md` in the printed run
directory. Verify that each selected question has 16 records, seeds 2026
through 2041, no transport failures, and at least one non-trivial problem has
more than one unique response. The report is labeled `NON-CANONICAL PILOT` and
must not be used in the investor score table.

**4. Full run** (30 problems × 16 attempts = 480 completions):

```bash
uv run scripts/run_eval.py configs/aime.yaml
```

The cap is 163,840 output tokens, not a target. Runtime depends on the actual
thinking lengths and can be hours. The runner writes the raw harness result,
raw attempts, log, resolved config, manifest, and Markdown report under
`results/raw/<benchmark>-<timestamp>/`.

**5. Record it:** only after the manifest validates all 30 questions and 480
attempts, add the canonical `avg@16` row to `results/summary.md`. Copy the
methodology disclosure with the score. Report `pass@16` only as the linked
best-of-16 secondary metric.

## Running a different benchmark

Point the runner at a different config:

```bash
uv run scripts/run_eval.py configs/<benchmark>.yaml
```

The config is the single source of truth: the runner builds the whole
lm-eval command from it (`--dry-run` prints the command without running
it). To change a setting, edit the YAML - never tweak flags ad hoc, or
the scoreboard stops being reproducible. The config format is
documented field by field in the top-level [README](../README.md).

Two practical notes:

- **Finding the exact task name:** the `task:` field must match a task the
  harness knows. List candidates with
  `uv run lm-eval ls tasks | grep -i aime`.
- **When the harness doesn't have the task** (like AIME 2026): define
  it yourself under `tasks/<name>/` - copy the closest built-in task
  YAML, swap the `dataset_path` to a Hugging Face dataset with the
  right schema, and set `include_path: tasks/<name>` in the benchmark
  config. `tasks/aime26/` is the worked example.
- **Chat endpoints need generative tasks.** Task variants scored by
  token probabilities (plain multiple-choice, names without `cot`/
  `generative`) fail with `Loglikelihood is not supported`. Check the
  task's output type with `lm-eval ls tasks` (want `generate_until`).

For benchmarks whose config declares a different `harness:` (e.g.
`terminal-bench`), the runner refuses on purpose - those are Phase 3
tools with their own CLIs, run on the GPU node; see
[eval-setup-plan.md](eval-setup-plan.md).

## Evaluating other weights

The canonical AIME command intentionally rejects a different model identity,
revision, context, or serving topology. Do not overwrite those fields and
reuse the investor protocol label for a LoRA or another checkpoint. Define a
separate non-canonical benchmark configuration and disclosure before comparing
other weights.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | tunnel not open - do step 1 |
| `Connection reset by peer` | vLLM still starting or crashed - `ssh ubuntu@192.222.52.206 'journalctl -u vllm -n 30'`; first start after a change can take minutes |
| reply contains `Thinking Process` / `</think>` | server missing `--reasoning-parser qwen3` - fix the checked-in unit and redeploy it |
| server rejects `max_tokens=163840` | checked-in service has not been deployed, or startup did not use `--max-model-len 262144` |
| manifest reports missing response metadata | do not publish the score; confirm the tracked adapter is selected in the dry-run command |
| 403 downloading a dataset | it's gated - log in on HF, accept the terms on its dataset page |
| `Loglikelihood is not supported` | wrong task variant - use the `_cot_zeroshot` (generative) task names |
