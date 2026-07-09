# Running Evals

Follow these steps in order. Every command is copy-paste ready.

## One-time setup (skip if done)

1. Get SSH access: someone with access adds your `~/.ssh/id_ed25519.pub`
   line to the node's `~/.ssh/authorized_keys`.
2. Get a free Hugging Face token (huggingface.co/settings/tokens), then:

   ```bash
   uvx --from "huggingface_hub[cli]" hf auth login --token hf_YOUR_TOKEN
   ```

   (AIME is ungated; only some datasets additionally require accepting
   terms on their HF page while logged in.)

3. Clone this repo and run `uv sync` inside it.

## Every session

**1. Open the tunnel** (makes the GPU node's endpoint appear on your laptop):

```bash
ssh -N -f -L 8000:localhost:8000 ubuntu@192.222.52.206
```

"Address already in use" = a tunnel is already open. That's fine, move on.

**2. Smoke test:**

```bash
uv run scripts/check_endpoint.py
```

Expect: `models endpoint OK` and a `pong` reply. If it fails, see
Troubleshooting below.

**3. Pilot run** (first 3 problems, a few minutes - always do this first):

```bash
uv run scripts/run_eval.py configs/aime.yaml --limit 3
```

Skim the `samples_*.jsonl` under the printed output path: answers
should contain `\boxed{...}` and the score fields should not all be
zero. A 0-score pilot usually means an extraction problem, not a bad
model - read the samples before believing any number.

**4. Full run** (AIME is 30 problems; expect tens of minutes - hard
problems think for a long time):

```bash
uv run scripts/run_eval.py configs/aime.yaml
```

The score prints at the end; results land in
`results/raw/<benchmark>-<timestamp>/` automatically.

**5. Record it:** add a row to `results/summary.md` with the score,
settings, and date.

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

- **Finding the exact task name:** the `task:` field must match a task
  the harness knows. List candidates with
  `uvx --from "lm-eval[api]" lm-eval ls tasks | grep -i aime`.
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

## Evaluating your own weights (e.g. a LoRA)

1. Copy them up: `rsync -avz ./my-lora/ ubuntu@192.222.52.206:checkpoints/my-lora/`
2. On the node, add to the ExecStart block of
   `/etc/systemd/system/vllm.service`:
   `--enable-lora --lora-modules my-lora=/home/ubuntu/checkpoints/my-lora \`
   then `sudo systemctl daemon-reload && sudo systemctl restart vllm`.
3. Set `model: my-lora` in `configs/endpoint.yaml`, then rerun
   steps 2-5 unchanged - the runner picks the new name up from there.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | tunnel not open - do step 1 |
| `Connection reset by peer` | vLLM still starting or crashed - `ssh ubuntu@192.222.52.206 'journalctl -u vllm -n 30'`; first start after a change can take minutes |
| reply contains `Thinking Process` / `</think>` | server missing `--reasoning-parser qwen3` - fix the unit, restart vllm |
| 403 downloading a dataset | it's gated - log in on HF, accept the terms on its dataset page |
| `Loglikelihood is not supported` | wrong task variant - use the `_cot_zeroshot` (generative) task names |
