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

**3. Pilot run** (5 questions, a few minutes - always do this first):

```bash
uvx --from "lm-eval[api]" lm_eval \
  --model local-chat-completions \
  --model_args "model=qwen3.5-9b,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=4" \
  --tasks aime26 --include_path tasks/aime26 \
  --apply_chat_template \
  --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=32768" \
  --limit 3 --log_samples --output_path results/raw/aime-pilot/
```

Skim the `samples_*.jsonl` it writes: answers should end with a clear
choice and the score fields should not be empty.

**4. Full run** (AIME is 30 problems; expect tens of minutes - hard
problems think for a long time):

```bash
uvx --from "lm-eval[api]" lm_eval \
  --model local-chat-completions \
  --model_args "model=qwen3.5-9b,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8" \
  --tasks aime26 --include_path tasks/aime26 \
  --apply_chat_template \
  --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=32768" \
  --log_samples --output_path results/raw/aime-$(date +%Y-%m-%d)/
```

The score prints at the end.

**5. Record it:** add a row to `results/summary.md` with the score,
settings, and date.

## Running a different benchmark

The command never changes shape; only the values do, and every value
comes from that benchmark's YAML in `configs/`. The mapping:

| In the command | Comes from |
|---|---|
| `model=...` and `base_url=...` in `--model_args` | `configs/endpoint.yaml` |
| `num_concurrent=...` in `--model_args` | `num_concurrent` in the benchmark YAML |
| `--tasks ...` | `task` in the benchmark YAML |
| `--gen_kwargs "temperature=...,top_p=...,max_gen_toks=..."` | the `generation:` block |
| `--output_path results/raw/<benchmark>-<date>/` | the `benchmark` name |

So to run another benchmark: open its YAML under `configs/`, and swap
the values in - task name, generation block, new output path. That's
the whole procedure. The YAMLs exist so that two people running
"the AIME eval" a month apart use identical settings and their numbers
compare fairly; if you change a setting, change it in the YAML first.

Two practical notes:

- **Finding the exact task name:** the `task:` field must match a task
  the harness knows. List candidates with
  `uvx --from "lm-eval[api]" lm-eval ls tasks | grep -i aime`.
- **When the harness doesn't have the task** (like AIME 2026): define
  it yourself under `tasks/<name>/` - copy the closest built-in task
  YAML, swap the `dataset_path` to a Hugging Face dataset with the
  right schema, and pass `--include_path tasks/<name>` when running.
  `tasks/aime26/` is the worked example.
- **Chat endpoints need generative tasks.** Task variants scored by
  token probabilities (plain multiple-choice, names without `cot`/
  `generative`) fail with `Loglikelihood is not supported`. Check the
  task's output type with `lm-eval ls tasks` (want `generate_until`).

For benchmarks with a different `harness:` in their YAML (e.g.
`terminal-bench`), the lm_eval command doesn't apply at all - those are
Phase 3 tools with their own CLIs, run on the GPU node; see
[eval-setup-plan.md](eval-setup-plan.md).

## Evaluating your own weights (e.g. a LoRA)

1. Copy them up: `rsync -avz ./my-lora/ ubuntu@192.222.52.206:checkpoints/my-lora/`
2. On the node, add to the ExecStart block of
   `/etc/systemd/system/vllm.service`:
   `--enable-lora --lora-modules my-lora=/home/ubuntu/checkpoints/my-lora \`
   then `sudo systemctl daemon-reload && sudo systemctl restart vllm`.
3. Rerun steps 2-5 with `model=my-lora` in `--model_args` (and in
   `configs/endpoint.yaml`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | tunnel not open - do step 1 |
| `Connection reset by peer` | vLLM still starting or crashed - `ssh ubuntu@192.222.52.206 'journalctl -u vllm -n 30'`; first start after a change can take minutes |
| reply contains `Thinking Process` / `</think>` | server missing `--reasoning-parser qwen3` - fix the unit, restart vllm |
| 403 downloading a dataset | it's gated - log in on HF, accept the terms on its dataset page |
| `Loglikelihood is not supported` | wrong task variant - use the `_cot_zeroshot` (generative) task names |
