# Running Evals

Follow these steps in order. Every command is copy-paste ready.

## One-time setup (skip if done)

1. Get SSH access: someone with access adds your `~/.ssh/id_ed25519.pub`
   line to the node's `~/.ssh/authorized_keys`.
2. Get a free Hugging Face token (huggingface.co/settings/tokens) and
   accept the terms at huggingface.co/datasets/Idavidrein/gpqa. Then:

   ```bash
   uvx --from "huggingface_hub[cli]" hf auth login --token hf_YOUR_TOKEN
   ```

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
  --tasks gpqa_diamond_cot_zeroshot \
  --apply_chat_template \
  --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=8192" \
  --limit 5 --log_samples --output_path results/raw/gpqa-pilot/
```

Skim the `samples_*.jsonl` it writes: answers should end with a clear
choice and the score fields should not be empty.

**4. Full run** (198 questions, tens of minutes):

```bash
uvx --from "lm-eval[api]" lm_eval \
  --model local-chat-completions \
  --model_args "model=qwen3.5-9b,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8" \
  --tasks gpqa_diamond_cot_zeroshot \
  --apply_chat_template \
  --gen_kwargs "temperature=0.6,top_p=0.95,max_gen_toks=32768" \
  --log_samples --output_path results/raw/gpqa-$(date +%Y-%m-%d)/
```

The score prints at the end.

**5. Record it:** add a row to `results/summary.md` with the score,
settings, and date.

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
| 403 downloading GPQA | HF terms not accepted or token missing - redo one-time setup step 2 |
| `Loglikelihood is not supported` | wrong task variant - use the `_cot_zeroshot` (generative) task names |
