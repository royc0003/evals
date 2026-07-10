# AIME 2026 GLM-Aligned Evaluation: Design

Status: approved on 2026-07-09.

## Goal

Produce an investor-presentable and reproducible AIME 2026 evaluation of
`Qwen/Qwen3.5-9B` served by vLLM. The run should match every disclosed
GLM-5.2 AIME setting that is practical, while using deterministic rule-based
grading instead of an LLM judge and disclosing that difference next to every
headline score.

The canonical result is a full 30-question run with 16 independent attempts
per question. It reports `avg@16` as the primary score and `pass@16` as a
secondary best-of-16 score.

## Fixed decisions

- Benchmark: AIME 2026, all 30 questions from `math-ai/aime26`, pinned to
  dataset revision `79037aebdb6580008fb960d17cb21fd3099083e3`.
- Model: `Qwen/Qwen3.5-9B` at revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, in thinking mode through its
  vLLM OpenAI-compatible chat-completions endpoint.
- Harness: EleutherAI lm-evaluation-harness, pinned to the currently validated
  `0.4.12` release.
- Grading: deterministic integer matching. No LLM judge.
- Tools: none.
- Attempts: 16 independent generations per question, 480 logical attempts in
  total.
- Primary metric: `avg@16`.
- Secondary metric: `pass@16`, always labeled as best-of-16 rather than
  ordinary accuracy.
- Sampling: temperature `1.0` and top-p `0.95`, matching the disclosed GLM
  values and the corresponding values in Qwen's thinking-mode
  recommendations.
- Maximum completion length: 163,840 tokens.
- Maximum total sequence length: 262,144 tokens.
- Serving hardware: one H100 80GB in tensor-parallel size 1. The current node
  contains two H100 80GB GPUs, but the second GPU is not required for this
  9B model and is outside the canonical serving topology.
- Concurrency: eight requests. A pilot must verify that the live server does
  not preempt requests at this setting.
- Each logical attempt permits three transport retries. Retries do not create
  additional logical attempts and are recorded separately.

## Why the long context is feasible

The live vLLM 0.24.0 server currently loads the model in BF16 on one H100. Its
startup log reports 17.66 GiB for model loading, 50.53 GiB available for KV
cache, and capacity for 1,631,118 cached tokens. Eight AIME requests with a
small prompt and a 163,840-token completion cap require about 1.32 million
cached tokens at their theoretical maximum, which fits within that measured
capacity.

`max_model_len` is the total prompt-plus-completion limit. It therefore must
be larger than `max_output_tokens`; the 262,144-token server limit leaves more
than 98,000 tokens for the prompt when the completion cap is 163,840.

The 163,840 value is a cap, not a target. The report must distinguish the
configured cap from the output lengths actually observed.

## GLM alignment and known deviations

| Protocol element | This evaluation | GLM-5.2 disclosure | Status |
|---|---|---|---|
| Benchmark year | AIME 2026 | AIME 2026 | Aligned |
| Response shape | Explanation, Exact Answer, Confidence | Same three-field structure | Aligned in structure |
| Temperature | 1.0 | 1.0 | Aligned |
| Top-p | 0.95 | 0.95 | Aligned |
| Maximum output | 163,840 | 163,840 | Aligned |
| Grading | Rule-based integer match | GPT-5.5-medium judge | Deliberate deviation |
| Aggregation | avg@16 and pass@16 | Not disclosed | Deliberate, disclosed choice |
| Attempts per question | 16 | Not disclosed | Deliberate, disclosed choice |
| Total context | 262,144 | AIME-specific value not disclosed | Disclosed local setting |
| Tools | None | Not disclosed for AIME | Disclosed local setting |

The repository must not describe the resulting score as a reproduction of
GLM-5.2's 99.2. It is a directional comparison under a closely aligned but
not identical protocol.

## Prompt contract

The harness sends a system message that mirrors the response structure
disclosed by GLM:

```text
Respond using exactly these fields:
Explanation: <your reasoning>
Exact Answer: <your final integer>
Confidence: <a percentage from 0% through 100%>
```

The AIME problem is the user message. There are no few-shot examples and the
reference answer is never included in the model input.

The logged sample artifact must preserve the final system and user messages so
the prompt actually sent to vLLM is auditable. The prompt text and its hash are
also included in the run manifest. Each sample also preserves its request
seed, API `finish_reason`, and API `usage.completion_tokens`; these fields are
the source of truth for independence, truncation, and output-length reporting.

## Independent sampling contract

Each problem receives exactly 16 stochastic attempts. Attempt `j` uses a
distinct deterministic seed derived from a recorded base seed, so reruns can
be reproduced without sending the same random request 16 times.

Before a full run, the pilot must demonstrate both of these properties:

1. the raw artifact contains 16 responses for every selected problem; and
2. the request seeds are distinct and at least one non-trivial pilot problem
   produces more than one unique response.

If lm-evaluation-harness cannot pass distinct per-attempt seeds through its
OpenAI-compatible adapter, the implementation must take control of request
generation rather than silently report repeated identical samples as
`avg@16`.

## Rule-based grading

For each response, the grader locates exactly one `Exact Answer:` field and
extracts its value. It accepts an integer from 0 through 999, optionally
surrounded by `\boxed{}`. Leading zeroes are normalized numerically, so `007`
and `7` are equivalent. The extracted integer is compared with the normalized
reference answer.

The explanation and confidence fields are retained for audit but do not affect
mathematical correctness. Their presence is reported separately as format
compliance.

An attempt receives a correctness value of zero when any of the following is
true:

- the request fails after its configured transport retries;
- the response is empty;
- generation ends because it reaches the maximum output length;
- the response has no `Exact Answer:` field or more than one such field;
- the extracted value is not a single integer from 0 through 999; or
- the integer does not match the reference answer.

No failed, empty, invalid, or truncated attempt is removed from the metric
denominator.

## Metrics

Let `c[i][j]` be 1 when attempt `j` for question `i` is correct and 0
otherwise. There are 30 questions and 16 attempts per question.

```text
avg@16  = sum(c[i][j]) / 480
pass@16 = sum(any(c[i][0:16]) for each question i) / 30
```

The harness-facing metric names are `avg_at_16` and `pass_at_16`; reports
render them as `avg@16` and `pass@16`.

`avg@16` estimates expected single-attempt correctness with reduced sampling
variance. `pass@16` answers whether at least one of 16 attempts solved a
problem and is an oracle upper bound. The report does not present `pass@16` as
a deployable single-attempt score. It also does not report a separate
estimated `pass@1`, because that would be numerically the same estimate as
`avg@16` here.

## Reproducible serving configuration

[`scripts/vllm.service`](../../../scripts/vllm.service) is the canonical
serving configuration. It will explicitly set:

```text
Qwen/Qwen3.5-9B
model revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
served model name: qwen3.5-9b
dtype: bfloat16
tensor parallel size: 1
reasoning parser: qwen3
host: 127.0.0.1
port: 8000
max model length: 262144
GPU memory utilization: 0.92
```

[`scripts/provision-lambda.sh`](../../../scripts/provision-lambda.sh) will pin
vLLM to the validated 0.24.0 release, install the checked-in systemd unit,
reload systemd, and enable the service. The committed unit is installed with:

```bash
sudo install -m 0644 "$(dirname "$0")/vllm.service" \
  /etc/systemd/system/vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now vllm
```

The provisioning script resolves `vllm.service` relative to its own location,
so it has the same behavior regardless of the caller's working directory.

After deployment, the runbook verifies `/health` and requires the startup log
to contain `Using max model len 262144`. It also sends a short smoke-test
request with `max_tokens=163840`; the response may stop normally after a few
tokens, but the server must accept the requested cap.

## Benchmark configuration

[`configs/aime.yaml`](../../../configs/aime.yaml) remains the canonical
request-level configuration. It records:

```yaml
generation:
  temperature: 1.0
  top_p: 0.95
  max_gen_toks: 163840
attempts_per_problem: 16
num_concurrent: 8
base_seed: 2026
max_retries: 3
grading: rule_based_integer
```

[`tasks/aime26/aime26.yaml`](../../../tasks/aime26/aime26.yaml) implements the
dataset revision `79037aebdb6580008fb960d17cb21fd3099083e3`, prompt,
16 repeats, and metric declarations. Its Python grader scores every response
in the `results` list; it must not select only `results[0]`.

The runner validates that the benchmark declares the expected attempts,
grading method, and maximum output before launching the canonical full run.
Pilot runs may reduce the number of questions through `--limit`, but they do
not reduce attempts per selected question.

## Run artifacts and reporting

Every run directory under `results/raw/<run-id>/` contains:

- the raw lm-evaluation-harness result JSON;
- the raw sample JSONL with all attempts;
- `run.log`;
- `resolved-config.yaml`, containing the endpoint and benchmark settings used;
- `manifest.json`, containing versions, revisions, hardware, prompt hash,
  attempt counts, metrics, failure counts, and token statistics; and
- `report.md`, a human-readable rendering of the manifest.

The manifest records at least:

- run ID and UTC timestamp;
- repository commit;
- model name and resolved model revision;
- dataset name, split, and resolved dataset revision;
- lm-evaluation-harness 0.4.12 and live vLLM version;
- serving-unit path and hash;
- GPU model, number of GPUs used, dtype, and tensor-parallel size;
- maximum model length and maximum output tokens;
- complete sampling parameters and seed policy;
- prompt text and hash;
- expected, completed, failed, invalid, and truncated attempts;
- total transport retries;
- `avg@16` and `pass@16`;
- mean, median, p95, and maximum observed completion tokens; and
- response-format compliance count.

Observed token statistics come from each API response's
`usage.completion_tokens`. Truncation comes from `finish_reason=length`. A
canonical report is invalid if the generation layer did not retain those
fields.

The tracked [`results/summary.md`](../../../results/summary.md) stays concise.
It contains one headline row per canonical run with these columns:

| Benchmark | Model | Ours | GLM-5.2 reported | Attempts | Grading | Max output | Context | Truncated |
|---|---|---:|---:|---:|---|---:|---:|---:|

The canonical AIME row is written only after the full run validates. It must
identify `avg@16` in the score cell, show 16 attempts per question and 480 in
total, identify rule-based integer grading, show the 163,840 output cap and
262,144 context, and give the exact truncated-attempt count out of 480.

`pass@16`, output-token statistics, versions, and the full protocol live in
the linked per-run report rather than widening the headline table further.

## Required methodology footnote

Every headline score, including an investor-facing slide copied from this
repository, carries a visible footnote marker. The footnote appears directly
below the score table and reads:

> **Methodology disclosure:** GLM-5.2's AIME 2026 score of 99.2 is
> self-reported by Z.ai. GLM used GPT-5.5-medium as an answer judge but did not
> disclose attempts per question or its aggregation method. Our evaluation
> uses deterministic rule-based numeric grading and reports avg@16 across 30
> questions and 480 completions. Invalid, missing, failed, and truncated
> answers are scored incorrect. Our run uses no tools, temperature 1.0,
> top-p 0.95, a 163,840-token maximum output, and a 262,144-token context
> window. Our structured response prompt mirrors GLM's disclosed three-field
> format. Consequently, the scores are directionally comparable but not
> methodologically identical.

The footnote is part of the result contract, not optional commentary.

## Components and data flow

1. `scripts/provision-lambda.sh` installs pinned vLLM and the checked-in
   service.
2. `scripts/vllm.service` starts Qwen3.5-9B with the canonical serving limits.
3. The preflight checks endpoint health, accepted output cap, and live serving
   metadata.
4. `scripts/run_eval.py` resolves the endpoint and AIME configuration, writes
   the initial run metadata, and invokes pinned lm-evaluation-harness.
5. The local AIME task builds the GLM-shaped messages and requests 16 seeded
   attempts per question.
6. The generation layer retains the request seed, `finish_reason`, completion
   token usage, and response text for every logical attempt.
7. The task grader scores all attempts and returns per-question `avg_at_16`
   and `pass_at_16` values.
8. A small reporting component validates run completeness, computes token and
   failure statistics, and writes `manifest.json` plus `report.md`.
9. A canonical full run is added to `results/summary.md` with its required
   methodology footnote.

## Error handling

- Serving startup fails visibly when the model cannot support the configured
  context or memory allocation. The deployment does not silently lower
  `max_model_len`.
- The preflight fails before evaluation when the endpoint rejects
  `max_tokens=163840` or when live serving metadata does not match the checked-
  in service.
- The full run fails validation when any selected question has a response
  count other than 16, seeds are duplicated, or the aggregate denominator is
  not 480.
- Individual generation failures remain scored attempts with correctness zero
  and are counted in the report.
- A canonical report is not produced when raw samples, resolved revisions, or
  required metadata are missing.

## Testing and verification

Implementation follows red-green-refactor and adds durable tests for:

- extracting valid plain and boxed integers from one `Exact Answer:` field;
- normalizing leading zeroes;
- rejecting missing, duplicate, non-integer, and out-of-range answer fields;
- scoring empty, failed, and truncated attempts as incorrect;
- computing `avg@16` and `pass@16` from mixed 16-attempt outcomes;
- preserving all 16 responses rather than reading only the first;
- building the system and user messages without exposing the target;
- constructing the pinned harness command with the canonical generation
  settings;
- rejecting inconsistent attempt, seed, context, and output settings; and
- producing a complete manifest, headline row, and exact disclosure footnote.

Repository verification runs all tests plus the existing Ruff and strict mypy
gates. Harness verification validates the custom task and performs a dry run.

Live verification then proceeds in this order:

1. install and start the checked-in vLLM service;
2. confirm the 262,144 context in the startup log;
3. confirm the endpoint accepts a 163,840-token completion cap;
4. run a two-question pilot, producing 32 attempts;
5. inspect prompt roles, seeds, response diversity, grading, token counts, and
   absence of preemption warnings; and
6. run the canonical 30-question evaluation only after the pilot passes.

## Out of scope

- Adding an LLM judge or reproducing GPT-5.5-medium grading.
- Tool-integrated AIME evaluation.
- Majority voting or heavy-thinking aggregation.
- Using the second H100, adding a load balancer, or changing the canonical
  tensor-parallel topology.
- Changing model weights or fine-tuning.
- Extending the reporting implementation to non-AIME benchmarks in this
  change.
