# Evals

This repo hosts an eval pipeline for benchmarking a locally served model (Qwen3.5 9B on a vLLM OpenAI-compatible endpoint, hosted on Lambda GPU cloud) against the benchmark suite reported in the GLM-5.2 release blogpost (Z.ai, June 2026).

## Documents

- [glm-5.2-benchmark-research.md](glm-5.2-benchmark-research.md) - What GLM-5.2 actually reported: the full benchmark table with scores, harnesses, sampling parameters, and judges, plus a reproducibility triage of which benchmarks we can and cannot run.
- [running-evals.md](running-evals.md) - The short runbook: tunnel, smoke test, pilot, full run, recording results, bringing your own weights. Start here to actually run something.

## TL;DR

1. From a local checkout, run
   `./scripts/deploy-vllm-service.sh ubuntu@192.222.52.206` to verify or install
   vLLM 0.24.0 on an existing GPU node, copy and restart the checked-in unit,
   and wait up to 15 minutes for health. SSH configuration supplies custom
   keys, ports, or jump hosts. Use `bash scripts/provision-lambda.sh` on the
   GPU node only when provisioning a fresh machine.
2. Start with the cheap reasoning evals (AIME, GPQA-Diamond, HMMT) via lm-eval-harness.
3. Graduate to agentic evals (Terminal-Bench 2.1, SWE-bench via mini-swe-agent, SWE-bench Pro via OpenHands).
4. Record every run's config and compare against GLM-5.2's reported numbers, with methodology deviations noted.
