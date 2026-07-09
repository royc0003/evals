# Evals

This repo hosts an eval pipeline for benchmarking a locally served model (Qwen3.5 9B on a vLLM OpenAI-compatible endpoint, hosted on Lambda GPU cloud) against the benchmark suite reported in the GLM-5.2 release blogpost (Z.ai, June 2026).

## Documents

- [glm-5.2-benchmark-research.md](glm-5.2-benchmark-research.md) - What GLM-5.2 actually reported: the full benchmark table with scores, harnesses, sampling parameters, and judges, plus a reproducibility triage of which benchmarks we can and cannot run.
- [running-evals.md](running-evals.md) - The short runbook: tunnel, smoke test, pilot, full run, recording results, bringing your own weights. Start here to actually run something.

## TL;DR

1. Serve the model: `vllm serve Qwen/Qwen3.5-9B` on a Lambda GPU instance. Every harness below talks to it through the OpenAI-compatible API.
2. Start with the cheap reasoning evals (AIME, GPQA-Diamond, HMMT) via lm-eval-harness.
3. Graduate to agentic evals (Terminal-Bench 2.1, SWE-bench via mini-swe-agent, SWE-bench Pro via OpenHands).
4. Record every run's config and compare against GLM-5.2's reported numbers, with methodology deviations noted.
