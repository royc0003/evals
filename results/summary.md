# Results Summary

Scores here are directional, not apples-to-apples: GLM-5.2 is a 753B
model evaluated with 256K-1M contexts and, in places, LLM judges. The
point is a repeatable pipeline; record every methodology deviation.

| Benchmark | Qwen3.5-9B (ours) | GLM-5.2 (reported) | Harness + version | Our settings | Deviations from GLM methodology |
|---|---|---|---|---|---|
| GPQA-Diamond | - | 91.2 | - | - | rule-based grading; smaller context |
| AIME | - | 99.2 (AIME 2026) | - | - | no LLM judge; 2025 set until harness ships 2026 |
| HMMT Nov 2025 | - | 94.4 | - | - | - |
| HMMT Feb 2026 | - | 92.5 | - | - | - |
| Terminal-Bench 2.1 | - | 81.0 (Terminus-2) | - | - | reduced episode/timeout budgets |
| SWE-bench Verified | - | n/a (Pro: 62.1) | - | - | different subset than Pro |

Raw harness outputs go in `results/raw/<benchmark>/<date>/` (gitignored)
together with a copy of the exact config used for the run.
