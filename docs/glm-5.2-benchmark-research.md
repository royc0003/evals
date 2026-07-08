# GLM-5.2 Blogpost Benchmarks: Research Reference

Research notes on the benchmark suite reported by Z.ai for the GLM-5.2 release (June 2026). This is the source-of-truth reference for which benchmarks exist, how Z.ai ran them, and which ones we can realistically reproduce against our own vLLM endpoint.

Sources consulted (2026-07-08):

- Official blogpost (HF mirror): https://huggingface.co/blog/zai-org/glm-52-blog
- Official repo with benchmark table and serving recipes: https://github.com/zai-org/GLM-5
- Z.ai developer docs (sampling params, OpenAI-compatible API): https://docs.z.ai/guides/llm/glm-5.2
- Model weights: https://huggingface.co/zai-org/GLM-5.2
- Community recap: https://www.latent.space/p/ainews-glm-gpt-glm-52-passes-vibe

Note: GLM-5.2 is a 753B-parameter, MIT-licensed open-weight model with a 1M-token context. Its scores are not a realistic target for a 9B model; the table below is our methodology reference, not our scoreboard.

## Reported benchmark table

### Reasoning

| Benchmark | GLM-5.2 | Opus 4.8 | GPT-5.5 | Gemini 3.1 Pro | Methodology notes |
|---|---|---|---|---|---|
| HLE (text-only) | 40.5 | 49.8 | 41.4 | 45.0 | max 163,840 gen tokens; temp=1.0, top_p=0.95 |
| HLE (w/ tools) | 54.7 | 57.9 | 52.2 | 51.4 | 300K context, no context management |
| CritPt | 16.7 | 20.9 | 27.1 | 17.7 | - |
| AIME 2026 | 99.2 | 95.7 | 98.3 | - | GPT-5.5 used as judge model |
| HMMT Nov 2025 | 94.4 | 96.5 | 96.5 | - | - |
| HMMT Feb 2026 | 92.5 | 96.7 | 96.7 | - | - |
| IMOAnswerBench | 91.0 | 83.5 | - | - | Qwen 3.7-Max scored 90.0 |
| GPQA-Diamond | 91.2 | 93.6 | - | 94.3 | - |

### Coding (agentic scaffolds)

| Benchmark | GLM-5.2 | Opus 4.8 | GPT-5.5 | GLM-5.1 | Harness | Key settings |
|---|---|---|---|---|---|---|
| SWE-bench Pro | 62.1 | 69.2 | 58.6 | 58.4 | OpenHands | temp=1, top_p=1, max_new_tokens=32k, 400K context |
| NL2Repo | 48.9 | 69.7 | 50.7 | 42.7 | Custom (anti-hack) | temp=1.0, top_p=1.0, max_new_tokens=48k, 400K context |
| DeepSWE | 46.2 | 58.0 | 70.0 | 18.0 | mini-swe-agent | 2h timeout/task, 400K context, isolated container (2 CPU, 8GB RAM, no internet) |
| ProgramBench | 63.7 | 71.9 | 70.8 | 50.9 | Claude Code 2.1.156 | max_tokens=64k, max_turns=2000, 6h timeout, sandboxed |
| Terminal-Bench 2.1 | 81.0 | 85.0 | 84.0 | 63.5 | Terminus-2 | temp=1.0, top_p=1.0, max_new_tokens=48k, max_episodes=500, 4h timeout, 256K context |
| Terminal-Bench 2.1 | 82.7 | 78.9 | 83.4 | - | Claude Code 2.1.167 | temp=1.0, top_p=0.95, max_new_tokens=128k, 5 runs averaged |

### Long-horizon

| Benchmark | GLM-5.2 | Opus 4.8 | GPT-5.5 | GLM-5.1 | Conductor | Settings |
|---|---|---|---|---|---|---|
| FrontierSWE | 74.4 | 75.1 | 72.6 | 30.5 | Proximal (third party) | 1M context, max effort, 128K max output |
| PostTrainBench | 34.3 | 37.2 | 28.4 | 20.1 | PostTrainBench team | 1M context, max effort, 128K max output |
| SWE-Marathon | 13.0 | 26.0 | 12.0 | 1.0 | Abundant AI (third party) | 1M context, max effort, 128K max output |

### Agentic / tool use

| Benchmark | GLM-5.2 | Opus 4.8 | GPT-5.5 | Gemini 3.1 Pro | Settings |
|---|---|---|---|---|---|
| MCP-Atlas (public set) | 76.8 | 77.8 | 75.3 | 69.2 | Think mode, 500 tasks, 10-min timeout/task, Gemini-3.0-Pro as judge |
| Tool-Decathlon | 48.2 | 59.9 | 55.6 | - | Official evaluation service, max_token=128K |

## Reproducibility triage

The suite splits into three tiers based on what it takes to run each benchmark against an arbitrary OpenAI-compatible endpoint.

### Tier 1: Endpoint-only, cheap. Run these first.

Static Q&A datasets scored by answer matching. Any standard harness (lm-eval-harness, lighteval) can run them against a `base_url`.

| Benchmark | Feasibility notes |
|---|---|
| AIME 2026 | 30 questions/exam. If the 2026 set isn't in the harness yet, use AIME 2025. Z.ai used GPT-5.5 as judge; rule-based answer extraction is the standard local substitute. |
| HMMT Nov 2025 / Feb 2026 | Same shape as AIME. Available in lm-eval-harness / lighteval math suites. |
| GPQA-Diamond | 198 multiple-choice questions, gated HF dataset (accept terms with your HF account). Rule-based scoring. |
| HLE (text-only) | Public dataset, but properly scoring it requires an LLM judge. Optional; run a subset with a judge model if desired. |
| IMOAnswerBench | Public answer-verifiable IMO-style problems; runnable if a harness task exists, otherwise skip. |
| CritPt | Physics research reasoning; needs an LLM judge for grading. Optional. |

### Tier 2: Agentic, harness-driven, but endpoint-compatible

These need an agent scaffold plus Docker sandboxes, but all three scaffolds Z.ai used (or their equivalents) speak litellm/OpenAI-compatible APIs, so they work against vLLM.

| Benchmark | Scaffold to use | Feasibility notes |
|---|---|---|
| Terminal-Bench 2.1 | `terminal-bench` CLI (`tb`) with the Terminus agent | Directly matches Z.ai's primary harness. Needs Docker on the runner. |
| SWE-bench (Verified/Lite → Pro) | mini-swe-agent | Matches the DeepSWE setup. Simplest scaffold (~100 lines of agent logic, litellm-based). Start with Lite/Verified subsets. |
| SWE-bench Pro | OpenHands | Matches Z.ai's SWE-bench Pro harness. Heavier to operate; treat as a stretch goal after mini-swe-agent works. |
| DeepSWE | mini-swe-agent | Same scaffold as above; run if the task set is publicly available. |

### Tier 3: Not practically reproducible. Documented as out of scope.

| Benchmark | Why not |
|---|---|
| FrontierSWE | Run by a third-party conductor (Proximal); no public self-serve harness. |
| SWE-Marathon | Run by Abundant AI; 1M-context long-horizon tasks, far beyond a 9B model's context anyway. |
| PostTrainBench | Requires GPU training runs inside the eval itself; conducted by the benchmark team. |
| NL2Repo | Custom internal "anti-hack" harness; not published. |
| ProgramBench | Claude Code scaffold with 6h/task timeouts; impractical cost and scaffold coupling. |
| MCP-Atlas | Requires a fleet of MCP servers plus Gemini-3.0-Pro as paid judge. |
| Tool-Decathlon | Scored through the benchmark's official evaluation service. |

## Caveats when comparing our numbers to the table

- GLM-5.2 ran with 256K-1M context windows and 32K-128K output budgets. Qwen3.5 9B has a far smaller context; every run must record the actual `--max-model-len` and `max_new_tokens` used.
- Z.ai's sampling (mostly temp=1.0, top_p=1.0) is tuned for their model. Use the Qwen3.5 model card's recommended sampling parameters instead, and record them.
- Several Z.ai scores use LLM judges (GPT-5.5, Gemini-3.0-Pro). Local runs using rule-based extraction are systematically stricter; note the grading method next to every score.
