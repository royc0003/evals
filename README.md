# evals

Benchmark pipeline for models behind OpenAI-compatible endpoints.
First target: Qwen3.5 9B served by vLLM on Lambda GPU cloud, measured
against the reproducible subset of the GLM-5.2 release benchmarks.

- Documentation and setup guides: [docs/README.md](docs/README.md)
- Endpoint smoke test: `uv run scripts/check_endpoint.py`
- Run configs live in `configs/`, scores in `results/summary.md`
