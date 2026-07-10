#!/usr/bin/env python3
"""Run lm-eval after registering the repository's vLLM adapter."""

from __future__ import annotations

from lm_eval.__main__ import cli_evaluate

import evals.vllm_adapter  # noqa: F401 - import registers the model adapter

if __name__ == "__main__":
    cli_evaluate()
