#!/usr/bin/env python3
"""Smoke-test the vLLM OpenAI-compatible endpoint.

Reads the endpoint settings from configs/endpoint.yaml, confirms the
server lists its models, then asks for one short chat completion and
prints the reply.

Run with the SSH tunnel open (or directly on the Lambda instance):

    uv run scripts/check_endpoint.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "endpoint.yaml"
TIMEOUT_SECONDS = 60.0
PROMPT = "Reply with the single word: pong"


def fail(message: str, hint: str) -> NoReturn:
    """Print an error with a hint, then exit with status 1."""
    print(f"error: {message}", file=sys.stderr)
    print(f"hint: {hint}", file=sys.stderr)
    sys.exit(1)


def load_endpoint() -> tuple[str, str]:
    """Return (base_url, model) from configs/endpoint.yaml."""
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text())
    except FileNotFoundError:
        fail(
            f"config not found: {CONFIG_PATH}",
            "run from the repo root and check the configs/ directory",
        )
    except yaml.YAMLError as exc:
        fail(f"config is not valid YAML: {exc}", f"fix {CONFIG_PATH}")
    if not isinstance(raw, dict):
        fail(
            f"expected a mapping in {CONFIG_PATH}",
            "the file should define base_url and model",
        )
    base_url = raw.get("base_url")
    model = raw.get("model")
    if not isinstance(base_url, str) or not isinstance(model, str):
        fail(
            "base_url and model must both be set as strings",
            f"edit {CONFIG_PATH}",
        )
    return base_url.rstrip("/"), model


def check_models(client: httpx.Client, base_url: str) -> None:
    """Confirm the server answers GET /models."""
    try:
        response = client.get(f"{base_url}/models")
    except httpx.HTTPError as exc:
        fail(
            f"cannot reach {base_url}: {exc}",
            "is the SSH tunnel open and the vllm service running?",
        )
    if response.status_code != 200:
        fail(
            f"GET /models returned HTTP {response.status_code}",
            "check `journalctl -u vllm -f` on the instance",
        )
    print("models endpoint OK")


def request_completion(client: httpx.Client, base_url: str, model: str) -> str:
    """Send one short chat completion and return the reply text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 512,
    }
    try:
        response = client.post(f"{base_url}/chat/completions", json=payload)
    except httpx.HTTPError as exc:
        fail(
            f"completion request failed: {exc}",
            "the server may still be loading the model; retry shortly",
        )
    if response.status_code != 200:
        fail(
            f"completion returned HTTP {response.status_code}",
            "check that the model matches --served-model-name",
        )
    try:
        reply = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        fail(
            f"unexpected response shape: {exc}",
            "inspect the raw response with curl",
        )
    if not isinstance(reply, str) or not reply.strip():
        fail(
            "the model returned an empty reply",
            "raise max_tokens or check the reasoning parser flags",
        )
    return reply.strip()


def main() -> int:
    """Run both checks and report the endpoint as usable."""
    base_url, model = load_endpoint()
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        check_models(client, base_url)
        reply = request_completion(client, base_url, model)
    print(f"chat completion OK, {model} replied: {reply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
