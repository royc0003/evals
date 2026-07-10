"""Test the vLLM endpoint smoke-check contract."""

from __future__ import annotations

from scripts import check_endpoint


def test_build_completion_payload_uses_the_canonical_output_cap() -> None:
    """Ask the server to accept the full configured completion limit."""
    payload = check_endpoint.build_completion_payload("qwen3.5-9b")

    assert payload == {
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": check_endpoint.PROMPT}],
        "max_tokens": 163840,
    }
