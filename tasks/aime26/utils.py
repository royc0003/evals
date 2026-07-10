"""Grade AIME 2026 responses with deterministic integer matching."""

from __future__ import annotations

import re

from evals.vllm_adapter import Completion

ANSWER_FIELD_PATTERN = re.compile(
    r"^\s*Exact Answer:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ANSWER_VALUE_PATTERN = re.compile(r"^(?:([0-9]+)|\\boxed\{\s*([0-9]+)\s*\})$")
EXPLANATION_FIELD_PATTERN = re.compile(
    r"^\s*Explanation:\s*(\S.*)$",
    re.IGNORECASE | re.MULTILINE,
)
CONFIDENCE_FIELD_PATTERN = re.compile(
    r"^\s*Confidence:\s*([0-9]{1,3})%\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_exact_answer(response: str) -> int | None:
    """Return the single valid AIME integer in a response."""
    fields = ANSWER_FIELD_PATTERN.findall(response)
    if len(fields) != 1:
        return None

    value_match = ANSWER_VALUE_PATTERN.fullmatch(fields[0])
    if value_match is None:
        return None

    digits = value_match.group(1) or value_match.group(2)
    value = int(digits)
    return value if 0 <= value <= 999 else None


def has_required_fields(response: str) -> bool:
    """Return whether a response follows the three-field prompt format."""
    explanations = EXPLANATION_FIELD_PATTERN.findall(response)
    confidences = CONFIDENCE_FIELD_PATTERN.findall(response)
    if len(explanations) != 1 or len(confidences) != 1:
        return False

    confidence = int(confidences[0])
    return confidence <= 100 and parse_exact_answer(response) is not None


def grade_completion(completion: Completion, target: int) -> bool:
    """Return whether one completion is valid and mathematically correct."""
    if completion.error is not None:
        return False
    if completion.finish_reason == "length":
        return False
    return parse_exact_answer(completion.text) == target


def process_results(
    doc: dict[str, object],
    results: list[list[Completion]],
) -> dict[str, float]:
    """Return avg@16, pass@16, and attempts for one AIME problem."""
    if len(results) != 1 or len(results[0]) != 16:
        raise ValueError("AIME scoring requires exactly 16 attempts")
    attempts = results[0]
    target = int(str(doc["answer"]))
    correct = [grade_completion(item, target) for item in attempts]

    return {
        "avg_at_16": sum(correct) / len(correct),
        "pass_at_16": float(any(correct)),
        "number_of_attempts": float(len(correct)),
    }
