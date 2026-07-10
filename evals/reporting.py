"""Validate and render auditable evaluation results."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import cast

from evals.vllm_adapter import Completion
from tasks.aime26.utils import (
    grade_completion,
    has_required_fields,
    parse_exact_answer,
)

METHODOLOGY_DISCLOSURE = (
    "**Methodology disclosure:** GLM-5.2's AIME 2026 score of 99.2 is\n"
    "self-reported by Z.ai. GLM used GPT-5.5-medium as an answer judge but "
    "did not\n"
    "disclose attempts per question or its aggregation method. Our "
    "evaluation\n"
    "uses deterministic rule-based numeric grading and reports avg@16 "
    "across 30\n"
    "questions and 480 completions. Invalid, missing, failed, and "
    "truncated\n"
    "answers are scored incorrect. Our run uses no tools, temperature 1.0,\n"
    "top-p 0.95, a 163,840-token maximum output, and a 262,144-token "
    "context\n"
    "window. Our structured response prompt mirrors GLM's disclosed "
    "three-field\n"
    "format. Consequently, the scores are directionally comparable but not\n"
    "methodologically identical."
)


def _require_int(record: dict[str, object], field: str) -> int:
    """Return a required integer attempt field."""
    value = record.get(field)
    if not isinstance(value, int):
        raise ValueError(f"attempt field {field!r} must be an integer")
    return value


def _require_string(record: dict[str, object], field: str) -> str:
    """Return a required string attempt field."""
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"attempt field {field!r} must be a string")
    return value


def summarize_attempts(
    records: list[dict[str, object]],
    attempts_per_problem: int,
    expected_questions: int | None = None,
) -> dict[str, object]:
    """Validate raw attempts and compute AIME metrics and diagnostics."""
    if not records:
        raise ValueError("attempt records must not be empty")

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[_require_int(record, "doc_id")].append(record)

    if expected_questions is not None and len(grouped) != expected_questions:
        raise ValueError(
            f"run must contain exactly {expected_questions} questions"
        )

    expected_numbers = set(range(1, attempts_per_problem + 1))
    seed_schedule: tuple[int, ...] | None = None
    for doc_id, attempts in grouped.items():
        attempt_numbers = {
            _require_int(record, "attempt") for record in attempts
        }
        if (
            len(attempts) != attempts_per_problem
            or attempt_numbers != expected_numbers
        ):
            raise ValueError(
                f"question {doc_id} must contain exactly "
                f"{attempts_per_problem} attempts"
            )
        ordered_attempts = sorted(
            attempts,
            key=lambda record: _require_int(record, "attempt"),
        )
        seeds = tuple(
            _require_int(record, "seed") for record in ordered_attempts
        )
        if len(set(seeds)) != attempts_per_problem:
            raise ValueError(
                f"question {doc_id} must use {attempts_per_problem} "
                "distinct seeds"
            )
        if seed_schedule is None:
            seed_schedule = seeds
        elif seeds != seed_schedule:
            raise ValueError("every question must use the same seed schedule")

    correct = 0
    passed_questions = 0
    completed = 0
    failed = 0
    invalid = 0
    truncated = 0
    format_compliant = 0
    transport_retries = 0
    questions_with_multiple_unique_responses = 0
    completion_tokens: list[int] = []

    for attempts in grouped.values():
        question_correct = False
        successful_responses: set[str] = set()
        for record in attempts:
            response = _require_string(record, "response")
            target = int(_require_string(record, "target"))
            transport_attempts = _require_int(
                record,
                "transport_attempts",
            )
            if transport_attempts < 1:
                raise ValueError("transport_attempts must be positive")
            transport_retries += transport_attempts - 1

            error = record.get("error")
            if error is not None and not isinstance(error, str):
                raise ValueError(
                    "attempt field 'error' must be a string or null"
                )
            finish_reason = record.get("finish_reason")
            tokens = record.get("completion_tokens")
            if error is None:
                completed += 1
                if not isinstance(finish_reason, str):
                    raise ValueError(
                        "successful attempt must retain finish_reason"
                    )
                if not isinstance(tokens, int):
                    raise ValueError(
                        "successful attempt must retain completion_tokens"
                    )
                completion_tokens.append(tokens)
                successful_responses.add(response)
                invalid += int(parse_exact_answer(response) is None)
                format_compliant += int(has_required_fields(response))
            else:
                failed += 1

            truncated += int(finish_reason == "length")
            completion = Completion(
                text=response,
                finish_reason=(
                    finish_reason if isinstance(finish_reason, str) else None
                ),
                completion_tokens=tokens if isinstance(tokens, int) else None,
                transport_attempts=transport_attempts,
                error=error,
            )
            is_correct = grade_completion(completion, target)
            correct += int(is_correct)
            question_correct = question_correct or is_correct

        passed_questions += int(question_correct)
        questions_with_multiple_unique_responses += int(
            len(successful_responses) > 1
        )

    expected = len(grouped) * attempts_per_problem
    ordered_tokens = sorted(completion_tokens)
    p95_index = math.ceil(0.95 * len(ordered_tokens)) - 1
    return {
        "metrics": {
            "avg_at_16": correct / expected,
            "pass_at_16": passed_questions / len(grouped),
        },
        "attempts": {
            "expected": expected,
            "completed": completed,
            "failed": failed,
            "invalid": invalid,
            "truncated": truncated,
            "format_compliant": format_compliant,
            "transport_retries": transport_retries,
            "questions_with_multiple_unique_responses": (
                questions_with_multiple_unique_responses
            ),
        },
        "completion_tokens": {
            "mean": statistics.mean(ordered_tokens),
            "median": statistics.median(ordered_tokens),
            "p95": ordered_tokens[p95_index],
            "maximum": ordered_tokens[-1],
        },
    }


def _number(mapping: dict[str, object], field: str) -> float:
    """Return a numeric report field."""
    value = mapping.get(field)
    if not isinstance(value, int | float):
        raise ValueError(f"report field {field!r} must be numeric")
    return float(value)


def render_report(manifest: dict[str, object]) -> str:
    """Render one investor-readable report from a validated manifest."""
    summary = cast(dict[str, object], manifest["summary"])
    metrics = cast(dict[str, object], summary["metrics"])
    attempts = cast(dict[str, object], summary["attempts"])
    token_stats = cast(dict[str, object], summary["completion_tokens"])
    avg_at_16 = _number(metrics, "avg_at_16")
    pass_at_16 = _number(metrics, "pass_at_16")
    expected = int(_number(attempts, "expected"))
    truncated = int(_number(attempts, "truncated"))

    benchmark = str(manifest["benchmark"])
    model = str(manifest["model"])
    reference_score = _number(manifest, "reference_score")
    attempts_per_problem = int(_number(manifest, "attempts_per_problem"))
    max_output_tokens = int(_number(manifest, "max_output_tokens"))
    max_model_len = int(_number(manifest, "max_model_len"))
    grading = str(manifest["grading"])

    diagnostics = (
        "## Run diagnostics\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Completed attempts | {int(_number(attempts, 'completed'))}"
        f"/{expected} |\n"
        f"| Failed attempts | {int(_number(attempts, 'failed'))} |\n"
        f"| Invalid answers | {int(_number(attempts, 'invalid'))} |\n"
        f"| Truncated attempts | {truncated} |\n"
        f"| Format compliant | "
        f"{int(_number(attempts, 'format_compliant'))}/{expected} |\n"
        f"| Transport retries | "
        f"{int(_number(attempts, 'transport_retries'))} |\n"
        f"| Questions with response variation | "
        f"{int(_number(attempts, 'questions_with_multiple_unique_responses'))}"
        " |\n\n"
        "## Observed completion tokens\n\n"
        "Values come from API `usage.completion_tokens`; the configured "
        "maximum is a cap.\n\n"
        "| Statistic | Tokens |\n"
        "|---|---:|\n"
        f"| Mean | {_number(token_stats, 'mean'):.2f} |\n"
        f"| Median | {_number(token_stats, 'median'):g} |\n"
        f"| p95 | {_number(token_stats, 'p95'):g} |\n"
        f"| Maximum | {_number(token_stats, 'maximum'):g} |\n"
    )

    run_type_value = manifest.get("run_type")
    if run_type_value == "pilot":
        return (
            f"# {benchmark} NON-CANONICAL PILOT\n\n"
            f"Diagnostic score: {avg_at_16:.1%} diagnostic avg@16.\n\n"
            f"Diagnostic best-of score: {pass_at_16:.1%} pass@16.\n\n"
            f"Attempts: {expected}; truncated: {truncated}. This pilot is "
            "not eligible for the investor headline table.\n\n"
            f"{diagnostics}"
        )

    if run_type_value == "model-comparison":
        fine_tuning = manifest.get("fine_tuning")
        endpoint_config_sha256 = manifest.get("endpoint_config_sha256")
        if not isinstance(fine_tuning, dict):
            raise ValueError(
                "model-comparison manifest must contain fine_tuning"
            )
        if not isinstance(endpoint_config_sha256, str):
            raise ValueError(
                "model-comparison manifest must contain endpoint config hash"
            )
        training_disclosure = fine_tuning.get("training_data_disclosure")
        if not isinstance(training_disclosure, str):
            raise ValueError(
                "model-comparison manifest must disclose training data"
            )
        return (
            f"# {benchmark} NON-CANONICAL MODEL COMPARISON\n\n"
            f"Score: {avg_at_16:.1%} avg@16.\n\n"
            f"Secondary metric: {pass_at_16:.1%} pass@16 "
            "(best-of-16).\n\n"
            "This fine-tuned result is not eligible for the canonical GLM "
            "headline table.\n\n"
            f"> {METHODOLOGY_DISCLOSURE}\n\n"
            f"{diagnostics}\n"
            "## Fine-tuning provenance\n\n"
            "| Component | Value |\n"
            "|---|---|\n"
            f"| Type | {fine_tuning.get('type')} |\n"
            f"| Base model | {fine_tuning.get('base_model')} |\n"
            f"| Base revision | {fine_tuning.get('base_revision')} |\n"
            f"| Artifact | {fine_tuning.get('artifact')} |\n"
            f"| Training data disclosure | {training_disclosure} |\n"
            f"| Endpoint config SHA-256 | {endpoint_config_sha256} |\n"
        )

    if run_type_value != "canonical":
        raise ValueError("manifest run_type is invalid")

    header = (
        "| Benchmark | Model | Ours | GLM-5.2 reported | Attempts | "
        "Grading | Max output | Context | Truncated |\n"
    )
    row = (
        f"| {benchmark} | {model} | {avg_at_16:.1%} avg@16 | "
        f"{reference_score:.1f} | {attempts_per_problem}/question "
        f"({expected} total) | {grading} | {max_output_tokens:,} | "
        f"{max_model_len:,} | {truncated}/{expected} |\n"
    )
    return (
        f"# {benchmark} evaluation report\n\n"
        f"{header}"
        "|---|---|---:|---:|---:|---|---:|---:|---:|\n"
        f"{row}\n"
        f"> {METHODOLOGY_DISCLOSURE}\n\n"
        f"Secondary metric: {pass_at_16:.1%} pass@16 (best-of-16).\n\n"
        f"{diagnostics}\n"
        "## Reproducibility\n\n"
        "| Component | Value |\n"
        "|---|---|\n"
        f"| lm-eval | {manifest['lm_eval_version']} |\n"
        f"| vLLM | {manifest['vllm_version']} |\n"
        f"| Model revision | {manifest['model_revision']} |\n"
        f"| Dataset revision | {manifest['dataset_revision']} |\n"
        f"| Repository commit | {manifest['repository_commit']} |\n"
        f"| Service SHA-256 | {manifest['service_sha256']} |\n"
        f"| Prompt SHA-256 | {manifest['prompt_sha256']} |\n"
    )


def load_attempts(path: Path) -> list[dict[str, object]]:
    """Load raw JSON Lines attempt records."""
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        loaded: object = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"{path}:{line_number} must contain a JSON object"
            )
        records.append(cast(dict[str, object], loaded))
    return records


def _validate_audit_contract(
    records: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    """Match raw prompts and seeds to the recorded run configuration."""
    prompt = metadata.get("prompt")
    seed_policy = metadata.get("seed_policy")
    if not isinstance(prompt, str) or not isinstance(seed_policy, dict):
        raise ValueError("manifest must contain prompt and seed policy")
    raw_schedule = seed_policy.get("seeds_per_problem")
    if not isinstance(raw_schedule, list) or not all(
        isinstance(seed, int) for seed in raw_schedule
    ):
        raise ValueError("manifest seed schedule must contain integers")
    seed_schedule = cast(list[int], raw_schedule)

    for record in records:
        attempt = _require_int(record, "attempt")
        seed = _require_int(record, "seed")
        if attempt > len(seed_schedule) or seed != seed_schedule[attempt - 1]:
            raise ValueError(
                "attempt does not match the manifest seed schedule"
            )

        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("attempt must retain system and user messages")
        system_message, user_message = messages
        if not isinstance(system_message, dict) or not isinstance(
            user_message, dict
        ):
            raise ValueError("attempt messages must be mappings")
        if system_message != {"role": "system", "content": prompt}:
            raise ValueError(
                "attempt system message differs from the manifest"
            )
        if user_message.get("role") != "user" or not isinstance(
            user_message.get("content"), str
        ):
            raise ValueError("attempt must retain one text user message")


def write_run_artifacts(
    output_path: Path,
    *,
    metadata: dict[str, object],
    attempts_per_problem: int,
    expected_questions: int,
) -> dict[str, object]:
    """Validate raw attempts and write the manifest and Markdown report."""
    records = load_attempts(output_path / "attempts.jsonl")
    _validate_audit_contract(records, metadata)
    summary = summarize_attempts(
        records,
        attempts_per_problem,
        expected_questions,
    )
    if metadata.get("run_type") == "pilot":
        attempt_summary = cast(dict[str, object], summary["attempts"])
        variation_count = int(
            _number(
                attempt_summary,
                "questions_with_multiple_unique_responses",
            )
        )
        if variation_count < 1:
            raise ValueError(
                "pilot must show response variation for independent sampling"
            )
    manifest = {**metadata, "summary": summary}
    (output_path / "manifest.json").write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n"
    )
    (output_path / "report.md").write_text(render_report(manifest))
    return manifest
