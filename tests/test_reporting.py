"""Test AIME run validation, aggregation, and report rendering."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals import reporting


def formatted_response(answer: int) -> str:
    """Return a response that satisfies the three-field prompt."""
    return (
        "Explanation: Test reasoning.\n"
        f"Exact Answer: {answer}\n"
        "Confidence: 80%"
    )


def make_record(
    doc_id: int,
    attempt: int,
    *,
    response: str | None = None,
    finish_reason: str | None = "stop",
    completion_tokens: int | None = None,
    transport_attempts: int = 1,
    error: str | None = None,
) -> dict[str, object]:
    """Return one raw attempt record for reporting tests."""
    return {
        "doc_id": doc_id,
        "attempt": attempt,
        "seed": 2025 + attempt,
        "messages": [
            {"role": "system", "content": "Use required fields."},
            {"role": "user", "content": f"Problem {doc_id}"},
        ],
        "target": "42",
        "response": formatted_response(42) if response is None else response,
        "finish_reason": finish_reason,
        "completion_tokens": (
            attempt
            if completion_tokens is None and error is None
            else completion_tokens
        ),
        "transport_attempts": transport_attempts,
        "error": error,
    }


def sample_records() -> list[dict[str, object]]:
    """Return two complete 16-attempt questions with mixed outcomes."""
    records = [make_record(0, attempt) for attempt in range(1, 9)]
    records.extend(
        make_record(0, attempt, response=formatted_response(41))
        for attempt in range(9, 13)
    )
    records.append(make_record(0, 13, response="No exact field"))
    records.append(
        make_record(
            0,
            14,
            finish_reason="length",
            transport_attempts=2,
        )
    )
    records.append(
        make_record(
            0,
            15,
            response="",
            finish_reason=None,
            completion_tokens=None,
            transport_attempts=4,
            error="HTTP 503 after 4 transport attempts",
        )
    )
    records.append(make_record(0, 16, response=""))
    records.extend(make_record(1, attempt) for attempt in range(1, 17))
    return records


def artifact_metadata() -> dict[str, object]:
    """Return canonical provenance for artifact-writing tests."""
    return {
        "run_id": "aime-2026-test",
        "timestamp_utc": "2026-07-09T20:00:00Z",
        "canonical": True,
        "run_type": "canonical",
        "benchmark": "AIME 2026",
        "model": "Qwen/Qwen3.5-9B",
        "model_revision": "model-sha",
        "dataset": "math-ai/aime26",
        "dataset_revision": "dataset-sha",
        "reference_score": 99.2,
        "attempts_per_problem": 16,
        "grading": "rule-based integer",
        "max_output_tokens": 163840,
        "max_model_len": 262144,
        "repository_commit": "commit-sha",
        "lm_eval_version": "0.4.12",
        "vllm_version": "0.24.0",
        "service_sha256": "service-sha",
        "prompt": "Use required fields.",
        "prompt_sha256": "prompt-sha",
        "seed_policy": {
            "base_seed": 2026,
            "seeds_per_problem": list(range(2026, 2042)),
        },
    }


def fine_tuned_artifact_metadata() -> dict[str, object]:
    """Return provenance for a complete fine-tuned model comparison."""
    metadata = artifact_metadata()
    metadata.update(
        {
            "run_id": "aime-my-finetuned-model-test",
            "canonical": False,
            "run_type": "model-comparison",
            "model": "my-org/my-finetuned-model",
            "model_revision": "checkpoint-sha",
            "endpoint_config_sha256": "endpoint-sha",
            "fine_tuning": {
                "type": "lora",
                "base_model": "Qwen/Qwen3.5-9B",
                "base_revision": "base-sha",
                "artifact": "/models/my-finetuned-model",
                "training_data_disclosure": "AIME 2026 was not used.",
            },
        }
    )
    del metadata["service_sha256"]
    return metadata


def test_summarize_attempts_reports_investor_metrics_and_diagnostics() -> None:
    """Compute avg@16, pass@16, attempt outcomes, retries, and token stats."""
    summary = reporting.summarize_attempts(
        sample_records(),
        attempts_per_problem=16,
        expected_questions=2,
    )

    assert summary["metrics"] == {
        "avg_at_16": 0.75,
        "pass_at_16": 1.0,
    }
    assert summary["attempts"] == {
        "expected": 32,
        "completed": 31,
        "failed": 1,
        "invalid": 2,
        "truncated": 1,
        "format_compliant": 29,
        "transport_retries": 4,
        "questions_with_multiple_unique_responses": 1,
    }
    assert summary["completion_tokens"] == {
        "mean": pytest.approx(257 / 31),
        "median": 8,
        "p95": 16,
        "maximum": 16,
    }


def test_summarize_attempts_rejects_an_incomplete_question() -> None:
    """Reject a report that silently drops one logical attempt."""
    records = sample_records()
    records.pop()

    with pytest.raises(ValueError, match="16 attempts"):
        reporting.summarize_attempts(records, attempts_per_problem=16)


def test_summarize_attempts_requires_success_metadata() -> None:
    """Reject a successful response whose API metadata was discarded."""
    records = sample_records()
    records[0]["completion_tokens"] = None

    with pytest.raises(ValueError, match="completion_tokens"):
        reporting.summarize_attempts(records, attempts_per_problem=16)


def test_summarize_attempts_rejects_a_missing_question() -> None:
    """Reject a report that silently drops one complete question."""
    records = sample_records()[:16]

    with pytest.raises(ValueError, match="2 questions"):
        reporting.summarize_attempts(
            records,
            attempts_per_problem=16,
            expected_questions=2,
        )


def test_summarize_attempts_requires_distinct_reproducible_seeds() -> None:
    """Reject repeated seeds within one question's 16 attempts."""
    records = sample_records()
    records[1]["seed"] = 2026

    with pytest.raises(ValueError, match="distinct seeds"):
        reporting.summarize_attempts(
            records,
            attempts_per_problem=16,
            expected_questions=2,
        )


def test_render_report_places_disclosure_below_headline_table() -> None:
    """Place the exact GLM methodology disclosure by the canonical score."""
    summary = reporting.summarize_attempts(
        sample_records(),
        attempts_per_problem=16,
        expected_questions=2,
    )
    manifest = artifact_metadata()
    manifest["summary"] = copy.deepcopy(summary)

    report = reporting.render_report(manifest)

    table_end = report.index("| AIME 2026 |")
    disclosure = report.index(reporting.METHODOLOGY_DISCLOSURE)
    assert table_end < disclosure
    assert disclosure - table_end < 300
    assert "75.0% avg@16" in report
    assert "100.0% pass@16 (best-of-16)" in report
    assert "| Failed attempts | 1 |" in report
    assert "| Invalid answers | 2 |" in report
    assert "| Format compliant | 29/32 |" in report
    assert "| Mean | 8.29 |" in report
    assert "| p95 | 16 |" in report
    assert "| lm-eval | 0.4.12 |" in report
    assert "| vLLM | 0.24.0 |" in report
    assert "| Model revision | model-sha |" in report


def test_write_run_artifacts_preserves_provenance_and_summary(
    tmp_path: Path,
) -> None:
    """Write the raw-derived manifest and human-readable report."""
    attempts_path = tmp_path / "attempts.jsonl"
    attempts_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in sample_records())
    )
    metadata = artifact_metadata()

    manifest = reporting.write_run_artifacts(
        tmp_path,
        metadata=metadata,
        attempts_per_problem=16,
        expected_questions=2,
    )

    saved_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert saved_manifest == manifest
    assert saved_manifest["repository_commit"] == "commit-sha"
    assert saved_manifest["summary"]["metrics"]["avg_at_16"] == 0.75
    report = (tmp_path / "report.md").read_text()
    assert reporting.METHODOLOGY_DISCLOSURE in report


def test_write_run_artifacts_requires_the_recorded_system_prompt(
    tmp_path: Path,
) -> None:
    """Reject raw messages that do not contain the configured prompt."""
    records = sample_records()
    messages = records[0]["messages"]
    assert isinstance(messages, list)
    system_message = messages[0]
    assert isinstance(system_message, dict)
    system_message["content"] = "Different prompt."
    (tmp_path / "attempts.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )

    with pytest.raises(ValueError, match="system message"):
        reporting.write_run_artifacts(
            tmp_path,
            metadata=artifact_metadata(),
            attempts_per_problem=16,
            expected_questions=2,
        )


def test_write_run_artifacts_requires_the_manifest_seed_schedule(
    tmp_path: Path,
) -> None:
    """Reject a raw seed schedule that differs from the manifest."""
    records = sample_records()
    for record in records:
        seed = record["seed"]
        assert isinstance(seed, int)
        record["seed"] = seed + 1
    (tmp_path / "attempts.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )

    with pytest.raises(ValueError, match="manifest seed schedule"):
        reporting.write_run_artifacts(
            tmp_path,
            metadata=artifact_metadata(),
            attempts_per_problem=16,
            expected_questions=2,
        )


def test_write_run_artifacts_rejects_a_pilot_without_response_variation(
    tmp_path: Path,
) -> None:
    """Reject a pilot that does not demonstrate independent sampling."""
    records = sample_records()
    for record in records:
        if record["error"] is None:
            record["response"] = formatted_response(42)
    (tmp_path / "attempts.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )
    metadata = artifact_metadata()
    metadata["canonical"] = False
    metadata["run_type"] = "pilot"

    with pytest.raises(ValueError, match="response variation"):
        reporting.write_run_artifacts(
            tmp_path,
            metadata=metadata,
            attempts_per_problem=16,
            expected_questions=2,
        )


def test_render_report_labels_pilots_without_a_canonical_headline() -> None:
    """Keep incomplete pilot metrics out of the investor score table."""
    summary = reporting.summarize_attempts(
        sample_records(),
        attempts_per_problem=16,
        expected_questions=2,
    )
    manifest: dict[str, object] = {
        "run_id": "aime-2026-pilot",
        "canonical": False,
        "run_type": "pilot",
        "benchmark": "AIME 2026",
        "model": "Qwen/Qwen3.5-9B",
        "reference_score": 99.2,
        "attempts_per_problem": 16,
        "grading": "rule-based integer",
        "max_output_tokens": 163840,
        "max_model_len": 262144,
        "summary": summary,
    }

    report = reporting.render_report(manifest)

    assert "NON-CANONICAL PILOT" in report
    assert "75.0% diagnostic avg@16" in report
    assert reporting.METHODOLOGY_DISCLOSURE not in report
    assert "| AIME 2026 |" not in report


def test_render_report_labels_a_complete_model_comparison() -> None:
    """Report fine-tuned scores without presenting them as canonical."""
    summary = reporting.summarize_attempts(
        sample_records(),
        attempts_per_problem=16,
        expected_questions=2,
    )
    manifest = fine_tuned_artifact_metadata()
    manifest["summary"] = summary

    report = reporting.render_report(manifest)

    assert "NON-CANONICAL MODEL COMPARISON" in report
    assert "75.0% avg@16" in report
    assert "100.0% pass@16" in report
    assert "Training data disclosure" in report
    assert "AIME 2026 was not used." in report
    assert "endpoint-sha" in report
    assert reporting.METHODOLOGY_DISCLOSURE in report
    assert "| AIME 2026 |" not in report


def test_model_comparison_does_not_require_pilot_response_variation(
    tmp_path: Path,
) -> None:
    """Apply the variation gate only to limited diagnostic runs."""
    records = sample_records()
    for record in records:
        if record["error"] is None:
            record["response"] = formatted_response(42)
    (tmp_path / "attempts.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )

    manifest = reporting.write_run_artifacts(
        tmp_path,
        metadata=fine_tuned_artifact_metadata(),
        attempts_per_problem=16,
        expected_questions=2,
    )

    assert manifest["run_type"] == "model-comparison"
