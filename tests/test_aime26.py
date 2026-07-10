"""Test the AIME 2026 rule-based grading contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evals.vllm_adapter import Completion
from tasks.aime26 import utils

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_REVISION = "79037aebdb6580008fb960d17cb21fd3099083e3"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SYSTEM_INSTRUCTION = """Respond using exactly these fields:
Explanation: <your reasoning>
Exact Answer: <your final integer>
Confidence: <a percentage from 0% through 100%>"""


class _TaskLoader(yaml.SafeLoader):
    """Load task YAML while retaining lm-eval function references."""


def _load_function(
    loader: yaml.SafeLoader,
    node: yaml.ScalarNode,
) -> str:
    """Return an lm-eval function tag as its scalar reference."""
    return loader.construct_scalar(node)


_TaskLoader.add_constructor("!function", _load_function)


def load_yaml(path: Path, *, task: bool = False) -> dict[str, object]:
    """Return a repository YAML document for contract tests."""
    loader = _TaskLoader if task else yaml.SafeLoader
    loaded = yaml.load(path.read_text(), Loader=loader)
    assert isinstance(loaded, dict)
    return loaded


def make_completion(
    text: str,
    *,
    finish_reason: str | None = "stop",
    error: str | None = None,
) -> Completion:
    """Return a completion for grader tests."""
    return Completion(
        text=text,
        finish_reason=finish_reason,
        completion_tokens=10,
        transport_attempts=1,
        error=error,
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Exact Answer: 7", 7),
        (r"Exact Answer: \boxed{7}", 7),
        ("Exact Answer: 007", 7),
        ("Exact Answer: 0", 0),
        ("Exact Answer: 999", 999),
    ],
)
def test_parse_exact_answer_accepts_one_aime_integer(
    response: str,
    expected: int,
) -> None:
    """Accept one integer in the AIME answer range."""
    assert utils.parse_exact_answer(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        "The answer is 7.",
        "Exact Answer: 7\nExact Answer: 8",
        "Exact Answer: -1",
        "Exact Answer: 1000",
        "Exact Answer: 3.0",
        "Exact Answer: 7 apples",
        r"Exact Answer: \boxed{\boxed{7}}",
    ],
)
def test_parse_exact_answer_rejects_invalid_fields(response: str) -> None:
    """Reject missing, duplicate, or non-integer answer fields."""
    assert utils.parse_exact_answer(response) is None


def test_has_required_fields_accepts_the_prompt_contract() -> None:
    """Accept one valid occurrence of each required response field."""
    response = (
        "Explanation: The calculation is shown here.\n"
        "Exact Answer: 42\n"
        "Confidence: 85%"
    )

    assert utils.has_required_fields(response)


@pytest.mark.parametrize(
    "response",
    [
        "Exact Answer: 42\nConfidence: 85%",
        "Explanation: Work\nExact Answer: 42",
        (
            "Explanation: Work\nExact Answer: 42\n"
            "Confidence: 85%\nConfidence: 90%"
        ),
        "Explanation: Work\nExact Answer: 42\nConfidence: 101%",
    ],
)
def test_has_required_fields_rejects_format_violations(
    response: str,
) -> None:
    """Reject missing, duplicate, or invalid response fields."""
    assert not utils.has_required_fields(response)


def test_grade_completion_rejects_transport_failure() -> None:
    """Score a failed request as incorrect even with a matching answer."""
    completion = make_completion(
        "Exact Answer: 42",
        finish_reason=None,
        error="connection reset",
    )

    assert not utils.grade_completion(completion, 42)


def test_grade_completion_rejects_truncation() -> None:
    """Score a length-truncated request as incorrect."""
    completion = make_completion(
        "Exact Answer: 42",
        finish_reason="length",
    )

    assert not utils.grade_completion(completion, 42)


def test_process_results_reports_avg_pass_and_attempt_count() -> None:
    """Aggregate correctness across all 16 attempts for one question."""
    correct = [make_completion("Exact Answer: 42") for _ in range(12)]
    truncated = make_completion(
        "Exact Answer: 42",
        finish_reason="length",
    )
    incorrect = [make_completion("Exact Answer: 41") for _ in range(3)]

    metrics = utils.process_results(
        {"answer": "42"},
        [correct + [truncated] + incorrect],
    )

    assert metrics == {
        "avg_at_16": 0.75,
        "pass_at_16": 1.0,
        "number_of_attempts": 16.0,
    }


def test_process_results_reports_zero_pass_when_all_attempts_fail() -> None:
    """Report pass@16 as zero when no attempt is correct."""
    attempts = [make_completion("Exact Answer: 41") for _ in range(16)]

    metrics = utils.process_results({"answer": "42"}, [attempts])

    assert metrics["avg_at_16"] == 0.0
    assert metrics["pass_at_16"] == 0.0
    assert metrics["number_of_attempts"] == 16.0


def test_process_results_requires_all_16_attempts() -> None:
    """Refuse to label an incomplete response set as avg@16."""
    attempts = [make_completion("Exact Answer: 42") for _ in range(15)]

    with pytest.raises(ValueError, match="exactly 16"):
        utils.process_results({"answer": "42"}, [attempts])


def test_aime_config_pins_the_canonical_protocol() -> None:
    """Pin sampling, attempts, grading, revisions, and prompt text."""
    benchmark = load_yaml(REPO_ROOT / "configs" / "aime.yaml")
    endpoint = load_yaml(REPO_ROOT / "configs" / "endpoint.yaml")

    assert benchmark["attempts_per_problem"] == 16
    assert benchmark["base_seed"] == 2026
    assert benchmark["max_retries"] == 3
    assert benchmark["grading"] == "rule_based_integer"
    assert benchmark["dataset_revision"] == DATASET_REVISION
    assert benchmark["system_instruction"] == SYSTEM_INSTRUCTION
    assert benchmark["generation"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_gen_toks": 163840,
    }
    assert endpoint["model_revision"] == MODEL_REVISION


def test_aime_task_sends_only_the_problem_and_scores_all_attempts() -> None:
    """Keep the target out of the prompt and retain all 16 responses."""
    task = load_yaml(
        REPO_ROOT / "tasks" / "aime26" / "aime26.yaml",
        task=True,
    )

    assert task["dataset_kwargs"] == {"revision": DATASET_REVISION}
    assert task["doc_to_text"] == "{{problem}}"
    assert "answer" not in str(task["doc_to_text"]).lower()
    assert "boxed" not in str(task["doc_to_text"]).lower()
    assert task["repeats"] == 16
    assert task["num_fewshot"] == 0

    generation = task["generation_kwargs"]
    assert isinstance(generation, dict)
    assert generation["do_sample"] is True
    assert generation["temperature"] == 1.0
    assert generation["top_p"] == 0.95
    assert generation["max_gen_toks"] == 163840

    assert task["filter_list"] == [
        {
            "name": "all_attempts",
            "filter": [{"function": "take_first_k", "k": 16}],
        }
    ]
    metrics = task["metric_list"]
    assert isinstance(metrics, list)
    assert [metric["metric"] for metric in metrics] == [
        "avg_at_16",
        "pass_at_16",
        "number_of_attempts",
    ]
