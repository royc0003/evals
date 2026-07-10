"""Test the single evaluation runner entry point."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import yaml

from scripts import run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent


def canonical_configs() -> tuple[dict[str, object], dict[str, object]]:
    """Load independent copies of the canonical endpoint and AIME config."""
    endpoint = run_eval.load_yaml(REPO_ROOT / "configs" / "endpoint.yaml")
    benchmark = run_eval.load_yaml(REPO_ROOT / "configs" / "aime.yaml")
    return copy.deepcopy(endpoint), copy.deepcopy(benchmark)


def fine_tuned_endpoint() -> dict[str, object]:
    """Return a complete non-canonical endpoint configuration."""
    endpoint, _ = canonical_configs()
    endpoint.update(
        {
            "evaluation": {
                "label": "my-finetuned-model",
                "canonical": False,
            },
            "model": "my-finetuned-model",
            "source_model": "my-org/my-finetuned-model",
            "model_revision": "checkpoint-sha",
            "fine_tuning": {
                "type": "lora",
                "base_model": "Qwen/Qwen3.5-9B",
                "base_revision": "base-sha",
                "artifact": "/models/my-finetuned-model",
                "training_data_disclosure": "AIME 2026 was not used.",
            },
        }
    )
    return endpoint


def command_value(command: list[str], flag: str) -> str:
    """Return the value immediately following a command-line flag."""
    return command[command.index(flag) + 1]


def test_canonical_endpoint_declares_its_evaluation_profile() -> None:
    """Identify the locked endpoint separately from comparison models."""
    endpoint, _ = canonical_configs()

    assert endpoint["evaluation"] == {
        "label": "qwen3.5-9b",
        "canonical": True,
    }


def test_endpoint_profile_returns_the_label_and_canonical_flag() -> None:
    """Return the validated identity used for naming and reporting."""
    endpoint, _ = canonical_configs()

    assert run_eval.endpoint_profile(
        endpoint,
        REPO_ROOT / "configs" / "endpoint.yaml",
    ) == ("qwen3.5-9b", True)


@pytest.mark.parametrize(
    "evaluation",
    [
        None,
        {},
        {"label": "Unsafe Label", "canonical": False},
        {"label": "safe-label", "canonical": "false"},
    ],
)
def test_endpoint_profile_rejects_invalid_evaluation(
    evaluation: object,
) -> None:
    """Reject profile values that cannot safely identify a run."""
    endpoint, _ = canonical_configs()
    endpoint["evaluation"] = evaluation

    with pytest.raises(SystemExit):
        run_eval.endpoint_profile(endpoint, Path("endpoint.yaml"))


@pytest.mark.parametrize("fine_tuning_type", ["lora", "merged"])
def test_validate_endpoint_accepts_supported_fine_tuning_types(
    fine_tuning_type: str,
) -> None:
    """Accept both adapter and complete-checkpoint fine-tuning artifacts."""
    endpoint = fine_tuned_endpoint()
    fine_tuning = endpoint["fine_tuning"]
    assert isinstance(fine_tuning, dict)
    fine_tuning["type"] = fine_tuning_type

    run_eval.validate_endpoint(endpoint, Path("fine-tuned.yaml"))


@pytest.mark.parametrize(
    "field",
    [
        "type",
        "base_model",
        "base_revision",
        "artifact",
        "training_data_disclosure",
    ],
)
def test_validate_endpoint_requires_fine_tuning_provenance(
    field: str,
) -> None:
    """Reject a comparison profile that cannot identify its training."""
    endpoint = fine_tuned_endpoint()
    fine_tuning = endpoint["fine_tuning"]
    assert isinstance(fine_tuning, dict)
    del fine_tuning[field]

    with pytest.raises(SystemExit):
        run_eval.validate_endpoint(endpoint, Path("fine-tuned.yaml"))


def test_validate_endpoint_rejects_unknown_fine_tuning_type() -> None:
    """Reject unsupported artifact semantics instead of guessing."""
    endpoint = fine_tuned_endpoint()
    fine_tuning = endpoint["fine_tuning"]
    assert isinstance(fine_tuning, dict)
    fine_tuning["type"] = "qlora"

    with pytest.raises(SystemExit):
        run_eval.validate_endpoint(endpoint, Path("fine-tuned.yaml"))


@pytest.mark.parametrize(
    "field",
    [
        "expected_vllm_version",
        "gpu",
        "gpus_used",
        "dtype",
        "tensor_parallel_size",
        "max_model_len",
    ],
)
def test_validate_endpoint_requires_serving_provenance(field: str) -> None:
    """Reject an endpoint that cannot describe its serving conditions."""
    endpoint = fine_tuned_endpoint()
    serving = endpoint["serving"]
    assert isinstance(serving, dict)
    del serving[field]

    with pytest.raises(SystemExit):
        run_eval.validate_endpoint(endpoint, Path("fine-tuned.yaml"))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("expected_vllm_version", ""),
        ("gpu", ""),
        ("dtype", ""),
        ("gpus_used", 0),
        ("tensor_parallel_size", 0),
        ("max_model_len", 0),
    ],
)
def test_validate_endpoint_rejects_invalid_serving_provenance(
    field: str,
    invalid: object,
) -> None:
    """Require meaningful serving strings and positive topology values."""
    endpoint = fine_tuned_endpoint()
    serving = endpoint["serving"]
    assert isinstance(serving, dict)
    serving[field] = invalid

    with pytest.raises(SystemExit):
        run_eval.validate_endpoint(endpoint, Path("fine-tuned.yaml"))


def test_fine_tuned_endpoint_example_is_valid() -> None:
    """Keep the copyable user template synchronized with validation."""
    endpoint_path = REPO_ROOT / "configs" / "endpoint-finetuned.example.yaml"
    endpoint = run_eval.load_yaml(endpoint_path)

    run_eval.validate_endpoint(endpoint, endpoint_path)

    assert run_eval.endpoint_profile(endpoint, endpoint_path) == (
        "my-qwen-finetune",
        False,
    )


def test_build_command_uses_the_tracked_pinned_harness(
    tmp_path: Path,
) -> None:
    """Build the pinned command with all auditable request settings."""
    endpoint, benchmark = canonical_configs()
    output_path = tmp_path / "run"

    command = run_eval.build_command(
        endpoint,
        REPO_ROOT / "configs" / "endpoint.yaml",
        benchmark,
        REPO_ROOT / "configs" / "aime.yaml",
        output_path,
        limit=3,
    )

    assert command[:2] == [
        sys.executable,
        str(REPO_ROOT / "scripts" / "lm_eval_entrypoint.py"),
    ]
    assert Path(command[1]).is_file()
    assert command_value(command, "--model") == (
        "tracked-local-chat-completions"
    )
    model_args = command_value(command, "--model_args")
    assert "model=qwen3.5-9b" in model_args
    assert "num_concurrent=8" in model_args
    assert "max_retries=3" in model_args
    assert "seed=2026" in model_args
    assert f"attempts_path={output_path / 'attempts.jsonl'}" in model_args
    assert (
        command_value(command, "--system_instruction")
        == benchmark["system_instruction"]
    )
    assert command_value(command, "--include_path") == str(
        REPO_ROOT / "tasks" / "aime26"
    )
    assert command_value(command, "--limit") == "3"
    assert "--apply_chat_template" in command
    assert "--log_samples" in command
    assert "uvx" not in command


def test_build_command_tracks_attempts_for_a_fine_tuned_endpoint(
    tmp_path: Path,
) -> None:
    """Keep the complete AIME audit contract when model weights change."""
    endpoint = fine_tuned_endpoint()
    _, benchmark = canonical_configs()
    output_path = tmp_path / "run"

    command = run_eval.build_command(
        endpoint,
        Path("fine-tuned.yaml"),
        benchmark,
        REPO_ROOT / "configs" / "aime.yaml",
        output_path,
        limit=3,
    )

    assert command_value(command, "--model") == (
        "tracked-local-chat-completions"
    )
    model_args = command_value(command, "--model_args")
    assert "model=my-finetuned-model" in model_args
    assert "seed=2026" in model_args
    assert f"attempts_path={output_path / 'attempts.jsonl'}" in model_args


def test_build_command_preserves_the_standard_harness_path(
    tmp_path: Path,
) -> None:
    """Keep non-AIME lm-eval configs on the stock chat adapter."""
    endpoint = run_eval.load_yaml(REPO_ROOT / "configs" / "endpoint.yaml")
    benchmark = run_eval.load_yaml(REPO_ROOT / "configs" / "gpqa-diamond.yaml")

    command = run_eval.build_command(
        endpoint,
        REPO_ROOT / "configs" / "endpoint.yaml",
        benchmark,
        REPO_ROOT / "configs" / "gpqa-diamond.yaml",
        tmp_path / "run",
        limit=3,
    )

    assert command_value(command, "--model") == "local-chat-completions"
    assert "--system_instruction" not in command
    assert "attempts_path=" not in command_value(command, "--model_args")


def test_validate_limit_only_caps_the_30_question_aime_dataset() -> None:
    """Preserve larger limits for other benchmark datasets."""
    run_eval.validate_limit({"benchmark": "gpqa-diamond"}, 50)

    with pytest.raises(SystemExit):
        run_eval.validate_limit({"benchmark": "aime"}, 31)


@pytest.mark.parametrize(
    ("canonical_endpoint", "limit", "expected"),
    [
        (True, None, "canonical"),
        (True, 3, "pilot"),
        (False, None, "model-comparison"),
        (False, 3, "pilot"),
    ],
)
def test_run_type(
    canonical_endpoint: bool,
    limit: int | None,
    expected: str,
) -> None:
    """Classify complete and limited runs independently of model identity."""
    assert run_eval.run_type(canonical_endpoint, limit) == expected


@pytest.mark.parametrize(
    (
        "canonical_endpoint",
        "limit",
        "expected",
    ),
    [
        (True, None, "aime-2026-07-10_120000"),
        (True, 3, "aime-pilot-2026-07-10_120000"),
        (False, None, "aime-my-model-2026-07-10_120000"),
        (False, 3, "aime-my-model-pilot-2026-07-10_120000"),
    ],
)
def test_build_run_name(
    canonical_endpoint: bool,
    limit: int | None,
    expected: str,
) -> None:
    """Keep canonical names stable and label comparison-model outputs."""
    assert (
        run_eval.build_run_name(
            "aime",
            "my-model",
            canonical_endpoint,
            limit,
            "2026-07-10_120000",
        )
        == expected
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("attempts_per_problem", 8),
        ("grading", "judge"),
        ("base_seed", 1),
        ("max_retries", 2),
        ("num_concurrent", 4),
    ],
)
def test_build_command_rejects_noncanonical_aime_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    """Reject a canonical AIME run whose fixed protocol was changed."""
    endpoint, benchmark = canonical_configs()
    benchmark[field] = invalid

    with pytest.raises(SystemExit):
        run_eval.build_command(
            endpoint,
            REPO_ROOT / "configs" / "endpoint.yaml",
            benchmark,
            REPO_ROOT / "configs" / "aime.yaml",
            tmp_path / "run",
            limit=None,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("temperature", 0.6),
        ("top_p", 0.9),
        ("max_gen_toks", 32768),
    ],
)
def test_build_command_rejects_noncanonical_generation(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    """Reject altered AIME sampling or output limits."""
    endpoint, benchmark = canonical_configs()
    generation = benchmark["generation"]
    assert isinstance(generation, dict)
    generation[field] = invalid

    with pytest.raises(SystemExit):
        run_eval.build_command(
            endpoint,
            REPO_ROOT / "configs" / "endpoint.yaml",
            benchmark,
            REPO_ROOT / "configs" / "aime.yaml",
            tmp_path / "run",
            limit=None,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("model", "my-lora"),
        ("source_model", "another/model"),
        ("model_revision", "main"),
    ],
)
def test_build_command_rejects_noncanonical_model_identity(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    """Keep the investor run pinned to the declared Qwen model."""
    endpoint, benchmark = canonical_configs()
    endpoint[field] = invalid

    with pytest.raises(SystemExit):
        run_eval.build_command(
            endpoint,
            REPO_ROOT / "configs" / "endpoint.yaml",
            benchmark,
            REPO_ROOT / "configs" / "aime.yaml",
            tmp_path / "run",
            limit=None,
        )


def test_build_command_rejects_noncanonical_context_limit(
    tmp_path: Path,
) -> None:
    """Keep the investor run pinned to the 262,144-token context."""
    endpoint, benchmark = canonical_configs()
    serving = endpoint["serving"]
    assert isinstance(serving, dict)
    serving["max_model_len"] = 131072

    with pytest.raises(SystemExit):
        run_eval.build_command(
            endpoint,
            REPO_ROOT / "configs" / "endpoint.yaml",
            benchmark,
            REPO_ROOT / "configs" / "aime.yaml",
            tmp_path / "run",
            limit=None,
        )


def test_build_run_metadata_records_the_reproducibility_contract() -> None:
    """Record revisions, serving topology, prompt hash, and seed policy."""
    endpoint, benchmark = canonical_configs()
    prompt = benchmark["system_instruction"]
    assert isinstance(prompt, str)

    metadata = run_eval.build_run_metadata(
        endpoint,
        benchmark,
        REPO_ROOT / "configs" / "endpoint.yaml",
        run_id="aime-2026-test",
        timestamp_utc="2026-07-09T20:00:00Z",
        run_type_value="canonical",
        repository_commit="commit-sha",
        live_vllm_version="0.24.0",
    )

    assert metadata["model"] == "Qwen/Qwen3.5-9B"
    assert metadata["served_model"] == "qwen3.5-9b"
    assert metadata["model_revision"] == endpoint["model_revision"]
    assert metadata["dataset"] == "math-ai/aime26"
    assert metadata["dataset_revision"] == benchmark["dataset_revision"]
    assert metadata["lm_eval_version"] == "0.4.12"
    assert metadata["vllm_version"] == "0.24.0"
    assert metadata["hardware"] == {
        "gpu": "H100 80GB",
        "gpus_used": 1,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
    }
    assert metadata["max_output_tokens"] == 163840
    assert metadata["max_model_len"] == 262144
    assert metadata["sampling"] == benchmark["generation"]
    assert metadata["seed_policy"] == {
        "base_seed": 2026,
        "seeds_per_problem": list(range(2026, 2042)),
    }
    assert metadata["prompt"] == prompt
    assert (
        metadata["prompt_sha256"]
        == hashlib.sha256(prompt.encode()).hexdigest()
    )
    assert isinstance(metadata["service_sha256"], str)


def test_build_run_metadata_records_fine_tuning_provenance(
    tmp_path: Path,
) -> None:
    """Attribute a model-comparison score to the selected checkpoint."""
    endpoint = fine_tuned_endpoint()
    _, benchmark = canonical_configs()
    endpoint_path = tmp_path / "endpoint.yaml"
    endpoint_path.write_text(yaml.safe_dump(endpoint), encoding="utf-8")

    metadata = run_eval.build_run_metadata(
        endpoint,
        benchmark,
        endpoint_path,
        run_id="aime-my-finetuned-model-test",
        timestamp_utc="2026-07-10T20:00:00Z",
        run_type_value="model-comparison",
        repository_commit="commit-sha",
        live_vllm_version="0.24.0",
    )

    assert metadata["canonical"] is False
    assert metadata["run_type"] == "model-comparison"
    assert metadata["evaluation_label"] == "my-finetuned-model"
    assert metadata["fine_tuning"] == endpoint["fine_tuning"]
    assert metadata["endpoint_config_path"] == str(endpoint_path)
    assert (
        metadata["endpoint_config_sha256"]
        == hashlib.sha256(endpoint_path.read_bytes()).hexdigest()
    )
    assert "service_path" not in metadata
    assert "service_sha256" not in metadata


def test_write_resolved_config_records_endpoint_benchmark_and_limit(
    tmp_path: Path,
) -> None:
    """Write the exact run inputs before generation begins."""
    endpoint, benchmark = canonical_configs()

    run_eval.write_resolved_config(
        tmp_path,
        endpoint=endpoint,
        benchmark=benchmark,
        limit=3,
    )

    resolved = yaml.safe_load((tmp_path / "resolved-config.yaml").read_text())
    assert resolved == {
        "endpoint": endpoint,
        "benchmark": benchmark,
        "limit": 3,
    }


def test_fetch_vllm_version_reads_the_live_server() -> None:
    """Record the version reported by the serving endpoint."""

    class Handler(BaseHTTPRequestHandler):
        """Serve the vLLM version response."""

        def do_GET(self) -> None:
            """Return the live version document."""
            body = json.dumps({"version": "0.24.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress the test server access log."""
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        version = run_eval.fetch_vllm_version(
            f"http://127.0.0.1:{server.server_port}/v1"
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert version == "0.24.0"


def test_run_with_spinner_logs_noninteractive_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve run.log while streaming output outside an interactive TTY."""
    log_path = tmp_path / "run.log"

    returncode = run_eval.run_with_spinner(
        [sys.executable, "-c", "print('harness output')"],
        log_path,
    )

    assert returncode == 0
    assert log_path.read_text() == "harness output\n"
    assert "harness output" in capsys.readouterr().out


def test_cli_selects_a_fine_tuned_endpoint_config(tmp_path: Path) -> None:
    """Select a comparison model with one endpoint YAML option."""
    endpoint_path = tmp_path / "endpoint.yaml"
    endpoint_path.write_text(
        yaml.safe_dump(fine_tuned_endpoint()),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_eval.py"),
            str(REPO_ROOT / "configs" / "aime.yaml"),
            "--endpoint-config",
            str(endpoint_path),
            "--limit",
            "3",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "model=my-finetuned-model" in result.stdout
    assert "aime-my-finetuned-model-pilot-" in result.stdout
