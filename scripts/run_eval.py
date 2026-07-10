#!/usr/bin/env python3
"""Run an eval from its config file.

Reads the selected endpoint YAML plus the given benchmark config, builds the
lm-eval command from them, and runs it. Results go to a timestamped directory
under results/raw/ automatically.

Usage:
    uv run scripts/run_eval.py configs/aime.yaml            # full run
    uv run scripts/run_eval.py configs/aime.yaml --limit 3  # pilot
    uv run scripts/run_eval.py configs/aime.yaml --dry-run  # show command
    uv run scripts/run_eval.py configs/aime.yaml \
        --endpoint-config configs/endpoint-my-model.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import httpx
import yaml

from evals.reporting import write_run_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_CONFIG = REPO_ROOT / "configs" / "endpoint.yaml"
LM_EVAL_ENTRYPOINT = REPO_ROOT / "scripts" / "lm_eval_entrypoint.py"
ENDPOINT_LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
CANONICAL_AIME_FIELDS: dict[str, object] = {
    "harness_version": "0.4.12",
    "attempts_per_problem": 16,
    "num_concurrent": 8,
    "base_seed": 2026,
    "max_retries": 3,
    "grading": "rule_based_integer",
}
CANONICAL_AIME_GENERATION: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_gen_toks": 163840,
}
CANONICAL_ENDPOINT_FIELDS = {
    "model": "qwen3.5-9b",
    "source_model": "Qwen/Qwen3.5-9B",
    "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
CANONICAL_SERVING_FIELDS: dict[str, object] = {
    "expected_vllm_version": "0.24.0",
    "gpu": "H100 80GB",
    "gpus_used": 1,
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "max_model_len": 262144,
}
FINE_TUNING_FIELDS = (
    "base_model",
    "base_revision",
    "artifact",
    "training_data_disclosure",
)
SERVING_STRING_FIELDS = ("expected_vllm_version", "gpu", "dtype")
SERVING_POSITIVE_INT_FIELDS = (
    "gpus_used",
    "tensor_parallel_size",
    "max_model_len",
)


def fail(message: str) -> NoReturn:
    """Print an error and exit with status 1."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def load_yaml(path: Path) -> dict[str, Any]:
    """Return the mapping stored in a YAML file."""
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        fail(f"config not found: {path}")
    except yaml.YAMLError as exc:
        fail(f"{path} is not valid YAML: {exc}")
    if not isinstance(raw, dict):
        fail(f"expected a mapping in {path}")
    return raw


def require_str(config: dict[str, Any], key: str, path: Path) -> str:
    """Return a required non-empty string field from a config."""
    value = config.get(key)
    if not isinstance(value, str) or not value:
        fail(f"{path} must set '{key}' as a non-empty string")
    return value


def endpoint_profile(
    endpoint: dict[str, Any],
    endpoint_path: Path,
) -> tuple[str, bool]:
    """Return the validated endpoint label and canonical flag."""
    evaluation = endpoint.get("evaluation")
    if not isinstance(evaluation, dict):
        fail(f"{endpoint_path} must set an 'evaluation' mapping")
    label = evaluation.get("label")
    canonical = evaluation.get("canonical")
    if (
        not isinstance(label, str)
        or ENDPOINT_LABEL_PATTERN.fullmatch(label) is None
    ):
        fail(f"{endpoint_path}: evaluation.label is not filesystem-safe")
    if not isinstance(canonical, bool):
        fail(f"{endpoint_path}: evaluation.canonical must be a boolean")
    return label, canonical


def validate_endpoint(
    endpoint: dict[str, Any],
    endpoint_path: Path,
) -> None:
    """Validate canonical pins or fine-tuning provenance."""
    _, canonical = endpoint_profile(endpoint, endpoint_path)
    require_str(endpoint, "base_url", endpoint_path)
    require_str(endpoint, "model", endpoint_path)
    require_str(endpoint, "source_model", endpoint_path)
    require_str(endpoint, "model_revision", endpoint_path)
    serving = endpoint.get("serving")
    if not isinstance(serving, dict):
        fail(f"{endpoint_path} must set a 'serving' mapping")
    for field in SERVING_STRING_FIELDS:
        require_str(serving, field, endpoint_path)
    for field in SERVING_POSITIVE_INT_FIELDS:
        value = serving.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            fail(f"{endpoint_path}: serving.{field} must be a positive int")

    if canonical:
        for (
            endpoint_field,
            endpoint_expected,
        ) in CANONICAL_ENDPOINT_FIELDS.items():
            if endpoint.get(endpoint_field) != endpoint_expected:
                fail(
                    f"{endpoint_path}: '{endpoint_field}' must be "
                    f"{endpoint_expected!r}"
                )
        for (
            serving_field,
            serving_expected,
        ) in CANONICAL_SERVING_FIELDS.items():
            if serving.get(serving_field) != serving_expected:
                fail(
                    f"{endpoint_path}: serving.{serving_field} must be "
                    f"{serving_expected!r}"
                )
        return

    fine_tuning = endpoint.get("fine_tuning")
    if not isinstance(fine_tuning, dict):
        fail(f"{endpoint_path} must set a 'fine_tuning' mapping")
    if fine_tuning.get("type") not in {"lora", "merged"}:
        fail(f"{endpoint_path}: fine_tuning.type must be 'lora' or 'merged'")
    for field in FINE_TUNING_FIELDS:
        require_str(fine_tuning, field, endpoint_path)


def validate_limit(benchmark: dict[str, Any], limit: int | None) -> None:
    """Validate a sample limit against the selected benchmark."""
    if limit is None:
        return
    if limit < 1:
        fail("--limit must be positive")
    if benchmark.get("benchmark") == "aime" and limit > 30:
        fail("--limit cannot exceed the 30 AIME questions")


def run_type(canonical_endpoint: bool, limit: int | None) -> str:
    """Return the report classification for one evaluation run."""
    if limit is not None:
        return "pilot"
    if canonical_endpoint:
        return "canonical"
    return "model-comparison"


def build_run_name(
    benchmark: str,
    endpoint_label: str,
    canonical_endpoint: bool,
    limit: int | None,
    timestamp: str,
) -> str:
    """Return a collision-resistant canonical or comparison run name."""
    name_parts = [benchmark]
    if not canonical_endpoint:
        name_parts.append(endpoint_label)
    if limit is not None:
        name_parts.append("pilot")
    name_parts.append(timestamp)
    return "-".join(name_parts)


def validate_aime_protocol(
    benchmark: dict[str, Any],
    config_path: Path,
) -> None:
    """Reject changes to the fixed investor-facing AIME protocol."""
    if benchmark.get("benchmark") != "aime":
        return

    for field, expected in CANONICAL_AIME_FIELDS.items():
        if benchmark.get(field) != expected:
            fail(f"{config_path}: '{field}' must be {expected!r}")

    generation = benchmark.get("generation")
    if not isinstance(generation, dict):
        fail(f"{config_path} must set a 'generation' mapping")
    for field, expected in CANONICAL_AIME_GENERATION.items():
        if generation.get(field) != expected:
            fail(f"{config_path}: generation.{field} must be {expected!r}")


def build_run_metadata(
    endpoint: dict[str, Any],
    benchmark: dict[str, Any],
    endpoint_path: Path,
    *,
    run_id: str,
    timestamp_utc: str,
    run_type_value: str,
    repository_commit: str,
    live_vllm_version: str,
) -> dict[str, object]:
    """Return the reproducibility metadata recorded in the manifest."""
    evaluation_label, canonical_endpoint = endpoint_profile(
        endpoint,
        endpoint_path,
    )
    source_model = require_str(endpoint, "source_model", endpoint_path)
    served_model = require_str(endpoint, "model", endpoint_path)
    model_revision = require_str(
        endpoint,
        "model_revision",
        endpoint_path,
    )
    serving = endpoint.get("serving")
    if not isinstance(serving, dict):
        fail(f"{endpoint_path} must set a 'serving' mapping")
    generation = benchmark.get("generation")
    reference = benchmark.get("reference")
    if not isinstance(generation, dict) or not isinstance(reference, dict):
        fail("AIME config must set generation and reference mappings")

    prompt = require_str(
        benchmark,
        "system_instruction",
        REPO_ROOT / "configs" / "aime.yaml",
    )
    base_seed = int(benchmark["base_seed"])
    attempts_per_problem = int(benchmark["attempts_per_problem"])
    metadata: dict[str, object] = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "canonical": run_type_value == "canonical",
        "run_type": run_type_value,
        "evaluation_label": evaluation_label,
        "benchmark": "AIME 2026",
        "model": source_model,
        "served_model": served_model,
        "model_revision": model_revision,
        "endpoint_config_path": str(endpoint_path),
        "endpoint_config_sha256": hashlib.sha256(
            endpoint_path.read_bytes()
        ).hexdigest(),
        "dataset": str(benchmark["dataset"]),
        "dataset_revision": str(benchmark["dataset_revision"]),
        "repository_commit": repository_commit,
        "lm_eval_version": str(benchmark["harness_version"]),
        "vllm_version": live_vllm_version,
        "hardware": {
            "gpu": str(serving["gpu"]),
            "gpus_used": int(serving["gpus_used"]),
            "dtype": str(serving["dtype"]),
            "tensor_parallel_size": int(serving["tensor_parallel_size"]),
        },
        "max_output_tokens": int(generation["max_gen_toks"]),
        "max_model_len": int(serving["max_model_len"]),
        "sampling": dict(generation),
        "seed_policy": {
            "base_seed": base_seed,
            "seeds_per_problem": list(
                range(base_seed, base_seed + attempts_per_problem)
            ),
        },
        "attempts_per_problem": attempts_per_problem,
        "grading": "rule-based integer",
        "tools": "none",
        "reference_score": float(reference["glm_5_2_reported"]),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    if canonical_endpoint:
        service_path = REPO_ROOT / "scripts" / "vllm.service"
        metadata["service_path"] = str(service_path.relative_to(REPO_ROOT))
        metadata["service_sha256"] = hashlib.sha256(
            service_path.read_bytes()
        ).hexdigest()
    else:
        fine_tuning = endpoint.get("fine_tuning")
        if not isinstance(fine_tuning, dict):
            fail(f"{endpoint_path} must set a 'fine_tuning' mapping")
        metadata["fine_tuning"] = dict(fine_tuning)
    return metadata


def write_resolved_config(
    output_path: Path,
    *,
    endpoint: dict[str, Any],
    benchmark: dict[str, Any],
    limit: int | None,
) -> None:
    """Write the endpoint and benchmark settings used by one run."""
    output_path.mkdir(parents=True, exist_ok=True)
    resolved = {
        "endpoint": endpoint,
        "benchmark": benchmark,
        "limit": limit,
    }
    (output_path / "resolved-config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )


def fetch_vllm_version(base_url: str) -> str:
    """Return the version reported by the live vLLM server."""
    server_url = base_url.rstrip("/").removesuffix("/v1")
    response = httpx.get(f"{server_url}/version", timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("vLLM /version response must be a mapping")
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("vLLM /version response is missing its version")
    return version


def repository_commit() -> str:
    """Return the repository commit used by the run."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_command(
    endpoint: dict[str, Any],
    endpoint_path: Path,
    bench: dict[str, Any],
    config_path: Path,
    output_path: Path,
    limit: int | None,
) -> list[str]:
    """Assemble the lm-eval command from the two configs."""
    harness = bench.get("harness")
    if harness != "lm-eval-harness":
        fail(
            f"{config_path} uses harness {harness!r}; this runner only"
            " supports lm-eval-harness (agentic harnesses run on the"
            " GPU node, see docs/eval-setup-plan.md)"
        )
    validate_aime_protocol(bench, config_path)
    validate_endpoint(endpoint, endpoint_path)
    base_url = require_str(endpoint, "base_url", endpoint_path)
    model = require_str(endpoint, "model", endpoint_path)
    task = require_str(bench, "task", config_path)
    is_aime = bench.get("benchmark") == "aime"
    system_instruction = bench.get("system_instruction")
    if is_aime:
        system_instruction = require_str(
            bench,
            "system_instruction",
            config_path,
        )
    elif system_instruction is not None and not isinstance(
        system_instruction, str
    ):
        fail(f"{config_path}: 'system_instruction' must be a string")

    generation = bench.get("generation")
    if not isinstance(generation, dict) or not generation:
        fail(f"{config_path} must set a non-empty 'generation' mapping")
    gen_kwargs = ",".join(f"{k}={v}" for k, v in generation.items())

    num_concurrent = bench.get("num_concurrent", 4)
    if not isinstance(num_concurrent, int) or num_concurrent < 1:
        fail(f"{config_path}: 'num_concurrent' must be a positive int")

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    model_args = (
        f"model={model},base_url={chat_url},num_concurrent={num_concurrent}"
    )
    adapter = "local-chat-completions"
    if is_aime:
        base_seed = bench.get("base_seed")
        max_retries = bench.get("max_retries")
        if not isinstance(base_seed, int):
            fail(f"{config_path}: 'base_seed' must be an int")
        if not isinstance(max_retries, int) or max_retries < 1:
            fail(f"{config_path}: 'max_retries' must be a positive int")
        attempts_path = output_path / "attempts.jsonl"
        model_args += (
            f",max_retries={max_retries},seed={base_seed},"
            f"attempts_path={attempts_path}"
        )
        adapter = "tracked-local-chat-completions"

    command = [
        sys.executable,
        str(LM_EVAL_ENTRYPOINT),
        "--model",
        adapter,
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--apply_chat_template",
        "--gen_kwargs",
        gen_kwargs,
        "--log_samples",
        "--output_path",
        str(output_path),
    ]
    if isinstance(system_instruction, str) and system_instruction:
        command.extend(["--system_instruction", system_instruction])
    include_path = bench.get("include_path")
    if isinstance(include_path, str) and include_path:
        command.extend(["--include_path", str(REPO_ROOT / include_path)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def run_with_spinner(command: list[str], log_path: Path) -> int:
    """Run the command with a live spinner; log output to a file.

    Falls back to plain streaming when stdout is not a terminal, so
    backgrounded runs keep their full log on stdout instead.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not sys.stdout.isatty():
        with log_path.open("w") as log:
            stream_process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert stream_process.stdout is not None
            for line in stream_process.stdout:
                sys.stdout.write(line)
                log.write(line)
            return stream_process.wait()

    frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    start = time.monotonic()
    with log_path.open("w") as log:
        spinner_process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT
        )
        try:
            while spinner_process.poll() is None:
                minutes, seconds = divmod(int(time.monotonic() - start), 60)
                sys.stdout.write(
                    f"\r{next(frames)} evaluating... {minutes:02d}:"
                    f"{seconds:02d} elapsed (log: {log_path})"
                )
                sys.stdout.flush()
                time.sleep(0.2)
        except KeyboardInterrupt:
            spinner_process.terminate()
            spinner_process.wait()
            sys.stdout.write("\ninterrupted\n")
            return 130
    sys.stdout.write("\r" + " " * 100 + "\r")
    tail = log_path.read_text().splitlines()[-8:]
    print("\n".join(tail))
    return spinner_process.returncode


def main() -> int:
    """Parse arguments, build the command, and run the eval."""
    parser = argparse.ArgumentParser(
        description="Run an eval defined by a config YAML."
    )
    parser.add_argument(
        "config", type=Path, help="benchmark YAML under configs/"
    )
    parser.add_argument(
        "--endpoint-config",
        type=Path,
        default=ENDPOINT_CONFIG,
        help="endpoint YAML; defaults to configs/endpoint.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N examples (pilot run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command instead of running it",
    )
    args = parser.parse_args()

    endpoint_path = args.endpoint_config
    endpoint = load_yaml(endpoint_path)
    bench = load_yaml(args.config)
    validate_limit(bench, args.limit)
    benchmark = require_str(bench, "benchmark", args.config)
    endpoint_label, canonical_endpoint = endpoint_profile(
        endpoint,
        endpoint_path,
    )
    run_type_value = run_type(canonical_endpoint, args.limit)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = build_run_name(
        benchmark,
        endpoint_label,
        canonical_endpoint,
        args.limit,
        timestamp,
    )
    output_path = REPO_ROOT / "results" / "raw" / run_name

    command = build_command(
        endpoint,
        endpoint_path,
        bench,
        args.config,
        output_path,
        args.limit,
    )
    print("command:", " ".join(command))
    if args.dry_run:
        return 0

    write_resolved_config(
        output_path,
        endpoint=endpoint,
        benchmark=bench,
        limit=args.limit,
    )
    metadata: dict[str, object] | None = None
    if bench.get("benchmark") == "aime":
        base_url = require_str(endpoint, "base_url", endpoint_path)
        try:
            live_vllm_version = fetch_vllm_version(base_url)
        except (httpx.HTTPError, ValueError) as exc:
            fail(f"could not read the live vLLM version: {exc}")
        serving = endpoint.get("serving")
        if not isinstance(serving, dict):
            fail(f"{endpoint_path} must set a 'serving' mapping")
        expected_vllm_version = serving.get("expected_vllm_version")
        if live_vllm_version != expected_vllm_version:
            fail(
                "live vLLM version does not match the pinned protocol: "
                f"expected {expected_vllm_version}, got {live_vllm_version}"
            )

        timestamp_utc = (
            datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        metadata = build_run_metadata(
            endpoint,
            bench,
            endpoint_path,
            run_id=run_name,
            timestamp_utc=timestamp_utc,
            run_type_value=run_type_value,
            repository_commit=repository_commit(),
            live_vllm_version=live_vllm_version,
        )
    returncode = run_with_spinner(command, output_path / "run.log")
    if returncode == 0 and metadata is not None:
        expected_questions = args.limit if args.limit is not None else 30
        write_run_artifacts(
            output_path,
            metadata=metadata,
            attempts_per_problem=int(bench["attempts_per_problem"]),
            expected_questions=expected_questions,
        )
    if returncode == 0:
        print(f"results written to {output_path}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
