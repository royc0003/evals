#!/usr/bin/env python3
"""Run an eval from its config file.

Reads configs/endpoint.yaml plus the given benchmark config, builds
the lm-eval command from them, and runs it. Results go to
results/raw/<benchmark>-<timestamp>/ automatically.

Usage:
    uv run scripts/run_eval.py configs/aime.yaml            # full run
    uv run scripts/run_eval.py configs/aime.yaml --limit 3  # pilot
    uv run scripts/run_eval.py configs/aime.yaml --dry-run  # show command
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_CONFIG = REPO_ROOT / "configs" / "endpoint.yaml"


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


def build_command(
    endpoint: dict[str, Any],
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
    base_url = require_str(endpoint, "base_url", ENDPOINT_CONFIG)
    model = require_str(endpoint, "model", ENDPOINT_CONFIG)
    task = require_str(bench, "task", config_path)

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
    command = [
        "uvx",
        "--from",
        "lm-eval[api]",
        "lm_eval",
        "--model",
        "local-chat-completions",
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
    if not sys.stdout.isatty():
        return subprocess.run(command, check=False, cwd=REPO_ROOT).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    start = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT
        )
        try:
            while process.poll() is None:
                minutes, seconds = divmod(int(time.monotonic() - start), 60)
                sys.stdout.write(
                    f"\r{next(frames)} evaluating... {minutes:02d}:"
                    f"{seconds:02d} elapsed (log: {log_path})"
                )
                sys.stdout.flush()
                time.sleep(0.2)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            sys.stdout.write("\ninterrupted\n")
            return 130
    sys.stdout.write("\r" + " " * 100 + "\r")
    tail = log_path.read_text().splitlines()[-8:]
    print("\n".join(tail))
    return process.returncode


def main() -> int:
    """Parse arguments, build the command, and run the eval."""
    parser = argparse.ArgumentParser(
        description="Run an eval defined by a config YAML."
    )
    parser.add_argument(
        "config", type=Path, help="benchmark YAML under configs/"
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

    endpoint = load_yaml(ENDPOINT_CONFIG)
    bench = load_yaml(args.config)
    benchmark = require_str(bench, "benchmark", args.config)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{benchmark}-{timestamp}"
    if args.limit is not None:
        run_name = f"{benchmark}-pilot-{timestamp}"
    output_path = REPO_ROOT / "results" / "raw" / run_name

    command = build_command(
        endpoint, bench, args.config, output_path, args.limit
    )
    print("command:", " ".join(command))
    if args.dry_run:
        return 0

    returncode = run_with_spinner(command, output_path / "run.log")
    if returncode == 0:
        print(f"results written to {output_path}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
