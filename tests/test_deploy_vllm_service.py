"""Tests for the existing-node vLLM service deployment command."""

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "deploy-vllm-service.sh"
USAGE = "Usage: scripts/deploy-vllm-service.sh ubuntu@HOST"


def run_script(
    *arguments: str,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the deployment script with the supplied arguments."""
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def write_executable(path: Path, content: str) -> None:
    """Write an executable test command."""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize("arguments", [(), ("-oProxyCommand=bad",)])
def test_rejects_invalid_target(arguments: tuple[str, ...]) -> None:
    """Reject a missing target or a target that can become an SSH option."""
    result = run_script(*arguments)

    assert result.returncode == 2
    assert USAGE in result.stderr


def test_deploys_pinned_vllm_service_over_ssh(tmp_path: Path) -> None:
    """Send the pinned install and service workflow to the requested node."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_executable(
        bin_dir / "scp",
        """#!/usr/bin/env bash
printf '%s\n' "$@" > "$DEPLOY_TEST_DIR/scp-args"
""",
    )
    write_executable(
        bin_dir / "ssh",
        """#!/usr/bin/env bash
printf '%s\n' "$@" > "$DEPLOY_TEST_DIR/ssh-args"
cat > "$DEPLOY_TEST_DIR/ssh-stdin"
""",
    )
    environment = os.environ.copy()
    environment["DEPLOY_TEST_DIR"] = str(tmp_path)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"

    result = run_script(
        "ubuntu@gpu.example",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert os.access(SCRIPT, os.X_OK)

    scp_arguments = (tmp_path / "scp-args").read_text(
        encoding="utf-8",
    ).splitlines()
    assert scp_arguments[0] == str(ROOT / "scripts" / "vllm.service")
    assert scp_arguments[1].startswith(
        "ubuntu@gpu.example:/tmp/evals-vllm.service.",
    )
    assert scp_arguments[1].endswith(".service")

    remote_unit = scp_arguments[1].split(":", maxsplit=1)[1]
    ssh_arguments = (tmp_path / "ssh-args").read_text(
        encoding="utf-8",
    ).splitlines()
    assert ssh_arguments == [
        "ubuntu@gpu.example",
        "bash",
        "-s",
        "--",
        remote_unit,
    ]

    remote_program = (tmp_path / "ssh-stdin").read_text(encoding="utf-8")
    expected_commands = (
        "VLLM_VERSION=0.24.0",
        "HEALTH_ATTEMPTS=180",
        "HEALTH_INTERVAL_SECONDS=5",
        "id -un",
        "command -v curl",
        "command -v uv",
        "https://astral.sh/uv/install.sh",
        'VENV="$HOME/vllm-env"',
        '[[ ! -x "$VENV/bin/python" ]]',
        '"vllm==${VLLM_VERSION}"',
        'sudo install -m 0644 "$REMOTE_UNIT"',
        "sudo systemctl daemon-reload",
        "sudo systemctl enable vllm",
        "sudo systemctl restart vllm",
        "http://127.0.0.1:8000/health",
        'sleep "$HEALTH_INTERVAL_SECONDS"',
        "sudo systemctl status vllm --no-pager",
        "sudo journalctl -u vllm -n 100 --no-pager",
        "trap cleanup EXIT",
        'rm -f -- "$REMOTE_UNIT"',
    )
    for command in expected_commands:
        assert command in remote_program
