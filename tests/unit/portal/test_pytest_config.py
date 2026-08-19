"""Guards on the test-suite configuration itself.

`-n auto` used to be forced through `addopts`. Every xdist worker pays a ~4s
larvaworld registry import, which dominated the actual test work: a 9-test run
took 104s instead of 5s, and even the full portal suite was slower in parallel
(838s vs 746s). Parallelism and coverage are therefore opt-in, and CI passes
them explicitly. These tests keep that from being undone by accident.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def addopts() -> str:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return config["tool"]["pytest"]["ini_options"]["addopts"]


def test_parallelism_is_not_forced_on_every_run(addopts: str) -> None:
    assert "-n auto" not in addopts
    assert "-n=" not in addopts


def test_coverage_is_not_forced_on_every_run(addopts: str) -> None:
    assert "--cov" not in addopts


@pytest.mark.skipif(not CI_WORKFLOW.is_file(), reason="CI workflow not present")
def test_ci_still_requests_parallelism_and_coverage() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    pytest_lines = [
        ln for ln in workflow.splitlines() if "pytest" in ln and "run:" in ln
    ]

    assert pytest_lines, "expected at least one pytest invocation in CI"
    for line in pytest_lines:
        assert "-n auto" in line, f"CI lost parallelism: {line.strip()}"
        assert "--cov=larvaworld" in line, f"CI lost coverage: {line.strip()}"


@pytest.mark.skipif(not CI_WORKFLOW.is_file(), reason="CI workflow not present")
def test_ci_enables_the_exhaustive_sweeps_skipped_locally() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "LARVAWORLD_EXHAUSTIVE_TESTS" in workflow
