"""
Test configuration and fixtures for Larvaworld tests.

This module provides essential fixtures for:
- Deterministic testing (fixed seeds)
- Headless operation (no GUI windows)
- Temporary directory for test outputs (via tmp_path)
- Registry isolation (via parallel execution with pytest-xdist)
"""

import functools
import os
import random
from pathlib import Path

import numpy as np
import pytest


TESTS_ROOT = Path(__file__).parent.resolve()
LEGACY_TEST_REL_PATHS = {
    # Original integration suite (pre-refactor)
    "integration/test_analysis.py",
    "integration/test_evaluation.py",
    "integration/cli/test_cli_entrypoints.py",
    "integration/process/test_import_aux.py",
    "integration/process/test_import_legacy.py",
    "integration/plot/test_plotting_legacy.py",
    "integration/sim/test_sim_box2d.py",
    "integration/sim/test_sim_experiments.py",
    "integration/sim/test_sim_ga.py",
    "integration/sim/test_sim_replay.py",
    # Original unit suite (pre-refactor)
    "unit/model/test_model_space_dict.py",
    "unit/param/test_param_classes.py",
    "unit/util/test_util_ang.py",
    "unit/util/test_util_fft.py",
}

INTEGRATION_SLOW_PATHS = {
    "integration/test_analysis.py",
    "integration/test_evaluation.py",
    "integration/test_calibration_real.py",
    "integration/plot/test_plotting_legacy.py",
    "integration/process/test_import_legacy.py",
    "integration/process/test_import_schleyer.py",
    "integration/sim/test_sim_box2d.py",
    "integration/sim/test_sim_experiments.py",
    "integration/sim/test_sim_ga.py",
    "integration/sim/test_sim_replay.py",
}

INTEGRATION_FAST_PATHS = {
    "integration/cli/test_cli_entrypoints.py",
    "integration/process/test_import_aux.py",
    "integration/reg/test_registry_bootstrap.py",
    "integration/stored_confs/test_essay_conf_integration.py",
}

OPTIONAL_DEP_PATHS = {
    "integration/sim/test_sim_box2d.py",
}


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for marker-aware filtering."""
    marker_definitions = {
        "fast": "Fast unit/logic tests without heavy sims or plotting.",
        "slow": "Heavier tests (simulations, GA runs, or end-to-end plotting).",
        "integration": "Tests that exercise real Larvaworld subsystems (registry, pipelines, simulations).",
        "optional_dep": "Tests that rely on optional third-party dependencies.",
        "legacy": "Legacy Larvaworld test suite (pre-refactor).",
    }
    for name, description in marker_definitions.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark tests based on their location for easier filtering."""
    for item in items:
        try:
            rel_path = Path(item.fspath).resolve().relative_to(TESTS_ROOT).as_posix()
        except ValueError:
            continue

        if rel_path in LEGACY_TEST_REL_PATHS:
            item.add_marker("legacy")

        if rel_path.startswith("unit/") and item.get_closest_marker("fast") is None:
            item.add_marker("fast")

        if rel_path.startswith("integration/"):
            if item.get_closest_marker("integration") is None:
                item.add_marker("integration")
            if rel_path in INTEGRATION_SLOW_PATHS:
                item.add_marker("slow")
            elif (
                rel_path in INTEGRATION_FAST_PATHS
                or item.get_closest_marker("fast") is None
            ):
                item.add_marker("fast")

        if rel_path in OPTIONAL_DEP_PATHS:
            item.add_marker("optional_dep")


@pytest.fixture(autouse=True, scope="session")
def deterministic():
    """Fix all randomness for reproducible tests"""
    random.seed(42)
    np.random.seed(42)


@pytest.fixture(autouse=True)
def headless_mode(monkeypatch):
    """Force headless backends to prevent GUI windows during tests"""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Reusable test data directory for session-scoped test data"""
    return tmp_path_factory.mktemp("test_data")


# Registry isolation strategy:
#
# With parallel execution (pytest -n auto), each worker runs in a separate process
# with its own registry. This automatically solves the test pollution problem.
#
# Benefits:
# - Each test gets a fresh Python interpreter
# - No shared state between tests
# - Fixes the CLI test failures (test_cli_replay_args, test_cli_evaluation_args)
#
# For debugging individual tests (pytest -n 0 or single test):
# - Tests may experience pollution in serial mode
# - Option A: Run tests in isolation: pytest tests/test_cli.py::test_specific_function
# - Option B: Use --forked flag (requires pytest-forked)
# - Option C: Clean cache between runs: python clean_cache_cold_start.py
#
# Note: We allow tests to use the real registry and data directories.
# Test artifacts should be cleaned up manually if needed (see clean_cache_cold_start.py)


# ============================================================================
# LarvaDatasetStub - Minimal, reusable mock for unit tests (no I/O)
# ============================================================================

import pandas as pd
from types import SimpleNamespace


class LarvaDatasetStub:
    """
    Minimal LarvaDataset stand-in for unit testing.

    Provides synthetic trajectory data with NO I/O dependency.
    For simple cases only (sim tests, geo helpers).

    For complex orchestrators (calibration.py), use surgical monkeypatching instead!

    Args:
        n: Number of timesteps
        fps: Frames per second
        seed: Random seed for determinism
    """

    def __init__(self, n=30, fps=10.0, seed=42):
        rng = np.random.default_rng(seed)

        # Time array
        self.timestamps = np.arange(n) / fps

        # Random walk trajectory
        x = np.cumsum(rng.normal(0, 0.05, n))
        y = np.cumsum(rng.normal(0, 0.05, n))
        self.xy = np.column_stack([x, y])

        # Mimic d.data structure
        self.data = (
            SimpleNamespace(),  # step_data
            SimpleNamespace(),  # endpoint_data
            {"fps": fps, "dt": 1.0 / fps},  # config
        )

        self.config = {"fps": fps, "dt": 1.0 / fps}
        self.step_data = SimpleNamespace()
        self.endpoint_data = SimpleNamespace()

    def load_traj(self):
        """Return trajectory array."""
        return self.xy

    def comp_spatial(self):
        """Compute basic spatial metrics."""
        dt = self.timestamps[1] - self.timestamps[0]
        vx = np.gradient(self.xy[:, 0], dt)
        vy = np.gradient(self.xy[:, 1], dt)
        self.velocity = np.hypot(vx, vy)
        self.angle = np.arctan2(vy, vx)
        self.step_data.v = self.velocity
        self.step_data.angle = self.angle
        return self


@pytest.fixture()
def ds_stub():
    """Provide a LarvaDatasetStub with precomputed spatial metrics."""
    d = LarvaDatasetStub(n=30, fps=10.0, seed=42)
    d.comp_spatial()
    return d


@pytest.fixture(scope="session")
def real_dataset(ensure_datasets_ready):
    """
    Provide a REAL LarvaDataset from registry for tests that need it.

    Requires ensure_datasets_ready fixture (datasets must exist).
    Uses exploration.30controls as minimal real dataset.

    Returns:
        LarvaDataset with full preprocessing applied.
    """
    from larvaworld.lib.process import LarvaDataset

    # Load real dataset from registry (datasets guaranteed ready via fixture)
    d = LarvaDataset(refID="exploration.30controls")

    # Full preprocessing pipeline
    d.comp_spatial()
    d.comp_orientations()
    d.comp_bend(mode="full")  # mode="full" computes spine angles
    d.comp_ang_moments()  # Computes angular velocities for spine angles

    return d


# ============================================================================
# Optional Auto-Setup for Integration Tests (ENV-GATED, NO autouse!)
# Based on GPT-5 robust FileLock solution for HDF5 race conditions
# ============================================================================

import time
import subprocess
import sys

try:
    from filelock import FileLock
except ImportError:
    FileLock = None  # Will skip if not available

# Configuration
DATA_ROOT = Path("src/larvaworld/data/SchleyerGroup/processed")
READY_FLAG = Path(".pytest_datasets_ready")  # Signal file for workers
LOCK_FILE = Path(".pytest_datasets_build.lock")  # Build synchronization


def processed_datasets_exist() -> bool:
    """
    Check if minimal processed artifacts exist.

    Stable predicates: checks representative files (data.h5, conf.txt).
    Keep this cheap & deterministic.
    """
    target = DATA_ROOT / "exploration" / "30controls" / "data"
    return (target / "data.h5").exists() and (target / "conf.txt").exists()


def build_processed_datasets():
    """
    Build datasets from raw → processed (idempotent).

    IMPORTANT: This must be idempotent and safe if called on warm start.
    Delegates to `larvaworld.tests.init_datasets` which handles registry bootstrap
    and Schleyer dataset imports. Keeps the heavy lifting outside the fixture so
    it can also be reused in CI preparation steps.
    """
    script = Path(__file__).with_name("init_datasets.py")
    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        timeout=900,  # 15 min max (generous for CI)
        capture_output=True,
    )
    _patch_dataset_io_lock()


def _patch_dataset_io_lock() -> None:
    """Wrap LarvaDataset HDF5 writes with FileLock to avoid concurrent writers."""
    if FileLock is None:
        return

    from larvaworld.lib.process import dataset as _dataset_module

    lock = FileLock(str(LOCK_FILE))
    patched_attr = "_pytest_dataset_io_patch"
    if getattr(_dataset_module, patched_attr, False):
        return

    def wrap_method(method_name: str) -> None:
        original = getattr(_dataset_module.LarvaDataset, method_name, None)
        if original is None:
            return

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            with lock:
                return original(self, *args, **kwargs)

        setattr(_dataset_module.LarvaDataset, method_name, wrapped)

    for name in ["annotate", "save", "_save_step", "_save_end", "store"]:
        wrap_method(name)

    setattr(_dataset_module, patched_attr, True)


@pytest.fixture()
def dataset_lock():
    """Ensure LarvaDataset I/O methods are patched without holding a re-entrant lock."""
    _patch_dataset_io_lock()
    yield


@pytest.fixture(scope="session")  # ✅ NO autouse=True!
def ensure_datasets_ready():
    """
    Session-level gate: create processed datasets ONCE.

    Solves HDF5 race condition on cold starts:
    - If datasets exist: instant return (warm start)
    - If cold start: ONE worker builds, others wait for READY_FLAG
    - After build: ALL workers read existing HDF5 (no write locks)

    Usage:
        @pytest.mark.usefixtures("ensure_datasets_ready")
        def test_analysis():
            # Test will wait until datasets ready

    Enable integration tests with: LARVAWORLD_INIT_DATA=1 pytest
    Disable explicitly with: LARVAWORLD_INIT_DATA=0 pytest
    """
    # Opt-in/opt-out via env flag
    if os.getenv("LARVAWORLD_INIT_DATA") == "0":
        pytest.skip("Datasets init disabled (LARVAWORLD_INIT_DATA=0)")

    # Warm start: instant bypass
    if processed_datasets_exist() or READY_FLAG.exists():
        return

    # Check FileLock availability
    if FileLock is None:
        pytest.skip("filelock not installed; cannot safely create datasets in parallel")

    lock = FileLock(str(LOCK_FILE))
    got_lock = False

    try:
        # Non-blocking acquire (timeout=0) - GPT-5 pattern
        got_lock = lock.acquire(timeout=0)
    except Exception:
        got_lock = False

    if got_lock:
        # This worker will BUILD datasets
        try:
            # Double-check after acquiring lock (race condition safety)
            if not processed_datasets_exist():
                build_processed_datasets()

            # Signal other workers that datasets are ready
            READY_FLAG.write_text("ok")
        finally:
            lock.release()
    else:
        # This worker will WAIT for builder to finish
        for _ in range(1200):  # 10 min max (0.5s * 1200 = 600s)
            if READY_FLAG.exists() or processed_datasets_exist():
                break
            time.sleep(0.5)
        else:
            pytest.fail("Timeout waiting for processed datasets ready flag")

    # Hint HDF5 for shared read locks (optional but harmless)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "TRUE")


def _bootstrap_datasets_once() -> None:
    """Ensure processed datasets (and registry defaults) exist before collection."""
    if os.getenv("LARVAWORLD_INIT_DATA") == "0":
        return

    ready = processed_datasets_exist()
    if READY_FLAG.exists():
        ready = True

    if ready and not READY_FLAG.exists():
        READY_FLAG.write_text("ok")

    if ready:
        _patch_dataset_io_lock()
        return

    if FileLock is None:
        build_processed_datasets()
        READY_FLAG.write_text("ok")
        return

    lock = FileLock(str(LOCK_FILE))
    with lock:
        if not processed_datasets_exist():
            build_processed_datasets()
        READY_FLAG.write_text("ok")

    try:
        from larvaworld.lib import reg

        reg.define_default_refID()
    except Exception:
        # Any registry bootstrap issues will surface later in the tests.
        pass


def _wait_for_datasets(timeout: float = 600.0) -> None:
    """Worker processes wait until READY flag appears."""
    if os.getenv("LARVAWORLD_INIT_DATA") == "0":
        return

    if processed_datasets_exist():
        _patch_dataset_io_lock()
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        if READY_FLAG.exists() or processed_datasets_exist():
            _patch_dataset_io_lock()
            return
        time.sleep(0.5)

    pytest.exit("Timeout waiting for dataset initialisation (READY flag missing).")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Bootstrap datasets/registry before pytest collection (supports xdist)."""
    if os.getenv("LARVAWORLD_INIT_DATA") == "0":
        return

    if hasattr(session.config, "workerinput"):
        _wait_for_datasets()
        _patch_dataset_io_lock()
    else:
        _bootstrap_datasets_once()
