"""
API consistency tests for LarvaDataset and related classes.

Pure import/identity checks - NO I/O, NO dependencies.
FASTEST tests in the suite (~50ms total runtime).
"""

import importlib
import pytest


@pytest.mark.fast
def test_larvadataset_alias_identity():
    """
    Verify LarvaDataset is the same object across all import paths.

    Tests:
    - lib.process.dataset.LarvaDataset
    - lib.process.LarvaDataset
    - lib.LarvaDataset
    """
    m_dataset = importlib.import_module("larvaworld.lib.process.dataset")
    m_process = importlib.import_module("larvaworld.lib.process")
    m_lib = importlib.import_module("larvaworld.lib")

    LD1 = m_dataset.LarvaDataset
    LD2 = m_process.LarvaDataset
    LD3 = m_lib.LarvaDataset

    assert LD1 is LD2, "dataset.LarvaDataset !== process.LarvaDataset"
    assert LD2 is LD3, "process.LarvaDataset !== lib.LarvaDataset"


@pytest.mark.fast
def test_larvadataset_collection_alias_identity():
    """Verify LarvaDatasetCollection alias identity."""
    m_dataset = importlib.import_module("larvaworld.lib.process.dataset")
    m_process = importlib.import_module("larvaworld.lib.process")
    m_lib = importlib.import_module("larvaworld.lib")

    LDC1 = m_dataset.LarvaDatasetCollection
    LDC2 = m_process.LarvaDatasetCollection
    LDC3 = m_lib.LarvaDatasetCollection

    assert LDC1 is LDC2
    assert LDC2 is LDC3


@pytest.mark.fast
def test_evaluation_importable():
    """Verify Evaluation and EvalRun can be imported."""
    from larvaworld.lib.process import Evaluation
    from larvaworld.lib.sim import EvalRun

    assert isinstance(Evaluation, type), "Evaluation is not a class"
    assert isinstance(EvalRun, type), "EvalRun is not a class"


@pytest.mark.fast
def test_autoplot_constructible():
    """Verify AutoPlot can be constructed without I/O."""
    from larvaworld.lib.plot.base import AutoPlot

    ap = AutoPlot(datasets=[], labels=[])

    assert hasattr(ap, "datasets")
    assert isinstance(ap.datasets, list)


@pytest.mark.fast
def test_lib_process_lazy_exports():
    """
    Verify lazy export maps in lib/process facades.

    Checks registry keys to ensure re-exports are maintained.
    Protects against future refactors breaking import paths.
    """
    m_process = importlib.import_module("larvaworld.lib.process")

    # Check expected exports exist in __all__ or __dir__
    expected_exports = [
        "LarvaDataset",
        "LarvaDatasetCollection",
        "Evaluation",
    ]

    available = dir(m_process)
    for export in expected_exports:
        assert export in available, f"{export} not in lib.process facade"


@pytest.mark.fast
def test_lib_lazy_exports():
    """Verify lazy export maps in lib facade."""
    m_lib = importlib.import_module("larvaworld.lib")

    expected_exports = [
        "LarvaDataset",
        "LarvaDatasetCollection",
    ]

    # Check __all__ for lazy-loaded exports (dir() doesn't show them until accessed)
    assert hasattr(m_lib, "__all__"), "lib module missing __all__"

    for export in expected_exports:
        # Check declared in __all__
        assert export in m_lib.__all__, f"{export} not in lib.__all__"

        # Verify lazy loading works (trigger __getattr__)
        obj = getattr(m_lib, export)
        assert obj is not None, f"{export} lazy loading failed"
