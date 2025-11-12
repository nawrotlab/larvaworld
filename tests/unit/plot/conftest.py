"""
Pytest configuration for plot tests.

This module configures matplotlib for headless plotting and ensures
all figures are properly closed after each test to prevent memory leaks.
"""

import matplotlib

matplotlib.use("Agg")  # Set headless backend before importing pyplot

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Auto-close all matplotlib figures after each test.

    This prevents memory bloat during parallel test execution and ensures
    clean state between tests.
    """
    yield
    plt.close("all")
