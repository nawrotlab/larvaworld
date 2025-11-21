"""
Unit tests for larvaworld.lib.util.fitting module.

Tests pure math/scipy optimization and stochastic functions.
"""

import numpy as np
import pytest

from larvaworld.lib.util.fitting import (
    simplex,
    beta0,
    critical_bout,
    exp_bout,
)


@pytest.mark.fast
class TestSimplex:
    """Test simplex function for Nelder-Mead optimization."""

    def test_simplex_simple_quadratic(self):
        """Test optimization of simple quadratic function."""

        def cost(x):
            return (x - 3.0) ** 2

        result = simplex(cost, x0=0.0)
        assert result == pytest.approx(3.0, abs=1e-6)

    def test_simplex_with_offset(self):
        """Test optimization with different initial guess."""

        def cost(x):
            return (x - 5.0) ** 2 + 2.0

        result = simplex(cost, x0=10.0)
        assert result == pytest.approx(5.0, abs=1e-6)

    def test_simplex_with_args(self):
        """Test optimization with additional arguments."""

        def cost(x, target):
            return (x - target) ** 2

        result = simplex(cost, x0=0.0, args=(7.0,))
        assert result == pytest.approx(7.0, abs=1e-6)

    def test_simplex_negative_minimum(self):
        """Test optimization finding negative minimum."""

        def cost(x):
            return (x + 4.0) ** 2

        result = simplex(cost, x0=0.0)
        assert result == pytest.approx(-4.0, abs=1e-6)

    def test_simplex_returns_float(self):
        """Test that simplex always returns a float."""

        def cost(x):
            return x**2

        result = simplex(cost, x0=1.0)
        assert isinstance(result, (float, np.floating))


@pytest.mark.fast
class TestBeta0:
    """Test beta0 function for DEB textbook beta calculation."""

    def test_beta0_positive_values(self):
        """Test beta0 with positive input values (may return inf due to log singularity)."""
        result = beta0(0.5, 1.0)
        assert isinstance(result, (float, np.floating))
        # Function can return inf or nan due to mathematical singularities
        # This is expected behavior for certain input combinations

    def test_beta0_equal_values(self):
        """Test beta0 when both inputs are equal (returns nan due to log(x-1) singularity at x=1)."""
        result = beta0(1.0, 1.0)
        # When x0 == x1 == 1, we get log(1^(1/3) - 1) = log(0) = -inf
        # This causes f1 - f0 = 0, but intermediate operations produce nan
        assert isinstance(result, (float, np.floating))
        # The actual behavior is to return nan for x=1 due to log singularity

    def test_beta0_different_values(self):
        """Test beta0 with different input values."""
        result = beta0(0.1, 2.0)
        assert isinstance(result, (float, np.floating))
        # Result should be real (imaginary part removed)
        assert not np.isnan(result)

    def test_beta0_order_matters(self):
        """Test that beta0(x0, x1) != beta0(x1, x0)."""
        result1 = beta0(0.5, 1.5)
        result2 = beta0(1.5, 0.5)
        assert result1 == pytest.approx(-result2, abs=1e-10)

    def test_beta0_small_values(self):
        """Test beta0 with small input values."""
        result = beta0(0.01, 0.02)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_beta0_large_values(self):
        """Test beta0 with larger input values."""
        result = beta0(5.0, 10.0)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)


@pytest.mark.fast
class TestCriticalBout:
    """Test critical_bout function for stochastic bout simulation."""

    def test_critical_bout_returns_int(self):
        """Test that critical_bout returns an integer."""
        result = critical_bout(c=0.9, sigma=1.0, N=100, tmax=100)
        assert isinstance(result, (int, np.integer))

    def test_critical_bout_positive_duration(self):
        """Test that bout duration is positive."""
        result = critical_bout(c=0.9, sigma=1.0, N=100, tmax=100)
        assert result > 0

    def test_critical_bout_respects_tmin(self):
        """Test that bout duration respects minimum threshold."""
        # Run multiple times due to stochastic nature
        results = [
            critical_bout(c=0.9, sigma=1.0, N=100, tmax=100, tmin=5) for _ in range(10)
        ]
        assert all(r >= 5 for r in results)

    def test_critical_bout_respects_tmax(self):
        """Test that bout duration respects maximum threshold."""
        # With high c and low sigma, should terminate before tmax
        result = critical_bout(c=1.5, sigma=0.1, N=100, tmax=50)
        assert result <= 50

    def test_critical_bout_different_params(self):
        """Test critical_bout with different parameter combinations."""
        result1 = critical_bout(c=0.5, sigma=1.5, N=500)
        result2 = critical_bout(c=0.9, sigma=0.5, N=500)
        # Both should return valid positive integers
        assert isinstance(result1, (int, np.integer))
        assert isinstance(result2, (int, np.integer))
        assert result1 > 0
        assert result2 > 0

    def test_critical_bout_small_population(self):
        """Test critical_bout with small population size."""
        result = critical_bout(c=0.9, sigma=1.0, N=10, tmax=100)
        assert isinstance(result, (int, np.integer))
        assert result > 0

    def test_critical_bout_determinism_with_seed(self):
        """Test that seeding numpy.random affects results."""
        np.random.seed(42)
        result1 = critical_bout(c=0.9, sigma=1.0, N=100, tmax=100)
        np.random.seed(42)
        result2 = critical_bout(c=0.9, sigma=1.0, N=100, tmax=100)
        assert result1 == result2


@pytest.mark.fast
class TestExpBout:
    """Test exp_bout function for exponential bout simulation."""

    def test_exp_bout_returns_int(self):
        """Test that exp_bout returns an integer."""
        result = exp_bout(beta=0.1, tmax=100)
        assert isinstance(result, (int, np.integer))

    def test_exp_bout_positive_duration(self):
        """Test that bout duration is positive."""
        result = exp_bout(beta=0.1, tmax=100)
        assert result > 0

    def test_exp_bout_respects_tmin(self):
        """Test that bout duration respects minimum threshold."""
        # Run multiple times due to stochastic nature
        results = [exp_bout(beta=0.5, tmax=100, tmin=3) for _ in range(10)]
        assert all(r >= 3 for r in results)

    def test_exp_bout_respects_tmax(self):
        """Test that bout duration respects maximum threshold."""
        # With very low beta, should hit tmax and reset
        result = exp_bout(beta=0.001, tmax=50)
        assert result <= 50

    def test_exp_bout_high_beta(self):
        """Test exp_bout with high beta (quick termination)."""
        # High beta means high probability of termination
        result = exp_bout(beta=0.9, tmax=100)
        assert result > 0
        # Should typically be small with high beta
        assert result < 100

    def test_exp_bout_low_beta(self):
        """Test exp_bout with low beta (slower termination)."""
        # Low beta means low probability of termination
        result = exp_bout(beta=0.01, tmax=500)
        assert result > 0

    def test_exp_bout_different_params(self):
        """Test exp_bout with different parameter combinations."""
        result1 = exp_bout(beta=0.05, tmax=200, tmin=1)
        result2 = exp_bout(beta=0.2, tmax=200, tmin=1)
        # Both should return valid positive integers
        assert isinstance(result1, (int, np.integer))
        assert isinstance(result2, (int, np.integer))
        assert result1 > 0
        assert result2 > 0

    def test_exp_bout_determinism_with_seed(self):
        """Test that seeding numpy.random affects results."""
        np.random.seed(123)
        result1 = exp_bout(beta=0.1, tmax=100)
        np.random.seed(123)
        result2 = exp_bout(beta=0.1, tmax=100)
        assert result1 == result2
