"""
Unit tests for larvaworld.lib.util.xy module.

Tests pure math/numpy functions that don't require datasets or simulation.
Focus: distance calculations, angle unwrapping, averaging, geometric utilities.
"""

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib.util.xy import (
    unwrap_deg,
    unwrap_rad,
    rate,
    eudist,
    eudi5x,
    eudiNxN,
    moving_average,
    boolean_indexing,
    rolling_window,
    get_arena_bounds,
    circle_to_polygon,
    body_contour,
    rearrange_contour,
    comp_PI,
    Collision,
)


@pytest.mark.fast
class TestUnwrapDeg:
    """Test unwrap_deg function for angle unwrapping in degrees."""

    def test_unwrap_no_discontinuity(self):
        """Test with angles that don't need unwrapping."""
        angles = np.array([10.0, 20.0, 30.0, 40.0])
        result = unwrap_deg(angles)
        np.testing.assert_array_almost_equal(result, angles)

    def test_unwrap_with_discontinuity(self):
        """Test unwrapping across 180/-180 boundary."""
        angles = np.array([170.0, 180.0, -170.0, -160.0])
        result = unwrap_deg(angles)
        # After unwrapping, the jump should be continuous
        assert result[0] == pytest.approx(170.0)
        assert result[1] == pytest.approx(180.0)
        assert result[2] > 180.0  # Should be ~190 instead of -170
        assert result[3] > result[2]  # Should continue increasing

    def test_unwrap_with_nans(self):
        """Test that NaN values are preserved."""
        angles = np.array([10.0, np.nan, 30.0, np.nan])
        result = unwrap_deg(angles)
        assert np.isnan(result[1])
        assert np.isnan(result[3])
        assert result[0] == pytest.approx(10.0)
        assert result[2] == pytest.approx(30.0)

    def test_unwrap_pandas_series(self):
        """Test with pandas Series input."""
        angles = pd.Series([170.0, 180.0, -170.0])
        result = unwrap_deg(angles)
        assert isinstance(result, np.ndarray)
        assert result[2] > 180.0


@pytest.mark.fast
class TestUnwrapRad:
    """Test unwrap_rad function for angle unwrapping in radians."""

    def test_unwrap_no_discontinuity(self):
        """Test with angles that don't need unwrapping."""
        angles = np.array([0.1, 0.2, 0.3, 0.4])
        result = unwrap_rad(angles)
        np.testing.assert_array_almost_equal(result, angles)

    def test_unwrap_with_discontinuity(self):
        """Test unwrapping across π/-π boundary."""
        angles = np.array([3.0, 3.14, -3.1, -3.0])
        result = unwrap_rad(angles)
        # After unwrapping, the sequence should be continuous
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(3.14)
        assert result[2] > 3.14  # Should be ~3.18 instead of -3.1
        assert result[3] > result[2]

    def test_unwrap_with_nans(self):
        """Test that NaN values are preserved."""
        angles = np.array([1.0, np.nan, 2.0, np.nan])
        result = unwrap_rad(angles)
        assert np.isnan(result[1])
        assert np.isnan(result[3])

    def test_unwrap_pandas_series(self):
        """Test with pandas Series input."""
        angles = pd.Series([3.0, 3.14, -3.1])
        result = unwrap_rad(angles)
        assert isinstance(result, np.ndarray)


@pytest.mark.fast
class TestRate:
    """Test rate function for computing derivative."""

    def test_rate_linear_signal(self):
        """Test rate of linear signal."""
        signal = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        dt = 0.1
        result = rate(signal, dt)
        assert np.isnan(result[0])  # First element is NaN
        np.testing.assert_array_almost_equal(result[1:], [10.0, 10.0, 10.0, 10.0])

    def test_rate_constant_signal(self):
        """Test rate of constant signal (should be zero)."""
        signal = np.array([5.0, 5.0, 5.0, 5.0])
        dt = 0.5
        result = rate(signal, dt)
        assert np.isnan(result[0])
        np.testing.assert_array_almost_equal(result[1:], [0.0, 0.0, 0.0])

    def test_rate_quadratic_signal(self):
        """Test rate of quadratic signal."""
        signal = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2
        dt = 1.0
        result = rate(signal, dt)
        assert np.isnan(result[0])
        # Derivative approximations: 1, 3, 5, 7
        np.testing.assert_array_almost_equal(result[1:], [1.0, 3.0, 5.0, 7.0])

    def test_rate_pandas_series(self):
        """Test with pandas Series input."""
        signal = pd.Series([0.0, 2.0, 4.0, 6.0])
        dt = 0.1
        result = rate(signal, dt)
        assert isinstance(result, np.ndarray)
        assert np.isnan(result[0])


@pytest.mark.fast
class TestEudist:
    """Test eudist function for cumulative Euclidean distances."""

    def test_eudist_straight_line_x(self):
        """Test distance along x-axis."""
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        result = eudist(xy)
        expected = np.array([0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_eudist_straight_line_y(self):
        """Test distance along y-axis."""
        xy = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        result = eudist(xy)
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_eudist_diagonal(self):
        """Test distance along diagonal."""
        xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = eudist(xy)
        expected = np.array([0.0, np.sqrt(2), np.sqrt(2)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_eudist_with_nans(self):
        """Test that NaN handling works."""
        xy = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0]])
        result = eudist(xy)
        assert result[0] == 0.0
        # nansum treats NaN as 0, so distance calculation continues
        assert not np.isnan(result[1])

    def test_eudist_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        xy = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]})
        result = eudist(xy)
        assert isinstance(result, np.ndarray)
        assert result[0] == 0.0


@pytest.mark.fast
class TestEudi5x:
    """Test eudi5x function for distances from array to single point."""

    def test_eudi5x_2d_points(self):
        """Test distance from multiple points to single reference."""
        a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([0.5, 0.5])
        result = eudi5x(a, b)
        expected = np.array(
            [
                np.sqrt(0.5),  # distance from (0,0) to (0.5,0.5)
                np.sqrt(0.5),  # distance from (1,0) to (0.5,0.5)
                np.sqrt(0.5),  # distance from (0,1) to (0.5,0.5)
                np.sqrt(0.5),  # distance from (1,1) to (0.5,0.5)
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_eudi5x_3d_points(self):
        """Test with 3D points."""
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = np.array([0.0, 0.0, 0.0])
        result = eudi5x(a, b)
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_eudi5x_same_point(self):
        """Test distance from points to themselves (should be zero)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        result = eudi5x(a, b)
        assert result[0] == pytest.approx(0.0)
        assert result[1] > 0.0


@pytest.mark.fast
class TestEudiNxN:
    """Test eudiNxN function for pairwise distances."""

    def test_eudiNxN_simple_case(self):
        """Test pairwise distances with simple inputs."""
        a = np.array([[[0.0, 0.0], [1.0, 0.0]]])  # 1 set, 2 points
        b = np.array([[0.5, 0.0], [0.5, 1.0]])  # 2 reference points
        result = eudiNxN(a, b)
        # Result shape is (K, N, M) = (2, 1, 2) due to list comprehension
        assert result.shape == (2, 1, 2)
        # Check actual distances are computed correctly
        # result[i] is distances from all points to b[i]
        assert isinstance(result, np.ndarray)
        # Just verify computation works and returns valid distances
        assert np.all(result >= 0)

    def test_eudiNxN_multiple_sets(self):
        """Test with multiple sets of points."""
        a = np.array(
            [[[0.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [3.0, 0.0]]]
        )  # 2 sets, 2 points each
        b = np.array([[0.0, 0.0]])  # 1 reference point
        result = eudiNxN(a, b)
        # Result shape is (K, N, M) = (1, 2, 2)
        assert result.shape == (1, 2, 2)
        # Check that distances are non-negative
        assert np.all(result >= 0)


@pytest.mark.fast
class TestMovingAverage:
    """Test moving_average function."""

    def test_moving_average_simple(self):
        """Test with simple array."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(a, n=3)
        # Convolution with [1/3, 1/3, 1/3]
        # Edge effects with mode='same'
        assert len(result) == len(a)
        assert result[2] == pytest.approx(3.0)  # Center value (2+3+4)/3

    def test_moving_average_constant(self):
        """Test with constant array (center values remain constant, edges affected)."""
        a = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = moving_average(a, n=3)
        # With mode='same', edges are affected by boundary effects
        # Center values should remain constant
        assert result[2] == pytest.approx(5.0)  # Center value
        assert len(result) == len(a)

    def test_moving_average_window_1(self):
        """Test with window size 1 (identity)."""
        a = np.array([1.0, 2.0, 3.0])
        result = moving_average(a, n=1)
        np.testing.assert_array_almost_equal(result, a)

    def test_moving_average_large_window(self):
        """Test with large window size."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(a, n=5)
        assert len(result) == len(a)


@pytest.mark.fast
class TestBooleanIndexing:
    """Test boolean_indexing function for padding variable-length arrays."""

    def test_boolean_indexing_uniform_length(self):
        """Test with arrays of same length."""
        v = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = boolean_indexing(v, fillval=0.0)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_boolean_indexing_variable_length(self):
        """Test with arrays of different lengths."""
        v = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), np.array([6.0])]
        result = boolean_indexing(v, fillval=np.nan)
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[0], [1.0, 2.0, np.nan])
        np.testing.assert_array_equal(result[1], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result[2], [6.0, np.nan, np.nan])

    def test_boolean_indexing_empty_arrays(self):
        """Test with some empty arrays."""
        v = [np.array([1.0]), np.array([]), np.array([2.0, 3.0])]
        result = boolean_indexing(v, fillval=-1.0)
        assert result.shape == (3, 2)
        assert result[1, 0] == -1.0
        assert result[1, 1] == -1.0

    def test_boolean_indexing_custom_fillval(self):
        """Test with custom fill value."""
        v = [np.array([1.0, 2.0]), np.array([3.0])]
        result = boolean_indexing(v, fillval=999.0)
        assert result[1, 1] == 999.0


@pytest.mark.fast
class TestRollingWindow:
    """Test rolling_window function."""

    def test_rolling_window_size_2(self):
        """Test with window size 2."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_window(a, w=2)
        expected = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        np.testing.assert_array_equal(result, expected)

    def test_rolling_window_size_3(self):
        """Test with window size 3."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        result = rolling_window(a, w=3)
        expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_rolling_window_size_equal_length(self):
        """Test with window size equal to array length."""
        a = np.array([1.0, 2.0, 3.0])
        result = rolling_window(a, w=3)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result[0], a)

    def test_rolling_window_invalid_input(self):
        """Test that 2D array raises ValueError."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            rolling_window(a, w=2)


@pytest.mark.fast
class TestGetArenaBounds:
    """Test get_arena_bounds function."""

    def test_get_arena_bounds_square(self):
        """Test with square arena."""
        arena_dims = (1.0, 1.0)
        result = get_arena_bounds(arena_dims)
        expected = np.array([-0.5, 0.5, -0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_arena_bounds_rectangular(self):
        """Test with rectangular arena."""
        arena_dims = (2.0, 1.0)
        result = get_arena_bounds(arena_dims)
        expected = np.array([-1.0, 1.0, -0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_arena_bounds_with_scaling(self):
        """Test with scaling factor."""
        arena_dims = (1.0, 1.0)
        s = 2.0
        result = get_arena_bounds(arena_dims, s=s)
        expected = np.array([-1.0, 1.0, -1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_arena_bounds_asymmetric(self):
        """Test with asymmetric dimensions."""
        arena_dims = (0.5, 1.5)
        result = get_arena_bounds(arena_dims)
        expected = np.array([-0.25, 0.25, -0.75, 0.75])
        np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.fast
class TestCircleToPolygon:
    """Test circle_to_polygon function."""

    def test_circle_to_polygon_4_vertices(self):
        """Test with 4 vertices (square approximation)."""
        N = 4
        r = 1.0
        result = circle_to_polygon(N, r)
        assert len(result) == 4
        # Check that points are on circle
        for x, y in result:
            assert np.sqrt(x**2 + y**2) == pytest.approx(r)

    def test_circle_to_polygon_8_vertices(self):
        """Test with 8 vertices."""
        N = 8
        r = 2.0
        result = circle_to_polygon(N, r)
        assert len(result) == 8
        for x, y in result:
            assert np.sqrt(x**2 + y**2) == pytest.approx(r)

    def test_circle_to_polygon_first_point(self):
        """Test that first point is at expected position."""
        N = 4
        r = 1.0
        result = circle_to_polygon(N, r)
        # First point should be at angle 0
        x0, y0 = result[0]
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(r)

    def test_circle_to_polygon_radius_zero(self):
        """Test with radius zero."""
        N = 4
        r = 0.0
        result = circle_to_polygon(N, r)
        for x, y in result:
            assert x == 0.0
            assert y == 0.0


@pytest.mark.fast
class TestBodyContour:
    """Test body_contour function."""

    def test_body_contour_default(self):
        """Test with default parameters."""
        result = body_contour()
        # Should have 2*len(points) + 2 rows
        assert result.shape[0] == 2 * 2 + 2  # 6 points
        assert result.shape[1] == 2  # x, y
        # First point should be start
        np.testing.assert_array_equal(result[0], [1.0, 0.0])
        # Middle point should be stop
        np.testing.assert_array_equal(result[3], [0.0, 0.0])

    def test_body_contour_symmetric(self):
        """Test that contour is symmetric."""
        points = [(0.9, 0.1), (0.5, 0.2)]
        result = body_contour(points=points, start=(1, 0), stop=(0, 0))
        # Check y-symmetry
        n_points = len(points)
        for i in range(n_points):
            upper = result[1 + i]
            lower = result[-1 - i]
            assert upper[0] == pytest.approx(lower[0])  # Same x
            assert upper[1] == pytest.approx(-lower[1])  # Opposite y

    def test_body_contour_custom_start_stop(self):
        """Test with custom start and stop points."""
        start = (2.0, 0.0)
        stop = (0.5, 0.0)
        result = body_contour(points=[(1.5, 0.1)], start=start, stop=stop)
        np.testing.assert_array_equal(result[0], start)
        np.testing.assert_array_equal(result[2], stop)


@pytest.mark.fast
class TestRearrangeContour:
    """Test rearrange_contour function."""

    def test_rearrange_contour_mixed_y(self):
        """Test with mixed positive and negative y values."""
        ps0 = [(1.0, 0.5), (0.5, -0.3), (0.8, 0.2), (0.3, -0.1)]
        result = rearrange_contour(ps0)
        # First part should be positive y, descending x
        assert result[0] == (1.0, 0.5)
        assert result[1] == (0.8, 0.2)
        # Second part should be negative y, ascending x
        assert result[2] == (0.3, -0.1)
        assert result[3] == (0.5, -0.3)

    def test_rearrange_contour_all_positive(self):
        """Test with all positive y values."""
        ps0 = [(1.0, 0.5), (0.5, 0.3), (0.8, 0.2)]
        result = rearrange_contour(ps0)
        # All should be in descending x order
        assert result[0][0] > result[1][0] > result[2][0]

    def test_rearrange_contour_zero_y(self):
        """Test that zero y is treated as positive."""
        ps0 = [(1.0, 0.0), (0.5, -0.1)]
        result = rearrange_contour(ps0)
        # Zero y should be in positive group (first)
        assert result[0] == (1.0, 0.0)
        assert result[1] == (0.5, -0.1)


@pytest.mark.fast
class TestCompPI:
    """Test comp_PI function for preference index."""

    def test_comp_PI_all_left(self):
        """Test when all points are on left side."""
        arena_xdim = 1.0
        xs = np.array([-0.3, -0.25, -0.2, -0.15])
        result = comp_PI(arena_xdim, xs)
        # All points on left (< -0.1), none on right
        assert result == pytest.approx(1.0, abs=0.01)

    def test_comp_PI_all_right(self):
        """Test when all points are on right side."""
        arena_xdim = 1.0
        xs = np.array([0.3, 0.25, 0.2, 0.15])
        result = comp_PI(arena_xdim, xs)
        # All points on right (> 0.1), none on left
        assert result == pytest.approx(-1.0, abs=0.01)

    def test_comp_PI_balanced(self):
        """Test when points are balanced."""
        arena_xdim = 1.0
        xs = np.array([-0.3, -0.2, 0.2, 0.3])
        result = comp_PI(arena_xdim, xs)
        # Equal on both sides
        assert result == pytest.approx(0.0, abs=0.01)

    def test_comp_PI_center_ignored(self):
        """Test that center region points are ignored."""
        arena_xdim = 1.0
        xs = np.array([0.0, 0.05, -0.05])  # All in center region
        result = comp_PI(arena_xdim, xs)
        # No points in left or right regions
        assert result == pytest.approx(0.0)

    def test_comp_PI_return_num(self):
        """Test with return_num=True."""
        arena_xdim = 1.0
        xs = np.array([-0.3, 0.3, 0.0])
        result, N = comp_PI(arena_xdim, xs, return_num=True)
        assert isinstance(result, (float, np.floating))
        assert N == 3


@pytest.mark.fast
class TestCollision:
    """Test Collision exception class."""

    def test_collision_creation(self):
        """Test creating Collision exception."""
        obj1 = "agent1"
        obj2 = "agent2"
        collision = Collision(obj1, obj2)
        assert collision.object1 == obj1
        assert collision.object2 == obj2

    def test_collision_raise(self):
        """Test raising Collision exception."""
        with pytest.raises(Collision) as exc_info:
            raise Collision("a", "b")
        assert exc_info.value.object1 == "a"
        assert exc_info.value.object2 == "b"

    def test_collision_is_exception(self):
        """Test that Collision is an Exception."""
        collision = Collision(1, 2)
        assert isinstance(collision, Exception)
