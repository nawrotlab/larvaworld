import numpy as np
from larvaworld.lib.util import fft_max, detect_strides


def test_fft_max_basic():
    a = np.array([0, 1, 0, -1])
    dt = 1.0
    fr, yf = fft_max(a, dt, return_amps=True)
    assert np.isclose(fr, 0.25), f"Expected dominant frequency to be 0.25, but got {fr}"
    assert len(yf) == 2, f"Expected length of yf to be 2, but got {len(yf)}"


def test_fft_max_with_nan():
    a = np.array([0, 1, np.nan, -1])
    dt = 1.0
    fr, yf = fft_max(a, dt, return_amps=True)
    assert np.isclose(fr, 0.25), f"Expected dominant frequency to be 0.25, but got {fr}"
    assert len(yf) == 2, f"Expected length of yf to be 2, but got {len(yf)}"


def test_fft_max_frequency_range():
    a = np.array([0, 1, 0, -1])
    dt = 1.0
    fr = fft_max(a, dt, fr_range=(0.1, 0.3))
    assert np.isclose(fr, 0.25), f"Expected dominant frequency to be 0.25, but got {fr}"


def test_fft_max_return_amps_false():
    a = np.array([0, 1, 0, -1])
    dt = 1.0
    fr = fft_max(a, dt, return_amps=False)
    assert np.isclose(fr, 0.25), f"Expected dominant frequency to be 0.25, but got {fr}"


def test_fft_max_empty_array():
    a = np.array([])
    dt = 1.0
    fr = fft_max(a, dt, return_amps=False)
    assert np.isnan(fr), f"Expected dominant frequency to be NaN, but got {fr}"


# def test_detect_strides_basic():
#     a = np.array([0, 1, 0, -1, 0, 1, 0, -1])
#     dt = 1.0
#     strides = detect_strides(a, dt)
#     expected_strides = np.array([[0, 4], [4, 8]])
#     assert np.array_equal(strides, expected_strides), f"Expected strides to be {expected_strides}, but got {strides}"

# def test_detect_strides_with_nan():
#     a = np.array([0, 1, np.nan, -1, 0, 1, 0, -1])
#     dt = 1.0
#     strides = detect_strides(a, dt)
#     expected_strides = np.array([[0, 4], [4, 8]])
#     assert np.array_equal(strides, expected_strides), f"Expected strides to be {expected_strides}, but got {strides}"

# def test_detect_strides_with_custom_vel_thr():
#     a = np.array([0, 0.5, 0, -0.5, 0, 0.5, 0, -0.5])
#     dt = 1.0
#     vel_thr = 0.2
#     strides = detect_strides(a, dt, vel_thr=vel_thr)
#     expected_strides = np.array([[0, 4], [4, 8]])
#     assert np.array_equal(strides, expected_strides), f"Expected strides to be {expected_strides}, but got {strides}"

# def test_detect_strides_with_custom_stretch():
#     a = np.array([0, 1, 0, -1, 0, 1, 0, -1])
#     dt = 1.0
#     stretch = (0.5, 1.5)
#     strides = detect_strides(a, dt, stretch=stretch)
#     expected_strides = np.array([[0, 4], [4, 8]])
#     assert np.array_equal(strides, expected_strides), f"Expected strides to be {expected_strides}, but got {strides}"

# def test_detect_strides_with_custom_fr():
#     a = np.array([0, 1, 0, -1, 0, 1, 0, -1])
#     dt = 1.0
#     fr = 0.5
#     strides = detect_strides(a, dt, fr=fr)
#     expected_strides = np.array([[0, 4], [4, 8]])
#     assert np.array_equal(strides, expected_strides), f"Expected strides to be {expected_strides}, but got {strides}"
