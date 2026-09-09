"""Regression tests for trajectory resolution in LarvaDataset.comp_spatial."""

from __future__ import annotations

import numpy as np
import pytest


def _strip_trajectory(d):
    """Remove every column comp_spatial could take the trajectory from."""
    c = d.config
    drop = [
        p for p in list(c.traj_xy) + list(c.centroid_xy) if p in d.step_data.columns
    ]
    d.step_data.drop(columns=drop, inplace=True)
    return drop


@pytest.mark.fast
def test_trajectory_falls_back_to_the_midline_centroid(real_dataset) -> None:
    """
    A lab format declaring point_idx=-1 asks for the centroid as its tracked point, but
    the centroid is averaged from contour points, which trackers recording only a midline
    do not provide. comp_spatial used to index the missing trajectory columns straight
    away and raise KeyError, making every such format fail to import. The midline's own
    mean is used instead.
    """
    d = real_dataset
    c = d.config
    c.point_idx = -1  # track the centroid
    _strip_trajectory(d)
    # Emulate a tracker that records a midline but no contour, so that no centroid can be
    # averaged from contour points and the tracked point stays unavailable.
    c.Ncontour = 0
    assert not c.traj_xy.exist_in(d.step_data)
    assert not c.point_xy.exist_in(d.step_data)

    d.comp_spatial()

    assert c.traj_xy.exist_in(d.step_data)
    expected = np.mean(d.midline_xy_data, axis=1)
    actual = d.step_data[c.traj_xy].to_numpy()
    valid = ~np.isnan(actual).any(axis=1) & ~np.isnan(expected).any(axis=1)
    assert valid.sum() > 0
    np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-9)


@pytest.mark.fast
def test_tracked_point_still_wins_over_the_midline_centroid(real_dataset) -> None:
    """
    Verify the fallback only applies when no tracked point is available, so a dataset that
    does carry one keeps taking its trajectory from it rather than from the midline.
    """
    d = real_dataset
    c = d.config
    c.point_idx = -1
    _strip_trajectory(d)
    c.Ncontour = 0
    # A centroid that is deliberately not the midline mean, so the two are told apart.
    sentinel = np.mean(d.midline_xy_data, axis=1) + 1000.0
    d.step_data[c.centroid_xy] = sentinel
    assert c.point_xy.exist_in(d.step_data)

    d.comp_spatial()

    np.testing.assert_allclose(d.step_data[c.traj_xy].to_numpy(), sentinel, atol=1e-9)
