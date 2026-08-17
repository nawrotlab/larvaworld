"""Regression tests for angular-kinematics computation in LarvaDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from larvaworld.lib.util import nam


def test_comp_orientations_matches_correct_arctan2_wrap(real_dataset) -> None:
    """
    `s[par] = np.arctan2(y, x) % 2 * np.pi` used to evaluate as
    `(arctan2(y, x) % 2) * np.pi` (operator precedence) -- a completely
    different, wrong transformation -- instead of the intended angle-wrap
    `arctan2(y, x) % (2 * np.pi)`. Both happen to land in [0, 2*pi), so a
    mere range check doesn't catch the bug; compare against an independent
    recomputation of the same x/y diffs instead.
    """
    d = real_dataset
    d.comp_orientations(mode="minimal", recompute=True)
    s, _, c = d.data

    vecs = list(c.vector_dict.keys())[:2]
    pars = nam.orient(vecs)
    mid = d.midline_xy_data

    for vec, par in zip(vecs, pars):
        idx1, idx2 = c.vector_dict[vec]
        x = mid[:, idx2, 0] - mid[:, idx1, 0]
        y = mid[:, idx2, 1] - mid[:, idx1, 1]
        expected = np.arctan2(y, x) % (2 * np.pi)

        actual = s[par].to_numpy()
        valid = ~np.isnan(actual) & ~np.isnan(expected)
        assert valid.sum() > 0
        np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-9)


def test_reconstruct_at_Nsegs_produces_a_separately_stored_comparable_dataset(
    real_dataset, tmp_path: Path
) -> None:
    d = real_dataset
    new_dir = str(tmp_path / "reconstructed_2seg")

    d2 = d.reconstruct_at_Nsegs(2, new_id="test_2seg", new_dir=new_dir)

    assert d2.id == "test_2seg"
    assert d2.dir == new_dir
    assert d2.config.dir == new_dir
    assert (tmp_path / "reconstructed_2seg" / "data" / "data.h5").exists()
    assert (tmp_path / "reconstructed_2seg" / "data" / "conf.txt").exists()

    # front/rear vectors now split the body at its midpoint, not the
    # original dataset's own (tighter) front_vector/rear_vector span.
    assert d2.config.front_vector != d.config.front_vector
    assert d2.config.rear_vector != d.config.rear_vector

    for k in ["bend", "front_orientation_velocity", "front_orientation_acceleration"]:
        assert k in d2.s.columns
        values = d2.s[k].dropna().to_numpy()
        assert values.size > 0
        assert np.all(np.isfinite(values))

    # The original dataset itself must be untouched (deepcopy, not mutation).
    assert d.config.dir != new_dir
    assert d.id == "30controls"


def test_angular_pars_plot_across_real_and_reconstructed_datasets(
    real_dataset, tmp_path: Path
) -> None:
    """
    Regression test for two real bugs hit when plotting a dataset against
    its own reconstruct_at_Nsegs() output (same config.color, genuinely
    different config.dir):

    - LarvaDatasetCollection.get_colors() compared a candidate replacement
      color (a numpy array from util.random_colors) against a list of
      plain color strings via `in`, raising "the truth value of an array
      with more than one element is ambiguous" -- triggered whenever two
      plotted datasets share the same config.color (the normal case for a
      dataset and a copy of it).
    - LarvaDatasetCollection.set_dir() used a bare `raise` (no active
      exception) when datasets don't share a common parent directory,
      always crashing instead of falling through to the same `None`
      fallback every other unresolved case in that method already uses.
    """
    from larvaworld.lib import reg

    d = real_dataset
    d2 = d.reconstruct_at_Nsegs(2, new_dir=str(tmp_path / "reconstructed_2seg"))
    assert d.config.color == d2.config.color  # the exact trigger condition

    fig, save_to, filename = reg.graphs.run(
        "angular pars",
        datasets=[d, d2],
        labels=["real", "reconstructed"],
        return_fig=True,
    )
    assert hasattr(fig, "savefig")
