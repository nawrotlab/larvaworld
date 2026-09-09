"""Regression tests for angular-kinematics computation in LarvaDataset."""

from __future__ import annotations

import copy
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

    # front_vector=(1, mid)/rear_vector=(-mid, -1) used to include the head
    # tip point (0-indexed point 0, the noisiest, most-deformable tracked
    # point) in the front vector and overlap the front/rear vectors at the
    # midpoint. Both must now be excluded. Resolve the actual 0-indexed
    # endpoints the same way the real computation does (SimMetricOps.
    # vector_dict), rather than hand-deriving the off-by-one convention
    # here, which is easy to get subtly wrong.
    Npoints = d.config.Npoints
    front_lo, front_hi = d2.config.vector_dict["front"]
    rear_lo, rear_hi = d2.config.vector_dict["rear"]
    front_points = set(range(min(front_lo, front_hi), max(front_lo, front_hi) + 1))
    rear_points = set(
        range(min(rear_lo, rear_hi) % Npoints, max(rear_lo, rear_hi) % Npoints + 1)
    )
    assert 0 not in front_points, "front vector must not include the head tip"
    assert (Npoints - 1) not in rear_points, "rear vector must not include the tail tip"
    assert front_points.isdisjoint(rear_points), "front/rear vectors overlap"

    for k in ["bend", "front_orientation_velocity", "front_orientation_acceleration"]:
        assert k in d2.s.columns
        values = d2.s[k].dropna().to_numpy()
        assert values.size > 0
        assert np.all(np.isfinite(values))

    # The original dataset itself must be untouched (deepcopy, not mutation).
    assert d.config.dir != new_dir
    assert d.id == "30controls"
    assert d2.config.provenance["origin"] == "derived"
    assert d2.config.provenance["lineage"][-1]["operation"] == ("reconstruct_at_Nsegs")


def test_reconstruct_at_Nsegs_bend_closely_matches_original(
    real_dataset, tmp_path: Path
) -> None:
    """
    The whole point of reconstruct_at_Nsegs is that a coarse, tip-excluding
    2-vector body approximation should reproduce angular kinematics close to
    the original, finely-tracked dataset's own -- not diverge from it. Use
    the pooled Kolmogorov-Smirnov distance (the same class of metric
    DataEvaluation.eval_datasets uses) between "bend" distributions as the
    closeness measure.

    Both sides are put through an explicit settling recompute (not just
    read from whatever state the fixture happens to be in) so the
    comparison isolates the front_vector/rear_vector geometry itself,
    not incidental state differences between a dataset's first-ever
    comp_orientations() call and a later recompute=True call.
    """
    from scipy.stats import ks_2samp

    d = copy.deepcopy(real_dataset)
    d.comp_orientations(recompute=True)
    d.comp_bend(recompute=True)

    d2 = d.reconstruct_at_Nsegs(2, new_dir=str(tmp_path / "reconstructed_2seg"))

    a = d.s["bend"].dropna().to_numpy()
    b = d2.s["bend"].dropna().to_numpy()
    ks_stat = ks_2samp(a, b).statistic

    # The previous tip-including, overlapping vectors gave a KS distance of
    # ~0.03 on the bundled reference dataset; the corrected, tip-excluding
    # vectors give 0.0 for that same dataset. Assert a comfortably loose
    # bound so this stays a genuine regression guard without being brittle
    # to unrelated numerical noise.
    assert ks_stat < 0.05, f"bend distribution diverged too much (KS={ks_stat})"


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
