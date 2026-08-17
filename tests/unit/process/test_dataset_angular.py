"""Regression test for comp_orientations' orientation-wrapping arithmetic."""

from __future__ import annotations

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
