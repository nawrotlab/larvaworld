"""Unit tests for larvaworld.lib.plot.hist."""

from __future__ import annotations

import pytest

from larvaworld.lib.plot.hist import module_endpoint_hists


@pytest.mark.fast
class TestModuleEndpointHists:
    def test_rotates_xticklabels_to_avoid_overlap(self, real_dataset) -> None:
        """
        Regression test: module_endpoint_hists' per-panel conf_ax call had
        no xticklabelrotation, so long float tick labels at 18pt in narrow
        (~1.6in) sub-axes would overlap/collide.
        """
        fig, _save_to, _filename = module_endpoint_hists(
            e=real_dataset.e, mkey="crawler", mode="realistic", return_fig=True
        )
        for ax in fig.axes:
            for label in ax.get_xticklabels():
                assert label.get_rotation() == 30
