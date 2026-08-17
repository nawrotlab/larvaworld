"""Unit tests for LarvaDatasetCollection.get_colors()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from larvaworld.lib.process.dataset import LarvaDatasetCollection
from larvaworld.lib import util


def _fake_collection(colors):
    datasets = [SimpleNamespace(config=SimpleNamespace(color=c)) for c in colors]
    return SimpleNamespace(datasets=datasets)


@pytest.mark.fast
class TestGetColors:
    def test_keeps_explicit_distinct_colors(self):
        fake = _fake_collection(["red", "blue", "green"])
        result = LarvaDatasetCollection.get_colors(fake)
        assert result == ["red", "blue", "green"]

    def test_fills_missing_colors_from_coordinated_batch_not_one_off_random(self):
        """
        Datasets with no config.color used to each draw their own one-off
        random_colors(1) fallback independently. Now they should draw from
        one N_colors(Ndatasets) batch sized to the whole collection, so the
        assigned colors are the same well-separated, deterministic palette
        N_colors already provides elsewhere in the codebase.
        """
        fake = _fake_collection([None, None, None])
        result = LarvaDatasetCollection.get_colors(fake)
        assert len(result) == 3
        assert len(set(result)) == 3
        assert result == util.N_colors(3)

    def test_resolves_duplicate_explicit_colors(self):
        fake = _fake_collection(["red", "red", None])
        result = LarvaDatasetCollection.get_colors(fake)
        assert len(set(result)) == 3
        assert result[0] == "red"
        assert result[1] != "red"

    def test_mixed_explicit_and_missing_colors_all_distinct(self):
        fake = _fake_collection(["purple", None, "purple", None])
        result = LarvaDatasetCollection.get_colors(fake)
        assert len(result) == 4
        assert len(set(result)) == 4
        assert result[0] == "purple"
