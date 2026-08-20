"""Regression tests for the agent-index clamping of the annotated track plots.

`track_annotated_data` defaults to plotting agents 3, 4, 5 and 6, which raised an
IndexError on any dataset holding fewer than seven of them - including every
short tutorial run and the `track` graphgroup built on top of it.
"""

import pytest

from larvaworld.lib import reg
from larvaworld.lib.plot.traj import _min_agent_count


class _FakeConfig:
    def __init__(self, n):
        self.agent_ids = [f"Larva_{i}" for i in range(n)]


class _FakeDataset:
    def __init__(self, n):
        self.config = _FakeConfig(n)


@pytest.mark.fast
class TestMinAgentCount:
    def test_returns_the_smallest_population(self):
        ds = [_FakeDataset(30), _FakeDataset(4), _FakeDataset(12)]
        assert _min_agent_count(datasets=ds) == 4

    def test_returns_none_without_resolvable_datasets(self):
        assert _min_agent_count() is None
        assert _min_agent_count(datasets=[]) is None


@pytest.mark.fast
class TestTrackGraphgroupRegistration:
    def test_track_plots_are_registered(self):
        assert reg.graphs.exists("stride track")
        assert reg.graphs.exists("turn track")

    def test_track_graphgroup_exists(self):
        assert reg.graphs.group_exists("track")

    @pytest.mark.parametrize(
        "exp,expected",
        [
            ("dish", ["traj", "general", "endpoint", "distro", "stride", "track"]),
            (
                "dispersion",
                ["traj", "general", "endpoint", "distro", "dsp", "stride", "track"],
            ),
        ],
    )
    def test_exploration_analysis_is_complete(self, exp, expected):
        groups = reg.graphs.get_analysis_graphgroups(exp, sources={})
        assert sorted(groups) == sorted(expected)

    def test_dish_analysis_omits_dispersal(self):
        """A 10 cm dish is too small for dispersal to be meaningful."""
        assert "dsp" not in reg.graphs.get_analysis_graphgroups("dish", sources={})

    def test_other_experiments_do_not_get_the_track_group(self):
        groups = reg.graphs.get_analysis_graphgroups("chemotaxis", sources={})
        assert "track" not in groups
        assert "chemo" in groups
