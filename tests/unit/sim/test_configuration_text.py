"""
Unit tests for BaseRun/ReplayRun's on-screen intro-text content.

Uses object.__new__ to build bare instances with only `.p` set, avoiding a
full simulation setup -- configuration_items/configuration_text only read
`self.p`.
"""

from __future__ import annotations

import pytest

from larvaworld.lib.sim.base_run import BaseRun, render_configuration_text
from larvaworld.lib.sim.dataset_replay import ReplayRun
from larvaworld.lib.util import AttrDict


def _bare_base_run(**p_kwargs):
    r = object.__new__(BaseRun)
    r.p = AttrDict(
        {
            "runtype": "Exp",
            "experiment": "dish",
            "id": "dish_0",
            "dir": "/tmp/dish_0",
            "duration": 5.0,
            "dt": 0.1,
            **p_kwargs,
        }
    )
    return r


def _bare_replay_run(**p_kwargs):
    r = object.__new__(ReplayRun)
    r.p = AttrDict(
        {
            "runtype": "Replay",
            "experiment": "replay",
            "id": "replay_0",
            "dir": "/tmp/replay_0",
            "duration": 2.5,
            "dt": 0.1,
            "refID": "None.30controls",
            "time_range": None,
            "transposition": None,
            "track_point": -1,
            **p_kwargs,
        }
    )
    return r


@pytest.mark.fast
class TestRenderConfigurationText:
    def test_drops_none_valued_fields(self):
        text = render_configuration_text("Title", {"A": 1, "B": None, "C": "x"})
        assert "A : 1" in text
        assert "B" not in text
        assert "C : x" in text


@pytest.mark.fast
class TestBaseRunConfigurationText:
    def test_includes_crucial_identity_fields(self):
        r = _bare_base_run()
        items = r.configuration_items
        assert items["Simulation mode"] == "Exp"
        assert items["Simulation ID"] == "dish_0"
        assert items["Storage directory"] == "/tmp/dish_0"
        assert items["Duration (min)"] == 5.0
        assert items["Timestep (sec)"] == 0.1


@pytest.mark.fast
class TestReplayRunConfigurationText:
    def test_extends_rather_than_replaces_base_fields(self):
        """
        ReplayRun used to entirely replace BaseRun's configuration_text,
        dropping simulation ID/type/storage dir. It must now keep them and
        add its own replay-specific fields on top.
        """
        r = _bare_replay_run()
        items = r.configuration_items

        # base fields preserved
        assert items["Simulation mode"] == "Replay"
        assert items["Simulation ID"] == "replay_0"
        assert items["Storage directory"] == "/tmp/replay_0"

        # replay-specific field added
        assert items["Reference Dataset"] == "None.30controls"

    def test_omits_inactive_optional_replay_fields(self):
        r = _bare_replay_run(time_range=None, transposition=None, track_point=-1)
        text = r.configuration_text
        assert "Time range" not in text
        assert "Transposition" not in text
        assert "Tracked midline point" not in text

    def test_includes_active_optional_replay_fields(self):
        r = _bare_replay_run(time_range=(0, 60), transposition="center", track_point=3)
        text = r.configuration_text
        assert "Time range (sec) : (0, 60)" in text
        assert "Transposition : center" in text
        assert "Tracked midline point : 3" in text
