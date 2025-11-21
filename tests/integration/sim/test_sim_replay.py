import pytest

import larvaworld
from larvaworld.lib import reg, sim, util

pytestmark = [pytest.mark.integration, pytest.mark.slow]

larvaworld.VERBOSE = 1


replay_confs = [
    ("normal", {}),
    # ("dispersal", {"transposition": "origin", "time_range": (0, 20)}),
    (
        "fixed_point",
        {
            "agent_ids": [0],
            "close_view": True,
            "fix_point": 6,
            # "time_range": (80, 100),
        },
    ),
    (
        "fixed_segment",
        {
            "agent_ids": [0],
            "close_view": True,
            "fix_point": 6,
            "fix_segment": "rear",
            # "time_range": (100, 130),
        },
    ),
    (
        "fixed_overlap",
        {
            "agent_ids": [0],
            "close_view": True,
            "fix_point": 6,
            "fix_segment": "front",
            "overlap_mode": True,
        },
    ),
    ("2segs", {"draw_Nsegs": 2}),
    ("all_segs", {"draw_Nsegs": 11}),
]


@pytest.mark.parametrize("id,conf", replay_confs)
def test_replay(id, conf):
    refID = reg.default_refID
    # d = reg.loadRef(refID, load=True)
    rep = sim.ReplayRun(
        parameters=reg.gen.Replay(refID=refID, **conf).nestedConf,
        # dataset=d,
        id=f"{refID}_replay_{id}",
        dir=f"./media/{id}",
        # screen_kws={"vis_mode": "video", "show_display": True},
    )
    output = rep.run()
    assert output.parameters.constants["id"] == rep.id


def test_replay_by_dir():
    """Run an experiment replay specifying the dataset by its directory path."""
    rep = sim.ReplayRun(
        parameters=reg.gen.Replay(
            refDir="SchleyerGroup/processed/exploration/30controls"
        ).nestedConf,
        id="replay_by_dir",
        dir="./media/replay_by_dir",
    )
    output = rep.run()
    assert output.parameters.constants["id"] == rep.id


def test_replay_visualization():
    """Run an experiment replay with visualization."""
    rep = sim.ReplayRun(
        parameters=reg.gen.Replay(
            refID=reg.default_refID, time_range=(40, 60)
        ).nestedConf,
        id=f"{reg.default_refID}_replay_visualization",
        dir=f"./media/replay_visualization",
        screen_kws={"vis_mode": "video", "show_display": True},
    )
    output = rep.run()
    assert output.parameters.constants["id"] == rep.id
