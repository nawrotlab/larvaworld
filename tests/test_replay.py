import pytest
import larvaworld
from larvaworld.lib import reg, sim, util

larvaworld.VERBOSE = 1




# @pytest.mark.parametrize("a", range(1000))
def test_replay():
    refID = reg.default_refID
    d = reg.conf.Ref.loadRef(refID)
    replay_kws = {
        "normal": {"time_range": (10, 80)},
        "dispersal": {"transposition": "origin", "time_range": (60, 120)},
        "fixed_point": {
            "agent_ids": [0],
            "close_view": True,
            "fix_point": 6,
            "time_range": (80, 100),
        },
        "fixed_segment": {
            "agent_ids": [0],
            "close_view": True,
            "fix_point": 6,
            "fix_segment": "rear",
            "time_range": (100, 130),
        },
        "fixed_overlap": {
            "agent_ids": [0],
            "close_view": True,
            "fix_point": 6,
            "fix_segment": "front",
            "overlap_mode": True,
        },
        "2segs": {"draw_Nsegs": 2},
        "all_segs": {"draw_Nsegs": d.config.Npoints - 1},
    }

    for mode, kws in replay_kws.items():
        print(mode)
        parameters = reg.gen.Replay(
            **util.AttrDict(
                {
                    "refID": refID,
                    # 'dataset' : dataset,
                    **kws,
                }
            )
        ).nestedConf
        rep = sim.ReplayRun(
            parameters=parameters,
            dataset=d,
            id=f"{refID}_replay_{mode}",
            dir=f"./media/{mode}",
        )
        output = rep.run()
        assert output.parameters.constants["id"] == rep.id
        # raise


def test_replay_by_dir():
    """Run an experiment replay specifying the dataset by its directory path."""
    rep = sim.ReplayRun(
        parameters=reg.gen.Replay(
            **util.AttrDict(
                {"refDir": "SchleyerGroup/processed/exploration/30controls"}
            )
        ).nestedConf,
        id="replay_by_dir",
        dir="./media/replay_by_dir",
    )
    output = rep.run()
    assert output.parameters.constants["id"] == rep.id
