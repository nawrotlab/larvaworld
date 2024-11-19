import larvaworld
from larvaworld.lib import reg, sim, util

larvaworld.VERBOSE = 1


def test_replay():
    refIDs = reg.conf.Ref.confIDs
    refID = refIDs[-1]
    d = reg.conf.Ref.loadRef(refID)
    replay_kws = {
        "normal": {"time_range": (10, 80)},
        "dispersal": {"transposition": "origin", "time_range": (60, 120)},
        "fixed_point": {
            "agent_ids": [1],
            "close_view": True,
            "fix_point": 6,
            "time_range": (80, 100),
        },
        "fixed_segment": {
            "agent_ids": [1],
            "close_view": True,
            "fix_point": 6,
            "fix_segment": "rear",
            "time_range": (100, 130),
        },
        "fixed_overlap": {
            "agent_ids": [1],
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
