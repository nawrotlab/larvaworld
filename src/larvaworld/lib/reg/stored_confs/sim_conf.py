import numpy as np

from ... import reg, util, funcs
from ...param import Epoch, Odor
from importlib import import_module
from ...util import AttrDict

__all__ = [
    "Trial_dict",
    "Exp_dict",
    "Env_dict",
    "Ga_dict",
    "Batch_dict",
]


@funcs.stored_conf("Trial")
def Trial_dict():
    def trial_conf(durs=[], qs=[]):
        cumdurs = np.cumsum([0] + durs)
        return util.ItemList(
            Epoch(age_range=(t0, t1), sub=[q, "standard"]).nestedConf
            for i, (t0, t1, q) in enumerate(zip(cumdurs[:-1], cumdurs[1:], qs))
        )

    return AttrDict(
        {
            "default": AttrDict({"epochs": trial_conf()}),
            "odor_preference": AttrDict(
                {"epochs": trial_conf([5.0] * 8, [1.0, 0.0] * 4)}
            ),
            "odor_preference_short": AttrDict(
                {"epochs": trial_conf([0.125] * 8, [1.0, 0.0] * 4)}
            ),
        }
    )


@funcs.stored_conf("Env")
def Env_dict():
    from ...reg import gen

    E = reg.gen.Env
    FC = reg.gen.FoodConf

    d = {
        "focus": E.rect(0.01),
        "dish": E.dish(0.1),
        "dish_40mm": E.dish(0.04),
        "arena_200mm": E.rect(0.2),
        "arena_500mm": E.rect(0.5),
        "arena_1000mm": E.rect(1.0),
        "odor_gradient": E.odor_gradient((0.1, 0.06), pos=(0.04, 0.0), o="G", c=2),
        # 'mid_odor_gaussian': E.odor_gradient((0.3, 0.3), o='G'),
        "mid_odor_gaussian": E.odor_gradient((0.1, 0.06), o="G"),
        "odor_gaussian_square": E.odor_gradient(0.2, o="G"),
        "mid_odor_diffusion": E.odor_gradient(0.3, r=0.03, o="D"),
        "4corners": E.foodNodor_4corners(o="G"),
        "food_at_bottom": E.rect(
            0.2,
            f=FC.sg(
                id="FoodLine",
                odor=Odor.oO(o="G"),
                a=0.002,
                r=0.001,
                N=20,
                sh="oval",
                s=(0.01, 0.0),
                mode="periphery",
            ),
            o="G",
        ),
        "thermo_arena": E.rect(0.3, th={}),
        "windy_arena": E.rect(0.3, w={"wind_speed": 10.0}),
        "windy_blob_arena": E.rect(
            (0.5, 0.2),
            f=FC.sgs(
                Ngs=4,
                qs=np.ones(4),
                cs=util.N_colors(4),
                N=1,
                s=(0.04, 0.02),
                loc=(0.005, 0.0),
                mode="uniform",
                sh="rect",
                can_be_displaced=True,
                regeneration=True,
                o="D",
                regeneration_pos={"loc": (0.005, 0.0), "scale": (0.0, 0.0)},
            ),
            w={"wind_speed": 100.0},
            o="D",
        ),
        "windy_arena_bordered": E.rect(
            0.3,
            w={"wind_speed": 10.0},
            bl={
                "Border": gen.Border(
                    vertices=[(-0.03, -0.01), (-0.03, -0.06)], width=0.005
                )
            },
        ),
        "puff_arena_bordered": E.rect(
            0.3,
            w={
                "puffs": {
                    "PuffGroup": {
                        "N": 100,
                        "duration": 300.0,
                        "start_time": 0,
                        "speed": 100,
                    }
                }
            },
            bl={
                "Border": gen.Border(
                    vertices=[(-0.03, -0.01), (-0.03, -0.06)], width=0.005
                )
            },
        ),
        "single_puff": E.rect(
            0.3,
            w={
                "puffs": {
                    "Puff": {"N": 1, "duration": 30.0, "start_time": 55, "speed": 100}
                }
            },
        ),
        "CS_UCS_on_food": E.CS_UCS(grid=gen.FoodGrid(), o="G"),
        "CS_UCS_on_food_x2": E.CS_UCS(grid=gen.FoodGrid(), N=2, o="G"),
        "CS_UCS_off_food": E.CS_UCS(o="G"),
        "patchy_food": E.rect(
            0.2,
            f=FC.sg(
                N=8, s=(0.07, 0.07), mode="periphery", a=0.001, odor=Odor.oO(o="G", c=2)
            ),
            o="G",
        ),
        "random_food": E.rect(
            0.1, f=FC.sgs(Ngs=4, N=1, s=(0.04, 0.04), mode="uniform", sh="rect")
        ),
        "uniform_food": E.dish(
            0.05, f=FC.sg(N=2000, s=(0.025, 0.025), a=0.01, r=0.0001)
        ),
        "patch_grid": E.rect(
            0.2,
            f=FC.sg(
                N=5 * 5,
                s=(0.2, 0.2),
                a=0.01,
                r=0.007,
                mode="grid",
                sh="rect",
                odor=Odor.oO(o="G", c=0.2),
            ),
            o="G",
        ),
        "food_grid": E.rect(0.02, f=FC(food_grid=gen.FoodGrid())),
        "single_odor_patch": E.rect(0.1, f=FC.patch(odor=Odor.oO(o="G")), o="G"),
        "single_patch": E.rect(0.05, f=FC.patch()),
        "multi_patch": E.rect(
            0.02, f=FC.sg(N=8, s=(0.007, 0.007), mode="periphery", a=0.1, r=0.0015)
        ),
        "double_patch": E.double_patch(dim=0.24, o="G"),
        "maze": E.maze(),
        "game": E.game(),
        "arena_50mm_diffusion": E.dish(0.05, o="D"),
    }
    return AttrDict({k: v.nestedConf for k, v in d.items()})


@funcs.stored_conf("Exp")
def Exp_dict():
    def d():
        from ...param import Odor
        from ...reg import gen

        # GTRvsS import - deep import required due to circular dependency
        from ...reg.larvagroup import GTRvsS

        ENR = reg.gen.EnrichConf

        def lg(id=None, **kwargs):
            l = reg.gen.LarvaGroup(**kwargs)
            if id is None:
                id = l.model
            return l.entry(id)

        def lgID(mID, **kwargs):
            return lg(mID=mID, **kwargs)

        def lgs(mIDs, ids=None, cs=None, **kwargs):
            if ids is None:
                ids = mIDs
            N = len(mIDs)
            if cs is None:
                cs = util.N_colors(N)
            return AttrDict.merge_dicts(
                [lg(id=id, c=c, mID=mID, **kwargs) for mID, c, id in zip(mIDs, cs, ids)]
            )

        def exp(
            id, env=None, l={}, en=ENR(), dur=5.0, c=[], c0=["pose", "brain"], **kwargs
        ):
            if env is None:
                env = id
            return gen.Exp(
                larva_groups=l,
                env_params=reg.conf.Env.get(env),
                experiment=id,
                enrichment=en,
                collections=c0 + c,
                duration=dur,
                **kwargs,
            ).nestedConf

        def fE(id, dur=10.0, en=ENR.source_proc(), env="patch_grid", **kwargs):
            return exp(id, dur=dur, en=en, env=env, **kwargs)

        def tE(id, dur=600.0, en=ENR.source_proc(), env="single_patch", **kwargs):
            return exp(id, dur=dur, en=en, env=env, **kwargs)

        def gE(id, dur=20.0, **kwargs):
            return exp(id, dur=dur, **kwargs)

        def dE(id, dur=5.0, env="food_grid", h_starved=0.0, age=72.0, q=1.0, **kwargs):
            return exp(
                id,
                dur=dur,
                env=env,
                c=["gut"],
                l=GTRvsS(age=age, q=q, h_starved=h_starved),
                en=ENR.spatial_proc(),
                **kwargs,
            )

        def thermo_exp(id, dur=10.0, **kwargs):
            return exp(id, dur=dur, **kwargs)

        def prE(id, mID, dur=5.0, env="CS_UCS_off_food", trialID="default", **kwargs):
            return exp(
                id,
                dur=dur,
                en=ENR.PI_proc(),
                l=lgID(mID, s=(0.005, 0.02)),
                trials=reg.conf.Trial.getID(trialID),
                env=env,
                **kwargs,
            )

        def game_groups(dim=0.1, N=10, x=0.4, y=0.0, mode="king"):
            x = np.round(x * dim, 3)
            y = np.round(y * dim, 3)
            if mode == "king":
                l = {
                    **lg(
                        id="Left",
                        N=N,
                        loc=(-x, y),
                        mID="gamer-5x",
                        c="darkblue",
                        odor=Odor.oG(id="Left_odor"),
                    ),
                    **lg(
                        id="Right",
                        N=N,
                        loc=(+x, y),
                        mID="gamer-5x",
                        c="darkred",
                        odor=Odor.oG(id="Right_odor"),
                    ),
                }
            elif mode == "flag":
                l = {
                    **lg(id="Left", N=N, loc=(-x, y), mID="gamer", c="darkblue"),
                    **lg(id="Right", N=N, loc=(+x, y), mID="gamer", c="darkred"),
                }
            elif mode == "catch_me":
                l = {
                    **lg(
                        id="Left",
                        N=1,
                        loc=(-0.01, 0.0),
                        mID="follower-L",
                        c="darkblue",
                        odor=Odor.oD(id="Left_odor"),
                    ),
                    **lg(
                        id="Right",
                        N=1,
                        loc=(+0.01, 0.0),
                        mID="follower-R",
                        c="darkred",
                        odor=Odor.oD(id="Right_odor"),
                    ),
                }
            return l

        def lgs_x4(N=5):
            return lgs(
                mIDs=["max_forager", "max_feeder", "navigator", "explorer"],
                ids=["forager", "Orco", "navigator", "explorer"],
                N=N,
            )

        d0 = {
            "tethered": {
                "env": "focus",
                "dur": 30.0,
                "l": lgID("immobile", N=1, ors=(90.0, 90.0)),
            },
            "focus": {"l": lgID("Levy", N=1, ors=(90.0, 90.0))},
            "dish": {"l": lgID("explorer", s=(0.02, 0.02))},
            "dispersion": {"env": "arena_200mm", "l": lgID("explorer")},
            "dispersion_x2": {
                "env": "arena_200mm",
                "l": lgs(mIDs=["explorer", "Levy"], ids=["CoupledOsc", "Levy"], N=5),
            },
        }

        d1 = {
            "chemotaxis": {
                "env": "odor_gradient",
                "l": lgID(
                    "navigator",
                    N=8,
                    loc=(-0.04, 0.0),
                    s=(0.005, 0.02),
                    ors=(-30.0, 30.0),
                ),
            },
            "chemorbit": {"env": "mid_odor_gaussian", "l": lgID("navigator", N=3)},
            "chemorbit_OSN": {
                "env": "mid_odor_gaussian",
                "l": lgID("OSNnavigator", N=3),
            },
            "chemorbit_x2": {
                "env": "odor_gaussian_square",
                "l": lgs(
                    mIDs=["navigator", "RLnavigator"], ids=["CoupledOsc", "RL"], N=10
                ),
            },
            "chemorbit_x4": {"env": "odor_gaussian_square", "l": lgs_x4()},
            "chemotaxis_diffusion": {
                "env": "mid_odor_diffusion",
                "l": lgID("navigator"),
            },
            "chemotaxis_RL": {
                "env": "mid_odor_diffusion",
                "l": lgID("RLnavigator", N=10, mode="periphery", s=(0.04, 0.04)),
            },
            "reorientation": {
                "env": "mid_odor_diffusion",
                "l": lgID("immobile", N=200, s=(0.05, 0.05)),
            },
            "food_at_bottom": {
                "l": lgs(
                    mIDs=["max_feeder", "max_forager"],
                    ids=["Orco", "control"],
                    N=5,
                    sh="oval",
                    loc=(0.0, 0.04),
                    s=(0.04, 0.01),
                )
            },
        }

        d2 = {
            "anemotaxis": {"env": "windy_arena"},
            "anemotaxis_bordered": {"env": "windy_arena_bordered"},
            "puff_anemotaxis_bordered": {"env": "puff_arena_bordered"},
        }

        d3 = {"single_puff": {"env": "single_puff", "dur": 2.5, "l": lgID("explorer")}}

        # d4= {
        #     'PItest_off': {'env': 'CS_UCS_off_food','mID': 'navigator_x2'},
        #     'PItest_off_OSN': {'env': 'CS_UCS_off_food','mID': 'OSNnavigator_x2'},
        #     'PItest_on': {'env': 'CS_UCS_off_food','mID': 'OSNnavigator_x2'},
        #     # 'PItest_on': prE('PItest_on', env='CS_UCS_on_food', mID='forager_x2'),
        #     'PItrain_mini': {'env': 'CS_UCS_on_food_x2','mID': 'forager_RL','dur': 1.0,'trialID': 'odor_preference_short'},
        #     # 'PItrain_mini': prE('PItrain_mini', env='CS_UCS_on_food_x2', dur=1.0, trialID='odor_preference_short',mID='forager_RL'),
        #     'PItrain': {'env': 'CS_UCS_on_food_x2','mID': 'forager_RL','dur':41.0,'trialID': 'odor_preference'},
        #     # 'PItrain': prE('PItrain', env='CS_UCS_on_food_x2', dur=41.0, trialID='odor_preference', mID='forager_RL'),
        #     'PItest_off_RL': {'env': 'CS_UCS_off_food','mID': 'RLnavigator','dur': 105.0},
        #     # 'PItest_off_RL': prE('PItest_off_RL', dur=105.0, mID='RLnavigator')
        #
        # }

        d = {
            "exploration": {id: exp(id=id, **kws) for id, kws in d0.items()},
            "chemotaxis": {
                id: exp(id=id, en=ENR.source_proc(), **kws) for id, kws in d1.items()
            },
            "anemotaxis": {
                id: exp(
                    id=id, en=ENR.wind_proc(), l=lgID("explorer", N=4), dur=0.5, **kws
                )
                for id, kws in d2.items()
            },
            "chemanemotaxis": {
                id: exp(id=id, en=ENR.sourcewind_proc(), **kws)
                for id, kws in d3.items()
            },
            "thermotaxis": {
                "thermotaxis": thermo_exp(
                    "thermotaxis", env="thermo_arena", l=lgID("thermo_navigator")
                ),
            },
            "odor_preference": {
                "PItest_off": prE("PItest_off", mID="navigator_x2"),
                "PItest_off_OSN": prE("PItest_off", mID="OSNnavigator_x2"),
                "PItest_on": prE("PItest_on", env="CS_UCS_on_food", mID="forager_x2"),
                "PItrain_mini": prE(
                    "PItrain_mini",
                    env="CS_UCS_on_food_x2",
                    dur=1.0,
                    trialID="odor_preference_short",
                    mID="forager_RL",
                ),
                "PItrain": prE(
                    "PItrain",
                    env="CS_UCS_on_food_x2",
                    dur=41.0,
                    trialID="odor_preference",
                    mID="forager_RL",
                ),
                "PItest_off_RL": prE("PItest_off_RL", dur=105.0, mID="RLnavigator"),
            },
            "foraging": {
                "patchy_food": fE("patchy_food", env="patchy_food", l=lgID("forager")),
                "patch_grid": fE("patch_grid", l=lgs_x4()),
                "MB_patch_grid": fE(
                    "MB_patch_grid",
                    l=lgs(mIDs=["max_forager0_MB", "max_forager_MB"], N=3),
                ),
                "noMB_patch_grid": fE(
                    "noMB_patch_grid", l=lgs(mIDs=["max_forager0", "max_forager"], N=4)
                ),
                "random_food": fE(
                    "random_food",
                    env="random_food",
                    l=lgs(
                        mIDs=["feeder", "forager_RL"],
                        ids=["Orco", "RL"],
                        N=5,
                        mode="uniform",
                        shape="rect",
                        s=(0.04, 0.04),
                    ),
                ),
                "uniform_food": fE(
                    "uniform_food",
                    env="uniform_food",
                    l=lgID("feeder", N=5, s=(0.005, 0.005)),
                ),
                "food_grid": fE("food_grid", env="food_grid", l=lgID("feeder", N=5)),
                "single_odor_patch": fE(
                    "single_odor_patch",
                    env="single_odor_patch",
                    l=lgs(
                        mIDs=["feeder", "forager"],
                        ids=["Orco", "control"],
                        N=5,
                        mode="periphery",
                        s=(0.01, 0.01),
                    ),
                ),
                "single_odor_patch_x4": fE(
                    "single_odor_patch_x4", env="single_odor_patch", l=lgs_x4()
                ),
                "double_patch": fE(
                    "double_patch",
                    env="double_patch",
                    l=GTRvsS(N=5),
                    en=ENR.patch_proc(),
                ),
                "4corners": exp(
                    "4corners",
                    env="4corners",
                    l=lgID("forager_RL", N=10, s=(0.04, 0.04)),
                ),
            },
            "tactile": {
                "tactile_detection": tE(
                    "tactile_detection",
                    l=lgID("toucher", mode="periphery", s=(0.03, 0.03)),
                ),
                "tactile_detection_x4": tE(
                    "tactile_detection_x4",
                    l=lgs(
                        mIDs=["RLtoucher_2", "RLtoucher", "toucher", "toucher_brute"],
                        ids=["RL_3sensors", "RL_1sensor", "control", "brute"],
                        N=10,
                    ),
                ),
                "multi_tactile_detection": tE(
                    "multi_tactile_detection",
                    env="multi_patch",
                    l=lgs(
                        mIDs=["RLtoucher_2", "RLtoucher", "toucher"],
                        ids=["RL_3sensors", "RL_1sensor", "control"],
                        N=4,
                    ),
                ),
            },
            "growth": {
                "growth": dE("growth", dur=24 * 60.0, age=0.0),
                "RvsS": dE("RvsS", dur=180.0, age=0.0),
                "RvsS_on": dE("RvsS_on", dur=20.0),
                "RvsS_off": dE("RvsS_off", env="arena_200mm", dur=20.0),
                "RvsS_on_q75": dE("RvsS_on_q75", q=0.75),
                "RvsS_on_q50": dE("RvsS_on_q50", q=0.50),
                "RvsS_on_q25": dE("RvsS_on_q25", q=0.25),
                "RvsS_on_q15": dE("RvsS_on_q15", q=0.15),
                "RvsS_on_1h_prestarved": dE("RvsS_on_1h_prestarved", h_starved=1.0),
                "RvsS_on_2h_prestarved": dE("RvsS_on_2h_prestarved", h_starved=2.0),
                "RvsS_on_3h_prestarved": dE("RvsS_on_3h_prestarved", h_starved=3.0),
                "RvsS_on_4h_prestarved": dE("RvsS_on_4h_prestarved", h_starved=4.0),
            },
            "games": {
                "maze": gE(
                    "maze",
                    env="maze",
                    l=lgID("navigator", N=5, loc=(-0.4 * 0.1, 0.0), ors=(-60.0, 60.0)),
                ),
                "keep_the_flag": gE(
                    "keep_the_flag", env="game", l=game_groups(mode="king")
                ),
                "capture_the_flag": gE(
                    "capture_the_flag", env="game", l=game_groups(mode="flag")
                ),
                "catch_me": gE(
                    "catch_me",
                    env="arena_50mm_diffusion",
                    l=game_groups(mode="catch_me"),
                ),
            },
            "other": {
                "realistic_imitation": exp(
                    "realistic_imitation",
                    env="dish",
                    l=lgID("imitator"),
                    Box2D=True,
                    c=["midline", "contour"],
                ),
                "prey_detection": exp(
                    "prey_detection",
                    env="windy_blob_arena",
                    l=lgID("zebrafish", N=4, s=(0.002, 0.005)),
                ),
                # 'imitation': imitation_exp('None.150controls', model='explorer'),
            },
        }

        return d

    return AttrDict.merge_dicts(list(d().values()))


@funcs.stored_conf("Ga")
def Ga_dict():
    def _ga_conf(
        name,
        env="arena_200mm",
        ks=["interference", "turner"],
        init_mode="random",
        refID=None,
        fit_kws={},
        cyc=[],
        ev=util.AttrDict(),
        m="explorer",
        fID=None,
        **kwargs,
    ):
        if refID is not None:
            kwargs["dt"] = reg.conf.Ref.getRef(refID).dt
        # if m0 is None:
        #     m0=reg.conf.Exp.getID(name).larva_groups.keylist[0]
        return reg.gen.Ga(
            ga_select_kws=reg.gen.GAselector(
                base_model=m, space_mkeys=ks, init_mode=init_mode
            ),
            ga_eval_kws=reg.gen.GAevaluation(
                fit_kws=fit_kws,
                cycle_curve_metrics=cyc,
                eval_metrics=ev,
                fitness_func_name=fID,
                refID=refID,
            ),
            env_params=env,
            experiment=name,
            **kwargs,
        ).nestedConf

    ev1 = AttrDict(
        {
            "angular kinematics": ["run_fov_mu", "pau_fov_mu", "b", "fov", "foa"],
            "spatial displacement": ["v_mu", "pau_v_mu", "run_v_mu", "v", "a"],
            "temporal dynamics": ["fsv", "ffov", "run_tr", "pau_tr"],
        }
    )
    dID = reg.default_refID
    # Lazy import to avoid reg<->sim cycles
    OptimizationOps = getattr(
        import_module("larvaworld.lib.sim.batch_run"), "OptimizationOps"
    )

    l = [
        _ga_conf("interference", refID=dID, cyc=["fov", "foa", "rov"]),
        _ga_conf("exploration", refID=dID, ev=ev1),
        _ga_conf(
            "realism",
            refID=dID,
            ev=ev1,
            cyc=["sv", "fov", "foa", "b"],
            init_mode="model",
        ),
        _ga_conf(
            "chemorbit",
            m="navigator",
            ks=["olfactor"],
            fID="dst2source",
            fit_kws={"source_xy": None},
            env="odor_gaussian_square",
        ),
        _ga_conf(
            "obstacle_avoidance",
            m="obstacle_avoider",
            ks=["sensorimotor"],
            fID="cum_dst",
            env="dish_40mm",
            scene="obstacle_avoidance_700",
        ),
    ]
    return AttrDict({c.experiment: c for c in l})


@funcs.stored_conf("Batch")
def Batch_dict():
    # Lazy import to avoid reg<->sim cycles
    OptimizationOps = getattr(
        import_module("larvaworld.lib.sim.batch_run"), "OptimizationOps"
    )

    def bb(exp, proc=[], ss={}, o=None, N=5, abs=False, min=True, thr=0.001, **kwargs):
        return AttrDict(
            exp=exp,
            exp_kws={
                "enrichment": reg.gen.EnrichConf(proc_keys=proc).nestedConf,
                "experiment": exp,
            },
            optimization=OptimizationOps(
                fit_par=o, max_Nsims=N, absolute=abs, minimize=min, threshold=thr
            ).nestedConf,
            space_search=ss,
            **kwargs,
        )

    l = [
        bb(
            "PItest_off",
            ss={
                "gain_dict.CS": [[-100.0, 100.0], 4],
                "gain_dict.UCS": [[-100.0, 100.0], 4],
            },
            proc=["PI"],
        ),
        bb(
            "patchy_food",
            ss={"EEB": [[0.0, 1.0], 3], "freq": [[1.5, 2.5], 3]},
            o="ingested_food_volume",
        ),
        bb(
            "food_grid",
            ss={"EEB": [[0.0, 1.0], 6], "EEB_decay": [[0.1, 2.0], 6]},
            o="ingested_food_volume",
        ),
        bb(
            "growth",
            ss={"EEB": [[0.5, 0.8], 8], "hunger_gain": [[0.0, 0.0], 1]},
            o="deb_f_deviation",
            N=20,
            abs=True,
        ),
        bb(
            "tactile_detection",
            ss={"initial_gain": [[25.0, 75.0], 10], "decay_coef": [[0.01, 0.5], 4]},
            o="cum_food_detected",
            N=600,
            thr=100000.0,
            min=False,
        ),
        bb(
            "anemotaxis",
            ss={
                f"windsensor.weights.{m1}_{m2}": [[-20.0, 20.0], 3]
                for m1, m2 in zip(["bend", "hunch"], ["ang", "lin"])
            },
            o="anemotaxis",
            N=100,
            thr=1000.0,
            min=False,
            proc=["wind"],
        ),
    ]
    l += [
        bb(
            exp,
            ss={"Odor.mean": [[300.0, 1300.0], 3], "decay_coef": [[0.1, 0.5], 3]},
            o="final_dst_to_Source",
            proc=["angular", "spatial", "source"],
        )
        for exp in ["chemotaxis", "chemorbit"]
    ]
    l += [
        bb(
            exp,
            ss={"input_noise": [[0.0, 0.4], 2], "decay_coef": [[0.1, 0.5], 2]},
            proc=["PI"],
        )
        for exp in ["PItrain_mini", "PItrain"]
    ]
    return AttrDict({c.exp: c for c in l})
