from __future__ import annotations
from typing import Any
import itertools
import random

import numpy as np
import param

from ... import reg, util, funcs
from ...param import NestedConf, class_defaults, class_objs
from ...util import AttrDict, SuperList
from .. import agents, deb
from . import (
    basic,
    crawl_bend_interference,
    crawler,
    feeder,
    intermitter,
    memory,
    sensor,
    turner,
)

__all__: list[str] = [
    "BrainModule",
    "BrainModuleDB",
    "LarvaModuleDB",
    "SpaceDict",
    "moduleDB",
]


class BrainModule(NestedConf):
    ModeShortNames = AttrDict(
        {
            "realistic": "RE",
            "square": "SQ",
            "gaussian": "GAU",
            "constant": "CON",
            "default": "DEF",
            "neural": "NEU",
            "sinusoidal": "SIN",
            "nengo": "NENGO",
            "phasic": "PHI",
            "branch": "BR",
            "osn": "OSN",
            "RL": "RL",
            "MB": "MB",
        }
    )

    mID = param.String(default=None, doc="The unique ID of the module")
    color = param.Color(
        default=None, doc="The background color when plotting module tables"
    )
    dict = param.Dict(
        default=util.AttrDict(), doc="A dictionary of implemented modes as classes"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.excluded = [basic.Effector, "phi", "name"]
        self.default_dict = AttrDict(
            {
                mode: class_defaults(A=self.dict[mode], excluded=self.excluded)
                for mode in self.modes
                if not isinstance(self.dict[mode], dict)
            }
        )

    @property
    def parent_class(self):
        return util.common_ancestor_class(list(self.dict.values()))

    @property
    def modes(self):
        return self.dict.keylist

    @property
    def short_modes(self):
        return SuperList(
            [
                self.ModeShortNames[m] if m in self.ModeShortNames else m
                for m in self.modes
            ]
        )

    def get_class(self, mode: str):
        if mode in self.short_modes:
            mode = [k for k in self.modes if self.ModeShortNames[k] == mode][0]
        if mode in self.modes:
            return self.dict[mode]
        else:
            return None

    def build_module(self, conf: Any, **kwargs: Any):
        if conf is not None and "mode" in conf:
            C = self.get_class(conf.mode)
            if C is not None:
                return C(**{k: conf[k] for k in conf if k != "mode"}, **kwargs)
        return None

    def module_conf(
        self, mode: str | None = None, include_mode: bool = True, **kwargs: Any
    ):
        if mode in self.short_modes:
            mode = [k for k in self.modes if self.ModeShortNames[k] == mode][0]
        if mode in self.default_dict:
            d = self.default_dict[mode]
            d.update_existingdict(kwargs)
            if include_mode:
                d["mode"] = mode
            return d
        else:
            return None

    def module_objects(
        self, mode: str | None = None, excluded: list[Any] | None = None
    ):
        if excluded is None:
            excluded = self.excluded
        C = self.get_class(mode=mode)
        if C is not None:
            return class_objs(A=C, excluded=excluded)

        else:
            return AttrDict()

    def module_pars(self, **kwargs: Any):
        return self.module_objects(**kwargs).keylist

    def as_entry(self, d: AttrDict):
        return AttrDict({f"brain.{self.mID}": d})


class BrainModuleDB(NestedConf):
    BrainModuleModes = AttrDict(
        {
            "crawler": {
                "constant": crawler.Crawler,
                "gaussian": crawler.GaussOscillator,
                "square": crawler.SquareOscillator,
                "realistic": crawler.PhaseOscillator,
                "nengo": basic.NengoEffector,
            },
            "interference": {
                "default": crawl_bend_interference.DefaultCoupling,
                "square": crawl_bend_interference.SquareCoupling,
                "phasic": crawl_bend_interference.PhasicCoupling,
            },
            "turner": {
                "neural": turner.NeuralOscillator,
                "sinusoidal": turner.SinTurner,
                "constant": turner.ConstantTurner,
                "nengo": basic.NengoEffector,
            },
            "intermitter": {
                "default": intermitter.Intermitter,
                "branch": intermitter.BranchIntermitter,
            },
            "feeder": {"default": feeder.Feeder, "nengo": basic.NengoEffector},
            "olfactor": {
                "default": sensor.Olfactor,
                "osn": sensor.OSNOlfactor,
            },
            "toucher": {
                "default": sensor.Toucher,
            },
            "windsensor": {
                "default": sensor.Windsensor,
            },
            "thermosensor": {
                "default": sensor.Thermosensor,
            },
            "memory": {
                "RL": {"olfaction": memory.RLOlfMemory, "touch": memory.RLTouchMemory},
                "MB": {
                    "olfaction": memory.RemoteBrianModelMemory,
                    "touch": memory.RemoteBrianModelMemory,
                },
            },
            # 'memory': {
            #     'RL': memory.RLmemory,
            #     'MB': memory.RemoteBrianModelMemory
            # },
        }
    )

    BrainModuleColors = AttrDict(
        {
            "crawler": "lightcoral",
            "turner": "indianred",
            "interference": "lightsalmon",
            "intermitter": "#a55af4",
            "olfactor": "palegreen",
            "windsensor": "plum",
            "thermosensor": "plum",
            "toucher": "pink",
            "feeder": "pink",
            "memory": "pink",
        }
    )

    def __init__(self, **kwargs: Any) -> None:
        self.LocoModsBasic = SuperList(
            ["crawler", "turner", "interference", "intermitter"]
        )
        self.LocoMods = SuperList(
            ["crawler", "turner", "interference", "intermitter", "feeder"]
        )
        self.SensorMods = SuperList(
            ["olfactor", "toucher", "windsensor", "thermosensor"]
        )
        self.BrainMods = self.BrainModuleModes.keylist
        self.brainDB = AttrDict(
            {
                k: BrainModule(
                    mID=k,
                    dict=self.BrainModuleModes[k],
                    color=self.BrainModuleColors[k],
                )
                for k in self.BrainMods
            }
        )

        super().__init__(**kwargs)

    def mod_modes(self, k: str, short: bool = False):
        if k not in self.BrainMods:
            return None
        else:
            if short:
                return self.brainDB[k].short_modes
            else:
                return self.brainDB[k].modes

    def build_module(
        self, mID: str | None = None, conf: Any | None = None, **kwargs: Any
    ):
        return (
            self.brainDB[mID].build_module(conf=conf, **kwargs)
            if mID in self.BrainMods
            else None
        )

    def build_modules(self, mIDs: list[str], conf: AttrDict, **kwargs: Any):
        return AttrDict(
            {
                mID: self.build_module(
                    mID=mID, conf=conf[mID] if mID in conf else None, **kwargs
                )
                for mID in mIDs
            }
        )

    def build_locomodules(self, conf: AttrDict, **kwargs: Any):
        return self.build_modules(mIDs=self.LocoMods, conf=conf, **kwargs)

    def build_sensormodules(self, conf: AttrDict, **kwargs: Any):
        return self.build_modules(mIDs=self.SensorMods, conf=conf, **kwargs)

    def module_conf(
        self,
        mID: str | None = None,
        mode: str | None = None,
        as_entry: bool = True,
        **kwargs: Any,
    ):
        M = self.brainDB[mID]
        conf = M.module_conf(mode=mode, **kwargs) if mID in self.BrainMods else None
        return M.as_entry(conf) if as_entry else conf

    def module_objects(
        self,
        mID: str | None = None,
        mode: str | None = None,
        as_entry: bool = True,
        **kwargs: Any,
    ):
        M = self.brainDB[mID]
        objs = (
            M.module_objects(mode=mode, **kwargs)
            if mID in self.BrainMods
            else AttrDict()
        )
        return M.as_entry(objs) if as_entry else objs

    def modules_objects(
        self, mIDs: list[str], conf: AttrDict, as_entry: bool = True, **kwargs: Any
    ):
        C = AttrDict(
            {
                mID: self.module_objects(
                    mID,
                    conf[mID] if mID in conf else AttrDict(),
                    as_entry=False,
                    **kwargs,
                )
                for mID in mIDs
            }
        )
        return (
            AttrDict({f"brain.{mID}": C[mID] for mID in C}).flatten() if as_entry else C
        )

    def module_pars(self, **kwargs: Any):
        return self.module_objects(**kwargs).flatten().keylist

    def modules_pars(self, **kwargs: Any):
        return self.modules_objects(**kwargs).keylist

    def brainConf(self, ms: AttrDict = AttrDict(), mkws: AttrDict = AttrDict()):
        C = AttrDict()
        for k in self.BrainMods:
            C[k] = self.brainDB[k].module_conf(
                mode=ms[k] if k in ms else None, **mkws[k] if k in mkws else {}
            )
        C.nengo = C.crawler is not None and C.crawler.mode == "nengo"
        return C

    def mcolor(self, k: str):
        return self.brainDB[k].color if k in self.BrainMods else None

    def mod_combs(self, ks: SuperList, short: bool = False, to_return: str = "yield"):
        ks = ks.existing(self.BrainMods)
        x = itertools.product(*[self.mod_modes(k, short=short) for k in ks])
        if to_return == "yield":
            return x
        elif to_return == "list":
            return list(x)

    def parent_class(self, k: str):
        return self.brainDB[k].parent_class if k in self.BrainMods else None

    def get_memory_class(self, mode: str, modality: str):
        try:
            return self.brainDB["memory"].dict[mode][modality]
        except:
            return None

    def memory_kws(
        self,
        mode: str = "RL",
        modality: str = "olfaction",
        as_entry: bool = True,
        **kwargs: Any,
    ):
        A = self.get_memory_class(mode, modality)
        if A is not None:
            c = class_defaults(
                A=A,
                excluded=["dt"],
                included={"mode": mode, "modality": modality},
                **kwargs,
            )
            return AttrDict({"brain.memory": c}) if as_entry else c
        else:
            return None

    def build_memory_module(self, conf: AttrDict, **kwargs: Any):
        if conf is not None and "mode" in conf and "modality" in conf:
            A = self.get_memory_class(conf.mode, conf.modality)
            if A is not None:
                return A(
                    **{k: conf[k] for k in conf if k not in ["mode", "modality"]},
                    **kwargs,
                )
        return None

    def detect_brainconf_modes(self, m: AttrDict):
        return AttrDict(
            {
                k: m[k].mode if (k in m and "mode" in m[k]) else None
                for k in self.BrainMods
            }
        )


class LarvaModuleDB(BrainModuleDB):
    LarvaModuleColors = AttrDict(
        {
            "body": "lightskyblue",
            "physics": "lightsteelblue",
            "energetics": "lightskyblue",
            "DEB": "lightskyblue",
            "gut": "lightskyblue",
            "Box2D": "lightcoral",
            "sensorimotor": "lightcoral",
        }
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ModuleColorDict = AttrDict(
            **self.BrainModuleColors, **self.LarvaModuleColors
        )
        self.LarvaModsBasic = SuperList(["body", "physics"])
        self.LarvaModsOptional = SuperList(["energetics", "sensorimotor", "Box2D"])
        self.LarvaMods = self.LarvaModsBasic + self.LarvaModsOptional
        self.AllModules = self.BrainMods + self.LarvaMods
        self.LarvaModsConfDict = AttrDict(
            {
                "body": self.body_kws,
                "physics": self.physics_kws,
                "energetics": self.energetics_kws,
                "sensorimotor": self.sensorimotor_kws,
                "Box2D": self.Box2D_kws,
            }
        )
        self.LarvaModsDefaultDict = AttrDict(
            {k: f() for k, f in self.LarvaModsConfDict.items()}
        )

    def sensorimotor_kws(self, **kwargs: Any):
        return class_defaults(
            agents.ObstacleLarvaRobot, excluded=[agents.LarvaRobot], **kwargs
        )

    def energetics_kws(
        self, gut_kws: AttrDict = AttrDict(), DEB_kws: AttrDict = AttrDict()
    ):
        return AttrDict(
            {
                "DEB": class_defaults(
                    deb.DEB, excluded=[deb.DEB_model, "substrate", "id"], **DEB_kws
                ),
                "gut": class_defaults(deb.Gut, **gut_kws),
            }
        )

    def body_kws(self, **kwargs: Any):
        return class_defaults(
            agents.LarvaSegmented,
            excluded=[
                agents.OrientedAgent,
                "vertices",
                "base_vertices",
                "width",
                "guide_points",
                "segs",
            ],
            **kwargs,
        )

    def physics_kws(self, **kwargs: Any):
        return class_defaults(agents.BaseController, **kwargs)

    def Box2D_kws(self, **kwargs: Any):
        d = AttrDict(
            {
                "joint_types": {
                    "friction": {"N": 0, "args": {}},
                    "revolute": {"N": 0, "args": {}},
                    "distance": {"N": 0, "args": {}},
                }
            }
        )
        return d.update_existingnestdict(kwargs)

    def larvaConf(self, ms: AttrDict = AttrDict(), mkws: AttrDict = AttrDict()):
        C = AttrDict({"brain": self.brainConf(ms=ms, mkws=mkws)})
        for k, c in self.LarvaModsDefaultDict.items():
            if k in self.LarvaModsOptional:
                if k not in mkws:
                    C[k] = None
                    continue
            if k not in mkws:
                mkws[k] = {}
            C[k] = c.update_existingnestdict(mkws[k])
        return C


moduleDB = LarvaModuleDB()


class SpaceDict(NestedConf):
    base_model = reg.conf.Model.confID_selector()
    space_mkeys = param.ListSelector(
        default=[],
        objects=moduleDB.AllModules,
        label="keys of modules to include in space search",
        doc="Keys of the modules where the optimization parameters are",
    )
    Pmutation = param.Magnitude(
        default=0.3,
        step=0.01,
        label="mutation probability",
        doc="Probability of mutation for each agent in the next generation",
    )
    Cmutation = param.Magnitude(
        default=0.1,
        step=0.01,
        label="mutation coeficient",
        doc="Fraction of allowed parameter range to mutate within",
    )

    init_mode = param.Selector(
        default="random",
        objects=["random", "model", "default"],
        label="mode of initial generation",
        doc="Mode of initial generation",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mConf0 = reg.conf.Model.getID(self.base_model)
        self.space_objs = self.build()
        self.space_ks = self.space_objs.keylist
        self.parclasses = AttrDict({p: self.parclass(p) for p in self.space_ks})

    def build(self):
        D = AttrDict()
        for k in self.space_mkeys:
            xx = self.mConf0.brain[k]
            if xx is not None:
                A = moduleDB.BrainModuleModes[k][xx.mode]
                Aobjs = class_objs(A, excluded=[basic.Effector, "phi", "name"])
                for p, obj in Aobjs.items():
                    if p in xx:
                        obj.default = xx[p]
                    D[f"brain.{k}.{p}"] = obj
        return D

    def obj_attr(self, k, flat=True):
        if flat:
            return AttrDict(
                {
                    p: getattr(obj, k) if hasattr(obj, k) else None
                    for p, obj in self.space_objs.items()
                }
            )
        else:
            return AttrDict(
                {
                    obj.name: getattr(obj, k) if hasattr(obj, k) else None
                    for p, obj in self.space_objs.items()
                }
            )

    def obj_min_max_value(self, p):
        obj = self.space_objs[p]
        v = obj.default
        if isinstance(v, tuple):
            if v[0] is None:
                v = v[1]
            elif v[1] is None:
                v = v[0]
            else:
                v = np.mean([v[0], v[1]])
        min, max = obj.bounds if hasattr(obj, "bounds") else (None, None)
        step = obj.step if hasattr(obj, "step") else None
        return param._utils._get_min_max_value(min, max, value=v, step=step)

    @property
    def defaults(self):
        return self.obj_attr("default")

    def parclass(self, p):
        obj = self.space_objs[p]
        c = type(obj)

        def check(k):
            m = getattr(param, k)
            ms = [m] + m.__subclasses__()
            return c in ms or issubclass(c, m)

        valid = [
            k
            for k in [
                "Magnitude",
                "Integer",
                "Number",
                "Selector",
                "Boolean",
                "Range",
                "Dict",
            ]
            if check(k)
        ]
        return valid[0]

    def randomize(self) -> AttrDict:
        """
        Randomizes the values of the parameters in the model based on their types.

        This method iterates over the parameters defined in `self.space_ks` and assigns
        new random values to them based on their respective classes (`self.parclasses`).
        The new values are chosen according to the following rules:

        - If the parameter class is "Selector", a random object from `obj.objects` is chosen.
        - If the parameter class is "Boolean", a random boolean value (True or False) is chosen.
        - If the parameter class is "Dict", the parameter is skipped.
        - If the parameter class is "Range", two random values within the bounds are chosen and clipped.
        - If the parameter class is "Integer", a random integer within the bounds is chosen and clipped.
        - For other parameter classes, a random float within the bounds is chosen and clipped.

        The bounds for the random values are determined by `obj.bounds` or `self.obj_min_max_value(p)`.

        Returns:
            None
        """
        g = self.defaults
        for p in self.space_ks:
            v = g[p]
            obj = self.space_objs[p]
            cl = self.parclasses[p]
            if cl in ["Selector"]:
                g[p] = random.choice(obj.objects)
            elif cl in ["Boolean"]:
                g[p] = random.choice([True, False])
            elif cl in ["Dict"]:
                pass
            else:
                vmin, vmax = obj.bounds
                if None in (vmin, vmax):
                    vmin0, vmax0, vv = self.obj_min_max_value(p)
                else:
                    vmin0, vmax0 = vmin, vmax

                if cl in ["Range"]:
                    vnew = np.clip(random.uniform(vmin0, vmax0), a_min=vmin, a_max=vmax)
                    vnew2 = np.clip(random.uniform(vnew, vmax0), a_min=vnew, a_max=vmax)
                    g[p] = (vnew, vnew2)
                elif cl in ["Integer"]:
                    g[p] = obj.crop_to_bounds(random.randint(vmin0, vmax0))
                else:
                    g[p] = obj.crop_to_bounds(random.uniform(vmin0, vmax0))
        return g

    def mutate(self, g: AttrDict) -> AttrDict:
        """
        Mutates the given genome `g` based on predefined mutation probabilities and rules.

        Args:
            g (dict): The genome to be mutated. It is a dictionary where keys are parameter names
                      and values are their corresponding values.

        Returns:
            dict: The mutated genome.

        Mutation Rules:
            - For parameters classified as "Selector", a random choice from the available objects is selected.
            - For parameters classified as "Boolean", a random boolean value is selected.
            - For parameters classified as "Dict", no mutation is performed.
            - For other parameters:
                - If the parameter has a defined step, the mutation range is calculated as step * 5.
                - If the parameter has defined bounds, the mutation range is calculated as half the difference between the bounds.
                - The mutation is applied using a Gaussian distribution with the calculated standard deviation.
                - For "Range" parameters, each bound is mutated separately and clipped to the original bounds.
                - For "Integer" parameters, the mutated value is converted to an integer and cropped to the original bounds.
                - For other types, the mutated value is cropped to the original bounds.
        """
        for p in self.space_ks:
            v = g[p]
            if random.random() < self.Pmutation:
                obj = self.space_objs[p]
                cl = self.parclasses[p]
                if cl in ["Selector"]:
                    g[p] = random.choice(obj.objects)
                elif cl in ["Boolean"]:
                    g[p] = random.choice([True, False])
                elif cl in ["Dict"]:
                    pass
                else:
                    if v is not None:
                        if hasattr(obj, "step") and obj.step is not None:
                            vr = obj.step * 5
                        else:
                            vmin, vmax = obj.bounds
                            if None in (vmin, vmax):
                                vmin, vmax, vv = self.obj_min_max_value(p)
                            vr = np.abs(vmax - vmin) * 0.5
                        s = self.Cmutation * vr
                        if cl in ["Range"]:
                            vmin, vmax = obj.bounds
                            vnew = np.clip(
                                random.gauss(v[0], s), a_min=vmin, a_max=vmax
                            )
                            vnew2 = np.clip(
                                random.gauss(v[1], s), a_min=vnew, a_max=vmax
                            )
                            g[p] = (vnew, vnew2)
                        elif cl in ["Integer"]:
                            g[p] = obj.crop_to_bounds(int(random.gauss(v, s)))
                        else:
                            g[p] = obj.crop_to_bounds(random.gauss(v, s))

        return g

    def create_first_generation(self, N):
        m = self.init_mode
        if m == "default":
            return [self.defaults] * N
        elif m == "model":
            return [AttrDict({k: self.mConf0.flatten()[k] for k in self.space_ks})] * N
        elif m == "random":
            return [self.randomize() for i in range(N)]
        else:
            raise ValueError("Not implemented")


@funcs.stored_conf("Model")
def Model_dict():
    MD = moduleDB
    LMs = MD.LocoModsBasic

    def olf_kws(g={"Odor": 150.0}, mode="default", **kwargs):
        return MD.module_conf(mID="olfactor", mode=mode, gain_dict=g, **kwargs)

    E = {}

    def new(id, id0, kws={}):
        try:
            E[id] = E[id0].new_dict(kws)
        except:
            pass

    def extend(id0, pref=None):
        if pref is None:
            pref = id0

        def new0(id, kws={}):
            new(id=id, id0=id0, kws=kws)

        for sg, g in zip(
            ["", "0", "_x2"],
            [{"Odor": 150.0}, {"Odor": 0.0}, {"CS": 150.0, "UCS": 0.0}],
        ):
            for sb, br in zip(["", "_brute"], [False, True]):
                idd = f"{pref}_navigator{sg}{sb}"
                o = olf_kws(g=g, brute_force=br)
                new0(idd, o)
                for k in ["RL", "MB"]:
                    new0(f"{idd}_{k}", {**o, **MD.memory_kws(k)})

        for ss, eeb in zip(["", "_max"], [0.5, 0.9]):
            f = AttrDict(
                {
                    **MD.module_conf(mID="feeder", mode="default"),
                    "brain.intermitter.feed_bouts": True,
                    "brain.intermitter.EEB": eeb,
                }
            )
            new0(f"{pref}{ss}_feeder", f)
            for sg, g in zip(
                ["", "0", "_x2"],
                [{"Odor": 150.0}, {"Odor": 0.0}, {"CS": 150.0, "UCS": 0.0}],
            ):
                idd = f"{pref}{ss}_forager{sg}"
                o = olf_kws(g=g)
                new0(idd, {**o, **f})
                for k in ["RL", "MB"]:
                    new0(f"{idd}_{k}", {**o, **f, **MD.memory_kws(k)})

        for mm in [f"{pref}_avg", f"{pref}_var", f"{pref}_var2"]:
            if mm in reg.conf.Model.confIDs:
                E[mm] = reg.conf.Model.getID(mm)

    for id, (Tm, ImM) in zip(
        ["Levy", "NEU_Levy", "NEU_Levy_continuous"],
        [("SIN", "DEF"), ("NEU", "DEF"), ("NEU", None)],
    ):
        E[id] = MD.larvaConf(
            ms=AttrDict(zip(LMs, ["CON", Tm, "DEF", ImM])),
            mkws={
                "interference": {"attenuation": 0.0},
                "intermitter": {"run_mode": "exec"},
            },
        )
        extend(id0=id)

    for mms in MD.mod_combs(LMs, short=True):
        kws = {
            "ms": AttrDict(zip(LMs, mms)),
            "mkws": {"interference": {"attenuation": 0.1, "attenuation_max": 0.6}}
            if mms[2] != "DEF"
            else {},
        }
        if "NENGO" in mms:
            if list(mms) != ["NENGO", "NENGO", "SQ", "DEF"]:
                continue
            id = "nengo_explorer"
            E[id] = MD.larvaConf(**kws)
            extend(id0=id, pref="nengo")
        else:
            id = "_".join(mms)
            E[id] = MD.larvaConf(**kws)
            if mms[0] == "RE" and mms[3] == "DEF":
                extend(id0=id)
                if mms[1] == "NEU" and mms[2] == "PHI":
                    for idd in [
                        "navigator",
                        "navigator_x2",
                        "forager",
                        "forager0",
                        "forager_x2",
                        "max_forager",
                        "max_forager0",
                        "forager_RL",
                        "forager0_RL",
                        "max_forager_RL",
                        "max_forager0_RL",
                        "forager_MB",
                        "forager0_MB",
                        "max_forager_MB",
                        "max_forager0_MB",
                        "feeder",
                        "max_feeder",
                    ]:
                        E[idd] = E[f"{id}_{idd}"]
                    E["explorer"] = E[id]
                    E["RLnavigator"] = E[f"{id}_navigator_RL"]

    for id, dd in zip(
        [
            "imitator",
            "zebrafish",
            "thermo_navigator",
            "OSNnavigator",
            "OSNnavigator_x2",
        ],
        [
            {"body.Nsegs": 11},
            {
                "body.body_plan": "zebrafish_larva",
                "Box2D": {
                    "joint_types": {
                        "revolute": {
                            "N": 1,
                            "args": {"maxMotorTorque": 10**5, "motorSpeed": 1},
                        }
                    }
                },
            },
            MD.module_conf(mID="thermosensor", mode="default"),
            olf_kws(mode="osn"),
            olf_kws({"CS": 150.0, "UCS": 0.0}, mode="osn"),
        ],
    ):
        new(id, "explorer", dd)
    for ss, kkws in zip(
        ["", "_2", "_brute"], [{}, {"touch_sensors": [0, 2]}, {"brute_force": True}]
    ):
        new(
            f"toucher{ss}",
            "explorer",
            MD.module_conf(mID="toucher", mode="default", **kkws),
        )
        new(f"RLtoucher{ss}", f"toucher{ss}", MD.memory_kws(modality="touch"))
    for id, gd in zip(
        ["follower-R", "follower-L", "gamer", "gamer-5x"],
        [
            {"Left_odor": 150.0, "Right_odor": 0.0},
            {"Left_odor": 0.0, "Right_odor": 150.0},
            {"Flag_odor": 150.0, "Left_base_odor": 0.0, "Right_base_odor": 0.0},
            {
                "Flag_odor": 150.0,
                "Left_base_odor": 0.0,
                "Right_base_odor": 0.0,
                "Left_odor": 0.0,
                "Right_odor": 0.0,
            },
        ],
    ):
        new(id, "forager", {"brain.olfactor.gain_dict": gd})

    new(
        "immobile",
        "navigator",
        {
            "brain.crawler": None,
            "brain.turner": None,
            "brain.intermitter": None,
            "brain.interference": None,
            **MD.module_conf(mID="toucher", mode="default"),
        },
    )
    new("obstacle_avoider", "navigator", {"sensorimotor": MD.sensorimotor_kws()})

    for id in ["explorer", "navigator", "feeder", "forager"]:
        new(
            f"{id}_sample",
            id,
            {k: "sample" for k in MD.module_pars(mID="crawler", mode="RE")},
        )

    # Build larva-models having a DEB energetics component
    for sp, k_abs, eeb in zip(["rover", "sitter"], [0.8, 0.4], [0.67, 0.37]):
        en_ws = AttrDict(
            {
                "energetics": MD.energetics_kws(
                    gut_kws={"k_abs": k_abs}, DEB_kws={"species": sp}
                )
            }
        )
        en_ws2 = {**en_ws, "brain.intermitter.EEB": eeb}
        new(f"{sp}_explorer", "explorer", en_ws)
        new(f"{sp}_navigator", "navigator", en_ws)
        new(f"{sp}_feeder", "feeder", en_ws2)
        new(f"{sp}_forager", "forager", en_ws2)
        new(sp, f"{sp}_feeder")

    return E
