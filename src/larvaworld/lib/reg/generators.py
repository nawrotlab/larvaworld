"""
Configuration and Generator classes for higher-order objects in the larvaworld package.
"""

import os
import shutil

import matplotlib
import numpy as np
import pandas as pd
import param

from ... import CONFTYPES, DATA_DIR, SIM_DIR, SIMTYPES, vprint
from .. import reg, util
from .larvagroup import LarvaGroup
from ..model import (
    AnalyticalValueLayer,
    Border,
    DiffusionValueLayer,
    Food,
    FoodGrid,
    GaussianValueLayer,
    OdorScape,
    Source,
    ThermoScape,
    WindScape,
)
from ..param import (
    AirPuff,
    Area,
    ClassAttr,
    ClassDict,
    EnrichConf,
    Epoch,
    Filesystem,
    NestedConf,
    Odor,
    OptionalPositiveInteger,
    OptionalPositiveRange,
    OptionalSelector,
    PreprocessConf,
    RuntimeOps,
    SimMetricOps,
    SimOps,
    Substrate,
    TrackerOps,
    class_generator,
)
from ..util import AttrDict, nam

__all__ = [
    "gen",
    "SimConfiguration",
    "SimConfigurationParams",
    "FoodConf",
    "EnvConf",
    "LabFormat",
    "ExpConf",
    "ReplayConfGroup",
    "ReplayConfUnit",
    "ReplayConf",
]

gen = AttrDict(
    {
        "FoodGroup": class_generator(Food, mode="Group"),
        "Food": class_generator(Food),
        "Source": class_generator(Source),
        "Arena": class_generator(Area),
        "Border": class_generator(Border),
        "Odor": class_generator(Odor),
        "Epoch": class_generator(Epoch),
        "Substrate": class_generator(Substrate),
        "FoodGrid": class_generator(FoodGrid),
        "WindScape": class_generator(WindScape),
        "ThermoScape": class_generator(ThermoScape),
        "OdorScape": class_generator(OdorScape),
        "AnalyticalValueLayer": class_generator(AnalyticalValueLayer),
        "DiffusionValueLayer": class_generator(DiffusionValueLayer),
        "GaussianValueLayer": class_generator(GaussianValueLayer),
        "AirPuff": class_generator(AirPuff),
    }
)

gen.LarvaGroup = class_generator(LarvaGroup)


class SimConfiguration(RuntimeOps, SimMetricOps, SimOps):
    """
    The configuration of a simulation run.
    """

    runtype = param.Selector(objects=SIMTYPES, doc="The simulation mode")

    def __init__(self, runtype, **kwargs):
        self.param.add_parameter("experiment", self.exp_selector_param(runtype))
        super().__init__(runtype=runtype, **kwargs)

        if "experiment" in kwargs and kwargs["experiment"] is not None:
            self.experiment = kwargs["experiment"]

        if self.id is None or not type(self.id) == str:
            self.id = self.generate_id(self.runtype, self.experiment)
        if self.dir is None:
            save_to = f"{self.path_to_runtype_data}/{self.experiment}"
            self.dir = f"{save_to}/{self.id}"

    @property
    def path_to_runtype_data(self):
        return f"{SIM_DIR}/{self.runtype.lower()}_runs"

    def generate_id(self, runtype, exp):
        idx = reg.config.next_idx(exp, conftype=runtype)
        return f"{exp}_{idx}"

    def exp_selector_param(self, runtype):
        defaults = {
            "Exp": "dish",
            "Batch": "PItest_off",
            "Ga": "exploration",
            "Eval": "dispersion",
            "Replay": "replay",
        }
        kws = {"default": defaults[runtype], "doc": "The experiment simulated"}
        if runtype in CONFTYPES:
            return param.Selector(objects=reg.conf[runtype].confIDs, **kws)
        else:
            return param.Parameter(**kws)


class SimConfigurationParams(SimConfiguration):
    """
    The configuration of a simulation run with parameters.
    """

    parameters = param.Parameter(default=None)

    def __init__(
        self,
        runtype="Exp",
        experiment=None,
        parameters=None,
        N=None,
        modelIDs=None,
        groupIDs=None,
        sample=None,
        **kwargs,
    ):
        if parameters is None:
            if runtype in CONFTYPES:
                ct = reg.conf[runtype]
                if experiment is None:
                    raise ValueError(
                        f"Either a parameter dictionary or the ID of an available {runtype} configuration must be provided"
                    )
                elif experiment not in ct.confIDs:
                    raise ValueError(
                        f"Experiment {experiment} not available in {runtype} configuration dictionary"
                    )
                else:
                    parameters = ct.getID(experiment)
            elif runtype in reg.gen:
                parameters = reg.gen[runtype]().nestedConf
            else:
                pass
        elif experiment is None and "experiment" in parameters:
            experiment = parameters["experiment"]

        if "env_params" in parameters and isinstance(parameters.env_params, str):
            parameters.env_params = reg.conf.Env.getID(parameters.env_params)

        if parameters is not None:
            for k in set(parameters).intersection(set(SimOps().nestedConf)):
                if k in kwargs:
                    parameters[k] = kwargs[k]
                else:
                    kwargs[k] = parameters[k]

        if "larva_groups" in parameters:
            from .larvagroup import update_larva_groups

            parameters.larva_groups = update_larva_groups(
                parameters.larva_groups,
                modelIDs=modelIDs,
                groupIDs=groupIDs,
                Ns=N,
                sample=sample,
            )
        super().__init__(
            runtype=runtype, experiment=experiment, parameters=parameters, **kwargs
        )


def source_generator(
    genmode,
    Ngs=2,
    ids=None,
    cs=None,
    rs=None,
    ams=None,
    o=None,
    qs=None,
    type="standard",
    **kwargs,
):
    """
    Generate a list of source units or groups.

    Args:
    - genmode (str): The generation mode, either 'Group' or 'Unit'.
    - Ngs (int): The number of sources to generate.
    - ids (list, optional): The IDs of the sources.
    - cs (list, optional): The colors of the sources.
    - rs (list, optional): The radii of the sources.
    - ams (list, optional): The amount of food of the sources.
    - o (str, optional): The odor of the sources.
    - qs (list, optional): The substrate qualities of the sources.
    - type (str, optional): The substrate type of the sources.
    - **kwargs: Additional keyword arguments to pass to the source generation function.

    Returns:
    - dict: A dictionary of generated source units or groups.

    """
    if genmode == "Group":
        id0 = "SourceGroup"
        _class = gen.FoodGroup
    elif genmode == "Unit":
        id0 = "Source"
        _class = gen.Food
    else:
        raise ValueError(f"mode arg must be Group or Unit, instead was : {genmode}")
    if ids is None:
        ids = [f"{id0}{i}" for i in range(Ngs)]
    if ams is None:
        ams = np.random.uniform(0.002, 0.01, Ngs)
    if rs is None:
        rs = ams
    if qs is None:
        qs = np.linspace(0.1, 1, Ngs)
    if cs is None:
        cs = [
            matplotlib.colors.rgb2hex(
                tuple(util.col_range(q, low=(255, 0, 0), high=(0, 128, 0)))
            )
            for q in qs
        ]
    l = [
        _class(
            c=cs[i],
            r=rs[i],
            a=ams[i],
            odor=Odor.oO(o=o, id=f"Odor{i}"),
            sub=[qs[i], type],
            **kwargs,
        ).entry(ids[i])
        for i in range(Ngs)
    ]
    result = {}
    for d in l:
        result.update(d)
    return result


class FoodConf(NestedConf):
    """
    The configuration of food sources in the arena.
    """

    source_groups = ClassDict(
        item_type=gen.FoodGroup,
        doc="The groups of odor or food sources available in the arena",
    )
    source_units = ClassDict(
        item_type=gen.Food, doc="The individual sources  of odor or food in the arena"
    )
    food_grid = ClassAttr(gen.FoodGrid, default=None, doc="The food grid in the arena")

    @classmethod
    def CS_UCS(
        cls, grid=None, sg={}, N=1, x=0.04, colors=["red", "blue"], o="G", **kwargs
    ):
        F = gen.Food
        CS_kws = {"odor": Odor.oO(o=o, id="CS"), "c": colors[0], **kwargs}
        UCS_kws = {"odor": Odor.oO(o=o, id="UCS"), "c": colors[1], **kwargs}

        if N == 1:
            su = {
                **F(pos=(-x, 0.0), **CS_kws).entry("CS"),
                **F(pos=(x, 0.0), **UCS_kws).entry("UCS"),
            }
        elif N == 2:
            su = {
                **F(pos=(-x, 0.0), **CS_kws).entry("CS_l"),
                **F(pos=(x, 0.0), **CS_kws).entry("CS_r"),
                **F(pos=(-x, 0.0), **UCS_kws).entry("UCS_l"),
                **F(pos=(x, 0.0), **UCS_kws).entry("UCS_r"),
            }
        else:
            raise

        return cls(source_groups=sg, source_units=su, food_grid=grid)

    @classmethod
    def double_patch(
        cls,
        grid=None,
        sg={},
        type="standard",
        q=1.0,
        c="green",
        x=0.06,
        r=0.025,
        a=0.1,
        o="G",
        **kwargs,
    ):
        F = gen.Food
        kws = {"odor": Odor.oO(o=o), "c": c, "r": r, "a": a, "sub": [q, type], **kwargs}
        su = {
            **F(pos=(-x, 0.0), **kws).entry("Left_patch"),
            **F(pos=(x, 0.0), **kws).entry("Right_patch"),
        }
        return cls(source_groups=sg, source_units=su, food_grid=grid)

    @classmethod
    def patch(
        cls,
        grid=None,
        sg={},
        id="Patch",
        type="standard",
        q=1.0,
        c="green",
        r=0.01,
        a=0.1,
        **kwargs,
    ):
        kws = {"c": c, "r": r, "a": a, "sub": [q, type], **kwargs}
        return cls.su(id=id, grid=grid, sg=sg, **kws)

    @classmethod
    def su(cls, id="Source", grid=None, sg={}, **kwargs):
        return cls(
            source_groups=sg, source_units=gen.Food(**kwargs).entry(id), food_grid=grid
        )

    @classmethod
    def sus(cls, grid=None, sg={}, **kwargs):
        return cls(
            source_groups=sg,
            source_units=source_generator(genmode="Unit", **kwargs),
            food_grid=grid,
        )

    @classmethod
    def sg(cls, id="SourceGroup", grid=None, su={}, **kwargs):
        return cls(
            source_groups=gen.FoodGroup(**kwargs).entry(id),
            source_units=su,
            food_grid=grid,
        )

    @classmethod
    def sgs(cls, grid=None, su={}, **kwargs):
        return cls(
            source_groups=source_generator(genmode="Group", **kwargs),
            source_units=su,
            food_grid=grid,
        )

    @classmethod
    def foodNodor_4corners(
        cls,
        d=0.05,
        colors=["blue", "red", "green", "magenta"],
        grid=None,
        sg={},
        o="D",
        **kwargs,
    ):
        ps = [(-d, -d), (-d, d), (d, -d), (d, d)]
        l = [
            gen.Food(
                pos=ps[i],
                a=0.01,
                odor=Odor.oO(o=o, id=f"Odor_{i}"),
                c=colors[i],
                r=0.01,
                **kwargs,
            ).entry(f"Source_{i}")
            for i in range(4)
        ]
        sus = {**l[0], **l[1], **l[2], **l[3]}
        return cls(source_groups=sg, source_units=sus, food_grid=grid)


gen.FoodConf = FoodConf
gen.EnrichConf = EnrichConf
# gen.EnrichConf = class_generator(EnrichConf)


class EnvConf(NestedConf):
    """
    The configuration of the simulation's virtual environment.
    """

    arena = ClassAttr(gen.Arena, doc="The arena configuration")
    food_params = ClassAttr(gen.FoodConf, doc="The food sources in the arena")
    border_list = ClassDict(item_type=gen.Border, doc="The obstacles in the arena")
    odorscape = ClassAttr(
        class_=(
            gen.GaussianValueLayer,
            gen.AnalyticalValueLayer,
            gen.DiffusionValueLayer,
        ),
        default=None,
        doc="The sensory odor landscape in the arena",
    )
    windscape = ClassAttr(
        gen.WindScape, default=None, doc="The wind landscape in the arena"
    )
    thermoscape = ClassAttr(
        gen.ThermoScape, default=None, doc="The thermal landscape in the arena"
    )

    def __init__(self, odorscape=None, **kwargs):
        if odorscape is not None and isinstance(odorscape, AttrDict):
            mode = odorscape.odorscape
            odorscape_classes = list(EnvConf.param.odorscape.class_)
            odorscape_modes = dict(
                zip(["Gaussian", "Analytical", "Diffusion"], odorscape_classes)
            )
            odorscape = odorscape_modes[mode](**odorscape)

        super().__init__(odorscape=odorscape, **kwargs)

    def visualize(self, **kwargs):
        """
        Visualize the environment by launching a simulation without agents
        """
        from ..sim.base_run import BaseRun

        BaseRun.visualize_Env(envConf=self.nestedConf, envID=self.name, **kwargs)

    @classmethod
    def food_params_class(cls):
        return EnvConf.param.food_params.class_

    @classmethod
    def arena_class(cls):
        return EnvConf.param.arena.class_

    @classmethod
    def maze(cls, n=15, h=0.1, o="G", **kwargs):
        def get_maze(nx=15, ny=15, ix=0, iy=0, h=0.1, return_points=False):
            from ..model.envs.maze import Maze

            m = Maze(nx, ny, ix, iy, height=h)
            m.make_maze()
            lines = m.maze_lines()
            if return_points:
                ps = []
                for l in lines:
                    ps.append(l.coords[0])
                    ps.append(l.coords[1])
                ps = [(np.round(x - h / 2, 3), np.round(y - h / 2, 3)) for x, y in ps]
                return ps
            else:
                return lines

        return cls.rect(
            h,
            f=cls.food_params_class().su(id="Target", odor=Odor.oO(o=o), c="blue"),
            bl=AttrDict(
                {
                    "Maze": EnvConf.param.border_list.item_type(
                        vertices=get_maze(nx=n, ny=n, h=h, return_points=True),
                        color="black",
                        width=0.001,
                    )
                }
            ),
            o=o,
            **kwargs,
        )

    @classmethod
    def game(cls, dim=0.1, x=0.4, y=0.0, o="G", **kwargs):
        x = np.round(x * dim, 3)
        y = np.round(y * dim, 3)
        F = gen.Food
        sus = {
            **F(
                c="green",
                can_be_carried=True,
                a=0.01,
                odor=Odor.oO(o=o, c=2, id="Flag_odor"),
            ).entry("Flag"),
            **F(pos=(-x, y), c="blue", odor=Odor.oO(o=o, id="Left_base_odor")).entry(
                "Left_base"
            ),
            **F(pos=(+x, y), c="red", odor=Odor.oO(o=o, id="Right_base_odor")).entry(
                "Right_base"
            ),
        }

        return cls.rect(dim, f=cls.food_params_class()(source_units=sus), o=o, **kwargs)

    @classmethod
    def foodNodor_4corners(cls, dim=0.2, o="D", **kwargs):
        return cls.rect(
            dim,
            f=cls.food_params_class().foodNodor_4corners(d=dim / 4, o=o, **kwargs),
            o=o,
        )

    @classmethod
    def CS_UCS(cls, dim=0.1, o="G", **kwargs):
        return cls.dish(
            dim, f=cls.food_params_class().CS_UCS(x=0.4 * dim, o=o, **kwargs), o=o
        )

    @classmethod
    def double_patch(cls, dim=0.24, o="G", **kwargs):
        return cls.rect(
            dim,
            f=cls.food_params_class().double_patch(x=0.25 * dim, o=o, **kwargs),
            o=o,
        )

    @classmethod
    def odor_gradient(cls, dim=(0.1, 0.06), o="G", c=1, **kwargs):
        return cls.rect(
            dim, f=cls.food_params_class().su(odor=Odor.oO(o=o, c=c), **kwargs), o=o
        )

    @classmethod
    def dish(cls, xy=0.1, **kwargs):
        assert isinstance(xy, float)
        return cls.scapes(
            arena=cls.arena_class()(geometry="circular", dims=(xy, xy)), **kwargs
        )

    @classmethod
    def rect(cls, xy=0.1, **kwargs):
        if isinstance(xy, float):
            dims = (xy, xy)
        elif isinstance(xy, tuple):
            dims = xy
        else:
            raise
        return cls.scapes(
            arena=cls.arena_class()(geometry="rectangular", dims=dims), **kwargs
        )

    @classmethod
    def scapes(cls, o=None, w=None, th=None, f=None, bl={}, **kwargs):
        if f is None:
            f = cls.food_params_class()()
        if o == "D":
            o = gen.DiffusionValueLayer()
        elif o == "G":
            o = gen.GaussianValueLayer()
        if w is not None:
            if "puffs" in w:
                for id, args in w["puffs"].items():
                    w["puffs"][id] = AirPuff(**args).nestedConf
            else:
                w["puffs"] = {}
            w = EnvConf.param.windscape.class_(**w)
        if th is not None:
            th = EnvConf.param.thermoscape.class_(**th)
        return cls(
            odorscape=o,
            windscape=w,
            thermoscape=th,
            food_params=f,
            border_list=bl,
            **kwargs,
        )


# gen.Env = class_generator(EnvConf)
gen.Env = EnvConf


class LabFormat(NestedConf):
    """
    The configuration of the lab format.
    """

    labID = param.String(doc="The identifier ID of the lab")
    tracker = ClassAttr(TrackerOps, doc="The dataset metadata")
    filesystem = ClassAttr(Filesystem, doc="The import-relevant lab-format filesystem")
    env_params = ClassAttr(EnvConf, doc="The environment configuration")
    preprocess = ClassAttr(PreprocessConf, doc="The preprocessing configuration")

    @property
    def path(self):
        return f"{DATA_DIR}/{self.labID}Group"

    @property
    def raw_folder(self):
        return f"{self.path}/raw"

    @property
    def processed_folder(self):
        return f"{self.path}/processed"

    def get_source_dir(self, parent_dir, raw_folder=None, merged=False):
        if raw_folder is None:
            raw_folder = self.raw_folder
        source_dir = f"{raw_folder}/{parent_dir}"
        if merged:
            source_dir = [f"{source_dir}/{f}" for f in os.listdir(source_dir)]
        return source_dir

    def get_store_sequence(self, mode="semifull"):
        if mode == "full":
            return self.filesystem.read_sequence[1:]
        elif mode == "minimal":
            return nam.xy(self.tracker.point)
        elif mode == "semifull":
            return (
                nam.midline_xy(self.tracker.Npoints, flat=True)
                + nam.contour_xy(self.tracker.Ncontour, flat=True)
                + ["collision_flag"]
            )
        elif mode == "points":
            return nam.xy(self.tracker.points, flat=True) + ["collision_flag"]
        else:
            raise

    @property
    def import_func(self):
        from ..process.importing import lab_specific_import_functions as d

        return d[self.labID]

    def import_data_to_dfs(
        self, parent_dir, raw_folder=None, merged=False, save_mode="semifull", **kwargs
    ):
        source_dir = self.get_source_dir(parent_dir, raw_folder, merged)
        if self.filesystem.structure == "per_larva":
            read_sequence = self.filesystem.read_sequence
            store_sequence = self.get_store_sequence(save_mode)
        return self.import_func(
            source_dir=source_dir,
            tracker=self.tracker,
            filesystem=self.filesystem,
            **kwargs,
        )

    def build_dataset(
        self,
        step,
        end,
        parent_dir,
        proc_folder=None,
        group_id=None,
        id=None,
        sample=None,
        color="black",
        epochs=[],
        age=0.0,
        refID=None,
    ):
        if group_id is None:
            group_id = parent_dir
        if id is None:
            id = f"{self.labID}_{group_id}_dataset"
        if proc_folder is None:
            proc_folder = self.processed_folder
        dir = f"{proc_folder}/{group_id}/{id}"

        conf = {
            "initialize": True,
            "load_data": False,
            "dir": dir,
            "id": id,
            "refID": refID,
            "color": color,
            "larva_group": gen.LarvaGroup(
                group_id=group_id,
                c=color,
                sample=sample,
                mID=None,
                N=end.index.values.shape[0],
                life=[age, epochs],
            ).nestedConf,
            "env_params": self.env_params.nestedConf,
            **self.tracker.nestedConf,
            "step": step,
            "end": end,
        }
        from ..process.dataset import LarvaDataset

        d = LarvaDataset(**conf)
        vprint(
            f"***-- Dataset {d.id} created with {len(d.config.agent_ids)} larvae! -----",
            1,
        )
        return d

    def import_dataset(
        self,
        parent_dir,
        raw_folder=None,
        merged=False,
        proc_folder=None,
        group_id=None,
        id=None,
        sample=None,
        color="black",
        epochs=[],
        age=0.0,
        refID=None,
        enrich_conf=None,
        save_dataset=False,
        **kwargs,
    ):
        """
         Imports a single experimental dataset defined by their ID from a source folder.

        Parameters
        ----------
         parent_dir: string
             The parent directory where the raw files are located.

         raw_folder: string, optional
             The directory where the raw files are located.
             If not provided it is set as the subfolder 'raw' under the lab-specific group directory.
          merged: boolean
             Whether to merge all raw datasets in the source folder in a single imported dataset.
             Defaults to False.

         proc_folder: string, optional
             The directory where the imported dataset will be placed.
             If not provided it is set as the subfolder 'processed' under the lab-specific group directory.
         group_id: string, optional
             The group ID of the dataset to be imported.
             If not provided it is set as the parent_dir argument.
         id: string, optional
             The ID under which to store the imported dataset.
             If not provided it is set by default.


         sample: string, optional
             The reference ID of the reference dataset from which the current is sampled.
         color: string
             The default color of the new dataset.
             Defaults to 'black'.
         epochs: dict
             Any discrete rearing epochs during the larvagroup's life history.
             Defaults to '{}'.
         age: float
             The post-hatch age of the larvae in hours.
             Defaults to '0.0'.

        refID: string, optional
             The reference IDs under which to store the imported dataset as reference dataset.
             If not provided the dataset is not stored in the reference database.
         save_dataset: boolean
             Whether to store the imported dataset to disc.
             Defaults to True.
         enrich_conf: dict, optional
             The configuration for enriching the imported dataset with secondary parameters.
         **kwargs: keyword arguments
             Additional keyword arguments to be passed to the lab_specific build-function.

        Returns
        -------
         lib.process.dataset.LarvaDataset
             The imported dataset in the common larvaworld format.

        """
        vprint("", 1)
        vprint(
            f"----- Importing experimental dataset by the {self.labID} lab-specific format. -----",
            1,
        )
        step, end = self.import_data_to_dfs(
            parent_dir, raw_folder=raw_folder, merged=merged, **kwargs
        )
        if step is None and end is None:
            vprint("xxxxx Failed to create dataset! -----", 1)
            return None
        else:
            step = step.astype(float)
            d = self.build_dataset(
                step,
                end,
                parent_dir,
                proc_folder=proc_folder,
                group_id=group_id,
                id=id,
                sample=sample,
                color=color,
                epochs=epochs,
                age=age,
                refID=refID,
            )
            if enrich_conf is None:
                enrich_conf = AttrDict()
            enrich_conf.pre_kws = self.preprocess.nestedConf
            d.enrich(**enrich_conf, is_last=False)
            vprint(
                f"****- Processed dataset {d.id} to derive secondary metrics -----", 1
            )

            if save_dataset:
                shutil.rmtree(d.config.dir, ignore_errors=True)
                d.save()
            return d

    def import_datasets(self, source_ids, ids=None, colors=None, refIDs=None, **kwargs):
        """
        Imports multiple experimental datasets defined by their IDs.

        Parameters
        ----------
        source_ids: list of strings
            The IDs of the datasets to be imported as appearing in the source files.
        ids: list of strings, optional
            The IDs under which to store the datasets to be imported.
            The source_ids are used if not provided.
        refIDs: list of strings, optional
            The reference IDs under which to store the imported datasets as reference datasets.
             If not provided the datasets are not stored in the reference database.
        colors: list of strings, optional
            The colors of the datasets to be imported.
            Randomly selected if not provided.
        **kwargs: keyword arguments
            Additional keyword arguments to be passed to the import_dataset function.

        Returns
        -------
        list of lib.process.dataset.LarvaDataset
            The imported datasets in the common larvaworld format.

        """
        Nds = len(source_ids)
        if colors is None:
            colors = util.N_colors(Nds)
        if ids is None:
            ids = source_ids
        if refIDs is None:
            refIDs = [None] * Nds

        assert len(ids) == Nds
        assert len(colors) == Nds
        assert len(refIDs) == Nds

        return [
            self.import_dataset(
                id=ids[i],
                color=colors[i],
                source_id=source_ids[i],
                refID=refIDs[i],
                **kwargs,
            )
            for i in range(Nds)
        ]

    def read_timeseries_from_raw_files_per_larva(
        self, files, read_sequence, store_sequence, inv_x=False
    ):
        """
        Reads timeseries data stored in txt files of the lab-specific Jovanic format and returns them as a pd.Dataframe.

        Parameters
        ----------
        files : list
            List of the absolute filepaths of the data files.
        read_sequence : list of strings
            The sequence of parameters found in each file
        store_sequence : list of strings
            The sequence of parameters to store
        inv_x : boolean
            Whether to invert x axis.
            Defaults to False

        Returns
        -------
        list of pandas.DataFrame

        """
        dfs = []
        for f in files:
            df = pd.read_csv(f, header=None, index_col=0, names=read_sequence)

            # If indexing is in strings replace with ascending floats
            if all([type(ii) == str for ii in df.index.values]):
                df.reset_index(inplace=True, drop=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            if inv_x:
                for x_par in [p for p in read_sequence if p.endswith("x")]:
                    df[x_par] *= -1
            df = df[store_sequence]
            dfs.append(df)
        return dfs


class ExpConf(SimOps):
    """
    The configuration of the experiment.
    """

    env_params = ClassAttr(gen.Env, doc="The environment configuration")
    experiment = reg.conf.Exp.confID_selector()
    trials = param.Dict(
        default=AttrDict({"epochs": util.ItemList()}),
        doc="Temporal epochs of the experiment",
    )
    collections = param.ListSelector(
        default=["pose"],
        objects=reg.parDB.output_keys,
        doc="The data to collect as output",
    )
    larva_groups = ClassDict(item_type=gen.LarvaGroup, doc="The larva groups")
    parameter_dict = param.Dict(
        default={}, doc="Dictionary of parameters to pass to the agents"
    )
    enrichment = ClassAttr(gen.EnrichConf, doc="The post-simulation processing")

    def __init__(self, id=None, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def imitation_exp(cls, refID, mID="explorer", **kwargs):
        c = reg.conf.Ref.getRef(refID)
        kws = {
            "sample": refID,
            "model": mID,
            "color": c.color,
            "distribution": {"N": c.N},
            "imitation": True,
        }
        return cls(
            dt=c.dt,
            duration=c.duration,
            env_params=gen.Env(**c.env_params),
            larva_groups=AttrDict({f"Imitation {refID}": gen.LarvaGroup(**kws)}),
            experiment="dish",
            **kwargs,
        )

    @property
    def agent_confs(self):
        confs = []
        for gID, gConf in self.larva_groups.items():
            lg = LarvaGroup(**gConf, id=gID)
            confs += lg(parameter_dict=self.parameter_dict)
        return confs


gen.Exp = ExpConf


class ReplayConfGroup(NestedConf):
    """
    The population-related configuration of the dataset replay.
    """

    agent_ids = param.List(
        item_type=int,
        doc="Whether to only display some larvae of the dataset, defined by their indexes.",
    )
    transposition = OptionalSelector(
        objects=["origin", "arena", "center"],
        doc="Whether to transpose the dataset spatial coordinates.",
    )
    track_point = param.Integer(
        default=-1,
        softbounds=(-1, 12),
        doc="The midline point to use for defining the larva position.",
    )
    env_params = reg.conf.Env.confID_selector()


class ReplayConfUnit(NestedConf):
    """
    The individual-related configuration of the dataset replay.
    """

    close_view = param.Boolean(
        False, doc="Whether to visualize a small arena on close range."
    )
    fix_segment = OptionalSelector(
        objects=["rear", "front"],
        doc="Whether to additionally fixate the above or below body segment.",
    )
    fix_point = OptionalPositiveInteger(
        softmin=1,
        softmax=12,
        doc="Whether to fixate a specific midline point to the center of the screen. Relevant when replaying a single larva track.",
    )


class ReplayConf(ReplayConfGroup, ReplayConfUnit):
    """
    The configuration of the dataset replay.
    """

    refID = reg.conf.Ref.confID_selector()
    refDir = param.String(None)
    time_range = OptionalPositiveRange(
        doc="Whether to only replay a defined temporal slice of the dataset."
    )
    overlap_mode = param.Boolean(
        False, doc="Whether to draw overlapped image of the track."
    )
    draw_Nsegs = OptionalPositiveInteger(
        softmin=1,
        softmax=12,
        doc="Whether to artificially simplify the experimentally tracked larva body to a segmented virtual body of the given number of segments.",
    )


gen.LabFormat = LabFormat
gen.Replay = class_generator(ReplayConf)
