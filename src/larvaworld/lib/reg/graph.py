"""
Graph/plotting registry for the larvaworld package.
This module provides a class for managing and creating plots and graphs.
"""

from __future__ import annotations
from typing import Any

import os

from .. import util, funcs, plot

__all__: list[str] = [
    "GraphRegistry",
]


class GraphRegistry:
    """
    Registry for managing and creating plots and graphs.

    Manages graph functions, graphgroups (collections of related plots),
    and provides methods for batch plot creation during analysis.

    Attributes:
        dict: Dictionary of graph functions
        required_data_dict: Dictionary of required data for each graph function
        graphgroups: Dictionary of graphgroups, each containing plot specifications

    Example:
        >>> graph_reg = GraphRegistry()
        >>> graph_reg.exists('trajectories')  # Check if graph exists
        >>> plots = graph_reg.run_group('traj')  # Run trajectory graphgroup
        >>> graph_reg.eval_graphgroups(['general', 'endpoint'], save_to='plots/')
    """

    def __init__(self) -> None:
        self.dict = funcs.graphs
        self.required_data_dict = funcs.graph_required_data
        self.graphgroups = self.build_graphgroups()

    @property
    def ks(self) -> util.SuperList:
        """
        Returns a sorted list of the keys of the graph functions.
        """
        return util.SuperList(self.dict.keys()).sorted

    def exists(self, ID: str) -> bool:
        """
        Checks if a graph function with the given ID exists.
        """
        if isinstance(ID, str) and ID in self.ks:
            return True
        else:
            return False

    def group_exists(self, gID: str) -> bool:
        """
        Checks if a graphgroup with the given ID exists.
        """
        if isinstance(gID, str) and gID in self.graphgroups:
            return True
        else:
            return False

    def eval_graphgroups(
        self, graphgroups: list[str] | dict, save_to: str | None = None, **kws: Any
    ) -> util.AttrDict:
        """
        Evaluates a list of graphgroups.

        Args:
        - graphgroups (list): A list of graphgroups to evaluate.
        - save_to (str, optional): The directory to save the plots to. Defaults to None.
        - **kws: Additional keyword arguments to pass to the graph functions.

        Returns:
        - dict: A dictionary of evaluated graphgroups.

        """
        kws.update({"subfolder": None})
        d = self.grouplist_to_dict(graphgroups)
        return util.AttrDict(
            {
                gID: self.eval_entries(
                    entries,
                    save_to=f"{save_to}/{gID}" if save_to is not None else None,
                    **kws,
                )
                for gID, entries in d.items()
            }
        )

    def grouplist_to_dict(self, groups: list | dict) -> util.AttrDict | dict:
        """
        Converts a list of graphgroups to a dictionary.

        Args:
        - groups (list): A list of graphgroups to convert.

        Returns:
        - dict: A dictionary of graphgroups.

        """
        if isinstance(groups, list):
            ds = util.AttrDict()
            for gg in groups:
                if isinstance(gg, str) and gg in self.graphgroups:
                    gg = {gg: self.graphgroups[gg]}
                assert isinstance(gg, dict)
                assert len(gg) == 1
                gID = list(gg)[0]
                ds[gID] = gg[gID]
            return ds
        elif isinstance(groups, dict):
            return groups

    def eval_entries(self, entries: list[dict], **kwargs: Any) -> util.AttrDict:
        """
        Evaluates a list of graph entries.

        Args:
        - entries (list): A list of graph entries to evaluate.

        Returns:
        - dict: A dictionary of evaluated graph entries.

        """
        return util.AttrDict(
            {e["key"]: self.run(ID=e["plotID"], **e["args"], **kwargs) for e in entries}
        )

    def run(self, ID: str, **kwargs: Any) -> Any:
        """
        Runs a graph function with the given ID.

        Args:
        - ID (str): The ID of the graph function to run.

        Returns:
        - plot.Plot: The plot created by the graph function.

        """
        assert self.exists(ID)
        return self.dict[ID](**kwargs)

    def run_group(self, gID: str, **kwargs: Any) -> util.AttrDict:
        """
        Runs a graphgroup with the given ID.

        Args:
        - gID (str): The ID of the graphgroup to run.

        Returns:
        - dict: A dictionary of plots created by the graphgroup.

        """
        assert self.group_exists(gID)
        return self.eval_entries(self.graphgroups[gID], **kwargs)

    def entry(self, ID: str, name: str | None = None, **kwargs: Any) -> dict:
        """
        Creates a graph entry with the given ID and optional name.

        Args:
        - ID (str): The ID of the graph function to create an entry for.
        - name (str, optional): The name of the graph entry. Defaults to None.

        Returns:
        - dict: A dictionary containing the key, plotID, and arguments for the graph entry.

        """
        assert self.exists(ID)
        args = kwargs
        if name is not None:
            args["name"] = name
            key = name
        else:
            key = ID
        return {"key": key, "plotID": ID, "args": args}

    def model_tables(
        self,
        mIDs: list[str],
        dIDs: list[str] | None = None,
        save_to: str | None = None,
        **kwargs: Any,
    ) -> util.AttrDict:
        """
        Creates tables for the given model IDs.

        Args:
        - mIDs (list): A list of model IDs to create tables for.
        - dIDs (list, optional): A list of display IDs to use for the model IDs. Defaults to None.
        - save_to (str, optional): The directory to save the tables to. Defaults to None.

        Returns:
        - dict: A dictionary of tables created for the given model IDs.

        """
        ds = {}
        ds["mdiff_table"] = self.dict["model diff"](
            mIDs, dIDs=dIDs, save_to=save_to, **kwargs
        )
        gfunc = self.dict["model table"]
        for mID in mIDs:
            try:
                ds[f"{mID}_table"] = gfunc(mID, save_to=save_to, **kwargs)
            except:
                print("TABLE FAIL", mID)
        if save_to is not None and len(ds) > 1:
            util.combine_pdfs(
                file_dir=save_to, save_as="_MODEL_TABLES_.pdf", include_subdirs=False
            )
        return util.AttrDict(ds)

    def model_summaries(
        self, mIDs: list[str], save_to: str | None = None, **kwargs: Any
    ) -> dict:
        """
        Creates summary plots for the given model IDs.

        Args:
        - mIDs (list): A list of model IDs to create summaries for.
        - save_to (str, optional): The directory to save the summaries to. Defaults to None.

        Returns:
        - dict: A dictionary of summary plots created for the given model IDs.

        """
        ds = {}
        for mID in mIDs:
            try:
                ds[f"{mID}_summary"] = self.dict["model summary"](
                    mID, save_to=save_to, **kwargs
                )
            except:
                print("SUMMARY FAIL", mID)
        if save_to is not None and len(ds) > 0:
            util.combine_pdfs(
                file_dir=save_to, save_as="_MODEL_SUMMARIES_.pdf", include_subdirs=False
            )
        return ds

    def store_model_graphs(self, mIDs: list[str], dir: str) -> util.AttrDict:
        """
        Stores model graphs for the given model IDs.

        Args:
        - mIDs (list): A list of model IDs to store graphs for.
        - dir (str): The directory to store the graphs in.

        Returns:
        - dict: A dictionary of stored model graphs.

        """
        f1 = f"{dir}/plots/model_tables"
        f2 = f"{dir}/plots/model_summaries"
        os.makedirs(f1, exist_ok=True)
        os.makedirs(f2, exist_ok=True)

        graphs = util.AttrDict(
            {
                "tables": self.model_tables(mIDs, save_to=f1),
                "summaries": self.model_summaries(
                    mIDs, Nids=10, refDataset=self, save_to=f2
                ),
            }
        )
        return graphs

    def source_graphgroup(
        self, source_ID: str, pos: tuple[float, float] | None = None, **kwargs: Any
    ) -> util.AttrDict:
        """
        Creates a graphgroup consisting of plots related to a given food/odor source.

        Args:
        - source_ID (str): The ID of the source to create a graphgroup for.
        - pos (tuple, optional): The position of the source. Defaults to None.

        Returns:
        - dict: A dictionary of plot entries for source-related plotting.

        """
        ID = source_ID
        gID = f"locomotion_relative_to_source_{ID}"
        d0 = [
            # FIXME Currently the bearing related plots are buggy
            # self.entry('bearing/turn', name=f'bearing to {ID}', min_angle=5.0, ref_angle=None, source_ID=ID, **kwargs),
            # self.entry('bearing/turn', name='bearing to 270deg', min_angle=5.0, ref_angle=270, source_ID=ID, **kwargs),
            *[
                self.entry("timeplot", name=p, pars=[p], **kwargs)
                for p in [
                    util.nam.bearing_to(ID),
                    util.nam.dst_to(ID),
                    util.nam.scal(util.nam.dst_to(ID)),
                ]
            ],
        ]
        # FIXME Currently the bearing related plots are buggy
        # for chunk in ['stride', 'pause', 'Lturn', 'Rturn']:
        #     for dur in [0.0, 0.5, 1.0]:
        #         d0.append(self.entry('bearing to source/epoch', name=f'{chunk}_bearing2_{ID}_min_{dur}_sec',
        #                              min_dur=dur, chunk=chunk, source_ID=ID, **kwargs))
        return util.AttrDict({gID: d0})

    def get_analysis_graphgroups(
        self, exp: str, sources: dict[str, Any], **kwargs: Any
    ) -> util.AttrDict | dict:
        """
        Determines the plots to be created during the analysis of a given experiment/simulation.

        Args:
        - exp (str): The experiment that has been completed for which the analysis is requested.
            The string that defines the type of experiment is associated with one or more graphgroups by their keys.
        - sources (dict): A dictionary of the IDs and positions of any food/odor sources in the arena.
            For certain experiments these are needed to create source related plots.

        Returns:
        - list: All graphgroups (by their keys) and optionally any single graphs to be created (evaluated) during analysis

        """
        groups = ["traj", "general"]

        if exp in ["random_food"]:
            groups.append("survival")
        else:
            dic = {
                "patch": ["patch"],
                "tactile": ["tactile"],
                "thermo": ["thermo"],
                "RvsS": ["deb", "intake"],
                "growth": ["deb", "intake"],
                "anemo": ["anemotaxis"],
                "puff": ["puff"],
                "chemo": ["chemo"],
                "RL": ["RL"],
                # 'dispersion': ['comparative_analysis'],
                "dispersion": ["endpoint", "distro", "dsp"],
            }
            for k, v in dic.items():
                if k in exp:
                    groups += v

        # FIXME Currently the source related plots don't work as expected. Timeplot takes forever and bearing related plots are buggy
        groups += [
            self.source_graphgroup(id, pos=pos, **kwargs) for id, pos in sources.items()
        ]
        return self.grouplist_to_dict(groups)

    def build_graphgroups(self) -> util.AttrDict:
        """
        Creates a dictionary of lists/groups of plots/graphs.
        Each such group of plots (graphgroup) can be accessed by a key so that all the plots included in the group can be created.
        The analysis of a given experiment/simulation can be associated with several such graphgroups in the get_analysis_graphgroups method.
        """
        d = util.AttrDict(
            {
                "general": [
                    # self.entry('ethogram', add_samples=False),
                    self.entry("pathlength", scaled=False),
                    # self.entry('navigation index'),
                    self.entry("epochs", stridechain_duration=True),
                ],
                "tactile": [
                    self.entry(
                        "endpoint hist", "time ratio on food (final)", ks=["on_food_tr"]
                    ),
                    self.entry(
                        "timeplot", "time ratio on food", ks=["on_food_tr"], unit="min"
                    ),
                    self.entry(
                        "timeplot", "time on food", ks=["cum_f_det"], unit="min"
                    ),
                    self.entry(
                        "timeplot",
                        "turner input",
                        ks=["A_tur"],
                        unit="min",
                        show_first=True,
                    ),
                    self.entry(
                        "timeplot",
                        "turner output",
                        ks=["Act_tur"],
                        unit="min",
                        show_first=True,
                    ),
                    self.entry(
                        "timeplot",
                        "tactile activation",
                        ks=["A_touch"],
                        unit="min",
                        show_first=True,
                    ),
                    self.entry("ethogram"),
                ],
                "chemo": [
                    # autotime(['sv', 'fov', 'b', 'a']),
                    self.entry(
                        "timeplots",
                        "chemosensation",
                        ks=["c_odor1", "dc_odor1", "A_olf", "A_T", "I_T"],
                        individuals=False,
                    ),
                    self.entry("trajectories"),
                    # self.entry('turn amplitude'),
                    # self.entry('angular pars', Npars=5),
                ],
                "intake": [
                    # 'deb_analysis',
                    # *[time(p) for p in ['sf_faeces_M', 'f_faeces_M', 'sf_abs_M', 'f_abs_M', 'f_am']],
                    self.entry("food intake (timeplot)", "food intake (raw)"),
                    self.entry(
                        "food intake (timeplot)",
                        "food intake (filtered)",
                        filt_amount=True,
                    ),
                    self.entry("pathlength", scaled=False),
                    self.entry("barplot", name="food intake (barplot)", ks=["f_am"]),
                    self.entry("ethogram"),
                ],
                "anemotaxis": [
                    *[
                        self.entry(
                            "nengo",
                            name=p,
                            group=p,
                            same_plot=True if p == "anemotaxis" else False,
                        )
                        for p in [
                            "anemotaxis",
                            "frequency",
                            "interference",
                            "velocity",
                            "crawler",
                            "turner",
                            "wind_effect_on_V",
                            "wind_effect_on_Fr",
                        ]
                    ],
                    self.entry("timeplots", "anemotaxis", ks=["A_wind", "anemotaxis"]),
                    self.entry(
                        "endpoint hist", name="final anemotaxis", ks=["anemotaxis"]
                    ),
                ],
                "thermo": [
                    self.entry("trajectories"),
                    self.entry(
                        "timeplots",
                        "thermosensation",
                        ks=["temp_W", "dtemp_W", "temp_C", "dtemp_C", "A_therm"],
                        show_first=True,
                    ),
                ],
                "puff": [
                    # self.entry('trajectories'),
                    # self.entry('ethogram', add_samples=False),
                    self.entry("pathlength", scaled=False),
                    # *[self.entry('timeplot', ks=[p], absolute=True) for p in ['fov', 'foa']],
                    self.entry(
                        "timeplots", "angular moments", ks=["fov", "foa"], absolute=True
                    ),
                    # *[time(p, abs=True) for p in ['fov', 'foa','b', 'bv', 'ba']],
                    # *[self.entry('timeplot', ks=[p]) for p in ['sv', 'sa']],
                    self.entry("timeplots", "translational moments", ks=["sv", "sa"]),
                    # *[time(p) for p in ['sv', 'sa', 'v', 'a']],
                ],
                "RL": [
                    self.entry(
                        "timeplot",
                        "olfactor_decay_table",
                        ks=["D_olf"],
                        table="best_gains",
                    ),
                    self.entry(
                        "timeplot",
                        "olfactor_decay_table_inds",
                        ks=["D_olf"],
                        table="best_gains",
                        individuals=True,
                    ),
                    self.entry(
                        "timeplot",
                        "reward_table",
                        ks=["cum_reward"],
                        table="best_gains",
                    ),
                    self.entry(
                        "timeplot",
                        "best_gains_table",
                        ks=["g_odor1"],
                        table="best_gains",
                    ),
                    self.entry(
                        "timeplot",
                        "best_gains_table_x2",
                        ks=["g_odor1", "g_odor2"],
                        table="best_gains",
                    ),
                ],
                "patch": [
                    self.entry(
                        "timeplots", "Y position", ks=["y"], legend_loc="lower left"
                    ),
                    self.entry("navigation index"),
                    self.entry("turn amplitude"),
                    self.entry("turn duration"),
                    self.entry(
                        "turn amplitude VS Y pos",
                        "turn angle VS Y pos (scatter)",
                        mode="scatter",
                    ),
                    self.entry(
                        "turn amplitude VS Y pos",
                        "turn angle VS Y pos (hist)",
                        mode="hist",
                    ),
                    self.entry(
                        "turn amplitude VS Y pos",
                        "bearing correction VS Y pos",
                        mode="hist",
                        ref_angle=270,
                    ),
                ],
                "survival": [
                    # 'foraging_list',
                    self.entry(
                        "timeplot", "time ratio on food", ks=["on_food_tr"], unit="min"
                    ),
                    self.entry("food intake (timeplot)", "food intake (raw)"),
                    self.entry("pathlength", scaled=False),
                ],
                "deb": [
                    *[
                        self.entry(
                            "deb",
                            name=f"DEB.{m} (hours)",
                            sim_only=False,
                            mode=m,
                            save_as=f"{m}_in_hours.pdf",
                        )
                        for m in ["energy", "growth", "full"]
                    ],
                    *[
                        self.entry(
                            "deb",
                            name=f"FEED.{m} (hours)",
                            sim_only=True,
                            mode=m,
                            save_as=f"{m}_in_hours.pdf",
                        )
                        for m in [
                            "feeding",
                            "reserve_density",
                            "assimilation",
                            "food_ratio_1",
                            "food_ratio_2",
                            "food_mass_1",
                            "food_mass_2",
                            "hunger",
                            "EEB",
                            "fs",
                        ]
                    ],
                ],
                "endpoint": [
                    self.entry(
                        "endpoint box",
                        ks=[
                            "l",
                            "str_N",
                            "dsp_0_40_max",
                            "run_tr",
                            "fv",
                            "ffov",
                            "v_mu",
                            "sv_mu",
                            "tor5_mu",
                            "tor5_std",
                            "tor20_mu",
                            "tor20_std",
                        ],
                    ),
                    self.entry("endpoint box", ks=["l", "fv", "v_mu", "run_tr"]),
                    self.entry("crawl pars"),
                ],
                "submission": [
                    self.entry("endpoint box", mode="tiny", Ncols=4),
                    self.entry("crawl pars"),
                    self.entry("epochs", stridechain_duration=True),
                    self.entry("dispersal", range=(20, 100)),
                    self.entry("dispersal", range=(10, 70)),
                    self.entry("dispersal", range=(20, 80)),
                    self.entry("dispersal", range=(20, 100), scaled=True),
                    self.entry("dispersal", range=(10, 70), scaled=True),
                    self.entry("dispersal", range=(20, 80), scaled=True),
                ],
                "distro": [
                    self.entry("distros", mode="box"),
                    self.entry("distros", mode="hist"),
                    self.entry("angular pars", Npars=5),
                ],
                "dsp": [
                    self.entry("dispersal", range=(0, 40)),
                    # self.entry('dispersal', range=(0, 60)),
                    self.entry("dispersal summary", range=(0, 40)),
                    # self.entry('dispersal summary', range=(0, 60)),
                ],
                "stride": [
                    self.entry("stride cycle"),
                    self.entry("stride cycle", individuals=True),
                ],
                "traj": [
                    self.entry("trajectories", mode="default", unit="mm"),
                    self.entry(
                        "trajectories",
                        name="aligned2origin",
                        mode="origin",
                        unit="mm",
                        single_color=True,
                    ),
                ],
                "track": [
                    self.entry("stride track"),
                    self.entry("turn track"),
                ],
            }
        )
        return d
