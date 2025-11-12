from __future__ import annotations
from typing import Any, Optional
import os
import time

import agentpy
import numpy as np
import pandas as pd

from ... import vprint
from .. import reg, util
from ..reg import LarvaGroup
from ..process import LarvaDatasetCollection
from .base_run import BaseRun
from .conditions import get_exp_condition

__all__: list[str] = [
    "ExpRun",
]


class ExpRun(BaseRun):
    def __init__(
        self,
        experiment: Optional[str] = None,
        parameters: Optional[dict] = None,
        parameter_dict: dict = {},
        **kwargs: Any,
    ) -> None:
        """
        Simulation mode 'Exp' launches a single simulation of a specified experiment type.

        Args:
            **kwargs: Arguments passed to the setup method

        """
        super().__init__(
            runtype="Exp", experiment=experiment, parameters=parameters, **kwargs
        )
        self.parameter_dict = parameter_dict

    def setup(self) -> None:
        """
        Sets up the simulation environment and agents for a single run.

        This method performs the following steps:
        1. Initializes simulation epochs based on the provided trials.
        2. Converts epoch age ranges to start and stop times in simulation steps.
        3. Builds the simulation environment using the provided environment parameters.
        4. Constructs agents based on the provided larva groups and parameter dictionary.
        5. Sets up data collectors for the simulation.
        6. Eliminates overlap between agents if larva collisions are not allowed.
        7. Determines the termination condition for the simulation.
        """
        self.sim_epochs = self.p.trials.epochs
        for ep in self.sim_epochs:
            t1, t2 = ep.age_range
            ep["start"] = int(t1 * 60 / self.dt)
            ep["stop"] = int(t2 * 60 / self.dt)
        self.build_env(self.p.env_params)
        self.build_agents(self.p.larva_groups, self.parameter_dict)
        self.set_collectors(self.p.collections)
        self.accessible_sources = None
        if not self.larva_collisions:
            self.eliminate_overlap()
        k = get_exp_condition(self.experiment)
        self.exp_condition = k(env=self) if k is not None else None

    def step(self) -> None:
        """Defines the models' events per simulation step."""
        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        if len(self.sources) > 10:
            self.space.accessible_sources_multi(self.agents)
        self.agents.step()
        if self.Box2D:
            self.space.Step(self.dt, 6, 2)
            self.agents.updated_by_Box2D()

    def update(self) -> None:
        """Record a dynamic variable."""
        self.agents.nest_record(self.collectors["step"])

    def end(self) -> None:
        """Repord an evaluation measure."""
        self.screen_manager.finalize()
        self.agents.nest_record(self.collectors["end"])

    def simulate(self, **kwargs: Any):
        """
        Simulates the larva world environment and collects data.

        This method initializes the simulation, runs it, collects the output data,
        and optionally enriches and stores the data.

        Returns:
            list: A list of datasets collected during the simulation.
        """
        vprint(f"--- Simulation {self.id} initialized!--- ", 1)
        start = time.time()
        self.run(**kwargs)
        self.data_collection = LarvaDatasetCollection.from_agentpy_output(self.output)
        self.datasets = self.data_collection.datasets
        end = time.time()
        dur = np.round(end - start).astype(int)
        vprint(f"--- Simulation {self.id} completed in {dur} seconds!--- ", 1)
        if self.p.enrichment:
            for d in self.datasets:
                vprint(f"--- Enriching dataset {d.id} ---", 1)
                d.enrich(**self.p.enrichment, is_last=False)
                vprint(f"--- Dataset {d.id} enriched ---", 1)
                vprint("--------------------------------", 1)
        if self.store_data:
            self.store()
        return self.datasets

    def build_agents(self, larva_groups: dict, parameter_dict: dict = {}) -> None:
        """
        Builds and places agent groups in the simulation.

        Args:
            larva_groups (dict): A dictionary containing larva group configurations.
            parameter_dict (dict, optional): A dictionary of parameters to be passed to each larva group. Defaults to an empty dictionary.

        Returns:
            None
        """
        vprint(f"--- Simulation {self.id} : Generating agent groups!--- ", 1)
        confs = util.SuperList(
            [
                LarvaGroup(**v)(parameter_dict=parameter_dict)
                for v in larva_groups.values()
            ]
        ).flatten
        self.place_agents(confs)

    def eliminate_overlap(self) -> None:
        """
        Adjusts the positions of larva agents to eliminate overlaps.

        This method iteratively checks for collisions among larva agents and
        adjusts their positions randomly until no collisions are detected.
        The scale parameter is used to determine the precision of collision
        detection and adjustment.

        The process involves:
        1. Checking for existing collisions at a given scale.
        2. If collisions are detected, updating the positions of the larva agents.
        3. Repeating the process until no collisions are found.
        """
        scale = 3.0
        while self.collisions_exist(scale=scale):
            self.larva_bodies = self.get_larva_bodies(scale=scale)
            for l in self.agents:
                dx, dy = np.random.randn(2) * l.length / 10
                overlap = True
                while overlap:
                    ids = self.detect_collisions(l.unique_id)
                    if len(ids) > 0:
                        l.move_body(dx, dy)
                        self.larva_bodies[l.unique_id] = l.get_polygon(scale=scale)
                    else:
                        break

    def collisions_exist(self, scale: float = 1.0) -> bool:
        """
        Check if any collisions exist among the agents.

        This method scales the larva bodies and checks each agent for collisions.
        If any agent has collisions, the method returns True, otherwise False.

        Args:
            scale (float, optional): The scale factor to apply to the larva bodies. Defaults to 1.0.

        Returns:
            bool: True if any collisions are detected, False otherwise.
        """
        self.larva_bodies = self.get_larva_bodies(scale=scale)
        for l in self.agents:
            ids = self.detect_collisions(l.unique_id)
            if len(ids) > 0:
                return True
        return False

    def detect_collisions(self, id: int):
        """
        Detects collisions between a given larva and other larvae in the simulation.

        Args:
            id (int): The identifier of the larva to check for collisions.

        Returns:
            list: A list of identifiers of larvae that are colliding with the given larva.
        """
        ids = []
        for id0, body0 in self.larva_bodies.items():
            if id0 == id:
                continue
            if self.larva_bodies[id].intersects(body0):
                ids.append(id0)
        return ids

    def get_larva_bodies(self, scale: float = 1.0):
        """
        Retrieve the shapes of all larva agents in the simulation.

        Args:
            scale (float, optional): A scaling factor to apply to the shapes. Defaults to 1.0.

        Returns:
            dict: A dictionary where the keys are the unique IDs of the larva agents and the values are their shapes, scaled by the given factor.
        """
        return {l.unique_id: l.get_shape(scale=scale) for l in self.agents}

    def analyze(self, **kwargs: Any) -> None:
        """
        Analyzes the datasets based on the specified experiment type and generates plots or results accordingly.

        Keyword Args:
            **kwargs: Additional keyword arguments to be passed to the plotting functions.

        Returns:
            None

        This method performs the following steps:
        1. Creates the directory for storing plots if it does not exist.
        2. Checks if the datasets are available and valid.
        3. If the experiment type includes "PI" (odor preference), it extracts PI and PI2 values from the datasets and stores them in the results attribute.
        4. If the experiment type includes "disp" (disperal), it loads reference datasets based on unique samples in the datasets.
        5. Retrieves the analysis graph groups based on the experiment type and source coordinates.
        6. Evaluates the graph groups and generates the corresponding figures, saving them to the plot directory.
        """
        os.makedirs(self.plot_dir, exist_ok=True)
        exp = self.experiment
        ds = self.datasets
        if ds is None or any([d is None for d in ds]):
            return

        if "PI" in exp:
            PIs = {}
            PI2s = {}
            for d in ds:
                PIs[d.id] = d.config.PI["PI"]
                PI2s[d.id] = d.config.PI2
            self.results = {"PIs": PIs, "PI2s": PI2s}
            return

        if "disp" in exp:
            samples = util.unique_list([d.config.sample for d in ds])
            ds += [reg.conf.Ref.loadRef(sd) for sd in samples if sd is not None]
        self.figs = reg.graphs.eval_graphgroups(
            self.graphgroups, datasets=ds, save_to=self.plot_dir, **kwargs
        )

    @property
    def graphgroups(self):
        """
        Retrieve the graph groups for the current experiment type.

        Returns:
            list: A list of graph groups corresponding to the experiment type.
        """
        return reg.graphs.get_analysis_graphgroups(self.experiment, self.p.source_xy)

    def store(self) -> None:
        """
        Stores the simulation output and datasets.

        This method performs the following steps:
        1. Attempts to save the simulation output using the parameters specified in `self.p.agentpy_output_kws`.
        2. Creates the directory specified by `self.data_dir` if it does not already exist.
        3. Iterates over the datasets in `self.datasets`, saving each one and storing their larva dictionaries.

        Note:
            If an exception occurs during the saving of the simulation output, it is silently ignored.

        Raises:
            Any exceptions raised during the creation of the directory or saving of datasets are not handled and will propagate.
        """
        try:
            self.output.save(**self.p.agentpy_output_kws)
        except:
            pass
        os.makedirs(self.data_dir, exist_ok=True)
        for d in self.datasets:
            d.save()
            d.store_larva_dicts()

    def load_agentpy_output(self):
        """
        Load and process the output from an AgentPy simulation.

        This method loads the simulation output data using the parameters specified
        in `self.p.agentpy_output_kws`, concatenates the variables along the rows,
        and drops the second level of the index. The resulting DataFrame's index
        is renamed to "Model".

        Returns:
            pd.DataFrame: A DataFrame containing the processed simulation output data.
        """
        df = agentpy.DataDict.load(**self.p.agentpy_output_kws)
        df1 = pd.concat(df.variables, axis=0).droplevel(1, axis=0)
        df1.index.rename("Model", inplace=True)
        return df1

    @classmethod
    def from_ID(cls, id: str, simulate: bool = True, **kwargs: Any):
        """
        Create an instance of the class from a given experiment ID.

        Args:
            id (str): The experiment ID to create the instance from.
            simulate (bool, optional): Whether to run the simulation. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            object: An instance of the class initialized with the given experiment ID.

        Raises:
            AssertionError: If the provided ID is not in the list of valid experiment IDs.
        """
        assert id in reg.conf.Exp.confIDs
        r = cls(experiment=id, **kwargs)
        if simulate:
            _ = r.simulate()
        return r
