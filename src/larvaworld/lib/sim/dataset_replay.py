from __future__ import annotations
from typing import Any

import copy

import numpy as np

from .. import reg, util
from .base_run import BaseRun, render_configuration_text
from .manifest import (
    RunManifestSession,
    deterministic_random_context,
    json_ready,
    prepare_master_seed,
)
from ..util import nam

__all__: list[str] = [
    "ReplayRun",
]


class ReplayRun(BaseRun):
    def __init__(
        self,
        parameters: Any,
        dataset: Any | None = None,
        screen_kws: dict = {},
        _record_manifest: bool = True,
        _source_manifest: str | None = None,
        **kwargs: Any,
    ):
        """
        Simulation mode 'Replay' reconstructs a real or simulated experiment from stored data.

        Args:
            parameters: Dictionary of configuration parameters to be passed to the ABM model
            dataset: The stored dataset to replay. If not specified it is retrieved using either the storage path (parameters.dir) or the respective unique reference ID (parameters.RefID)
            experiment: The type of experiment. Defaults to 'replay'
            **kwargs: Arguments passed to parent class

        """
        # The screen settings below are derived from the dataset, so they are
        # written into a copy: `screen_kws` is the caller's dict, and when the
        # argument is omitted it is the shared default of this constructor,
        # which would carry one replay's settings into the next.
        screen_kws = dict(screen_kws)
        self._record_manifest = _record_manifest
        self._source_manifest = _source_manifest
        self._manifest_parameters = copy.deepcopy(parameters)
        dir = parameters.refDir if "refDir" in parameters else None
        d = reg.conf.Ref.retrieve_dataset(dataset=dataset, id=parameters.refID, dir=dir)

        # Configure the dataset to replay (deprecated)
        # self.refDataset = copy.deepcopy(d)
        # self.refDataset, screen_kws["background_motion"] = self.smaller_dataset(
        #     p=parameters, d=self.refDataset
        # )

        # Configure the dataset to replay (refactored)
        self.refDataset, screen_kws["background_motion"] = d.smaller_dataset(
            p=parameters
        )

        c = self.refDataset.config
        parameters.steps = c.Nsteps
        kwargs.update(**{"duration": c.duration, "dt": c.dt, "Nsteps": c.Nsteps})

        if parameters.draw_Nsegs == "all":
            parameters.draw_Nsegs = c.Npoints - 1

        if parameters.overlap_mode:
            screen_kws["vis_mode"] = "image"
            screen_kws["image_mode"] = "overlap"

        super().__init__(
            runtype="Replay", parameters=parameters, screen_kws=screen_kws, **kwargs
        )

    def _manifest_invocation(
        self, seed: int, steps: Any | None, display: bool
    ) -> dict[str, Any]:
        return {
            "run_class": f"{type(self).__module__}:{type(self).__qualname__}",
            "resolved_parameters": json_ready(self.p),
            "constructor": {"replay_parameters": json_ready(self._manifest_parameters)},
            "runtime_options": {
                "store_data": bool(self.store_data),
                "screen_kws": json_ready(self.screen_kws),
            },
            "execute": {
                "method": "run",
                "kwargs": {"steps": steps, "seed": seed, "display": display},
            },
        }

    def run(
        self,
        steps: Any | None = None,
        seed: Any | None = None,
        display: bool = True,
    ):
        master_seed = prepare_master_seed(seed)
        if not self._record_manifest:
            with deterministic_random_context(master_seed):
                return super().run(steps=steps, seed=master_seed, display=display)
        session = RunManifestSession(
            run=self,
            invocation=self._manifest_invocation(master_seed, steps, display),
            seed=master_seed,
            source_manifest=self._source_manifest,
        )
        try:
            with deterministic_random_context(master_seed):
                result = super().run(steps=steps, seed=master_seed, display=display)
            if getattr(self, "aborted", False):
                session.abort()
            else:
                session.finish(datasets=[self.refDataset])
            return result
        except KeyboardInterrupt as exc:
            session.abort(str(exc) or "Replay interrupted")
            raise
        except BaseException as exc:
            session.fail(exc)
            raise

    @property
    def configuration_items(self) -> dict[str, Any]:
        """
        Extends `BaseRun.configuration_items` (sim type/ID/storage dir/
        timing) with the replay-specific fields that are actually active
        for this replay -- optional fields left at their "unused" default
        (no time slice, no transposition, no fixed track point) are
        omitted rather than shown as clutter.
        """
        c = self.p
        items = dict(super().configuration_items)
        items["Reference Dataset"] = c.refID
        items["Time range (sec)"] = c.time_range
        items["Transposition"] = c.transposition
        if c.track_point is not None and c.track_point != -1:
            items["Tracked midline point"] = c.track_point
        return items

    @property
    def configuration_text(self) -> str:
        return render_configuration_text(
            "Dataset Replay configuration", self.configuration_items
        )

    def setup(self) -> None:
        self.draw_Nsegs = self.p.draw_Nsegs
        self.build_env(self.p.env_params)
        self.build_agents(d=self.refDataset)

    def build_agents(self, d: Any) -> None:
        s, e, c = d.data

        if "length" in e.columns:
            ls = e["length"].values
        else:
            ls = np.ones(c.N) * 0.005

        assert util.cols_exist(["front_orientation", "rear_orientation"], s)
        if self.p.draw_Nsegs is not None:
            if self.p.draw_Nsegs == 2:
                pass
            elif self.p.draw_Nsegs == c.Npoints - 1:
                seg_orientD = d.midline_seg_orients_data_byID
                midlineD = d.midline_seg_xy_data_byID
            else:
                raise
        else:
            contourD = d.contour_xy_data_byID
            midlineD = d.midline_xy_data_byID

        confs = []
        for i, id in enumerate(c.agent_ids):
            conf = util.AttrDict(
                {"unique_id": id, "length": ls[i], "color": d.config.color}
            )
            data = util.AttrDict()
            ss = s.xs(id, level="AgentID", drop_level=True)
            xy = ss[["x", "y"]].values
            data.pos = util.np2Dtotuples(xy)
            fo, ro = ss["front_orientation"].values, ss["rear_orientation"].values
            data.front_orientation = fo
            data.rear_orientation = ro
            if self.p.draw_Nsegs is not None:
                conf.Nsegs = self.p.draw_Nsegs
                if conf.Nsegs == 2:
                    data.seg_orientations = np.vstack([fo, ro]).T
                    l1, l2 = conf.length / 2, conf.length / 2
                    p1 = xy + util.rotationMatrix(-fo).T @ (l1 / 2, 0)
                    p2 = xy - util.rotationMatrix(-ro).T @ (l2 / 2, 0)
                    data.midline = np.hstack([p1, p2]).reshape([-1, 2, 2])
                elif conf.Nsegs == c.Npoints - 1:
                    data.seg_orientations = seg_orientD[id]
                    data.midline = midlineD[id]
                else:
                    raise
            else:
                data.contour = contourD[id]
                data.midline = midlineD[id]
            conf.data = data
            confs.append(conf)
        self.place_agents(confs)

    def step(self) -> None:
        """Defines the models' events per simulation step."""
        self.agents.step()

    def end(self) -> None:
        self.screen_manager.finalize()
