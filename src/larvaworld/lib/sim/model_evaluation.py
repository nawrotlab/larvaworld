from __future__ import annotations

from typing import Any
import os
import warnings

from ... import vprint
from ..param import class_generator
from ..process import DataEvaluation

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from .. import reg, util
from ..reg import SimConfiguration
from ..reg import LarvaGroupMutator
from ..util import AttrDict

__all__: list[str] = [
    "EvalRun",
    "evalNplot",
    "modelConf_analysis",
]


class EvalConf(LarvaGroupMutator, DataEvaluation):
    def __init__(self, dataset=None, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.target.id = "experiment"
        self.target.config.id = "experiment"
        self.target.color = "grey"
        self.target.config.color = "grey"


# TODO this should be a subclass of SimConfigurationParams and not of SimConfiguration.
# This should be adjusted in order to remove also the need for the LarvaGroupMutator parent class of EvalConf
# (note that the args N,modelIDS, groupIDs are common in LarvaGroupMutator and SimConfigurationParams)
class EvalRun(EvalConf, SimConfiguration):
    def __init__(
        self, enrichment: bool = True, screen_kws: dict[str, Any] = {}, **kwargs: Any
    ) -> None:
        """Model evaluation mode. This mode is used to evaluate a number of larva models
        for similarity with a preexisting reference dataset, most often one retained
        via monitoring real experiments. The evaluation is done by comparing the
        trajectories of the models with the reference dataset, via several evaluation metrics.

        Args:
            enrichment (bool, optional): Whether to enrich the simulated datasets by deriving secondary pareameters. Defaults to True.
            screen_kws (dict, optional): The screen visualization parameters. Defaults to {}.
        """

        EvalConf.__init__(self, runtype="Eval", **kwargs)
        kwargs["dt"] = self.target.config.dt
        if "duration" not in kwargs:
            kwargs["duration"] = self.target.config.Nticks * kwargs["dt"] / 60
        SimConfiguration.__init__(self, runtype="Eval", **kwargs)

        # TODO This should be removed
        if self.groupIDs is None:
            self.groupIDs = self.modelIDs

        self.screen_kws = screen_kws
        self.enrichment = enrichment
        self.figs = AttrDict(
            {
                "errors": {},
                "hist": {},
                "boxplot": {},
                "stride_cycle": {},
                "loco": {},
                "epochs": {},
                "models": {"table": {}, "summary": {}},
            }
        )
        self.error_plot_dir = f"{self.plot_dir}/errors"

    def simulate(self) -> Any:
        kws = {
            "dt": self.dt,
            "duration": self.duration,
        }

        Nm = len(self.modelIDs)
        if self.offline is None:
            from .agent_simulations import sim_models

            vprint(
                f"Simulating offline {Nm} models : {self.groupIDs} with {self.N} larvae each",
                2,
            )
            temp = self.s_pars + self.e_pars
            tor_durs = np.unique(
                [
                    int(ii[len("tortuosity") + 1 :])
                    for ii in temp
                    if ii.startswith("tortuosity")
                ]
            )
            dsp = reg.getPar("dsp")
            dsp_temp = [
                ii[len(dsp) + 1 :].split("_") for ii in temp if ii.startswith(f"{dsp}_")
            ]
            dsp_starts = np.unique([int(ii[0]) for ii in dsp_temp]).tolist()
            dsp_stops = np.unique([int(ii[1]) for ii in dsp_temp]).tolist()
            c = self.target.config
            lgs = c.larva_group.new_groups(
                Ns=self.N,
                modelIDs=self.modelIDs,
                groupIDs=self.groupIDs,
                sample=self.refID,
            )
            self.datasets = sim_models(
                modelIDs=self.modelIDs,
                tor_durs=tor_durs,
                dsp_starts=dsp_starts,
                dsp_stops=dsp_stops,
                groupIDs=self.groupIDs,
                lgs=lgs,
                enrichment=self.enrichment,
                Nids=self.N,
                env_params=c.env_params,
                refDataset=self.target,
                data_dir=self.data_dir,
                **kws,
            )
        else:
            from .single_run import ExpRun

            vprint(
                f"Simulating {Nm} models : {self.groupIDs} with {self.N} larvae each", 2
            )
            kws0 = AttrDict(
                {
                    "dir": self.dir,
                    "store_data": self.store_data,
                    "experiment": self.experiment,
                    "id": self.id,
                    "offline": self.offline,
                    "modelIDs": self.modelIDs,
                    "groupIDs": self.groupIDs,
                    "N": self.N,
                    "sample": self.refID,
                    # 'parameters': conf,
                    "screen_kws": self.screen_kws,
                    **kws,
                }
            )
            run = ExpRun(**kws0)
            self.datasets = run.simulate()
        self.analyze()
        if self.store_data:
            self.store()
        return self.datasets

    def get_error_plots(self, error_dict: Any, mode: str = "pooled") -> AttrDict:
        GD = reg.graphs.dict
        label_dic = {
            "1:1": {"end": "RSS error", "step": r"median 1:1 distribution KS$_{D}$"},
            "pooled": {
                "end": "Pooled endpoint values KS$_{D}$",
                "step": "Pooled distributions KS$_{D}$",
            },
        }
        labels = label_dic[mode]
        dic = AttrDict()
        for norm in self.norm_modes:
            d = self.norm_error_dict(error_dict, mode=norm)
            df0 = pd.DataFrame.from_dict(
                {k: df.mean(axis=1) for i, (k, df) in enumerate(d.items())}
            )
            kws = {"save_to": f"{self.error_plot_dir}/{norm}"}
            bars = {}
            tabs = {}
            for k, df in d.items():
                tabs[k] = GD["error table"](data=df, k=k, title=labels[k], **kws)
            tabs["mean"] = GD["error table"](
                data=df0, k="mean", title="average error", **kws
            )
            bars["full"] = GD["error barplot"](
                error_dict=d, evaluation=self.evaluation, labels=labels, **kws
            )
            # Summary figure with barplots and tables for both endpoint and timeseries metrics
            bars["summary"] = GD["error summary"](
                norm_mode=norm,
                eval_mode=mode,
                error_dict=d,
                evaluation=self.evaluation,
                **kws,
            )
            dic[norm] = {"tables": tabs, "barplots": bars}
        return AttrDict(dic)

    def analyze(self, **kwargs: Any) -> None:
        vprint("Evaluating all models", 1)
        os.makedirs(self.plot_dir, exist_ok=True)

        for m in self.eval_modes:
            self.error_dicts[m] = self.eval_datasets(self.datasets, mode=m, **kwargs)
            self.figs.errors[m] = self.get_error_plots(self.error_dicts[m], m)

    def store(self) -> None:
        if self.data_dir is not None:
            util.save_dict(self.error_dicts, f"{self.data_dir}/error_dicts.txt")
            vprint(f"Results saved at {self.data_dir}", 1)

    def plot_models(self, **kwargs: Any) -> None:
        GD = reg.graphs.dict
        save_to = self.plot_dir
        for mID in self.modelIDs:
            self.figs.models.table[mID] = GD["model table"](
                mID=mID, save_to=save_to, figsize=(14, 11), **kwargs
            )
            self.figs.models.summary[mID] = GD["model summary"](
                mID=mID, save_to=save_to, refID=self.refID, **kwargs
            )

    @property
    def existing_dispersion_ranges(self) -> Any:
        ds = [self.target] + self.datasets
        return util.SuperList([d.existing_dispersion_ranges for d in ds]).flatten.unique

    def plot_results(
        self,
        plots: list[str] = [
            "hists",
            "trajectories",
            "dispersion",
            "bouts",
            "fft",
            "boxplots",
        ],
        **kwargs: Any,
    ) -> None:
        GD = reg.graphs.dict

        self.target.load(h5_ks=["epochs", "angular", "dspNtor"])
        kws = {
            "datasets": [self.target] + self.datasets,
            "save_to": self.plot_dir,
            **kwargs,
        }
        kws1 = {"subfolder": None, **kws}

        kws2 = {
            "target": self.target,
            "datasets": self.datasets,
            "save_to": self.plot_dir,
            **kwargs,
        }
        self.figs.summary = GD["eval summary"](**kws2)
        self.figs.stride_cycle.norm = GD["stride cycle"](
            shorts=["sv", "fov", "rov", "foa", "b"], individuals=True, **kws
        )
        if "dispersion" in plots:
            for r0, r1 in self.existing_dispersion_ranges:
                # for r0, r1 in itertools.product(self.dsp_starts, self.dsp_stops):
                self.figs.loco[f"dsp_{r0}_{r1}"] = AttrDict(
                    {
                        "plot": GD["dispersal"](range=(r0, r1), **kws1),
                        "traj": GD["trajectories"](
                            name=f"traj_{r0}_{r1}",
                            range=(r0, r1),
                            mode="origin",
                            **kws1,
                        ),
                        "summary": GD["dispersal summary"](range=(r0, r1), **kws2),
                    }
                )
        if "bouts" in plots:
            self.figs.epochs.turn = GD["epochs"](turns=True, **kws)
            self.figs.epochs.runNpause = GD["epochs"](stridechain_duration=True, **kws)
        if "fft" in plots:
            self.figs.loco.fft = GD["freq powerspectrum"](**kws)
        if "hists" in plots:
            self.figs.hist.ang = GD["angular pars"](
                half_circles=False,
                absolute=False,
                Nbins=100,
                Npars=3,
                include_rear=False,
                **kws1,
            )
            self.figs.hist.crawl = GD["crawl pars"](pvalues=False, **kws1)
        if "trajectories" in plots:
            self.figs.loco.trajectories = GD["trajectories"](**kws1)
        if "boxplots" in plots:
            pass
            # self.figs.boxplot.end = self.plot_data(mode='end', type='box')
            # self.figs.boxplot.step = self.plot_data(mode='step', type='box')


# TODO The class generator is not working properly. Arguments changed after initialization are not being updated. In this case groupIDs.
reg.gen.Eval = class_generator(EvalConf)
# reg.gen.Eval = EvalConf


def evalNplot(show: bool = True, **kwargs: Any) -> EvalRun:
    E = EvalRun(**kwargs)
    E.simulate()
    E.plot_models(show=show)
    E.plot_results(show=show)
    return E


def adapt_mID(d, mID0, mID, ks):
    from ..model import moduleDB

    vprint(f"Adapting {mID0} on {d.refID} as {mID}, fitting {ks} modules", 1)
    ps = ["body.length"]

    m0 = reg.conf.Model.getID(mID0)
    if "intermitter" in ks:
        m0 = d.config.get_sample_bout_distros(m0.get_copy())

    ps = moduleDB.modules_pars(mIDs=ks, conf=m0, as_entry=True)
    m0 = m0.update_nestdict(
        AttrDict(
            {p: np.median(vs) for p, vs in d.sample_larvagroup(N=100, ps=ps).items()}
        )
    )
    reg.conf.Model.setID(mID, m0)


def modelConf_analysis(d: Any) -> None:
    from collections import ChainMap

    from ..model.modules.module_modes import moduleDB as MD
    from .genetic_algorithm import GAevaluation, optimize_mID

    warnings.filterwarnings("ignore")

    fit_kws = {
        "eval_metrics": {
            "angular kinematics": ["b", "fov", "foa"],
            "spatial displacement": [
                "v_mu",
                "pau_v_mu",
                "run_v_mu",
                "v",
                "a",
                "dsp_0_40_max",
                "dsp_0_60_max",
            ],
            "temporal dynamics": ["fsv", "ffov", "run_tr", "pau_tr"],
        },
        "cycle_curve_metrics": ["fov", "foa", "b"],
    }

    def comp_D():
        kws2 = {
            "evaluator": GAevaluation(dataset=d, refID=d.config.refID, **fit_kws),
            "dir": f"{d.config.dir}/model/optimization",
            "dt": d.config.dt,
        }

        def fit_3modules():
            ee = []
            ks0 = ["crawler", "turner", "interference"]
            ks = MD.ids.nonexisting(ks0)
            for c, t, f in MD.mod_combs(ks0, short=True):
                if c != "CON":
                    mID0 = f"{c}_{t}_{f}_DEF"
                    mID = f"{mID0}_fit"
                    print(mID0)
                    adapt_mID(d=d, mID0=mID0, mID=mID, ks=ks)
                    # print(mID0)
                    ee.append(optimize_mID(mID0=mID, ks=ks0, id=mID, **kws2))
            return AttrDict(ChainMap(*ee))

        def fit_average():
            ks0 = ["turner", "interference"]
            ks = MD.ids.nonexisting(ks0)
            ee = [
                adapt_mID(d=d, mID0=f"RE_{t}_{f}_DEF", mID=f"{f}on{t}", ks=ks)
                for t, f in MD.mod_combs(ks0, short=True)
            ]
            return AttrDict(ChainMap(*ee))

        def fit_variable(mIDs_avg):
            A = AttrDict()
            CM = reg.conf.Model
            sample_kws = {
                f"brain.crawler.{k}": "sample"
                for k in [
                    "stride_dst_mean",
                    "stride_dst_std",
                    "max_scaled_vel",
                    "max_vel_phase",
                    "freq",
                ]
            }
            for (mID0,) in mIDs_avg:
                mID = f"{mID0}_var"
                m0 = CM.getID(mID0).get_copy()
                A[mID] = m0.update_existingnestdict(sample_kws)
                CM.setID(mID, A[mID])

            return A

        D = d.config.modelConfs
        D["3modules"] = fit_3modules()
        D.average = fit_average()
        D.variable = fit_variable(list(D.average))
        d.save_config()

    def eval_D():
        D = d.config.modelConfs
        mIDs_avg = list(D.average)
        mIDs_var = list(D.variable.keys())
        Dataevaluator = DataEvaluation(dataset=d, refID=d.config.refID, **fit_kws)
        kws1 = {
            "dir": f"{d.config.dir}/model/evaluation",
            "N": 5,
            "duration": 1,
            "refID": d.config.refID,
        }
        ts = MD.mod_modes("turner", short=True)
        for c, f in MD.mod_combs(["crawler", "intermitter"], short=True):
            mIDs = [f"{c}_{t}_{f}_DEF_fit" for t in ts]
            evalNplot(
                modelIDs=mIDs,
                groupIDs=ts,
                id=f"Tmod_variable_Cmod_{c}_Ifmod_{f}",
                **kws1,
            )
        evalNplot(modelIDs=mIDs_avg, id="6mIDs_avg", **kws1)
        evalNplot(modelIDs=mIDs_var, id="6mIDs_var", **kws1)
        evalNplot(modelIDs=mIDs_avg[:3] + mIDs_var[:3], id="3mIDs_avgVSvar1", **kws1)
        evalNplot(modelIDs=mIDs_avg[3:] + mIDs_var[3:], id="3mIDs_avgVSvar2", **kws1)
        reg.graphs.store_model_graphs(list(D["3modules"]), d.dir)
        reg.graphs.store_model_graphs(list(D.average), d.dir)

    comp_D()
