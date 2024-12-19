"""
Methods for dataset evaluation/comparison
"""

from typing import List

import numpy as np
import pandas as pd
import param
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .. import reg, util
from ..param import EndpointDataFrame, NestedConf, StepDataFrame
from ..util import AttrDict, SuperList
from .dataset import LarvaDataset

__all__ = [
    "eval_end_fast",
    "eval_distro_fast",
    "eval_fast",
    "RSS",
    "RSS_dic",
    "eval_RSS",
    "col_df",
    "get_target_data",
    "arrange_evaluation",
    "Evaluation",
    "DataEvaluation",
]


def eval_end_fast(ee: EndpointDataFrame, e_data, e_sym, mode="pooled") -> dict:
    """Fast evaluation of endpoint data"""
    E = {}
    for p, sym in e_sym.items():
        e_vs = e_data[p]
        E[sym] = None
        if p in ee.columns:
            if mode == "1:1":
                E[sym] = ((e_vs - ee[p]) ** 2).mean() ** 0.5
            elif mode == "pooled":
                E[sym] = ks_2samp(e_vs.values, ee[p].values)[0]
    return E


def eval_distro_fast(
    ss: StepDataFrame, s_data, s_sym, mode="pooled", min_size=10
) -> dict:
    """Fast evaluation of step data"""
    if mode == "1:1":
        E = {}
        for p, sym in s_sym.items():
            if p in ss.columns:
                pps = []
                for id in s_data.index:
                    sp, ssp = (
                        s_data[p].loc[id].values,
                        ss[p].xs(id, level="AgentID").dropna().values,
                    )
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        pps.append(ks_2samp(sp, ssp)[0])

                E[sym] = np.median(pps)
    elif mode == "pooled":
        E = {}
        for p, sym in s_sym.items():
            if p in ss.columns:
                spp, sspp = s_data[p].values, ss[p].dropna().values
                if spp.shape[0] > min_size and sspp.shape[0] > min_size:
                    E[sym] = ks_2samp(spp, sspp)[0]
    elif mode == "1:pooled":
        ids = ss.index.unique("AgentID").values
        E = {id: {} for id in ids}
        for id in ids:
            sss = ss.xs(id, level="AgentID")
            for p, sym in s_sym.items():
                if p in ss.columns:
                    sp, ssp = s_data[p].dropna().values, sss[p].dropna().values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        E[id][sym] = ks_2samp(sp, ssp)[0]
    else:
        raise

    return E


def eval_fast(
    datasets: List[LarvaDataset],
    data: AttrDict,
    symbols: AttrDict,
    mode="pooled",
    min_size=20,
) -> AttrDict:
    """Fast evaluation of datasets"""
    GEend = {
        d.id: eval_end_fast(d.e, data.end, symbols.end, mode=mode) for d in datasets
    }
    GEdistro = {
        d.id: eval_distro_fast(
            d.s, data.step, symbols.step, mode=mode, min_size=min_size
        )
        for d in datasets
    }
    # if mode == '1:1':
    #     labels = ['RSS error', r'median 1:1 distribution KS$_{D}$']
    # elif mode == 'pooled':
    #     labels = ['pooled endpoint KS$_{D}$', 'pooled distribution KS$_{D}$']
    # elif mode == '1:pooled':
    #     labels = ['individual endpoint KS$_{D}$', 'individual distribution KS$_{D}$']
    E = AttrDict(
        {
            "end": pd.DataFrame.from_records(GEend).T,
            "step": pd.DataFrame.from_records(GEdistro).T,
        }
    )
    E.end.index.name = "metric"
    E.step.index.name = "metric"
    return E


def RSS(vs0: np.array, vs: np.array) -> float:
    """Root sum of squares"""
    er = vs - vs0
    r = np.abs(np.max(vs0) - np.min(vs0))
    ee = (er / r) ** 2
    MSE = np.mean(np.sum(ee))
    return np.round(np.sqrt(MSE), 2)


def RSS_dic(dd: LarvaDataset, d: LarvaDataset) -> float:
    """Calculate RSS for a dictionary of curves"""
    f = d.pooled_cycle_curves
    ff = dd.pooled_cycle_curves

    def RSS0(ff, f, sh, mode):
        vs0 = np.array(f[sh][mode])
        vs = np.array(ff[sh][mode])
        return RSS(vs0, vs)

    def RSS1(ff, f, sh):
        dic = {}
        for mode in f[sh]:
            dic[mode] = RSS0(ff, f, sh, mode)
        return dic

    dic = {}
    for sh in f:
        dic[sh] = RSS1(ff, f, sh)

    stat = np.round(np.mean([dic[sh]["norm"] for sh in f if sh != "sv"]), 2)
    dd.pooled_cycle_curves_errors = AttrDict({"dict": dic, "stat": stat})
    return stat


def eval_RSS(rss, rss_target, rss_sym, mode="1:pooled") -> dict:
    """Evaluate RSS for a dictionary"""
    assert mode == "1:pooled"
    RSS_dic = {}
    for id, rrss in rss.items():
        RSS_dic[id] = {}
        for p, sym in rss_sym.items():
            if p in rrss:
                RSS_dic[id][sym] = RSS(rrss[p], rss_target[p])
    return RSS_dic


def col_df(shorts, groups):
    """Create a dataframe for coloring evaluation metrics"""
    import matplotlib as plt

    group_col_dic = {
        "angular kinematics": "Blues",
        "spatial displacement": "Greens",
        "temporal dynamics": "Reds",
        "dispersal": "Purples",
        "tortuosity": "Purples",
        "epochs": "Oranges",
        "stride cycle": "Oranges",
    }
    group_label_dic = {
        "angular kinematics": r"$\bf{angular}$ $\bf{kinematics}$",
        "spatial displacement": r"$\bf{spatial}$ $\bf{displacement}$",
        "temporal dynamics": r"$\bf{temporal}$ $\bf{dynamics}$",
        "dispersal": r"$\bf{dispersal}$",
        "tortuosity": r"$\bf{tortuosity}$",
        "epochs": r"$\bf{epochs}$",
        "stride cycle": r"$\bf{stride}$ $\bf{cycle}$",
    }
    df = pd.DataFrame(
        {
            "group": groups,
            "group_label": [group_label_dic[g] for g in groups],
            "shorts": shorts,
            "pars": [reg.getPar(sh) for sh in shorts],
            "symbols": [reg.getPar(sh, to_return="l") for sh in shorts],
            "group_color": [group_col_dic[g] for g in groups],
        }
    )

    if groups != [] and shorts != []:
        df["cols"] = df.apply(
            lambda row: [(row["group"], p) for p in row["symbols"]], axis=1
        )
        df["par_colors"] = df.apply(
            lambda row: [
                plt.colormaps[row["group_color"]](i)
                for i in np.linspace(0.4, 0.7, len(row["pars"]))
            ],
            axis=1,
        )
    else:
        df["cols"] = []
        df["par_colors"] = []
    df.set_index("group", inplace=True)
    return df


def cycle_curve_dict(s, dt, shs=["sv", "fov", "rov", "foa", "b"]) -> AttrDict:
    """Create a dictionary of cycle curves"""
    strides = util.detect_strides(s[reg.getPar("sv")], dt)
    da = np.array(
        [
            np.trapz(s[reg.getPar("fov")][s0:s1].dropna())
            for ii, (s0, s1) in enumerate(strides)
        ]
    )
    dic = {sh: util.mean_stride_curve(s[reg.getPar(sh)], strides, da) for sh in shs}
    return AttrDict(dic)


def cycle_curve_dict_multi(
    s: StepDataFrame, dt, shs=["sv", "fov", "rov", "foa", "b"]
) -> AttrDict:
    """Create a dictionary of cycle curves for multiple agents"""
    return AttrDict(
        {
            id: cycle_curve_dict(s.xs(id, level="AgentID"), dt=dt, shs=shs)
            for id in s.index.unique("AgentID").values
        }
    )


def get_target_data(d: LarvaDataset, eval_metrics):
    """Get target data for evaluation"""
    s, e, c = d.data
    all_ks = SuperList(eval_metrics.values()).flatten.unique
    all_ps = SuperList(reg.getPar(all_ks[:]))
    Eps = all_ps.existing(e)
    Dps = all_ps.existing(s)
    Dps = Dps.nonexisting(Eps)
    return AttrDict(
        {"step": {p: s[p].dropna() for p in Dps}, "end": {p: e[p] for p in Eps}}
    )


def arrange_evaluation(data, eval_metrics):
    """Arrange evaluation data"""
    Eks = reg.getPar(p=list(data.end.keys()), to_return="k")
    Dks = reg.getPar(p=list(data.step.keys()), to_return="k")
    dic = AttrDict(
        {"end": {"shorts": [], "groups": []}, "step": {"shorts": [], "groups": []}}
    )
    for g, shs in eval_metrics.items():
        Eshorts, Dshorts = util.existing_cols(shs, Eks), util.existing_cols(shs, Dks)

        if len(Eshorts) > 0:
            dic.end.shorts.append(Eshorts)
            dic.end.groups.append(g)
        if len(Dshorts) > 0:
            dic.step.shorts.append(Dshorts)
            dic.step.groups.append(g)
    return AttrDict({k: col_df(**v) for k, v in dic.items()})


class Evaluation(NestedConf):
    """Evaluation class"""

    refID = reg.conf.Ref.confID_selector()
    refDir = param.String(
        default=None,
        label="reference directory",
        doc="The directory containing the reference dataset",
    )
    eval_metrics = param.Dict(
        default=AttrDict(
            {
                "angular kinematics": [
                    "run_fov_mu",
                    "pau_fov_mu",
                    "b",
                    "fov",
                    "foa",
                    "rov",
                    "roa",
                    "tur_fou",
                ],
                "spatial displacement": [
                    "cum_d",
                    "run_d",
                    "str_c_l",
                    "v_mu",
                    "pau_v_mu",
                    "run_v_mu",
                    "v",
                    "a",
                    "dsp_0_40_max",
                ],
                "temporal dynamics": [
                    "fsv",
                    "ffov",
                    "run_t",
                    "pau_t",
                    "run_tr",
                    "pau_tr",
                ],
                "stride cycle": [
                    "str_d_mu",
                    "str_d_std",
                    "str_sv_mu",
                    "str_fov_mu",
                    "str_fov_std",
                    "str_N",
                    "str_t",
                    "str_d",
                    "str_sd",
                ],
                # 'epochs': ['run_t', 'pau_t'],
                "tortuosity": ["tor5", "tor20", "tor5_mu", "tor20_mu"],
            }
        ),
        doc="Evaluation metrics to use",
    )
    cycle_curve_metrics = param.List(
        default=["sv", "fov", "rov", "foa", "b"], doc="Stride-cycle metrics to evaluate"
    )

    def __init__(self, dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.target = reg.conf.Ref.retrieve_dataset(
            dataset=dataset,
            id=self.refID,
            dir=self.refDir,
            load=True,
            h5_ks=["epochs", "base_spatial", "angular", "dspNtor"],
        )
        self.build(d=self.target)

    def build(self, d: LarvaDataset) -> None:
        self.target_data = get_target_data(d, eval_metrics=self.eval_metrics)
        self.evaluation = arrange_evaluation(
            data=self.target_data, eval_metrics=self.eval_metrics
        )
        self.eval_symbols = AttrDict(
            {
                "step": dict(zip(self.s_pars, self.s_symbols)),
                "end": dict(zip(self.e_pars, self.e_symbols)),
            }
        )

        if len(self.cycle_curve_metrics) > 0 and d.pooled_cycle_curves is not None:
            cycle_dict = {
                "sv": "abs",
                "fov": "norm",
                "rov": "norm",
                "foa": "norm",
                "b": "norm",
            }
            self.cycle_modes = {sh: cycle_dict[sh] for sh in self.cycle_curve_metrics}
            self.cycle_curve_target = AttrDict(
                {
                    sh: np.array(d.pooled_cycle_curves[sh][mod])
                    for sh, mod in self.cycle_modes.items()
                }
            )
            self.rss_sym = {sh: sh for sh in self.cycle_curve_metrics}

    @property
    def s_pars(self):
        return SuperList(self.evaluation.step.pars).flatten

    @property
    def e_pars(self):
        return SuperList(self.evaluation.end.pars).flatten

    @property
    def s_symbols(self):
        return SuperList(self.evaluation.step.symbols).flatten

    @property
    def e_symbols(self):
        return SuperList(self.evaluation.end.symbols).flatten

    @property
    def func_eval_metric_solo(self):
        def func(ss):
            return AttrDict(
                {
                    "KS": {
                        sym: ks_2samp(
                            self.target_data.step[p].values, ss[p].dropna().values
                        )[0]
                        for p, sym in self.eval_symbols.step.items()
                    }
                }
            )

        return func

    @property
    def func_eval_metric_multi(self):
        def gfunc(s):
            return AttrDict(
                {
                    "KS": eval_distro_fast(
                        s,
                        self.target_data.step,
                        self.eval_symbols.step,
                        mode="1:pooled",
                        min_size=10,
                    )
                }
            )

        return gfunc

    @property
    def func_cycle_curve_solo(self):
        def func(ss):
            c0 = cycle_curve_dict(
                s=ss, dt=self.target.config.dt, shs=self.cycle_curve_metrics
            )
            eval_curves = AttrDict(
                {sh: c0[sh][mode] for sh, mode in self.cycle_modes.items()}
            )
            return AttrDict(
                {
                    "RSS": {
                        sh: RSS(ref_curve, eval_curves[sh])
                        for sh, ref_curve in self.cycle_curve_target.items()
                    }
                }
            )

        return func

    @property
    def func_cycle_curve_multi(self):
        def gfunc(s):
            rss0 = cycle_curve_dict_multi(
                s=s, dt=self.target.config.dt, shs=self.cycle_curve_metrics
            )
            rss = AttrDict(
                {
                    id: {sh: dic[sh][mod] for sh, mod in self.cycle_modes.items()}
                    for id, dic in rss0.items()
                }
            )
            return AttrDict(
                {
                    "RSS": eval_RSS(
                        rss, self.cycle_curve_target, self.rss_sym, mode="1:pooled"
                    )
                }
            )

        return gfunc

    @property
    def fit_func_multi(self):
        def fit_func(s):
            fit_dicts = AttrDict()
            if len(self.cycle_curve_metrics) > 0:
                fit_dicts.update(self.func_cycle_curve_multi(s))
            if len(self.eval_metrics) > 0:
                fit_dicts.update(self.func_eval_metric_multi(s))
            return fit_dicts

        return fit_func

    @property
    def fit_func_solo(self):
        def fit_func(ss):
            fit_dicts = AttrDict()
            if len(self.cycle_curve_metrics) > 0:
                fit_dicts.update(self.func_cycle_curve_solo(ss))
            if len(self.eval_metrics) > 0:
                fit_dicts.update(self.func_eval_metric_solo(ss))
            return fit_dicts

        return fit_func

    def eval_datasets(self, ds, mode, min_size=20):
        return eval_fast(
            datasets=ds,
            data=self.target_data,
            symbols=self.eval_symbols,
            mode=mode,
            min_size=min_size,
        )


class DataEvaluation(Evaluation):
    """Data evaluation class"""

    norm_modes = param.ListSelector(
        default=["raw", "minmax"],
        objects=["raw", "minmax", "std"],
        doc="Normalization modes to use",
    )
    eval_modes = param.ListSelector(
        default=["pooled"],
        objects=["pooled", "1:1", "1:pooled"],
        doc="Evaluation modes to use",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.error_dicts = AttrDict()

    def norm_error_dict(self, error_dict, mode="raw"):
        if mode == "raw":
            return error_dict
        elif mode == "minmax":
            return AttrDict(
                {
                    k: pd.DataFrame(
                        MinMaxScaler().fit(df).transform(df),
                        index=df.index,
                        columns=df.columns,
                    )
                    for k, df in error_dict.items()
                }
            )
        elif mode == "std":
            return AttrDict(
                {
                    k: pd.DataFrame(
                        StandardScaler().fit(df).transform(df),
                        index=df.index,
                        columns=df.columns,
                    )
                    for k, df in error_dict.items()
                }
            )
