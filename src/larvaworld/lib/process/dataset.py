"""
Basic classes for larvaworld-format datasets
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import copy
import itertools
import os
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import param
from scipy.signal import find_peaks

from ... import vprint, DATA_DIR
from .. import reg, util
from ..reg import LarvaGroup
from ..param import (
    ClassAttr,
    ClassDict,
    EndpointDataFrame,
    SimMetricOps,
    StepDataFrame,
    BoundedArea,
    RuntimeDataOps,
    SimTimeOps,
    RandomizedColor,
    OptionalPositiveInteger,
    OptionalPositiveNumber,
)
from ..util import AttrDict, SuperList, nam

__all__: list[str] = [
    "DatasetConfig",
    "ParamLarvaDataset",
    "BaseLarvaDataset",
    "LarvaDataset",
    "LarvaDatasetCollection",
]


class DatasetConfig(RuntimeDataOps, SimMetricOps, SimTimeOps):
    """
    The configuration of a LarvaDataset.
    """

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        super().__init__(**kwargs)

    Nticks = OptionalPositiveInteger(default=None)
    refID = param.String(None, doc="The unique ID of the reference dataset")
    group_id = param.String(None, doc="The unique ID of the group")
    color = RandomizedColor(
        default="black", doc="The color of the dataset", instantiate=True
    )
    env_params = ClassAttr(reg.gen.Env, doc="The environment configuration")
    larva_group = ClassAttr(LarvaGroup, doc="The larva group object")
    agent_ids = param.List(doc="The unique IDs of the agents in the dataset")
    N = OptionalPositiveInteger(softmax=500, doc="The number of agents in the group")
    sample = reg.conf.Ref.confID_selector()
    filtered_at = OptionalPositiveNumber(
        doc="Whether data has been low-pass filtered at a certain cut-off frequency during preprocessing"
    )
    rescaled_by = OptionalPositiveNumber(
        doc="Whether data has been rescaled by a certain value during preprocessing"
    )
    pooled_cycle_curves = param.Dict(
        doc="The average across-larvae curves of diverse parameters during the stridecycle"
    )
    bout_distros = param.Dict(
        doc="The temporal distributions of the diverse types of behavioral bouts"
    )
    intermitter = param.Dict(doc="The fitted parameters for the intermittency module")
    modelConfs = param.Dict(
        default=AttrDict(
            {"average": {}, "variable": {}, "individual": {}, "3modules": {}}
        ),
        doc="The fitted model configurations",
    )
    EEB_poly1d = param.Parameter(
        doc="The polynomial describing the exploration-exploitation balance."
    )

    @property
    def h5_kdic(self):
        """Returns the keys of the h5 file that store the parameters of the dataset"""

        def epochs_ps():
            cs = [
                "turn",
                "Lturn",
                "Rturn",
                "pause",
                "exec",
                "stride",
                "stridechain",
                "run",
            ]
            pars = [
                "id",
                "start",
                "stop",
                "dur",
                "dst",
                nam.scal("dst"),
                "length",
                nam.max("vel"),
                "count",
            ]
            return SuperList([nam.chunk_track(c, pars) for c in cs]).flatten

        def dspNtor_ps():
            tor_ps = [
                f"tortuosity_{dur}"
                for dur in [1, 2, 5, 10, 20, 30, 60, 100, 120, 240, 300]
            ]
            dsp_ps = [
                f"dispersion_{t0}_{t1}"
                for (t0, t1) in itertools.product(
                    [0, 5, 10, 20, 30, 60], [30, 40, 60, 90, 120, 240, 300]
                )
            ]
            pars = SuperList(tor_ps + dsp_ps + nam.scal(dsp_ps))
            return pars

        def base_spatial_ps(p=""):
            d, v, a = ps = [nam.dst(p), nam.vel(p), nam.acc(p)]
            ld, lv, la = lps = nam.lin(ps)
            ps0 = nam.xy(p) + ps + lps + nam.cum([d, ld])
            return SuperList(ps0 + nam.scal(ps0))

        def ang_pars(angs):
            avels = nam.vel(angs)
            aaccs = nam.acc(angs)
            uangs = nam.unwrap(angs)
            avels_min, avels_max = nam.min(avels), nam.max(avels)
            return SuperList(avels + aaccs + uangs + avels_min + avels_max)

        def angular(N):
            Nangles = np.clip(N - 2, a_min=0, a_max=None)
            Nsegs = np.clip(N - 1, a_min=0, a_max=None)
            ors = nam.orient(
                ["front", "rear", "head", "tail"] + nam.midline(Nsegs, type="seg")
            )
            ang = ors + [f"angle{i}" for i in range(Nangles)] + ["bend"]
            return SuperList(ang + ang_pars(ang)).unique

        dic = AttrDict(
            {
                "contour": nam.contour_xy(self.Ncontour, flat=True),
                "midline": nam.midline_xy(self.Npoints, flat=True),
                "epochs": epochs_ps(),
                "base_spatial": base_spatial_ps(self.point),
                "angular": angular(self.Npoints),
                "dspNtor": dspNtor_ps(),
            }
        )
        return dic

    @param.depends("agent_ids", watch=True)
    def update_Nagents(self):
        self.N = len(self.agent_ids)

    @property
    def arena_vertices(self):
        a = self.env_params.arena
        vs = BoundedArea(dims=a.dims, geometry=a.geometry).vertices
        return np.array(vs)

    def get_sample_bout_distros(self, m):
        B = self.bout_distros
        dic = {
            "pause_dist": "pause_dur",
            "stridechain_dist": "run_count",
            "run_dist": "run_dur",
        }
        I = m.brain.intermitter
        if I:
            for k, v in dic.items():
                if k in I and v in B and B[v] is not None:
                    I[k] = B[v]
        return m


class ParamLarvaDataset(param.Parameterized):
    config = ClassAttr(DatasetConfig, doc="The dataset metadata")
    step_data = StepDataFrame(doc="The timeseries data")
    endpoint_data = EndpointDataFrame(doc="The endpoint data")
    config2 = ClassDict(
        default=AttrDict(), item_type=None, doc="Additional dataset metadata"
    )

    def __init__(self, **kwargs: Any) -> None:
        if "config" not in kwargs:
            kws = AttrDict()
            for k in DatasetConfig().param_keys:
                if k in kwargs:
                    kws[k] = kwargs[k]
                    kwargs.pop(k)
            kwargs["config"] = DatasetConfig(**kws)
        assert "config2" not in kwargs

        ks = list(kwargs.keys())
        kws2 = AttrDict()
        for k in ks:
            if k not in self.param.objects(instance=False):
                kws2[k] = kwargs[k]
                kwargs.pop(k)
        kwargs["config2"] = AttrDict(kws2)
        super().__init__(**kwargs)
        self.merge_configs()
        self.epoch_dict = AttrDict({"pause": None, "run": None})
        self.larva_dicts = {}
        self.__dict__.update(self.config.nestedConf)
        self._epoch_dicts = None
        self._chunk_dicts = None
        self._pooled_epochs = None
        self._fitted_epochs = None

        self._cycle_curves = None

    def validate_IDs(self):
        try:
            s1 = self.s.index.unique("AgentID").tolist()
            s2 = self.e.index.values.tolist()
            assert len(s1) == len(s2)
            assert set(s1) == set(s2)
            assert s1 == s2
            self.c.agent_ids = s1
        except:
            pass

    def update_ids_in_data(self):
        self.set_data(
            step=self.s.loc[(slice(None), self.ids), :], end=self.e.loc[self.ids]
        )

    @param.depends("step_data", watch=True)
    def update_Nticks(self):
        self.c.Nticks = self.s.index.unique("Step").size
        self.c.duration = self.c.dt * self.c.Nticks / 60

    @property
    def c(self):
        return self.config

    @property
    def ids(self):
        return self.config.agent_ids

    @property
    def s(self):
        if self.step_data is None:
            self.load()
        return self.step_data

    @property
    def e(self):
        if self.endpoint_data is None:
            self.load(step=False)
        return self.endpoint_data

    @property
    def end_ps(self):
        return SuperList(self.e.columns).sorted

    @property
    def step_ps(self):
        return SuperList(self.s.columns).sorted

    @property
    def end_ks(self):
        return SuperList(reg.getPar(d=self.end_ps, to_return="k")).sorted

    @property
    def step_ks(self):
        return SuperList(reg.getPar(d=self.step_ps, to_return="k")).sorted

    @property
    def min_tick(self):
        return self.s.index.unique("Step").min()

    def timeseries_slice(self, time_range=None, df=None):
        if df is None:
            df = self.s
        if time_range is None:
            return df
        else:
            t0, t1 = time_range
            s0 = int(t0 / self.c.dt)
            s1 = int(t1 / self.c.dt)
            df_slice = df.loc[(slice(s0, s1), slice(None)), :]
            return df_slice

    def required(**pars):
        def wrap(f):
            def wrapped_f(self, *args, **kwargs):
                if self.data_exists(**pars):
                    f(self, *args, **kwargs)

            return wrapped_f

        return wrap

    def valid(required=None, returned=None):
        _verbose = -3

        def wrap(f):
            def wrapped_f(self, *args, **kwargs):
                vprint("_______________________________", _verbose)
                vprint(f"Checking method {f.__name__}", _verbose)
                if required is not None:
                    if self.data_exists(**required):
                        vprint("   Required columns exist. Proceeding ...", _verbose)
                    else:
                        vprint("   Required columns not found. Aborting...", _verbose)
                        return wrapped_f
                if returned is not None:
                    if not self.data_exists(**returned):
                        vprint(
                            "   Columns to be returned do not exist. Executing ...",
                            _verbose,
                        )

                    else:
                        if kwargs.get("recompute"):
                            vprint("   Forced to recompute. Executing...", _verbose)
                            f(self, *args, **kwargs)
                        else:
                            vprint(
                                "   Columns to be returned exist and not forced to recompute. Aborting...",
                                _verbose,
                            )
                            return wrapped_f

                f(self, *args, **kwargs)

            return wrapped_f

        return wrap

    def data_exists(self, ks=[], ps=[], eks=[], eps=[], config_attrs=[], attrs=[]):
        if not all([hasattr(self, attr) for attr in attrs]):
            return False
        spars = SuperList(
            ps + reg.getPar(ks) + [getattr(self.c, attr) for attr in config_attrs]
        ).flatten.unique
        if not spars.exist_in(self.s):
            return False
        epars = SuperList(eps + reg.getPar(eks))
        return epars.exist_in(self.s)

    @property
    def chunk_dicts(self):
        try:
            assert self._chunk_dicts is not None
        except AssertionError:
            self._chunk_dicts = AttrDict(self.read("chunk_dicts"))
        except KeyError:
            self.detect_bouts()
        finally:
            return self._chunk_dicts

    @chunk_dicts.setter
    def chunk_dicts(self, d):
        self._chunk_dicts = d
        self.store(d, "chunk_dicts")
        vprint("Chunk dictionaries stored.", 1)

    @property
    def epoch_dicts(self):
        try:
            assert self._epoch_dicts is not None
        except AssertionError:
            self._epoch_dicts = AttrDict(self.read("epoch_dicts"))
        except KeyError:
            self.comp_pooled_epochs()
        finally:
            return self._epoch_dicts

    @epoch_dicts.setter
    def epoch_dicts(self, d):
        self._epoch_dicts = d
        self.store(d, "epoch_dicts")

    @property
    def fitted_epochs(self):
        try:
            assert self._fitted_epochs is not None
        except AssertionError:
            self._fitted_epochs = AttrDict(self.read("fitted_epochs"))
        except KeyError:
            self.fit_pooled_epochs()
        finally:
            return self._fitted_epochs

    @fitted_epochs.setter
    def fitted_epochs(self, d):
        self._fitted_epochs = d
        self.store(d, "fitted_epochs")

    @property
    def pooled_epochs(self):
        try:
            assert self._pooled_epochs is not None
        except AssertionError:
            self._pooled_epochs = util.load_dict(f"{self.c.data_dir}/pooled_epochs.txt")
        except KeyError:
            self.comp_pooled_epochs()

        finally:
            return self._pooled_epochs

    @pooled_epochs.setter
    def pooled_epochs(self, d):
        self._pooled_epochs = d
        self.save_dict(d, "pooled_epochs.txt")

    @property
    def cycle_curves(self):
        try:
            assert self._cycle_curves is not None
        except AssertionError:
            self._cycle_curves = AttrDict(self.read("cycle_curves"))
        except KeyError:
            self.comp_interference()
        finally:
            return self._cycle_curves

    @cycle_curves.setter
    def cycle_curves(self, d):
        self._cycle_curves = d
        self.store(d, "cycle_curves")

    @property
    def pooled_cycle_curves(self):
        try:
            assert self.c.pooled_cycle_curves is not None
        except AssertionError:
            self.comp_pooled_cycle_curves()
        finally:
            return self.c.pooled_cycle_curves

    @pooled_cycle_curves.setter
    def pooled_cycle_curves(self, d):
        self.c.pooled_cycle_curves = d

    def track_par_in_chunk(self, chunk, par):
        A = self.empty_df(dim3=3)
        for i, id in enumerate(self.ids):
            E = self.epoch_dicts[chunk][id]
            if E.shape[0] > 0:
                S = self.s[par].xs(id, level="AgentID")
                t0s, t1s = E[:, 0], E[:, 1]
                b0s = S.loc[t0s].values
                b1s = S.loc[t1s].values
                A[t0s, i, 0] = b0s
                A[t1s, i, 1] = b1s
                A[t1s, i, 2] = b1s - b0s
        self.s[nam.atStartStopChunk(par, chunk)] = A.reshape([-1, 3])

    def epochs_pose_by_ID(self, chunk, id):
        E = self.epoch_dicts[chunk][id]
        if E.shape[0] > 0:
            S = self.s[self.c.traj_xy + [nam.unwrap(nam.orient("front"))]].xs(
                id, level="AgentID"
            )
            return S.loc[E[:, 0]].values, S.loc[E[:, 1]].values
        else:
            return np.array([[], [], []]), np.array([[], [], []])

    def epochs_bearing_by_ID(self, chunk, id, loc=(0.0, 0.0)):
        b0s, b1s = self.epochs_pose_by_ID(chunk, id)
        p0 = np.array([util.comp_bearing_solo(x, y, o, loc=loc) for x, y, o in b0s])
        p1 = np.array([util.comp_bearing_solo(x, y, o, loc=loc) for x, y, o in b1s])
        return p0, p1

    def epoch_durs(self, epochs):
        return (np.diff(epochs).flatten()) * self.c.dt

    def epoch_amps(self, epochs, a):
        return np.array(
            [
                np.trapz(a[p][~np.isnan(a[p])], dx=self.c.dt)
                for p in util.epoch_slices(epochs)
            ]
        )

    def epoch_maxs(self, epochs, a):
        return np.array([np.max(a[p]) for p in util.epoch_slices(epochs)])

    def epoch_idx(self, epochs):
        slices = util.epoch_slices(epochs)
        if len(slices) == 0:
            return []
        elif len(slices) == 1:
            return slices[0]
        else:
            return np.concatenate(slices)

    def comp_chunk_bearing(self, chunk):
        for n, loc in self.c.sources.items():
            A = self.empty_df(dim3=3)
            for i, id in enumerate(self.ids):
                ep = self.epoch_dicts[chunk][id]
                if ep.shape[0] > 0:
                    b0s, b1s = self.epochs_bearing_by_ID(chunk, id, loc=loc)
                    A[ep[:, 0], i, 0] = b0s
                    A[ep[:, 1], i, 1] = b1s
                    A[ep[:, 1], i, 2] = b1s - b0s
            self.s[nam.atStartStopChunk(nam.bearing_to(n), chunk)] = A.reshape([-1, 3])

    def detect_epochs(self, idx, min_dur=None):
        dt = self.c.dt
        if min_dur is None:
            min_dur = 2 * dt
        p0s = idx[np.where(np.diff(idx, prepend=[-np.inf]) != 1)[0]]
        p1s = idx[np.where(np.diff(idx, append=[np.inf]) != 1)[0]]
        epochs = np.vstack([p0s, p1s]).T
        return epochs[self.epoch_durs(epochs) >= min_dur]

    def detect_runs(self, a, vel_thr=0.3, min_dur=0.5):
        """
        Annotates crawl-runs in timeseries.

        Extended description of function.

        Parameters
        ----------
        a : array
            1D np.array : forward velocity timeseries
        vel_thr : float
            Maximum velocity threshold
         min_dur : float, optional
            The minimum required duration for a turn

        Returns
        -------
        runs : list
            A list of pairs of the start-end indices of the runs.


        """
        return self.detect_epochs(np.where(a >= vel_thr)[0], min_dur)

    def detect_pauses(self, a, vel_thr=0.3, runs=None, min_dur=None):
        """
        Annotates crawl-pauses in timeseries.

        Extended description of function.

        Parameters
        ----------
        a : array
            1D np.array : forward velocity timeseries
        vel_thr : float
            Maximum velocity threshold
        runs : list
            A list of pairs of the start-end indices of the runs.
            If provided pauses that overlap with runs will be excluded.
        min_dur : float, optional
            The minimum required duration for a turn

        Returns
        -------
        pauses : list
            A list of pairs of the start-end indices of the pauses.

        """
        idx = np.where(a <= vel_thr)[0]
        if runs is not None:
            for r0, r1 in runs:
                idx = idx[(idx <= r0) | (idx >= r1)]
        return self.detect_epochs(idx, min_dur)

    def detect_strides(
        self, a, vel_thr=0.3, stretch=(0.75, 2.0), fr=None, return_extrema=True
    ):
        """
        Annotates strides-runs and pauses in timeseries.

        Extended description of function.

        Parameters
        ----------
        a : array
            1D np.array : forward velocity timeseries
        vel_thr : float
            Maximum velocity threshold
        stretch : Tuple[float,float]
            The min-max stretch of a stride relative to the default derived from the dominnt frequency
        fr : float, optional
            The dominant crawling frequency.
        return_extrema : boolean
            Whether to additionally return the stride extrema


        Returns
        -------
        strides : list
            A list of pairs of the start-end indices of the strides.
        i_min : array
            Indices of the local minima.
        i_max : array
            Indices of the local maxima

        """
        dt = self.c.dt
        if fr is None:
            fr = util.fft_max(a, dt, fr_range=(1, 2.5))
        tmin = stretch[0] // (fr * dt)
        tmax = stretch[1] // (fr * dt)
        i_min = find_peaks(-a, height=-3 * vel_thr, distance=tmin)[0]
        i_max = find_peaks(a, height=vel_thr, distance=tmin)[0]
        strides = []
        for m in i_max:
            try:
                s0, s1 = [i_min[i_min < m][-1], i_min[i_min > m][0]]
                if ((s1 - s0) <= tmax) and ([s0, s1] not in strides):
                    strides.append([s0, s1])
            except:
                pass
        strides = np.array(strides)
        if return_extrema:
            return i_min, i_max, strides
        else:
            return strides

    def detect_stridechains(self, strides):
        """
        Annotates stridechains-runs by concatenating consecutive strides.

        Extended description of function.

        Parameters
        ----------
        strides : array
            2D np.array : the start-end tics of the stride epochs

        Returns
        -------
        runs : list
             A list of pairs of the start-end indices of the runs/stridechains.
        run_counts : list
             Stride-counts of the runs/stridechains.

        """
        runs, run_counts = [], []
        s00, s11 = None, None

        count = 0
        for ii, (s0, s1) in enumerate(strides.tolist()):
            if ii == 0:
                s00, s11 = s0, s1
                count = 1
                continue
            if s11 == s0:
                s11 = s1
                count += 1
            else:
                runs.append([s00, s11])
                run_counts.append(count)
                count = 1
                s00, s11 = s0, s1
            if ii == len(strides) - 1:
                runs.append([s00, s11])
                run_counts.append(count)
                break
        runs = np.array(runs)
        return runs, run_counts

    def detect_turns(self, a, min_dur=None):
        """
        Annotates turns in timeseries.

        Extended description of function.

        Parameters
        ----------
        a : array
            1D np.array : angular velocity timeseries
        min_dur : float, optional
            The minimum required duration for a turn

        Returns
        -------
        Lturns : list
            A list of pairs of the start-end indices of the Left turns.
        Rturns : list
            A list of pairs of the start-end indices of the Right turns.


        """
        dt = self.c.dt
        if type(a) != pd.core.series.Series:
            a = pd.Series(a)
        if min_dur is None:
            min_dur = 2 * dt
        i_zeros = np.where(np.sign(a).diff().ne(0) == True)[0]
        Rturns, Lturns = [], []
        for s0, s1 in zip(i_zeros[:-1], i_zeros[1:]):
            if (s1 - s0) <= 2:
                continue
            elif np.isnan(np.sum(a[s0:s1])):
                continue
            else:
                if all(a[s0:s1] >= 0):
                    Lturns.append([s0, s1])
                elif all(a[s0:s1] <= 0):
                    Rturns.append([s0, s1])
        Lturns = np.array(Lturns)
        Rturns = np.array(Rturns)
        return Lturns[self.epoch_durs(Lturns) >= min_dur], Rturns[
            self.epoch_durs(Rturns) >= min_dur
        ]

    def crawl_annotation(
        self, strides_enabled: bool = True, vel_thr: float = 0.3
    ) -> AttrDict:
        if self.c.Npoints <= 1:
            strides_enabled = False
        l, v, sv, fov = reg.getPar(["l", "v", "sv", "fov"])
        (
            str_d_mu,
            str_d_std,
            str_sd_mu,
            str_sd_std,
            run_tr,
            pau_tr,
            cum_run_t,
            cum_pau_t,
            cum_t,
        ) = reg.getPar(
            [
                "str_d_mu",
                "str_d_std",
                "str_sd_mu",
                "str_sd_std",
                "run_tr",
                "pau_tr",
                "cum_run_t",
                "cum_pau_t",
                "cum_t",
            ]
        )
        Sps = (
            [str_d_mu, str_d_std]
            + reg.getPar(["str_sv_mu", "str_N", "run_v_mu", "pau_v_mu"])
            + [cum_run_t, cum_pau_t]
        )
        Svs = np.zeros([self.c.N, len(Sps)]) * np.nan
        DD = AttrDict()
        for jj, id in enumerate(self.ids):
            D = AttrDict()
            S = self.s.xs(id, level="AgentID")
            a_v = S[v].values
            a_fov = S[fov].values
            if strides_enabled:
                a = S[sv].values
                D.vel_minima, D.vel_maxima, D.stride = self.detect_strides(
                    a, vel_thr=vel_thr
                )
                D.exec, D.run_count = self.detect_stridechains(D.stride)
                D.stride_Dor = np.array(
                    [np.trapz(a_fov[s0 : s1 + 1]) for s0, s1 in D.stride]
                )
                D.stride_dur = self.epoch_durs(D.stride)
                D.stride_dst = self.epoch_amps(D.stride, a_v)
                D.stride_idx = self.epoch_idx(D.stride)
            else:
                D.vel_minima, D.vel_maxima, D.stride, D.run_count = (
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                )
                D.stride_Dor, D.stride_dur, D.stride_dst, D.stride_idx = (
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    [],
                )
                a = a_v
                D.exec = self.detect_epochs(np.where(a_v >= vel_thr)[0])

            D.run_dur = self.epoch_durs(D.exec)
            D.run_dst = self.epoch_amps(D.exec, a_v)
            D.run_idx = self.epoch_idx(D.exec)
            D.pause = self.detect_pauses(a, vel_thr=vel_thr, runs=D.exec)
            D.pause_dur = self.epoch_durs(D.pause)
            D.pause_idx = self.epoch_idx(D.pause)
            Svs[jj, :] = [
                np.nanmean(D.stride_dst),
                np.nanstd(D.stride_dst),
                np.nanmean(a[D.stride_idx]),
                np.nansum(D.run_count),
                np.mean(a_v[D.run_idx]),
                np.mean(a_v[D.pause_idx]),
                np.sum(D.run_dur),
                np.sum(D.pause_dur),
            ]
            DD[id] = D
        self.e[Sps] = Svs
        self.e[run_tr] = self.e[cum_run_t] / self.e[cum_t]
        self.e[pau_tr] = self.e[cum_pau_t] / self.e[cum_t]
        if l in self.end_ps:
            self.e[str_sd_mu] = self.e[str_d_mu] / self.e[l]
            self.e[str_sd_std] = self.e[str_d_std] / self.e[l]
        return DD

    def turn_annotation(self, min_dur=None):
        S = self.s[reg.getPar("fov")]
        ps = reg.getPar(["Ltur_N", "Rtur_N", "tur_N", "tur_H"])
        vs = np.zeros([self.c.N, len(ps)]) * np.nan
        DD = {}
        for j, id in enumerate(self.ids):
            D = AttrDict()
            a = S.xs(id, level="AgentID")
            D.Lturn, D.Rturn = self.detect_turns(a, min_dur=min_dur)
            D.Lturn_dur = self.epoch_durs(D.Lturn)
            D.Rturn_dur = self.epoch_durs(D.Rturn)
            D.Lturn_amp = self.epoch_amps(D.Lturn, a.values)
            D.Rturn_amp = self.epoch_amps(D.Rturn, a.values)
            Lmaxs = self.epoch_maxs(D.Lturn, a.values)
            Rmaxs = self.epoch_maxs(D.Rturn, a.values)
            D.turn_dur = np.concatenate([D.Lturn_dur, D.Rturn_dur])
            D.turn_amp = np.concatenate([D.Lturn_amp, D.Rturn_amp])
            D.turn_vel_max = np.concatenate([Lmaxs, Rmaxs])
            LN, RN = D.Lturn.shape[0], D.Rturn.shape[0]
            N = LN + RN
            H = LN / N if N != 0 else 0
            vs[j, :] = [LN, RN, N, H]
            DD[id] = D
        self.e[ps] = vs
        return DD

    def turn_mode_annotation(self):
        wNh = {}
        wNh_ps = [
            "weathervane_q25_amp",
            "weathervane_q75_amp",
            "headcast_q25_amp",
            "headcast_q75_amp",
        ]
        for id in self.ids:
            D = self.chunk_dicts[id]
            T = util.epoch_slices(D.Lturn) + util.epoch_slices(D.Rturn)
            wvane_idx = [
                ii for ii, t in enumerate(T) if all([tt in D.run_idx for tt in t])
            ]
            cast_idx = [
                ii for ii, t in enumerate(T) if all([tt in D.pause_idx for tt in t])
            ]
            Awvane = D.turn_amp[wvane_idx]
            Acast = D.turn_amp[cast_idx]
            wNh[id] = (
                np.nanquantile(Awvane, 0.25),
                np.nanquantile(Awvane, 0.75),
                np.nanquantile(Acast, 0.25),
                np.nanquantile(Acast, 0.75),
            )
        self.e[wNh_ps] = pd.DataFrame.from_dict(wNh).T

    def patch_residency_annotation(self):
        dst, on_tr, on_t_mu, cum_on_d, on_d_mu, on_v_mu, cum_on_t, cum_t = reg.getPar(
            [
                "d",
                "on_food_tr",
                "on_food_t_mu",
                "cum_on_food_d",
                "on_food_d_mu",
                "on_food_v_mu",
                "cum_on_food_t",
                "cum_t",
            ]
        )
        Sps = [cum_on_t, on_t_mu, cum_on_d, on_d_mu]
        Svs = np.zeros([self.c.N, len(Sps)]) * np.nan
        DD = {}
        for jj, id in enumerate(self.ids):
            S = self.s.xs(id, level="AgentID")
            D = AttrDict()
            D.on_food = np.array([])
            if nam.on_food in self.s.columns:
                D.on_food = self.detect_epochs(
                    np.where(S[nam.on_food].values == True)[0]
                )
            D.on_food_dur = self.epoch_durs(D.on_food)
            D.on_food_dst = self.epoch_amps(D.on_food, S[dst].values)
            DD[id] = D
            Svs[jj, :] = [
                np.nansum(D.on_food_dur),
                np.nanmean(D.on_food_dur),
                np.nansum(D.on_food_dst),
                np.nanmean(D.on_food_dst),
            ]
        self.e[Sps] = Svs
        self.e[on_tr] = self.e[cum_on_t] / self.e[cum_t]
        self.e[on_v_mu] = self.e[cum_on_d] / self.e[cum_t]
        return DD

    def detect_epoch_on_food_overlap(self, chunk):
        on = nam.on_food
        CT = self.e[nam.cum(nam.dur(on))]
        D0 = self.epoch_dicts["on_food"]
        cdur_on = f"{nam.cum(nam.dur(chunk))}_{on}"
        cc_N_on = f"{nam.num(chunk)}_{on}"
        Sps = [cdur_on, cc_N_on]
        Svs = np.zeros([self.c.N, len(Sps)]) * np.nan
        D = self.epoch_dicts[chunk]
        for jj, id in enumerate(self.ids):
            valid = util.epoch_overlap(D[id], D0[id])
            durs = self.epoch_durs(valid)
            Svs[jj, :] = [np.nansum(durs), durs.shape[0]]
        self.e[Sps] = Svs
        self.e[f"{nam.dur_ratio(chunk)}_{on}"] = self.e[cdur_on] / CT
        self.e[f"{nam.mean(nam.num(chunk))}_{on}"] = self.e[cc_N_on] / CT

    def detect_bouts(self, vel_thr=0.3, strides_enabled=True, castsNweathervanes=True):
        self.comp_freqs()
        Dtur = self.turn_annotation()
        vprint("Turn annotation complete.", 1)
        Dcr = self.crawl_annotation(strides_enabled=strides_enabled, vel_thr=vel_thr)
        vprint("Crawl annotation complete.", 1)
        Dpa = self.patch_residency_annotation()
        vprint("Patch residency annotation complete.", 1)
        self.chunk_dicts = AttrDict(
            {id: {**Dtur[id], **Dcr[id], **Dpa[id]} for id in self.ids}
        )
        if castsNweathervanes:
            self.turn_mode_annotation()
        vprint("Completed bout detection.", 1)

    def comp_pooled_epochs(self):
        """
        Compute pooled epochs from chunk dictionaries.

        This method processes the `chunk_dicts` attribute to create `epoch_dicts` and `pooled_epochs`.
        It first extracts unique epoch keys from the chunk dictionaries and then constructs a dictionary
        of epochs (`epoch_dicts`) where each key corresponds to a dictionary of chunk data.

        The method then defines an inner function `get_vs` to concatenate values from the dictionaries,
        handling cases where the values have different shapes. If the majority of the values have a shape
        of 2 dimensions, it filters out those with a shape of 1 dimension before concatenation.

        Finally, it creates the `pooled_epochs` attribute by concatenating the values for each epoch key,
        excluding specific keys such as "turn_slice", "pause_idx", "run_idx", and "stride_idx".

        Attributes:
            chunk_dicts (dict): A dictionary containing chunk data.
            epoch_dicts (AttrDict): A dictionary of epochs with chunk data.
            pooled_epochs (AttrDict): A dictionary of concatenated epoch data.

        Raises:
            Exception: If there is an issue with concatenating the values in `get_vs`.

        Prints:
            "Completed bout detection." upon successful completion.
        """
        d0 = self.chunk_dicts
        epoch_ks = SuperList([list(dic.keys()) for dic in d0.values()]).flatten.unique
        self.epoch_dicts = AttrDict(
            {k: {id: d0[id][k] for id in list(d0)} for k in epoch_ks}
        )

        def get_vs(dic):
            l = SuperList(dic.values())

            # Filter out empty arrays or lists
            l = SuperList(
                [
                    ll
                    for ll in l
                    if (isinstance(ll, np.ndarray) and ll.size > 0)
                    or (isinstance(ll, list) and len(ll) > 0)
                ]
            )

            if len(l) == 0:
                return np.array([])
            else:
                return np.concatenate(l)

        self.pooled_epochs = AttrDict(
            {
                k: get_vs(dic)
                for k, dic in self.epoch_dicts.items()
                # if k not in ["turn_slice", "pause_idx", "run_idx", "stride_idx"]
            }
        )

        vprint("Completed bout detection.", 1)

    def fit_pooled_epochs(self):
        try:
            D = self.pooled_epochs
            assert D is not None
            d = {}
            for k in D:
                try:
                    x0 = np.abs(D[k])
                    d[k] = reg.fit_bout_distros(
                        x0,
                        bout=k,
                        combine=False,
                        discrete=True if k == "run_count" else False,
                    )
                except:
                    d[k] = None
            self.fitted_epochs = AttrDict(d)
            vprint("Fitted pooled epoch durations.", 1)
        except:
            vprint("Failed to fit pooled epoch durations.", 1)

    def generate_pooled_epochs(self, mID):
        m = reg.conf.Model.getID(mID)
        Im = self.c.get_sample_bout_distros(m.get_copy()).brain.intermitter
        try:
            D = self.pooled_epochs
            assert D is not None
            d = {}
            for n, k in zip(
                ["pause", "run", "stridechain"], ["pause_dur", "run_dur", "run_count"]
            ):
                try:
                    x0 = reg.BoutGenerator(
                        **Im[f"{n}_dist"], dt=1 if k == "run_count" else self.c.dt
                    ).sample(D[k].shape[0])
                    d[k] = reg.fit_bout_distros(
                        x0,
                        bout=k,
                        combine=False,
                        discrete=True if k == "run_count" else False,
                    )
                except:
                    d[k] = None
            vprint("Generated pooled epoch durations.", 1)
            return AttrDict(d)
        except:
            vprint("Failed to generate pooled epoch durations.", 1)

    def comp_bout_distros(self):
        c = self.config
        c.bout_distros = AttrDict()
        for k, dic in self.fitted_epochs.items():
            try:
                c.bout_distros[k] = dic["best"]
                vprint(f"Completed {k} bout distribution analysis.", 1)
            except:
                c.bout_distros[k] = None
                vprint(f"Failed to complete {k} bout distribution analysis.", 1)
        self.register_bout_distros()

    def register_bout_distros(self):
        s, e, c = self.data
        from ..model.modules.intermitter import get_EEB_poly1d

        try:
            c.intermitter = {
                nam.freq("crawl"): e[nam.freq(nam.scal(nam.vel("")))].mean(),
                nam.freq("feed"): e[nam.freq("feed")].mean()
                if nam.freq("feed") in self.end_ps
                else 2.0,
                "dt": c.dt,
                "feed_bouts": True,
                "stridechain_dist": c.bout_distros.run_count,
                "pause_dist": c.bout_distros.pause_dur,
                "run_dist": c.bout_distros.run_dur,
                "feeder_reoccurence_rate": None,
            }
            c.EEB_poly1d = get_EEB_poly1d(**c.intermitter).c.tolist()
        except:
            pass

    def comp_cycle_curves(self, Nbins=64):
        CC = AttrDict()
        for sh in ["sv", "fov", "rov", "foa", "b"]:
            ss = self.s[reg.getPar(sh)]
            CC[sh] = AttrDict(
                {
                    "abs": np.zeros([self.c.N, Nbins]) * np.nan,
                    "plus": np.zeros([self.c.N, Nbins]) * np.nan,
                    "minus": np.zeros([self.c.N, Nbins]) * np.nan,
                    "norm": np.zeros([self.c.N, Nbins]) * np.nan,
                }
            )

            for jj, id in enumerate(self.ids):
                D = self.chunk_dicts[id]
                aa = util.stride_interp(
                    ss.xs(id, level="AgentID").values, D.stride, Nbins
                )
                aa_minus = aa[D.stride_Dor < 0]
                aa_plus = aa[D.stride_Dor > 0]
                aa_norm = np.vstack([aa_plus, -aa_minus])
                CC[sh].abs[jj, :] = np.nanquantile(np.abs(aa), q=0.5, axis=0)
                CC[sh].plus[jj, :] = np.nanquantile(aa_plus, q=0.5, axis=0)
                CC[sh].minus[jj, :] = np.nanquantile(aa_minus, q=0.5, axis=0)
                CC[sh].norm[jj, :] = np.nanquantile(aa_norm, q=0.5, axis=0)

        return CC

    def comp_attenuation(self, Nbins=64):
        p_sv, pau_fov_mu = reg.getPar(["sv", "pau_fov_mu"])
        x = np.linspace(0, 2 * np.pi, Nbins)
        CC = self.cycle_curves
        att0s, att1s = np.min(CC["fov"].abs, axis=1), np.max(CC["fov"].abs, axis=1)

        self.e[nam.phi(nam.max("attenuation"))] = x[np.argmax(CC["fov"].abs, axis=1)]
        self.e[nam.phi(nam.max(p_sv))] = x[np.argmax(CC["sv"].abs, axis=1)]
        self.e[reg.getPar("str_sv_max")] = np.max(CC["sv"].abs, axis=1)
        try:
            self.e["attenuation"] = att0s / self.e[pau_fov_mu]
            self.e[nam.max("attenuation")] = (att1s - att0s) / self.e[pau_fov_mu]
        except:
            pass

    def comp_interference(self, Nbins=64):
        try:
            self.cycle_curves = self.comp_cycle_curves(Nbins=Nbins)
            self.comp_attenuation(Nbins=Nbins)
            self.comp_pooled_cycle_curves()
            vprint("Completed stridecycle interference analysis.", 1)
        except:
            vprint("Failed to complete stridecycle interference analysis.", 1)

    def comp_pooled_cycle_curves(self):
        try:
            self.pooled_cycle_curves = AttrDict(
                {
                    k: {
                        mode: np.nanquantile(vs, q=0.5, axis=0).tolist()
                        for mode, vs in dic.items()
                    }
                    for k, dic in self.cycle_curves.items()
                }
            )
            vprint(
                "Computed average curves during stridecycle for diverse parameters.", 1
            )
        except:
            vprint(
                "Failed to compute average curves during stridecycle for diverse parameters.",
                1,
            )

    def annotate(
        self,
        anot_keys=["bout_detection", "bout_distribution", "interference"],
        is_last=False,
        **kwargs,
    ):
        if "bout_detection" in anot_keys:
            self.detect_bouts()
            self.comp_pooled_epochs()
        if "bout_distribution" in anot_keys:
            self.fit_pooled_epochs()
            self.comp_bout_distros()
        if "interference" in anot_keys:
            self.comp_interference()
        if "source_attraction" in anot_keys:
            for chunk in ["stride", "pause", "Lturn", "Rturn", "turn"]:
                try:
                    self.comp_chunk_bearing(chunk)
                except:
                    pass
        if "patch_residency" in anot_keys:
            on = nam.on_food
            for chunk in ["Lturn", "Rturn", "pause"]:
                self.detect_epoch_on_food_overlap(chunk)
            self.e[f"handedness_score_{on}"] = self.e[f"{nam.num('Lturn')}_{on}"] / (
                self.e[f"{nam.num('Lturn')}_{on}"] + self.e[f"{nam.num('Rturn')}_{on}"]
            )
        if is_last:
            self.save()

    def interpolate_nan_values(self):
        s, e, c = self.data
        pars = c.all_xy.existing(s)
        Npars = len(pars)
        for id in self.ids:
            A = np.zeros([c.Nticks, Npars])
            ss = s.xs(id, level="AgentID")
            for i, p in enumerate(pars):
                A[:, i] = util.interpolate_nans(ss[p].values)
            s.loc[(slice(None), id), pars] = A
        vprint("All parameters interpolated", 1)

    def filter(self, filter_f=2.0, recompute=False):
        s, e, c = self.data
        assert isinstance(filter_f, float)
        if c.filtered_at is not None and not recompute:
            vprint(
                f"Dataset already filtered at {c.filtered_at}. To apply additional filter set recompute to True",
                1,
            )
            return
        c.filtered_at = filter_f

        pars = c.all_xy.existing(s)
        data = np.dstack(
            list(s[pars].groupby("AgentID").apply(pd.DataFrame.to_numpy))
        ).astype(None)
        f_array = util.apply_filter_to_array_with_nans_multidim(
            data, freq=filter_f, fr=1 / c.dt
        )
        for j, p in enumerate(pars):
            s[p] = f_array[:, j, :].flatten()
        vprint(f"All spatial parameters filtered at {filter_f} Hz", 1)

    def rescale(self, recompute=False, rescale_by=1.0):
        s, e, c = self.data
        assert isinstance(rescale_by, float)
        if c.rescaled_by is not None and not recompute:
            vprint(
                f"Dataset already rescaled by {c.rescaled_by}. To rescale again set recompute to True",
                1,
            )
            return
        c.rescaled_by = rescale_by
        points = c.midline_points + ["centroid", ""]
        pars = (
            c.all_xy + nam.dst(points) + nam.vel(points) + nam.acc(points) + ["length"]
        )
        for p in util.existing_cols(pars, s):
            s[p] = s[p].apply(lambda x: x * rescale_by)
        if "length" in e.columns:
            e["length"] = e["length"].apply(lambda x: x * rescale_by)
        vprint(f"Dataset rescaled by {rescale_by}.", 1)

    def exclude_rows(self, flag="collision_flag", accepted=[0], rejected=None):
        s, e, c = self.data
        if accepted is not None:
            s.loc[s[flag] != accepted[0]] = np.nan
        if rejected is not None:
            s.loc[s[flag] == rejected[0]] = np.nan
        for id in self.ids:
            e.loc[id, "cum_dur"] = (
                len(s.xs(id, level="AgentID", drop_level=True).dropna()) * c.dt
            )
        vprint(f"Rows excluded according to {flag}.", 1)

    def smaller_dataset(self, p):
        """
        Generate a smaller dataset based on the given ReplayConf parameters.

        Args:
            p (ReplayConf): The configuration for dataset replay.

        Returns:
            LarvaDataset: A subset of the original dataset.
        """
        d = copy.deepcopy(self)
        c = d.config
        # Ensure the dataset is loaded
        d.load(h5_ks=["contour", "midline", "angular"])

        # Update point tracking configuration
        if p.track_point is not None:
            c.point_idx = p.track_point

        # Fix a specific point if required
        if p.fix_point is not None:
            c.fix_point = c.get_track_point(p.fix_point)
            if c.fix_point == "centroid" or p.fix_segment is None:
                c.fix_point2 = None
            else:
                P2_idx = p.fix_point + 1 if p.fix_segment == "rear" else p.fix_point - 1
                c.fix_point2 = c.get_track_point(P2_idx)
        else:
            c.fix_point = None

        # Select specific agent IDs if provided
        if p.agent_ids not in [None, []]:
            if isinstance(p.agent_ids, list) and all(
                isinstance(i, int) for i in p.agent_ids
            ):
                p.agent_ids = [c.agent_ids[i] for i in p.agent_ids]
            elif isinstance(p.agent_ids, int):
                p.agent_ids = [c.agent_ids[p.agent_ids]]
            c.agent_ids = p.agent_ids

        # If a fixation point is provided, only keep the first agent
        if c.fix_point is not None:
            c.agent_ids = c.agent_ids[:1]

        # Update dataset based on selected agents
        d.update_ids_in_data()

        # Apply time slicing if specified
        if p.time_range is not None:
            d.step_data = d.timeseries_slice(time_range=p.time_range)

        # Align trajectory to match tracking point
        xy_pars = nam.xy(c.point)
        if xy_pars.exist_in(d.step_data):
            d.step_data[["x", "y"]] = d.step_data[xy_pars]

        # Update environment parameters
        if p.env_params is None:
            p.env_params = c.env_params.nestedConf

        # Reduce arena size for close view
        if p.close_view:
            p.env_params.arena = reg.gen.Arena(dims=(0.01, 0.01)).nestedConf

        # Fix larva orientation if required
        if c.fix_point is not None:
            d.step_data, bg = util.fixate_larva(
                d.step_data,
                c,
                arena_dims=p.env_params.arena.dims,
                P1=c.fix_point,
                P2=c.fix_point2,
            )
        else:
            bg = None

        # Apply spatial transposition if specified
        if p.transposition is not None:
            d.step_data = d.align_trajectories(
                transposition=p.transposition, replace=True
            )
            xy_max = 2 * np.max(
                d.step_data[nam.xy(c.point)].dropna().abs().values.flatten()
            )
            p.env_params.arena = reg.gen.Arena(dims=(xy_max, xy_max)).nestedConf

        return d, bg

    def align_trajectories(
        self, track_point=None, arena_dims=None, transposition="origin", replace=True
    ):
        s, e, c = self.data

        assert transposition in ["arena", "origin", "center"]
        mode = transposition

        xy_flat = c.all_xy.existing(s)
        xy_pairs = xy_flat.in_pairs

        if replace:
            ss = s
        else:
            ss = copy.deepcopy(s[xy_flat])

        if mode == "arena":
            vprint("Centralizing trajectories in arena center")
            if arena_dims is None:
                arena_dims = c.env_params.arena.dims
            for x, y in xy_pairs:
                ss[x] -= arena_dims[0] / 2
                ss[y] -= arena_dims[1] / 2
            return ss
        else:
            # s = self._load_step(h5_ks=["contour", "midline"])
            if track_point is None:
                track_point = c.point
            XY = (
                nam.xy(track_point)
                if util.cols_exist(nam.xy(track_point), s)
                else c.traj_xy
            )
            if not util.cols_exist(XY, s):
                raise ValueError(
                    "Defined point xy coordinates do not exist. Can not align trajectories! "
                )
            if mode == "origin":
                vprint("Aligning trajectories to common origin")
                xy = [
                    s[XY].xs(id, level="AgentID").dropna().values[0]
                    if not s[XY].xs(id, level="AgentID").dropna().empty
                    else [0, 0]
                    for id in self.ids
                ]
            elif mode == "center":
                vprint(
                    "Centralizing trajectories in trajectory center using min-max positions"
                )
                xy = [
                    (
                        s[XY].xs(id, level="AgentID").max().values
                        - s[XY].xs(id, level="AgentID").min().values
                    )
                    / 2
                    for id in self.ids
                ]
            else:
                raise ValueError('Supported modes are "arena", "origin" and "center"!')
            xs = np.array([x for x, y in xy] * c.Nticks)
            ys = np.array([y for x, y in xy] * c.Nticks)

            for x, y in xy_pairs:
                ss[x] = ss[x].values - xs
                ss[y] = ss[y].values - ys
            return ss

    def preprocess(
        self,
        drop_collisions=False,
        interpolate_nans=False,
        filter_f=None,
        rescale_by=None,
        transposition=None,
        recompute=False,
    ):
        if drop_collisions:
            self.exclude_rows()
        if interpolate_nans:
            self.interpolate_nan_values()
        if filter_f is not None:
            self.filter(filter_f=filter_f, recompute=recompute)
        if rescale_by is not None:
            self.rescale(rescale_by=rescale_by, recompute=recompute)
        if transposition is not None:
            self.align_trajectories(transposition=transposition)

    def merge_configs(self):
        d = param.guess_param_types(**self.config2)
        for n, p in d.items():
            self.config.param.add_parameter(n, p)

    def set_data(self, step=None, end=None, agents=None, **kwargs):
        if step is not None:
            self.step_data = step.sort_index(level=self.param.step_data.levels)
        if end is not None:
            self.endpoint_data = end.sort_index()
        if agents is not None:
            self.larva_dicts = get_larva_dicts(agents, validIDs=self.ids)
        self.validate_IDs()

    @property
    def data(self):
        return self.s, self.e, self.c

    def path_to_file(self, file="data.h5"):
        f = self.c.data_dir
        if f is not None:
            return f"{f}/{file}"
        else:
            return None

    @property
    def path_to_config(self):
        return self.path_to_file("conf.txt")

    def store(self, df, key, file="data.h5"):
        path = self.path_to_file(file)
        if path is not None:
            if not isinstance(df, pd.DataFrame):
                pd.DataFrame(df).to_hdf(path, key)
            else:
                df.to_hdf(path, key)

    def save_dict(self, d, file):
        path = self.path_to_file(file)
        if path is not None:
            util.save_dict(d, path)

    def read(self, key, file="data.h5"):
        path = self.path_to_file(file)
        if path is not None:
            try:
                return pd.read_hdf(path, key)
            except:
                return None
        else:
            return None

    def load(self, step=True, h5_ks=None):
        s = self._load_step(h5_ks=h5_ks) if step else None
        e = self.read("end")
        self.set_data(step=s, end=e)

    def _load_step(self, h5_ks=None):
        s = self.read("step")
        if h5_ks is None:
            h5_ks = list(self.config.h5_kdic.keys())
        for h5_k in h5_ks:
            ss = self.read(h5_k)
            if ss is not None:
                ps = util.nonexisting_cols(ss.columns.values, s)
                if len(ps) > 0:
                    s = s.join(ss[ps])
        return s

    def _save_step(self, s):
        s = s.loc[:, ~s.columns.duplicated()]
        stored_ps = []
        for h5_k, ps in self.c.h5_kdic.items():
            pps = ps.unique.existing(s)
            if len(pps) > 0:
                self.store(s[pps], h5_k)
                stored_ps += pps

        self.store(s.drop(stored_ps, axis=1, errors="ignore"), "step")

    def save(self, refID=None):
        if self.s is not None:
            self._save_step(s=self.s)
        if self.e is not None:
            self.store(self.e, "end")
        self.save_config(refID=refID)
        vprint(f"***** Dataset {self.c.id} stored.-----", 1)

    def save_config(self, refID=None):
        c = self.c
        if refID is not None:
            c.refID = refID
        if c.refID is not None:
            reg.conf.Ref.setID(c.refID, c.dir)
            vprint(f"Saved reference dataset under : {c.refID}", 1)
        self.save_dict(c.nestedConf, "conf.txt")

    def load_traj(self, mode="default"):
        key = f"traj.{mode}"
        df = self.read(key)
        if df is None:
            if mode == "default":
                df = self.s[["x", "y"]]
            elif mode in ["origin", "center"]:
                df = self.align_trajectories(replace=False, transposition=mode)[
                    ["x", "y"]
                ]
            else:
                raise ValueError("Not implemented")
            self.store(df, key)
        return df

    def load_dicts(self, type, ids=None):
        """
        Load dictionaries based on the specified type and optional IDs.

        Args:
            type (str): The type of dictionaries to load.
            ids (list, optional): A list of IDs to load. If None, uses self.ids.

        Returns:
            list: A list of dictionaries corresponding to the specified type and IDs.

        Notes:
            - If the specified type and IDs are found in self.larva_dicts, the dictionaries are loaded from there.
            - Otherwise, the dictionaries are loaded from files located in the directory specified by self.config.data_dir.
        """
        if ids is None:
            ids = self.ids
        ds0 = self.larva_dicts
        if type in ds0 and all([id in ds0[type] for id in ids]):
            ds = [ds0[type][id] for id in ids]
        else:
            path = f"{self.config.data_dir}/individuals/{type}"
            ds = [util.load_dict(f"{path}/{id}.txt") for id in ids]
        return ds

    def store_dicts(self, type, dicts):
        """
        Stores a dictionary of dictionaries to individual files.

        Args:
            type (str): The type/category of the dictionaries to be stored.
            dicts (dict): A dictionary where keys are identifiers and values are dictionaries to be stored.

        Example:
            >>> store_dicts('example_type', {'id1': {'key1': 'value1'}, 'id2': {'key2': 'value2'}})
            This will create files 'id1.txt' and 'id2.txt' in the directory specified by self.config.data_dir/individuals/example_type.
        """
        path = f"{self.config.data_dir}/individuals/{type}"
        if path is not None:
            os.makedirs(path, exist_ok=True)
            for id, d in dicts.items():
                util.save_dict(d, f"{path}/{id}.txt")

    def store_larva_dicts(self):
        """
        Stores larva dictionaries by iterating over the items in `self.larva_dicts`.

        This method retrieves each type and its corresponding dictionary from
        `self.larva_dicts` and passes them to the `store_dicts` method for storage.

        Returns:
            None
        """
        for type, dicts in self.larva_dicts.items():
            self.store_dicts(type, dicts)

    @property
    def contour_xy_data_byID(self):
        if self.c.Ncontour == 0:
            return AttrDict(
                {id: np.zeros([self.c.Nticks, 2]) * np.nan for id in self.ids}
            )
        xy = self.c.contour_xy
        assert xy.exist_in(self.s)
        grouped = self.s[xy].groupby("AgentID")
        return AttrDict(
            {id: df.values.reshape([-1, self.c.Ncontour, 2]) for id, df in grouped}
        )

    @property
    def midline_xy_data_byID(self):
        if self.c.Npoints == 0:
            return AttrDict(
                {id: np.zeros([self.c.Nticks, 2]) * np.nan for id in self.ids}
            )
        xy = self.c.midline_xy
        # assert xy.exist_in(self.step_data)
        grouped = self.s[xy].groupby("AgentID")
        return AttrDict(
            {id: df.values.reshape([-1, self.c.Npoints, 2]) for id, df in grouped}
        )

    @property
    def traj_xy_data_byID(self):
        return self.data_by_ID(self.s[self.c.traj_xy])

    def data_by_ID(self, data):
        grouped = data.groupby("AgentID")
        return AttrDict({id: df.values for id, df in grouped})

    @property
    def midline_xy_data(self):
        return self.s[self.c.midline_xy].values.reshape([-1, self.c.Npoints, 2])

    @property
    def contour_xy_data(self):
        return self.s[self.c.contour_xy].values.reshape([-1, self.c.Ncontour, 2])

    def empty_df(self, dim3=1):
        c = self.c
        if dim3 == 1:
            return np.zeros([c.Nticks, c.N]) * np.nan
        elif dim3 > 1:
            return np.zeros([c.Nticks, c.N, dim3]) * np.nan

    def apply_per_agent(self, pars, func, time_range=None, **kwargs):
        """
        Apply a function to each subdataframe of a MultiIndex DataFrame after grouping by the agentID.

        Parameters
        ----------
        s : pandas.DataFrame
            A MultiIndex DataFrame with levels ['Step', 'AgentID'].
        func : function
            The function to apply to each subdataframe.

        **kwargs : dict
            Additional keyword arguments to pass to the 'func' function.

        Returns
        -------
        numpy.ndarray
            An array of dimensions [N_ticks, N_ids], where N_ticks is the number of unique 'Step' values,
            and N_ids is the number of unique 'AgentID' values.

        Notes
        -----
        This function groups the DataFrame 's' by the specified 'level', applies 'func' to each subdataframe, and
        returns the results as a numpy array.

        """
        level = "AgentID"
        s = self.timeseries_slice(time_range)[pars]
        Nt = s.index.unique("Step").size
        s0 = s.index.unique("Step").min() - self.min_tick

        A = None

        for i, (v, ss) in enumerate(s.groupby(level=level)):
            ss = ss.droplevel(level)
            Ai = func(ss, **kwargs)
            if A is None:
                A = self.empty_df(dim3=len(Ai.shape))
            A[s0 : s0 + Nt, i] = Ai
        return A

    def midline_xy_1less(self, mid):
        mid2 = copy.deepcopy(mid[:, :-1, :])
        for i in range(mid.shape[1] - 1):
            mid2[:, i, :] = (mid[:, i, :] + mid[:, i + 1, :]) / 2
        return mid2

    @property
    def midline_seg_xy_data_byID(self):
        g = self.midline_xy_data_byID
        return AttrDict({id: self.midline_xy_1less(mid) for id, mid in g.items()})

    @property
    def midline_seg_orients_data_byID(self):
        g = self.midline_xy_data_byID
        return AttrDict(
            {id: self.midline_seg_orients_from_mid(mid) for id, mid in g.items()}
        )

    def midline_seg_orients_from_mid(self, mid):
        """
        Calculate the orientation of midline segments from midline coordinates.

        Parameters:
        mid (numpy.ndarray): A 3D array of shape (Nticks, N, 2) where Nticks is the number of timesteps,
                             N is the number of midline points, and 2 represents the
                             x and y coordinates of each point.

        Returns:
        numpy.ndarray: A 2D array of shape (Nticks, N-1) containing the orientation angles (in radians)
                       of each segment for each timestep, with values in the range [0, 2π).
        """
        Ax, Ay = mid[:, :, 0], mid[:, :, 1]
        Adx = np.diff(Ax)
        Ady = np.diff(Ay)
        return np.arctan2(Ady, Adx) % (2 * np.pi)

    def comp_freq(self, par, fr_range=(0.0, +np.inf)):
        """
        Compute the frequency of a parameter for each agent.

        This method calculates the dominant frequency of a given parameter for each agent
        in the dataset. It uses the Fast Fourier Transform (FFT) to find the frequency
        with the highest amplitude within a specified frequency range.

        Parameters:
        par (str): The name of the parameter to compute the frequency for.
        fr_range (tuple, optional): A tuple specifying the frequency range to consider.
                                    Defaults to (0.0, +np.inf).

        Returns:
        None: The result is stored in the endpoint dataframe with the frequency name
              as the key.
        """
        self.e[nam.freq(par)] = (
            self.s[par]
            .groupby("AgentID")
            .apply(util.fft_max, dt=self.c.dt, fr_range=fr_range)
        )

    def comp_freqs(self):
        """
        Compute dominant frequencies for translational and angular velocities.
        The frequency ranges (in Hz) are (1.0, 2.5) and (0.1, 0.8) respectively.

        Parameters:
        None

        Returns:
        None
        """
        v = reg.getPar("v")
        if v in self.step_ps:
            self.comp_freq(par=v, fr_range=(1.0, 2.5))
        sv = nam.scal(v)
        if sv in self.step_ps:
            self.comp_freq(par=sv, fr_range=(1.0, 2.5))
        fov = reg.getPar("fov")
        if fov in self.step_ps:
            self.comp_freq(par=fov, fr_range=(0.1, 0.8))

    @valid(required={"config_attrs": ["midline_xy"]}, returned={"ks": ["fo", "ro"]})
    def comp_orientations(self, mode="minimal", recompute=False):
        """
        Compute the orientations of body segments for each timestep, for each agent in the dataset.

        Parameters:
        mode (str): Determines whether to compute only front and rear orientations
                    or one for each body segment.
                    Options are "minimal" (default) or "full".
        recompute (bool): If True, recompute the orientations even if they already exist.
                          Default is False.

        Returns:
        None
        """
        s, e, c = self.data
        all_vecs = list(c.vector_dict.keys())
        vecs = all_vecs[:2] if mode == "minimal" else all_vecs
        pars = nam.orient(vecs)
        if pars.exist_in(s) and not recompute:
            vprint(
                "Vector orientations are already computed. If you want to recompute them, set recompute to True",
                1,
            )
        else:
            mid = self.midline_xy_data
            for vec, par in zip(vecs, pars):
                (idx1, idx2) = c.vector_dict[vec]
                x, y = (
                    mid[:, idx2, 0] - mid[:, idx1, 0],
                    mid[:, idx2, 1] - mid[:, idx1, 1],
                )
                s[par] = np.arctan2(y, x) % 2 * np.pi

        if mode == "full":
            mid = self.midline_xy_data
            s[c.seg_orientations] = self.midline_seg_orients_from_mid(mid)

    def comp_angular(self, is_last=False, **kwargs):
        """
        Perform angular analysis on the dataset.

        This method computes orientations, bends, and angular moments for the dataset.
        If `is_last` is set to True, the results are saved after computation.

        Parameters:
        is_last (bool): Flag to indicate if this is the last computation step. If True, the results are saved.
        **kwargs: Additional keyword arguments passed to the computation methods.

        Returns:
        None
        """
        self.comp_orientations(**kwargs)
        self.comp_bend(**kwargs)
        self.comp_ang_moments(**kwargs)
        if is_last:
            self.save()
        vprint("Angular analysis complete.", 1)

    def comp_bend(self, mode="minimal", recompute=False):
        """
        Compute the body bending angle for each timestep, for each agent in the dataset.

        Parameters:
        mode (str): Determines whether to compute a single angle or one for each intersegmental joint.
                    Options are "minimal" (default) or "full".
        recompute (bool): If True, forces recomputation of the bending angles
                          even if they are already computed. Default is False.

        Raises:
        Exception: If the bending angle computation method specified in the
                   configuration is not recognized.

        Notes:
        - If the bending angles are already computed and recompute is set to False,
          a message will be printed and the function will exit without recomputing.
        - The bending angle can be computed in two ways:
          1. "from_vectors": As the difference between front and rear orientations.
          2. "from_angles": As the sum of the first N front angles, where N is
             specified in the configuration.
        - The computed bending angles are stored in the step dataframe.
        """
        if "bend" in self.step_ps and not recompute:
            vprint(
                "Vector orientations are already computed. If you want to recompute them, set recompute to True",
                1,
            )
        else:
            s, e, c = self.data
            if c.bend == "from_vectors":
                vprint(
                    "Computing bending angle as the difference between front and rear orients"
                )
                fo, ro = nam.orient(["front", "rear"])
                a = np.remainder(s[fo] - s[ro], 2 * np.pi)
                a[a > np.pi] -= 2 * np.pi
            elif c.bend == "from_angles":
                vprint(
                    f"Computing bending angle as the sum of the first {c.Nbend_angles} front angles"
                )
                Ada = np.diff(s[c.seg_orientations]) % (2 * np.pi)
                Ada[Ada > np.pi] -= 2 * np.pi
                a = np.sum(Ada[:, : c.Nbend_angles], axis=1)
                if mode == "full":
                    s[c.angles] = Ada
            else:
                raise

            s["bend"] = a

    def comp_ang_moments(self, pars=None, mode="minimal", recompute=False):
        s, e, c = self.data
        if pars is None:
            ho, to, fo, ro = nam.orient(["head", "tail", "front", "rear"])
            if c.Npoints > 1:
                base_pars = ["bend", ho, to, fo, ro]
                pars = base_pars + c.angles + c.seg_orientations
            else:
                pars = [ho]

        pars = util.existing_cols(util.unique_list(pars), s)

        for p in pars:
            vel = nam.vel(p)
            acc = nam.acc(p)
            # ss = s[p]
            if p.endswith("orientation"):
                p_unw = nam.unwrap(p)
                s[p_unw] = self.apply_per_agent(pars=p, func=util.unwrap_deg).flatten()
                pp = p_unw
            else:
                pp = p
            s[vel] = self.apply_per_agent(pars=pp, func=util.rate, dt=c.dt).flatten()
            s[acc] = self.apply_per_agent(pars=vel, func=util.rate, dt=c.dt).flatten()

            self.comp_operators(pars=[p, vel, acc])

    def comp_xy_moments(self, point="", **kwargs):
        s, e, c = self.data
        xy = nam.xy(point)
        if not xy.exist_in(s):
            return

        dst = nam.dst(point)
        vel = nam.vel(point)
        acc = nam.acc(point)
        cdst = nam.cum(dst)

        sdst = nam.scal(dst)
        svel = nam.scal(vel)
        csdst = nam.cum(sdst)

        s[dst] = self.apply_per_agent(pars=xy, func=util.eudist).flatten()
        s[vel] = s[dst] / c.dt
        s[acc] = self.apply_per_agent(pars=vel, func=util.rate, dt=c.dt).flatten()

        self.scale_to_length(pars=[dst, vel, acc])

        s[cdst] = s[dst].groupby("AgentID").cumsum()
        s[csdst] = s[sdst].groupby("AgentID").cumsum()

        e[cdst] = s[dst].dropna().groupby("AgentID").sum()
        e[nam.mean(vel)] = s[vel].dropna().groupby("AgentID").mean()

        e[csdst] = s[sdst].dropna().groupby("AgentID").sum()
        e[nam.mean(svel)] = s[svel].dropna().groupby("AgentID").mean()

    @valid(required={"ps": ["x", "y", "dst"]})
    def comp_tortuosity(self, dur=20, **kwargs):
        s, e, c = self.data
        p = reg.getPar(f"tor{dur}")
        w = int(dur / c.dt / 2)
        ticks = np.arange(c.Nticks)
        s[p] = self.apply_per_agent(
            pars=["x", "y", "dst"],
            func=util.straightness_index,
            rolling_ticks=util.rolling_window(ticks, w),
            **kwargs,
        ).flatten()
        e[nam.mean(p)] = s[p].groupby("AgentID").mean()
        e[nam.std(p)] = s[p].groupby("AgentID").std()
        vprint("Tortuosity analysis complete.", 1)

    @valid(required={"config_attrs": ["traj_xy"]})
    def comp_dispersal(self, t0=0, t1=60, **kwargs):
        s, e, c = self.data
        p = reg.getPar(f"dsp_{int(t0)}_{int(t1)}")
        s[p] = self.apply_per_agent(
            pars=c.traj_xy,
            func=util.compute_dispersal_solo,
            time_range=(t0, t1),
            **kwargs,
        ).flatten()
        self.scale_to_length(pars=[p])
        sp = nam.scal(p)
        self.comp_operators(pars=[p, sp])
        vprint("Dispersal analysis complete.", 1)

    def comp_operators(self, pars):
        s, e, c = self.data
        for p in pars:
            g = s[p].dropna().groupby("AgentID")
            e[nam.max(p)] = g.max()
            e[nam.mean(p)] = g.mean()
            e[nam.std(p)] = g.std()
            e[nam.initial(p)] = g.first()
            e[nam.final(p)] = g.last()
            e[nam.cum(p)] = g.sum()

    @valid(
        required={"config_attrs": ["contour_xy"]},
        returned={"config_attrs": ["centroid_xy"]},
    )
    def comp_centroid(self, **kwargs):
        c = self.config
        if c.Ncontour > 0:
            self.step_data[c.centroid_xy] = (
                np.sum(self.contour_xy_data, axis=1) / c.Ncontour
            )

    @valid(required={"config_attrs": ["midline_xy"]}, returned={"eks": ["l"]})
    def comp_length(self, mode="minimal", recompute=False):
        if "length" in self.end_ps and not recompute:
            vprint(
                "Length is already computed. If you want to recompute it, set recompute_length to True",
                1,
            )
        else:
            self.step_data["length"] = np.sum(
                np.sum(np.diff(self.midline_xy_data, axis=1) ** 2, axis=2) ** (1 / 2),
                axis=1,
            )
            self.endpoint_data["length"] = (
                self.step_data["length"].groupby("AgentID").quantile(q=0.5)
            )

    def comp_spatial(self, **kwargs):
        s, e, c = self.data
        self.comp_centroid(**kwargs)
        self.comp_length(**kwargs)
        if not c.traj_xy.exist_in(s) and c.point_xy.exist_in(s):
            s[c.traj_xy] = s[c.point_xy]
        self.comp_operators(pars=c.traj_xy)
        for point in ["", "centroid"]:
            self.comp_xy_moments(point, **kwargs)
        vprint("Spatial analysis complete.", 1)

    def scale_to_length(self, pars=None, keys=None):
        s, e, c = self.data
        if "length" not in self.end_ps:
            self.comp_length()
        l = e["length"]
        if pars is None:
            if keys is not None:
                pars = reg.getPar(keys)
            else:
                raise ValueError("No parameter names or keys provided.")
        s_pars = util.existing_cols(pars, s)

        if len(s_pars) > 0:
            ids = s.index.get_level_values("AgentID").values
            ls = l.loc[ids].values
            s[nam.scal(s_pars)] = (s[s_pars].values.T / ls).T
        e_pars = util.existing_cols(pars, e)
        if len(e_pars) > 0:
            e[nam.scal(e_pars)] = (e[e_pars].values.T / l.values).T

    @required(ks=["fo"], config_attrs=["traj_xy"])
    def comp_source_metrics(self):
        s, e, c = self.data
        fo = reg.getPar("fo")
        for n, pos in c.source_xy.items():
            vprint(f"Computing bearing and distance to {n} based on xy position")
            o, d = nam.bearing_to(n), nam.dst_to(n)
            pabs = nam.abs(o)
            temp = np.array(pos) - s[c.traj_xy].values
            s[o] = (
                s[fo] + 180 - np.rad2deg(np.arctan2(temp[:, 1], temp[:, 0]))
            ) % 360 - 180
            s[pabs] = s[o].abs()
            s[d] = util.eudi5x(s[c.traj_xy].values, pos)
            self.comp_operators(pars=[d, pabs])
            if "length" in e.columns:
                l = e["length"]

                def rowIndex(row):
                    return row.name[1]

                def rowLength(row):
                    return l.loc[rowIndex(row)]

                def rowFunc(row):
                    return row[d] / rowLength(row)

                sd = nam.scal(d)
                s[sd] = s.apply(rowFunc, axis=1)
                self.comp_operators(pars=[sd])

            vprint("Bearing and distance to source computed", 1)

    def comp_wind(self):
        w = self.config.env_params.windscape
        if w is not None:
            wo = w.wind_direction
            woo = np.deg2rad(wo)
            try:
                self.comp_wind_metrics(woo, wo)
            except:
                self.comp_final_anemotaxis(woo)

    def comp_wind_metrics(self, woo, wo):
        s, e, c = self.data
        for id in self.ids:
            xy = s[c.traj_xy].xs(id, level="AgentID", drop_level=True).values
            origin = e[[nam.initial("x"), nam.initial("y")]].loc[id]
            d = util.eudi5x(xy, origin)
            dx = xy[:, 0] - origin[0]
            dy = xy[:, 1] - origin[1]
            angs = np.arctan2(dy, dx)
            a = np.array([util.angle_dif(ang, woo) for ang in angs])
            s.loc[(slice(None), id), "anemotaxis"] = d * np.cos(a)
        s[nam.bearing_to("wind")] = s.apply(
            lambda r: util.angle_dif(r[nam.orient("front")], wo), axis=1
        )
        e["anemotaxis"] = s["anemotaxis"].groupby("AgentID").last()

    def comp_final_anemotaxis(self, woo):
        s, e, c = self.data
        xy0 = s[c.traj_xy].groupby("AgentID").first()
        xy1 = s[c.traj_xy].groupby("AgentID").last()
        dx = xy1.values[:, 0] - xy0.values[:, 0]
        dy = xy1.values[:, 1] - xy0.values[:, 1]
        d = np.sqrt(dx**2 + dy**2)
        angs = np.arctan2(dy, dx)
        a = np.array([util.angle_dif(ang, woo) for ang in angs])
        e["anemotaxis"] = d * np.cos(a)

    def comp_PI2(self, xys, x=0.04):
        Nticks = xys.index.unique("Step").size
        ids = xys.index.unique("AgentID").values
        N = len(ids)
        dLR = np.zeros([N, Nticks]) * np.nan
        for i, id in enumerate(ids):
            xy = xys.xs(id, level="AgentID").values
            dL = util.eudi5x(xy, [-x, 0])
            dR = util.eudi5x(xy, [x, 0])
            dLR[i, :] = (dR - dL) / (2 * x)
        dLR_mu = np.mean(dLR, axis=1)
        mu_dLR_mu = np.mean(dLR_mu)
        return mu_dLR_mu

    def comp_PI(self, arena_xdim, xs, return_num=False):
        N = len(xs)
        r = 0.2 * arena_xdim
        xs = np.array(xs)
        N_l = len(xs[xs <= -r / 2])
        N_r = len(xs[xs >= +r / 2])
        # N_m = len(xs[(xs <= +r / 2) & (xs >= -r / 2)])
        pI = np.round((N_l - N_r) / N, 3)
        if return_num:
            return pI, N
        else:
            return pI

    def comp_dataPI(self):
        s, e, c = self.data
        if "x" in self.end_ps:
            xs = e["x"].values
        elif nam.final("x") in self.end_ps:
            xs = e[nam.final("x")].values
        elif "x" in self.step_ps:
            xs = s["x"].dropna().groupby("AgentID").last().values
        elif "centroid_x" in self.step_ps:
            xs = s["centroid_x"].dropna().groupby("AgentID").last().values
        else:
            raise ValueError("No x coordinate found")
        PI, N = self.comp_PI(
            xs=xs, arena_xdim=c.env_params.arena.dims[0], return_num=True
        )
        c.PI = {"PI": PI, "N": N}
        try:
            c.PI2 = self.comp_PI2(xys=s[nam.xy("")])
        except:
            pass

    def process(
        self,
        proc_keys=["angular", "spatial"],
        dsp_starts=[0],
        dsp_stops=[40, 60],
        tor_durs=[5, 10, 20],
        is_last=False,
        **kwargs,
    ):
        if "angular" in proc_keys:
            self.comp_angular()
        if "spatial" in proc_keys:
            self.comp_spatial()
        for t0, t1 in itertools.product(dsp_starts, dsp_stops):
            self.comp_dispersal(t0, t1)
        for dur in tor_durs:
            self.comp_tortuosity(dur)
        if "source" in proc_keys:
            self.comp_source_metrics()
        if "wind" in proc_keys:
            self.comp_wind()
        if "PI" in proc_keys:
            self.comp_dataPI()
        if is_last:
            self.save()

    def get_par(self, par=None, k=None, key="step"):
        if par is None and k is not None:
            par = reg.getPar(k)

        def read_key(key, par):
            res = self.read(key)[par]
            if res is not None:
                return res

        if key == "end" and par in self.end_ps:
            return self.e[par]
        if key == "step" and par in self.step_ps:
            return self.s[par]
        try:
            return read_key(key, par)
        except:
            if k is None:
                k = reg.getPar(p=par, to_return="k")
            return reg.par.get(k=k, d=self, compute=True)

    def sample_larvagroup(self, N=1, ps=[]):
        ps = reg.sample_ps(ps, self.e)
        E = self.e[ps]
        if len(ps) == 0:
            return {}
        elif len(ps) == 1:
            vs = np.atleast_2d(np.random.normal(E.mean(), E.std(), N))
        else:
            base = E.dropna().values.T
            cov = np.cov(base)
            ms = [E[p].mean() for p in ps]
            vs = np.random.multivariate_normal(ms, cov, N).T
        return AttrDict(
            {p: v for p, v in zip(reg.getPar(d=ps, to_return="flatname"), vs)}
        )

    def imitate_larvagroup(self, N=None, ps=None):
        if N is None:
            N = self.c.N
        e = self.e
        ids = random.sample(e.index.values.tolist(), N)
        poss = util.np2Dtotuples(e[reg.getPar(["x0", "y0"])].loc[ids].values)
        try:
            ors = e[reg.getPar("fo0")].loc[ids].values.tolist()
        except:
            ors = np.random.uniform(low=0, high=2 * np.pi, size=len(ids)).tolist()

        if ps is None:
            ps = list(reg.SAMPLING_PARS.keys())
        ps = self.end_ps.existing(ps)
        vs = [
            [e[p].loc[id] if not np.isnan(e[p].loc[id]) else e[p].mean() for id in ids]
            for p in ps
        ]
        dic = AttrDict(
            {p: v for p, v in zip(reg.getPar(d=ps, to_return="flatname"), vs)}
        )
        return ids, poss, ors, dic

    @property
    def existing_dispersion_ranges(self):
        l = util.ItemList(self.step_ps.pref("disp")).split("_")
        return [(int(ll[1]), int(ll[2])) for ll in l]

    def convert_to_pint(self):
        from pint_pandas.pint_array import PintDataFrameAccessor

        s = reg.par.df_to_pint(self.s)
        ss = PintDataFrameAccessor(s)
        self.step_data = ss.dequantify()
        e = reg.par.df_to_pint(self.e)
        ee = PintDataFrameAccessor(e)
        self.endpoint_data = ee.dequantify()


class BaseLarvaDataset(ParamLarvaDataset):
    @staticmethod
    def initGeo(to_Geo: bool = False, **kwargs: Any) -> "BaseLarvaDataset":
        if to_Geo:
            try:
                from importlib import import_module

                GeoLarvaDataset = getattr(
                    import_module("larvaworld.lib.process.dataset_geo"),
                    "GeoLarvaDataset",
                )
                return GeoLarvaDataset(**kwargs)
            except:
                pass
        return LarvaDataset(**kwargs)

    def __init__(
        self,
        dir: Optional[str] = None,
        refID: Optional[str] = None,
        load_data: bool = True,
        config: Optional[AttrDict] = None,
        step: Optional[pd.DataFrame] = None,
        end: Optional[pd.DataFrame] = None,
        agents: Optional[list[str]] = None,
        initialize: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Dataset class that stores a single experiment, real or simulated.
        Metadata and configuration parameters are stored in the 'config' dictionary.
        This can be provided as an argument, retrieved from a stored experiment or generated for a new experiment.

        The default pipeline goes as follows :
        The dataset needs the config file to be initialized. If it is not provided as an argument there are two ways to retrieve it.
        First if "dir" is an existing directory of a stored dataset the config file will be loaded from the default location
        within the dataset's file structure, specifically from a "conf.txt" in the "data" directory under the root "dir".
        As it is not handy to provide an absolute directory as an argument, the root "dir" locations of a number of stored "reference" datasets
        are stored in a file and loaded as a dictionary where the keys are unique "refID" strings holding the root "dir" value.

        Accessing the reference path dictionary is extremely easy through the "reg.stored" registry class with the following methods :
        -   getRefDir(id) returns the "root" directory stored in the "larvaworld/lib/reg/confDicts/Ref.txt" under the unique id
        -   getRef(id=None, dir=None) returns the config dictionary stored at the "root" directory. Accepts either the "dir" path or the "refID" id
        -   loadRef(id) first retrieves the config dictionary and then initializes the dataset.
            By setting load_data=True there is an attempt to load the data from the disc if found at the provided root directory.

        In the case that none of the above attempts yielded a config dictionary, a novel one is generated using any additional keyword arguments are provided.
        This is the default way that a new dataset is initialized. The data content is set after initialization via the "set_data(step, end)"
        method with which we provide the both the step-wise timeseries and the endpoint single-per-agent measurements

        Endpoint measurements are loaded always as a pd.Dataframe 'endpoint_data' with 'AgentID' indexing

        The timeseries data though can be initialized and processed in two ways :
        -   in the default mode  a pd.Dataframe 'step_data' with a 2-level index : 'Step' for the timestep index and 'AgentID' for the agent unique ID.
            Data is stored as a single HDF5 file or as nested dictionaries. The core file is 'data.h5' with keys like 'step' for timeseries and 'end' for endpoint metrics.
        -   in the trajectory mode a "movingpandas.TrajectoryCollection" is adjusted to the needs of the larva-tracking data format via the
            "lib.process.GeoLarvaDataset" class

        Args:
            dir: Path to stored data. Ignored if 'config' is provided. Defaults to None for no storage to disc
            load_data: Whether to load stored data
            config: The metadata dictionary. Defaults to None for attempting to load it from disc or generate a new.
            **kwargs: Any arguments to store in a novel configuration dictionary

        """
        if initialize:
            assert config is None
            kws = {
                "dir": dir,
                "refID": refID,
                # 'config':config,
                **kwargs,
            }
        else:
            if config is None:
                try:
                    config = reg.conf.Ref.getRef(dir=dir, id=refID)
                    config.update(**kwargs)
                except:
                    config = self.generate_config(dir=dir, refID=refID, **kwargs)
            kws = config

        super().__init__(**kws)

        if load_data:
            self.load()
        elif step is not None or end is not None:
            self.set_data(step=step, end=end, agents=agents)

    def generate_config(self, **kwargs):
        c0 = AttrDict(
            {
                "id": "unnamed",
                "group_id": None,
                "refID": None,
                "dir": None,
                "fr": None,
                "dt": None,
                "duration": None,
                "Nsteps": None,
                "Npoints": 3,
                "Ncontour": 0,
                "u": "m",
                "x": "x",
                "y": "y",
                "sample": None,
                "color": None,
                "metric_definition": None,
                "env_params": {},
                "larva_groups": {},
                "source_xy": {},
                "life_history": None,
            }
        )

        c0.update(kwargs)
        if c0.dt is not None:
            c0.fr = 1 / c0.dt
        if c0.fr is not None:
            c0.dt = 1 / c0.fr
        if c0.metric_definition is None:
            c0.metric_definition = SimMetricOps().nestedConf

        points = nam.midline(c0.Npoints, type="point")

        try:
            c0.point = points[c0.metric_definition.spatial.point_idx - 1]
        except:
            c0.point = "centroid"

        if len(c0.larva_groups) == 1:
            c0.group_id, gConf = list(c0.larva_groups.items())[0]
            c0.color = gConf["default_color"]
            c0.sample = gConf["sample"]
            c0.model = gConf["model"]
            c0.life_history = gConf["life_history"]

        vprint(f"Generated new conf {c0.id}", 1)
        return c0

    def delete(self):
        shutil.rmtree(self.config.dir)
        vprint(f"Dataset {self.id} deleted", 2)

    def set_id(self, id, save=True):
        self.id = id
        self.config.id = id
        if save:
            self.save_config()


class LarvaDataset(BaseLarvaDataset):
    def __init__(self, **kwargs: Any) -> None:
        """
        This is the default dataset class. Timeseries are stored as a pd.Dataframe 'step_data' with a 2-level index : 'Step' for the timestep index and 'AgentID' for the agent unique ID.
        Data is stored as a single HDF5 file or as nested dictionaries. The core file is 'data.h5' with keys like 'step' for timeseries and 'end' for endpoint metrics.
        To lesser the burdain of loading and saving all timeseries parameters as columns in a single pd.Dataframe, the most common parameters have been split in a set of groupings,
         available via keys that access specific entries of the "data.h5". The keys of "self.h5_kdic" dictionary store the parameters that every "h5key" keeps :
        -   'contour': The contour xy coordinates,
        -   'midline': The midline xy coordinates,
        -   'epochs': The behavioral epoch detection and annotation,
        -   'base_spatial': The most basic spatial parameters,
        -   'angular': The angular parameters,
        -   'dspNtor':  Dispersal and tortuosity,

        All parameters not included in any of these groups stays with the original "step" key that is always saved and loaded
        """
        super().__init__(**kwargs)

    def visualize(self, parameters={}, **kwargs):
        from ..sim import ReplayRun

        kwargs["dataset"] = self
        rep = ReplayRun(parameters=parameters, **kwargs)
        rep.run()

    def enrich(
        self,
        pre_kws={},
        proc_keys=[],
        anot_keys=[],
        is_last=True,
        mode="minimal",
        recompute=False,
        **kwargs,
    ):
        warnings.filterwarnings("ignore")
        self.preprocess(**pre_kws, recompute=recompute)
        self.process(
            proc_keys=proc_keys, is_last=False, mode=mode, recompute=recompute, **kwargs
        )
        self.annotate(anot_keys=anot_keys, is_last=False, recompute=recompute, **kwargs)

        if is_last:
            self.save()
        return self

    @property
    def epoch_bound_dicts(self):
        d = AttrDict()
        for k, dic in self.epoch_dicts.items():
            try:
                if all([vs.shape.__len__() == 2 for id, vs in dic.items()]):
                    d[k] = dic
            except:
                pass
        return d

    def get_chunk_par(self, chunk, k=None, par=None, min_dur=0, mode="distro"):
        s, e, c = self.data
        epochs = self.epoch_dicts[chunk]
        if min_dur != 0:
            epoch_durs = self.epoch_dicts[f"{chunk}_dur"]
            epochs = AttrDict(
                {id: epochs[id][epoch_durs[id] >= min_dur] for id in self.ids}
            )
        if par is None:
            par = reg.getPar(k)
        grouped = s[par].groupby("AgentID")
        if mode == "extrema":
            c01s = [
                [df.loc[epochs[id][:, 0]].values, df.loc[epochs[id][:, 1]].values]
                for id, df in grouped
                if epochs[id].shape > 0
            ]
            c0s = np.concatenate([c01[0] for c01 in c01s])
            c1s = np.concatenate([c01[1] for c01 in c01s])
            dc01s = c1s - c0s
            return c0s, c1s, dc01s
        elif mode == "distro":

            def get_idx(eps):
                Nepochs = eps.shape[0]
                if Nepochs == 0:
                    idx = []
                elif Nepochs == 1:
                    idx = np.arange(epochs[0][0], epochs[0][1] + 1, 1)
                else:
                    slices = [np.arange(r0, r1 + 1, 1) for r0, r1 in eps]
                    idx = np.concatenate(slices)
                return idx

            return np.concatenate(
                [df.loc[get_idx(epochs[id])].dropna().values for id, df in grouped]
            )


class LarvaDatasetCollection:
    def __init__(
        self,
        labels: Optional[list[str]] = None,
        colors: Optional[list[Any]] = None,
        add_samples: bool = False,
        config: Optional[AttrDict] = None,
        **kwargs: Any,
    ) -> None:
        ds = self.get_datasets(**kwargs)

        for d in ds:
            assert isinstance(d, BaseLarvaDataset)
        if labels is None:
            labels = ds.id

        if add_samples:
            targetIDs = SuperList(ds.config.sample).unique.existing(
                reg.conf.Ref.confIDs
            )
            ds += reg.conf.Ref.loadRefs(ids=targetIDs)
            labels += targetIDs
        self.config = config
        self.datasets = ds
        self.labels = labels
        self.Ndatasets = len(ds)
        if colors is None:
            colors = self.get_colors()
        self.colors = colors
        assert self.Ndatasets == len(self.labels)
        # print(self.labels, self.colors)
        self.group_ids = SuperList(ds.config.group_id).unique
        self.Ngroups = len(self.group_ids)
        self.dir = self.set_dir()

    def set_dir(self, dir=None):
        if dir is not None:
            return dir
        elif self.config and "dir" in self.config:
            return self.config.dir
        elif self.Ndatasets > 1 and self.Ngroups == 1:
            dir0 = util.unique_list(
                [
                    os.path.dirname(os.path.abspath(d.config.dir))
                    for d in self.datasets
                    if d.config.dir is not None
                ]
            )
            if len(dir0) == 1:
                return dir0[0]
            else:
                raise

    @property
    def plot_dir(self):
        return f"{self.dir}/group_plots"

    def plot(self, ids=[], gIDs=[], **kwargs):
        kws = {
            "datasets": self.datasets,
            "save_to": self.plot_dir,
            "show": False,
            "subfolder": None,
        }
        kws.update(**kwargs)
        plots = AttrDict()
        for id in ids:
            plots[id] = reg.graphs.run(id, **kws)
        for gID in gIDs:
            plots[gID] = reg.graphs.run_group(gID, **kws)
        return plots

    def get_datasets(self, datasets=None, refIDs=None, dirs=None, group_id=None):
        if datasets:
            pass
        elif refIDs:
            datasets = reg.conf.Ref.loadRefs(refIDs)
        elif dirs:
            datasets = [
                LarvaDataset(dir=f"{DATA_DIR}/{dir}", load_data=False) for dir in dirs
            ]
        elif group_id:
            datasets = reg.conf.Ref.loadRefGroup(group_id, to_return="list")
        return util.ItemList(datasets)

    def get_colors(self):
        colors = []
        for d in self.datasets:
            color = d.config.color
            while color is None or color in colors:
                color = util.random_colors(1)[0]
            colors.append(color)
        return colors

    @property
    def data_dict(self):
        return dict(zip(self.labels, self.datasets))

    @property
    def data_palette(self):
        return zip(self.labels, self.datasets, self.colors)

    @property
    def data_palette_with_N(self):
        return zip(self.labels_with_N, self.datasets, self.colors)

    @property
    def color_palette(self):
        return dict(zip(self.labels, self.colors))

    @property
    def Nticks(self):
        Nticks_list = [d.config.Nticks for d in self.datasets]
        return int(np.max(util.unique_list(Nticks_list)))

    @property
    def N(self):
        N_list = [d.config.N for d in self.datasets]
        return int(np.max(util.unique_list(N_list)))

    @property
    def labels_with_N(self):
        return [f"{l} (N={d.config.N})" for l, d in self.data_dict.items()]

    @property
    def fr(self):
        fr_list = [d.fr for d in self.datasets]
        return np.max(util.unique_list(fr_list))

    @property
    def dt(self):
        dt_list = util.unique_list([d.dt for d in self.datasets])
        return np.max(dt_list)

    @property
    def duration(self):
        return int(self.Nticks * self.dt)

    @property
    def tlim(self):
        return 0, self.duration

    def trange(self, unit="min"):
        if unit == "min":
            T = 60
        elif unit == "sec":
            T = 1
        t0, t1 = self.tlim
        x = np.linspace(t0 / T, t1 / T, self.Nticks)
        return x

    @property
    def arena_dims(self):
        dims = np.array([d.env_params.arena.dims for d in self.datasets])
        if self.Ndatasets > 1:
            dims = np.max(dims, axis=0)
        else:
            dims = dims[0]
        return tuple(dims)

    @property
    def arena_geometry(self):
        geos = util.unique_list([d.env_params.arena.geometry for d in self.datasets])
        if len(geos) == 1:
            return geos[0]
        else:
            return None

    def concat_data(self, key):
        return util.concat_datasets(dict(zip(self.labels, self.datasets)), key=key)

    @classmethod
    def from_agentpy_output(cls, output=None, agents=None, to_Geo=False):
        """Convert agentpy output to a LarvaDataset"""
        config0 = AttrDict(output.parameters["constants"])
        ds = []
        for gID, df in output.variables.items():
            assert "sample_id" not in df.index.names
            df.index.set_names(["AgentID", "Step"], inplace=True)
            df = df.reorder_levels(order=["Step", "AgentID"], axis=0)
            df.sort_index(level=["Step", "AgentID"], inplace=True)
            config = config0.get_copy()
            kws = {
                # 'larva_groups': {gID: gConf},
                # 'df': df,
                "group_id": config0.id,
                "id": gID,
                "refID": None,
                # 'refID': f'{config0.id}/{gID}',
                "dir": None,
                # 'color': None,
                # 'sample': gConf.sample,
                # 'life_history': gConf.life_history,
                # 'model': gConf.model,
            }
            if "larva_groups" in config0:
                gConf = config0.larva_groups[gID]
                kws.update(
                    **{
                        "larva_group": gConf,
                        # 'df': df,
                        # 'group_id': config0.id,
                        # 'id': gID,
                        # 'refID': None,
                        # 'refID': f'{config0.id}/{gID}',
                        "dir": f"{config0.dir}/data/{gID}",
                        "color": gConf.color,
                        # 'sample': gConf.sample,
                        # 'life_history': gConf.life_history,
                        # 'model': gConf.model,
                    }
                )
            config.update(**kws)
            d = BaseLarvaDataset.initGeo(
                to_Geo=to_Geo,
                load_data=False,
                end=df[config0["collectors"]["end"]].xs(
                    df.index.get_level_values("Step").max(), level="Step"
                ),
                step=df[config0["collectors"]["step"]],
                agents=agents,
                initialize=True,
                **config,
            )

            ds.append(d)

        return cls(datasets=ds, config=config0)


def get_larva_dicts(ls, validIDs=None):
    """Returns the individual-specific dictionaries of the larva datasets"""
    deb_dicts = {}
    nengo_dicts = {}
    bout_dicts = {}
    for id, l in ls.items():
        if validIDs and id not in validIDs:
            continue
        if hasattr(l, "deb") and l.deb is not None:
            deb_dicts[id] = l.deb.finalize_dict()
        try:
            from ..model import NengoBrain

            if isinstance(l.brain, NengoBrain):
                if l.brain.dict is not None:
                    nengo_dicts[id] = l.brain.dict
        except:
            pass
        if l.brain.locomotor.intermitter is not None:
            bout_dicts[id] = l.brain.locomotor.intermitter.build_dict()

    dic0 = AttrDict(
        {
            "deb": deb_dicts,
            "nengo": nengo_dicts,
            "bouts": bout_dicts,
        }
    )

    return AttrDict({k: v for k, v in dic0.items() if len(v) > 0})
