"""
Larvaworld parameter database
"""

from __future__ import annotations
from typing import Any, Optional

import numpy as np
import param

from .. import reg, util, funcs
from ..util import AttrDict, SuperList, nam

__all__: list[str] = [
    "output_keys",
    "output_dict",
    "ParamClass",
    "ParamRegistry",
]

#: Dictionary mapping module names to their output parameters.
#:
#: Each entry contains 'step' (per-timestep) and 'endpoint' (final) outputs.
#:
#: Example:
#:     >>> output_dict['feeder']['step']
#:     ['l', 'f_am', 'EEB', 'on_food', ...]
output_dict: AttrDict = AttrDict(
    {
        "olfactor": {
            "step": ["c_odor1", "dc_odor1", "c_odor2", "dc_odor2", "A_olf"],
            "endpoint": [],
        },
        "loco": {
            "step": [
                "A_CT",
                "A0_CT",
                "Amax_CT",
                "A_T",
                "I_T",
                "phi_T",
                "A_C",
                "I_C",
                "phi_C",
                "phi_Amax_CT",
            ],
            "endpoint": [],
        },
        "thermo": {
            "step": ["temp_W", "dtemp_W", "temp_C", "dtemp_C", "A_therm"],
            "endpoint": [],
        },
        "toucher": {"step": ["on_food_tr", "on_food"], "endpoint": ["on_food_tr"]},
        "wind": {"step": ["A_wind"], "endpoint": []},
        "feeder": {
            "step": [
                "l",
                "f_am",
                "EEB",
                "on_food",
                "fee_reocc",
                "beh",
                "phi_F",
                "A_F",
                "I_F",
            ],
            "endpoint": [
                "l",
                "f_am",
                "on_food_tr",
                "pau_N",
                "str_N",
                "run_N",
                "fee_N",
                "str_c_N",
                "fee_c_N",
                "fee_N_success",
                "fee_N_fail",
            ],
        },
        "gut": {
            "step": [
                "sf_am_Vg",
                "sf_am_V",
                "f_am_V",
                "sf_am_A",
                "sf_am_M",
                "sf_abs_M",
                "f_abs_M",
                "sf_faeces_M",
                "f_faeces_M",
                "f_am",
            ],
            "endpoint": [
                "sf_am_Vg",
                "sf_am_V",
                "sf_am_A",
                "sf_am_M",
                "sf_abs_M",
                "f_abs_M",
                "sf_faeces_M",
                "f_faeces_M",
                "f_am",
            ],
        },
        "pose": {"step": ["x", "y", "b", "fo", "ro"], "endpoint": ["l", "cum_t", "x"]},
        "memory": {"step": [], "endpoint": []},
        "midline": nam.midline_xy(3, flat=True),
        "contour": nam.contour_xy(0, flat=True),
    }
)

ks = ["loco", "olfactor", "feeder", "thermo", "wind", "memory", "toucher"]
output_dict.brain = AttrDict(
    {
        "step": SuperList([output_dict[k].step for k in ks]).flatten.unique,
        "endpoint": SuperList([output_dict[k].endpoint for k in ks]).flatten.unique,
    }
)

#: List of all output module names (keys from output_dict).
#:
#: Example:
#:     >>> output_keys
#:     ['olfactor', 'loco', 'thermo', 'toucher', 'wind', 'feeder', 'gut', ...]
output_keys: list[str] = list(output_dict.keys())


class ParamClass:
    """
    Parameter database for storing and managing Larvaworld parameters.

    Provides a comprehensive database of all parameters used in larvaworld simulations,
    including physical parameters, behavioral parameters, and configuration options.
    Supports parameter lookup, unit management, and symbolic notation.

    Attributes:
        dict: Dictionary of parameter definitions
        ks: List of parameter keys
        kdict: Flattened parameter dictionary for quick lookup

    Example:
        >>> pclass = ParamClass()
        >>> pclass.dict['length']
        {'p': 'body_length', 'k': 'l', 'd': 'Body length', ...}
    """

    def __init__(self) -> None:
        self.func_dict = funcs.param_computing
        self.k_ops = AttrDict(
            {
                "mean": ["bar", "_mu"],
                "std": ["tilde", "std"],
                "var": ["tilde", "var"],
                "min": ["sub", "min"],
                "max": ["sub", "max"],
                "final": ["sub", "_fin"],
                "initial": ["sub", "0"],
                "cum": ["sub", "cum"],
            }
        )
        self.build()

    def update_kdict(self, ks: list[str]) -> None:
        """
        Update the kdict with the parameters in the ks list
        """
        for k in ks:
            if k in self.ks and k not in self.kdict:
                self.kdict[k] = reg.get_LarvaworldParam(**self.dict[k])

    def finalize(self) -> None:
        """
        Finalize the parameter database
        """
        for k, prepar in self.dict.items():
            self.kdict[k] = reg.get_LarvaworldParam(**prepar)

    @property
    def dkeys(self) -> SuperList:
        return SuperList([p.d for k, p in self.dict.items()]).sorted

    @property
    def pkeys(self) -> SuperList:
        return SuperList([p.p for k, p in self.dict.items()]).sorted

    @property
    def ks(self) -> SuperList:
        return SuperList(self.dict.keys()).sorted

    def build(self) -> None:
        self.dict = AttrDict()
        self.kdict = AttrDict()
        self.build_initial()
        self.build_angular()
        self.build_spatial()
        self.build_chunks()
        self.build_sim_pars()
        self.build_deb_pars()
        self.p2k_dict = AttrDict({p.p: p.k for k, p in self.dict.items()})
        self.d2k_dict = AttrDict({p.d: p.k for k, p in self.dict.items()})

    def add(self, **kwargs: Any) -> None:
        prepar = reg.prepare_LarvaworldParam(**kwargs)
        self.dict[prepar.k] = prepar

    def build_initial(self) -> None:
        kws1 = {
            "vfunc": param.Number,
            "lim": (0.0, None),
            "dtype": float,
            "u": reg.units.s,
        }
        self.add(**{"p": "t", "sym": "$t$", "v0": 0.0, **kws1})
        self.add_operators(k0="t")
        self.add(**{"p": "model.dt", "d": "dt", "sym": "$dt$", "v0": 0.1, **kws1})
        self.add(
            **{
                "p": "cum_dur",
                "k": nam.cum("t"),
                "sym": nam.tex.sub("t", "cum"),
                "v0": 0.0,
                **kws1,
            }
        )

        kws2 = {
            "vfunc": param.Integer,
            "lim": (0, None),
            "v0": 0,
            "dtype": int,
            "u": reg.units.dimensionless,
        }
        self.add(**{"p": "num_ts", "k": "N_ts", "sym": nam.tex.sub("N", "ts"), **kws2})
        self.add(**{"p": "tick", "sym": "$tick$", **kws2})
        self.add(
            **{
                "p": "num_ticks",
                "k": "N_ticks",
                "sym": nam.tex.sub("N", "ticks"),
                **kws2,
            }
        )

        self.add_operators(k0="tick")

    def add_rate(
        self,
        k0: Optional[str] = None,
        k_time: str = "t",
        p: Optional[str] = None,
        k: Optional[str] = None,
        d: Optional[str] = None,
        sym: Optional[str] = None,
        k_num: Optional[str] = None,
        k_den: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if k0 is not None:
            b = self.dict[k0]
            if p is None:
                p = f"d_{k0}"
            if k is None:
                k = f"d_{k0}"
            if d is None:
                d = f"{b.d} rate"
            if sym is None:
                sym = nam.tex.dot(b.sym)
            if k_num is None:
                k_num = f"D_{k0}"
        if k_den is None:
            k_den = f"D_{k_time}"

        b_num = self.dict[k_num]
        b_den = self.dict[k_den]

        kws = {
            "p": p,
            "k": k,
            "d": d,
            "sym": sym,
            "u": b_num.u / b_den.u,
            "vfunc": param.Number,
            "required_ks": [k_num, k_den],
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_operators(self, k0: str) -> None:
        b = self.dict[k0]
        kws0 = {"u": b.u, "required_ks": [k0]}

        def cum_disp(k0):
            if k0 == "d":
                disp = "pathlength"
            elif k0 == "sd":
                disp = "scaled pathlength"
            else:
                disp = f"total {b.disp}"
            return disp

        for k, (tex, k2) in self.k_ops.items():
            f = nam[k]

            kws = {
                "d": f(b.d),
                "p": f(b.p),
                "sym": nam.tex[tex](b.sym) if tex != "sub" else nam.tex.sub(b.sym, k2),
                "disp": f"{k} {b.disp}" if k != "cum" else cum_disp(k0),
                "func": self.func_dict[k](b.d),
                "k": f"{b.k}{k2}" if k in ["mean", "final", "initial"] else f(b.k),
            }
            self.add(**kws, **kws0)

    def add_chunk(
        self, pc: str, kc: str, func: Any = None, required_ks: list[str] = []
    ) -> None:
        f_kws = {"func": func, "required_ks": required_ks}

        ptr = nam.dur_ratio(pc)
        pl = nam.length(pc)
        pN = nam.num(pc)
        pN_mu = nam.mean(pN)
        ktr = f"{kc}_tr"
        kl = f"{kc}_l"
        kN = f"{kc}_N"
        kN_mu = f"{kN}_mu"
        kt = f"{kc}_t"
        self.add(p=pc, k=kc, sym=f"${kc}$")
        self.add(
            p=ptr,
            k=ktr,
            sym=nam.tex.sub("r", kc),
            disp=f"time fraction in {pc}s",
            vfunc=param.Magnitude,
            required_ks=[nam.cum(nam.dur(pc)), nam.cum(nam.dur(""))],
            func=self.func_dict.tr(pc),
        )
        self.add(
            p=pN,
            k=kN,
            sym=nam.tex.sub("N", f"{pc}s"),
            disp=f"# {pc}s",
            vfunc=param.Integer,
            codename=f"brain.locomotor.intermitter.N{pc}s",
            dtype=int,
            **f_kws,
        )
        self.add(
            p=nam.dur(pc),
            k=kt,
            sym=nam.tex.sub(nam.tex.Delta("t"), kc),
            disp=f"{pc} duration",
            vfunc=param.Number,
            u=reg.units.s,
            **f_kws,
        )

        for ii in ["on", "off"]:
            self.add(p=f"{pN_mu}_{ii}_food", k=f"{kN_mu}_{ii}_food")
            self.add(p=f"{ptr}_{ii}_food", k=f"{ktr}_{ii}_food", vfunc=param.Magnitude)

        self.add_rate(
            k_num=kN,
            k_den=nam.cum("t"),
            k=kN_mu,
            p=pN_mu,
            sym=nam.tex.bar(kN),
            disp=f"avg. # {pc}s per sec",
            func=func,
        )
        self.add_operators(k0=kt)

        if str.endswith(pc, "chain"):
            self.add(
                **{"p": pl, "k": kl, "sym": nam.tex.sub("l", kc), "dtype": int, **f_kws}
            )
            self.add_operators(k0=kl)

    def add_chunk_track(self, kc: str, k: str, pc: Optional[str] = None) -> None:
        if pc is None:
            pc = self.dict[kc].p
        # bc = self.dict[kc]
        b = self.dict[k]
        kws = {"func": self.func_dict.track_par(pc, b.p), "u": b.u}
        k01 = f"{kc}_{k}"
        p0, p1, pdp = nam.atStartStopChunk(b.p, pc)

        kws0 = {
            "p": p0,
            "k": f"{kc}_{k}0",
            "disp": f"{b.disp} at {pc} start",
            "sym": nam.tex.subsup(b.sym, kc, "0"),
            **kws,
        }
        kws1 = {
            "p": p1,
            "k": f"{kc}_{k}1",
            "disp": f"{b.disp} at {pc} stop",
            "sym": nam.tex.subsup(b.sym, kc, "1"),
            **kws,
        }

        kws01 = {
            "p": pdp,
            "k": k01,
            "disp": f"{b.disp} during {pc}s",
            "sym": nam.tex.sub(nam.tex.Delta(b.sym), kc),
            **kws,
        }
        if kws01["k"] == "tur_fou":
            kws01["disp"] = "turn amplitude"

        self.add(**kws0)
        self.add(**kws1)
        self.add(**kws01)
        self.add_operators(k0=k01)

    def add_velNacc(
        self,
        k0: str,
        p_v: Optional[str] = None,
        k_v: Optional[str] = None,
        d_v: Optional[str] = None,
        sym_v: Optional[str] = None,
        disp_v: Optional[str] = None,
        p_a: Optional[str] = None,
        k_a: Optional[str] = None,
        d_a: Optional[str] = None,
        sym_a: Optional[str] = None,
        disp_a: Optional[str] = None,
        func_v: Any = None,
    ) -> None:
        b = self.dict[k0]
        b_dt = self.dict["dt"]
        if p_v is None:
            p_v = nam.vel(b.p)
        if p_a is None:
            p_a = nam.acc(b.p)
        if d_v is None:
            d_v = nam.vel(b.d)
        if d_a is None:
            d_a = nam.acc(b.d)
        if k_v is None:
            k_v = f"{b.k}v"
        if k_a is None:
            k_a = f"{b.k}a"
        if sym_v is None:
            sym_v = nam.tex.dot(b.sym)
        if sym_a is None:
            sym_a = nam.tex.dot(sym_v)

        if func_v is None:

            def func_v(d):
                s, e, c = d.data
                s[d_v] = util.apply_per_level(s[b.d], util.rate, dt=c.dt).flatten()

        self.add(
            **{
                "p": p_v,
                "k": k_v,
                "d": d_v,
                "u": b.u / b_dt.u,
                "sym": sym_v,
                "disp": disp_v,
                "required_ks": [k0],
                "func": func_v,
            }
        )

        def func_a(d):
            s, e, c = d.data
            s[d_a] = util.apply_per_level(s[d_v], util.rate, dt=c.dt).flatten()

        self.add(
            **{
                "p": p_a,
                "k": k_a,
                "d": d_a,
                "u": b.u / b_dt.u**2,
                "sym": sym_a,
                "disp": disp_a,
                "required_ks": [k_v],
                "func": func_a,
            }
        )

    def add_scaled(self, k0: str, **kwargs: Any) -> None:
        b = self.dict[k0]
        b_l = self.dict["l"]

        def func(d):
            d.scale_to_length(pars=[b.d])

        kws = {
            "p": nam.scal(b.p),
            "k": f"s{k0}",
            "d": nam.scal(b.d),
            "u": b.u / b_l.u,
            "sym": nam.tex.mathring(b.sym),
            "disp": f"scaled {b.disp}",
            "vfunc": param.Number,
            "required_ks": [k0],
            "func": func,
        }

        kws.update(kwargs)
        self.add(**kws)

    def add_unwrap(self, k0: str, **kwargs: Any) -> None:
        b = self.dict[k0]

        kws = {
            "p": nam.unwrap(b.p),
            "d": nam.unwrap(b.d),
            "k": f"{b.k}u",
            "u": b.u,
            "sym": b.sym,
            "disp": b.disp,
            "lim": None,
            "required_ks": [k0],
            "dv": b.dv,
            "v0": b.v0,
            "vfunc": param.Number,
            "func": self.func_dict.unwrap(b.d, in_deg=False),
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_dst(self, point: str = "", **kwargs: Any) -> None:
        xd, yd = nam.xy(point)
        xk, bx = [(k, p) for k, p in self.dict.items() if p.d == xd][0]
        yk, by = [(k, p) for k, p in self.dict.items() if p.d == yd][0]

        if bx.u == by.u:
            u = bx.u
        else:
            raise
        if bx.dv == by.dv:
            dv = bx.dv
        else:
            raise

        kws = {
            "p": nam.dst(point),
            "d": nam.dst(point),
            "k": nam.d(point),
            "u": u,
            "sym": nam.tex.sub("d", point),
            "disp": f"{point} distance",
            "lim": (0.0, None),
            "required_ks": [xk, yk],
            "dv": dv,
            "v0": 0.0,
            "vfunc": param.Number,
            "func": self.func_dict.dst(point=point),
        }
        kws.update(kwargs)

        self.add(**kws)

    def add_freq(self, k0: str, **kwargs: Any) -> None:
        b = self.dict[k0]
        kws = {
            "p": nam.freq(b.p),
            "d": nam.freq(b.d),
            "k": f"f{b.k}",
            "u": reg.units.Hz,
            "sym": nam.tex.sub(b.sym, "freq"),
            "disp": f"{b.disp} frequency",
            "vfunc": param.Number,
            "required_ks": [k0],
            "func": self.func_dict.freq(b.d),
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_phi(self, k0: str, **kwargs: Any) -> None:
        b = self.dict[k0]
        kws = {
            "p": nam.phi(b.p),
            "d": nam.phi(b.d),
            "k": f"phi_{b.k}",
            "u": reg.units.rad,
            "sym": nam.tex.sub("Phi", b.sym),
            "disp": f"{b.disp} phase",
            "lim": (0, 2 * np.pi),
            "vfunc": param.Number,
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_dsp(self, range: tuple[int, int] = (0, 40)) -> None:
        a = "dispersion"
        k0 = "dsp"
        s0 = nam.tex.circledast("d")
        r0, r1 = range
        dur = int(r1 - r0)
        p = f"{a}_{r0}_{r1}"
        k = f"{k0}_{r0}_{r1}"
        self.add(
            p=p,
            k=k,
            u=reg.units.m,
            sym=nam.tex.subsup(s0, f"{r0}", f"{r1}"),
            disp=f'dispersal in {dur}"',
            vfunc=param.Number,
            func=self.func_dict.dsp(range),
            required_ks=["x", "y"],
        )
        self.add_scaled(k0=k)
        self.add_operators(k0=k)
        self.add_operators(k0=f"s{k}")

    def add_tor(self, dur: int) -> None:
        p0 = "tortuosity"
        k0 = "tor"
        k = f"{k0}{dur}"
        self.add(
            p=f"{p0}_{dur}",
            k=k,
            sym=nam.tex.sub(k0, str(dur)),
            disp=f"{p0} over {dur}''",
            vfunc=param.Magnitude,
            func=self.func_dict.tor(dur),
        )
        self.add_operators(k0=k)

    def build_angular(self) -> None:
        kws = {
            "dv": np.round(np.pi / 180, 2),
            "u": reg.units.rad,
            "v0": 0.0,
            "vfunc": param.Number,
            "dtype": float,
        }
        self.add(
            **{
                "p": "bend",
                "codename": "body_bend",
                "k": "b",
                "sym": nam.tex.theta("b", sep="_"),
                "disp": "bending angle",
                "lim": (-np.pi, np.pi),
                **kws,
            }
        )
        self.add_velNacc(
            k0="b",
            sym_v=nam.tex.omega("b", sep="_"),
            disp_v="bending angular velocity",
            disp_a="bending angular acceleration",
        )

        angs = [
            ["f", "front", "", ""],
            ["r", "rear", "r", "rear "],
            ["h", "head", "h", "head "],
            ["t", "tail", "t", "tail "],
        ]

        for suf, psuf, ksuf, lsuf in angs:
            p0 = nam.orient(psuf)
            p_v, p_a = nam.vel(p0), nam.acc(p0)
            ko = f"{suf}o"
            kou = f"{ko}u"
            self.add(
                **{
                    "p": p0,
                    "k": ko,
                    "sym": nam.tex.theta(ksuf, sep="_"),
                    "disp": f"{lsuf}orientation",
                    "lim": (0, 2 * np.pi),
                    **kws,
                }
            )

            self.add_unwrap(k0=ko)

            self.add_velNacc(
                k0=kou,
                k_v=f"{suf}ov",
                k_a=f"{suf}oa",
                p_v=p_v,
                d_v=p_v,
                p_a=p_a,
                d_a=p_a,
                sym_v=nam.tex.omega(ksuf, sep="_"),
                disp_v=f"{lsuf}angular velocity",
                disp_a=f"{lsuf}angular acceleration",
            )
        for k0 in ["b", "bv", "ba", "fov", "foa", "rov", "roa", "fo", "ro", "ho", "to"]:
            self.add_freq(k0=k0)
            self.add_operators(k0=k0)

    def build_spatial(self) -> None:
        kws = {"u": reg.units.m, "vfunc": param.Number}
        self.add(
            **{
                "p": "length",
                "k": "l",
                "disp": "body length",
                "flatname": "body.length",
                "sym": "$l$",
                "v0": 0.004,
                "lim": (0.0005, 0.01),
                "dv": 0.0005,
                **kws,
            }
        )
        self.add_freq(k0="l")
        self.add_operators(k0="l")

        for point in ["", "centroid"]:
            px, py = nam.xy(point)
            self.add(
                **{"p": px, "disp": f"{point} X position", "sym": f"{point} X", **kws}
            )
            self.add(
                **{"p": py, "disp": f"{point} Y position", "sym": f"{point} Y", **kws}
            )
            self.add_dst(point=point)
            d_d, d_v, d_a = nam.dst(point), nam.vel(point), nam.acc(point)
            k_d, k_v, k_a = nam.d(point), nam.v(point), nam.a(point)
            d_sd, d_sv, d_sa = nam.scal([d_d, d_v, d_a])
            k_sd, k_sv, k_sa = f"s{k_d}", f"s{k_v}", f"s{k_a}"
            self.add_velNacc(
                k0=k_d,
                k_v=k_v,
                k_a=k_a,
                p_v=d_v,
                d_v=d_v,
                p_a=d_a,
                d_a=d_a,
                sym_v=k_v,
                disp_v=f"{point} crawling speed",
                disp_a=f"{point} crawling acceleration",
                func_v=self.func_dict.vel(d_d, d_v),
            )
            for k0 in [px, py, k_d]:
                self.add_scaled(k0=k0)
            self.add_velNacc(
                k0=k_sd,
                k_v=k_sv,
                k_a=k_sa,
                p_v=d_sv,
                d_v=d_sv,
                p_a=d_sa,
                d_a=d_sa,
                sym_v=nam.tex.mathring(k_v),
                disp_v=f"scaled {point} crawling speed",
                disp_a=f"scaled {point} crawling acceleration",
                func_v=self.func_dict.vel(d_sd, d_sv),
            )
            for k0 in [k_d, k_v, k_a, k_sd, k_sv, k_sa, px, py]:
                self.add_freq(k0=k0)
                self.add_operators(k0=k0)
                if k0 in [k_v, k_a, k_sv, k_sa]:
                    for k0_ext in [f"{k0}_min", f"{k0}_max"]:
                        self.add_phi(k0=k0_ext)
            for k0 in [nam.cum(k_d)]:
                self.add_scaled(k0=k0)

        self.add(
            **{
                "p": "dispersion",
                "k": "dsp",
                "sym": nam.tex.circledast("d"),
                "disp": "dispersal",
                **kws,
            }
        )

        for i in [
            (0, 40),
            (0, 60),
            (0, 80),
            (10, 60),
            (10, 80),
            (20, 60),
            (20, 80),
            (20, 100),
            (0, 120),
            (0, 240),
            (60, 120),
        ]:
            self.add_dsp(range=i)
        self.add(
            **{"p": "tortuosity", "k": "tor", "vfunc": param.Magnitude, "sym": "tor"}
        )
        for dur in [1, 2, 5, 10, 20, 60]:
            self.add_tor(dur=dur)
        self.add(**{"p": "anemotaxis", "sym": "anemotaxis"})

    def build_chunks(self) -> None:
        d0 = {
            "str": "stride",
            "pau": "pause",
            "run": "run",
            "fee": "feed",
            "tur": "turn",
            "Ltur": "Lturn",
            "Rtur": "Rturn",
            # 'exec': 'exec',
            "str_c": nam.chain("stride"),
            "fee_c": nam.chain("feed"),
            "on_food": "on_food",
            # 'off_food' : 'off_food'
        }
        for kc, pc in d0.items():
            temp = self.func_dict.chunk(kc)
            # func = temp.func
            # required_ks = temp.required_ks

            self.add_chunk(pc=pc, kc=kc, func=temp.func, required_ks=temp.required_ks)
            for k in [
                "fov",
                "rov",
                "x",
                "y",
                "fo",
                "fou",
                "ro",
                "rou",
                "b",
                "bv",
                "v",
                "sv",
                # 'a','sa','foa', 'roa','ba',
                "d",
                "sd",
            ]:
                self.add_chunk_track(kc=kc, k=k, pc=pc)
            self.add(p=f"handedness_score_{kc}", k=f"tur_H_{kc}")
            if kc == "fee":
                self.add_freq(k0=kc)

    def build_sim_pars(self) -> None:
        L = "brain.locomotor"
        IF = f"{L}.interference"
        Im = f"{L}.intermitter"

        for ii, jj in zip(["C", "T", "F"], ["crawler", "turner", "feeder"]):
            self.add(
                **{
                    "p": f"{L}.{jj}.output",
                    "k": f"A_{ii}",
                    "d": f"{jj} output",
                    "sym": nam.tex.sub("A", ii),
                }
            )
            self.add(
                **{
                    "p": f"{L}.{jj}.input",
                    "k": f"I_{ii}",
                    "d": f"{jj} input",
                    "sym": nam.tex.sub("I", ii),
                }
            )
            self.add(
                **{
                    "p": f"{L}.{jj}.phi",
                    "k": f"phi_{ii}",
                    "d": f"{jj} phase",
                    "sym": nam.tex.sub("Phi", ii),
                    "u": reg.units.rad,
                }
            )
        self.add(
            **{
                "p": "cur_attenuation",
                "codename": f"{IF}.cur_attenuation",
                "k": "A_CT",
                "l": "C->T suppression",
                "sym": nam.tex.sub("A", "CT"),
                "disp": "CRAWLER:TURNER interference suppression.",
            }
        )
        self.add(
            **{
                "p": "attenuation",
                "codename": f"{IF}.attenuation",
                "k": "A0_CT",
                "l": "C->T baseline suppression",
                "sym": nam.tex.sub("A0", "CT"),
                "disp": "CRAWLER:TURNER baseline interference suppression.",
            }
        )
        self.add(
            **{
                "p": nam.max("attenuation"),
                "codename": f"{IF}.attenuation_max",
                "k": "Amax_CT",
                "l": "C->T max suppression",
                "sym": nam.tex.sub("Amax", "CT"),
                "disp": "CRAWLER:TURNER maximum interference suppression.",
            }
        )
        self.add(
            **{
                "p": nam.phi(nam.max("attenuation")),
                "codename": f"{IF}.max_attenuation_phase",
                "k": "phi_Amax_CT",
                "u": reg.units.rad,
                "l": "C->T max suppression phase",
                "sym": nam.tex.sub("Phi_Amax", "CT"),
                "disp": "CRAWLER:TURNER maximum interference suppression phase.",
            }
        )
        # self.add(**{'p': 'brain.locomotor.cur_ang_suppression', 'k': 'c_CT', 'd': 'ang_suppression',
        #             'disp': 'angular suppression output', 'sym': sub('c', 'CT'), 'lim': (0.0, 1.0)})

        self.add(
            **{
                "p": f"{Im}.EEB",
                "k": "EEB",
                "d": "exploitVSexplore_balance",
                "lim": (0.0, 1.0),
                "sym": "EEB",
            }
        )
        self.add(
            **{
                "p": f"{Im}.feeder_reoccurence_rate",
                "k": "fee_reocc",
                "d": "feeder_reoccurence_rate",
                "lim": (0.0, 1.0),
                "sym": "fee_reocc",
            }
        )
        self.add(
            **{
                "p": f"{Im}.cur_state",
                "k": "beh",
                "d": "behavioral_state",
                "vs": ["exec", "pause", "feed"],
                "sym": "beh",
            }
        )
        self.add(
            **{
                "p": f"{Im}.Nfeeds_success",
                "k": "fee_N_success",
                "d": "successful_feeds",
                "dtype": int,
                "disp": "# successful feeds",
                "sym": "fee_N_success",
            }
        )
        self.add(
            **{
                "p": f"{Im}.Nfeeds_fail",
                "k": "fee_N_fail",
                "d": "failed_feeds",
                "dtype": int,
                "disp": "# failed feeds",
                "sym": "fee_N_fail",
            }
        )

        for ii, jj in zip(["1", "2"], ["first", "second"]):
            k = f"c_odor{ii}"
            dk = f"d{k}"
            sym = nam.tex.subsup("C", "odor", ii)
            dsym = nam.tex.subsup(nam.tex.delta("C"), "odor", ii)
            ddisp = f'{sym} sensed (C/{nam.tex.sub("C", "0")} - 1)'
            self.add(
                **{
                    "p": f"brain.olfactor.{jj}_odor_concentration",
                    "k": k,
                    "d": k,
                    "disp": sym,
                    "sym": sym,
                    "u": reg.units.micromol,
                }
            )
            self.add(
                **{
                    "p": f"brain.olfactor.{jj}_odor_concentration_change",
                    "k": dk,
                    "d": dk,
                    "disp": ddisp,
                    "sym": dsym,
                }
            )

        for ii, jj in zip(["W", "C"], ["warm", "cool"]):
            k = f"temp_{ii}"
            dk = f"d{k}"

            self.add(
                **{
                    "p": f"brain.thermosensor.{jj}_sensor_input",
                    "k": k,
                    "d": k,
                    "disp": f"{jj} sensor input",
                    "sym": nam.tex.sub("Temp", ii),
                }
            )
            self.add(
                **{
                    "p": f"brain.thermosensor.{jj}_sensor_perception",
                    "k": dk,
                    "d": dk,
                    "lim": (-0.1, 0.1),
                    "disp": f"{jj} sensor perception",
                    "sym": nam.tex.sub(nam.tex.Delta("Temp"), ii),
                }
            )

        for ii, jj in zip(
            ["olf", "tou", "wind", "therm"],
            ["olfactor", "toucher", "windsensor", "thermosensor"],
        ):
            self.add(
                **{
                    "p": f"brain.{jj}.output",
                    "k": f"A_{ii}",
                    "d": f"{jj} output",
                    "lim": (0.0, 1.0),
                    "sym": nam.tex.sub("A", ii),
                }
            )

        self.add_rate(
            k_num="Ltur_N",
            k_den="tur_N",
            k="tur_H",
            p="handedness_score",
            disp=f'handedness score ({nam.tex.sub("N", "Lturns")} / {nam.tex.sub("N", "turns")})',
            sym=nam.tex.sub("H", "tur"),
            lim=(0.0, 1.0),
        )
        for ii in ["on", "off"]:
            k = f"{ii}_food"
            self.add(**{"p": k, "k": k, "dtype": bool})
            self.add(**{"p": nam.dur(k), "k": f"{k}_t", "disp": f"time {ii} food"})
            self.add(
                **{
                    "p": nam.cum(nam.dur(k)),
                    "k": nam.cum(f"{k}_t"),
                    "disp": f"total time {ii} food",
                }
            )
            self.add(
                **{
                    "p": nam.dur_ratio(k),
                    "k": f"{k}_tr",
                    "lim": (0.0, 1.0),
                    "disp": f"time fraction {ii} food",
                }
            )
            self.add(
                **{
                    "p": f"handedness_score_{k}",
                    "k": f"tur_H_{k}",
                    "disp": f"handedness score {ii} food",
                }
            )
            for kk in [
                "fov",
                "rov",
                "foa",
                "roa",
                "x",
                "y",
                "fo",
                "fou",
                "ro",
                "rou",
                "b",
                "bv",
                "ba",
                "v",
                "sv",
                "a",
                "v_mu",
                "sv_mu",
                "sa",
                "d",
                "sd",
            ]:
                b = self.dict[kk]
                k0 = f"{kk}_{k}"
                p0 = f"{b.p}_{k}"
                self.add(**{"p": p0, "k": k0, "disp": f"{b.disp} {ii} food"})

    def build_deb_pars(self) -> None:
        ks = ["f_am", "sf_am_Vg", "f_am_V", "sf_am_V", "sf_am_A", "sf_am_M"]
        ps = [
            "amount_eaten",
            "deb.ingested_gut_volume_ratio",
            "deb.volume_ingested",
            "deb.ingested_body_volume_ratio",
            "deb.ingested_body_area_ratio",
            "deb.ingested_body_mass_ratio",
        ]
        ds = [
            "amount_eaten",
            "ingested_gut_volume_ratio",
            "ingested_volume",
            "ingested_body_volume_ratio",
            "ingested_body_area_ratio",
            "ingested_body_mass_ratio",
        ]
        disps = [
            "food consumed",
            "ingested food as gut volume fraction",
            "ingested food volume",
            "ingested food as body volume fraction",
            "ingested food as body area fraction",
            "ingested food as body mass fraction",
        ]
        for k, p, d, disp in zip(ks, ps, ds, disps):
            self.add(**{"p": p, "k": k, "d": d, "disp": disp})


class ParamRegistry(ParamClass):
    """
    Extended parameter registry with computation and display functionality.

    Extends ParamClass with methods for parameter computation from datasets,
    performance index (PI) tracking, and parameter display utilities. Used
    as the central registry for all larvaworld parameter operations.

    Attributes:
        PI: AttrDict for storing Performance Index calculations

    Example:
        >>> preg = ParamRegistry()
        >>> preg.get('l', dataset)  # Get body length parameter
        >>> preg.compute(ks=['vel', 'acc'], d=dataset)
    """

    def __init__(self) -> None:
        super().__init__()
        self.PI = AttrDict()

    def get(self, k: str, d: Any, compute: bool = True):
        if k not in self.ks:
            raise ValueError(f'parameter key "{k}" not in database')
        self.update_kdict(ks=[k])
        p = self.kdict[k]
        res = p.exists(d)

        if res["step"]:
            if hasattr(d, "step_data"):
                return d.step_data[p.d]
            else:
                return d.read("step")[p.d]
        elif res["end"]:
            if hasattr(d, "endpoint_data"):
                return d.endpoint_data[p.d]
            else:
                return d.read("end")[p.d]
        else:
            for key in res.keys():
                if key not in ["step", "end"] and res[key]:
                    return d.read(f"{key}.{p.d}")

        if compute:
            self.compute(k, d)
            return self.get(k, d, compute=False)
        else:
            print(f"Parameter {p.disp} not found")

    def compute(self, k: str, d: Any) -> None:
        p = self.kdict[k]
        res = p.exists(d)
        if not any(list(res.values())):
            k0s = p.required_ks
            for k0 in k0s:
                self.compute(k0, d)
            p.compute(d)

    def getPar(
        self,
        k: Optional[str] = None,
        p: Optional[str | list[str]] = None,
        d: Optional[str | list[str]] = None,
        to_return: str | list[str] = "d",
    ):
        """
        Retrieve the values of specific keys from a given parameter entry.
        Takes as argument the key by which to look up the parameter entry in the parameter database.

        Args:
            k (optional): Look up by short-key.
            p (optional): Look up by natural-language name.
            d (optional): Look up by dataset-based name.
            to_return (str, optional): Specifies the keys for which to return the values. Defaults to 'd'.

        Returns:
            The values associated with the selected keys from the parameter entry.

        """
        if k is None:
            if d is not None:
                if isinstance(d, str):
                    k = self.d2k_dict[d] if d in self.d2k_dict else None
                else:
                    k = [self.d2k_dict[dd] for dd in d if dd in self.d2k_dict]
            elif p is not None:
                if isinstance(p, str):
                    k = self.p2k_dict[p] if p in self.p2k_dict else None
                else:
                    k = [self.p2k_dict[pp] for pp in p if pp in self.p2k_dict]
        if k is None:
            raise

        if isinstance(k, str):
            self.update_kdict(ks=[k])
            par = self.kdict[k]
            if isinstance(to_return, list):
                return [getattr(par, i) for i in to_return]
            elif isinstance(to_return, str):
                return getattr(par, to_return)
        else:
            self.update_kdict(ks=k)
            pars = [self.kdict[kk] for kk in k]
            if isinstance(to_return, list):
                return [[getattr(par, i) for par in pars] for i in to_return]
            elif isinstance(to_return, str):
                return [getattr(par, to_return) for par in pars]

    def runtime_pars(self) -> list[str]:
        return [v.d for k, v in self.kdict.items()]

    def auto_load(self, ks: list[str], datasets: list[Any]) -> AttrDict:
        dic = {}
        for k in ks:
            dic[k] = {}
            for d in datasets:
                vs = self.get(k=k, d=d, compute=True)
                dic[k][d.id] = vs
        return AttrDict(dic)

    def df_to_pint(self, df):
        """
        Method to convert a pandas dataframe to a pint-pandas dataframe by assigning a pint unit to every column (parameter).
        The pint-pandas readable pint unit is a string formatted by the unit registered in the parameter class (check "upint" property of the LarvaworldParam class in lib.util.data_aux)

        """
        from pint_pandas.pint_array import PintType

        valid_pars = [
            col
            for col in self.dkeys.existing(df.columns)
            if not isinstance(df.dtypes[col], PintType)
        ]
        pint_dtypes = {
            par: PintType(f'pint[{self.getPar(d=par, to_return="u")}]')
            for par in valid_pars
        }
        df[valid_pars] = df[valid_pars].astype(dtype=pint_dtypes)
        return df

    def output_reporters(self, ks: list[str], agents: list[Any]) -> AttrDict:
        self.update_kdict(ks=ks)
        D = self.kdict
        dic = {}
        for k in ks:
            if k in D:
                d, p = D[k].d, D[k].codename
                try:
                    temp = [util.rgetattr(l, p) for l in agents]
                    dic.update({d: p})
                except:
                    pass
        return AttrDict(dic)

    def get_reporters(
        self, agents: list[Any], cs: Optional[list[str]] = None
    ) -> AttrDict:
        O = output_dict
        if cs is None:
            cs = ["pose"]
        return AttrDict(
            {
                "step": self.output_reporters(
                    ks=SuperList(O[c]["step"] for c in cs).flatten.unique, agents=agents
                ),
                "end": self.output_reporters(
                    ks=SuperList(O[c]["endpoint"] for c in cs).flatten.unique,
                    agents=agents,
                ),
            }
        )

    def select_output(self, pref: str) -> AttrDict:
        return AttrDict(
            {
                p.d: p.codename
                for k, p in reg.par.dict.items()
                if p.codename.startswith(pref)
            }
        )

    @property
    def brain_output(self) -> AttrDict:
        return self.select_output(pref="brain")
