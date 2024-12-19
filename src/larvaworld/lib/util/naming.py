"""
Class managing parameter naming
"""

import numpy as np

__all__ = [
    "NamingRegistry",
    # 'tex',
    # 'sub',
    # 'sup',
    # 'subsup',
]

from .dictsNlists import AttrDict, SuperList


def join(s, p, loc, c="_"):
    if loc == "suf":
        return f"{p}{c}{s}"
    elif loc == "pref":
        return f"{s}{c}{p}"
    elif loc == "sep":
        return f"{s}{c}{p}"


def name(s, ps, loc="suf", c="_"):
    if isinstance(ps, str):
        if ps == "":
            return s
        else:
            return join(s, ps, loc, c)
    elif isinstance(ps, list):
        return SuperList([join(s, p, loc, c) if p != "" else s for p in ps])


def _tex(p):
    return p.replace("$", "")


def tex_sym(symbol, p, sep=""):
    return rf"$\{symbol}{sep}{{{_tex(p)}}}$"


def tex(p, q, sep=""):
    return rf"${{{_tex(p)}}}{sep}{{{_tex(q)}}}$"


def sub(p, q):
    return tex(p, q, sep="_")


def sup(p, q):
    return tex(p, q, sep="^")


def subsup(p, q, z):
    return rf"${{{_tex(p)}}}_{{{_tex(q)}}}^{{{_tex(z)}}}$"


class TexNaming:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.symbols = ["bar", "tilde", "dot", "ddot", "mathring"]
        self.letters = ["theta", "omega", "Delta", "sum", "delta"]

    def get_func(self, symbol):
        assert symbol in self.symbols + self.letters

        def func(p, **kwargs):
            return tex_sym(symbol=symbol, p=p, **kwargs)

        return func

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, k):
        return self.get_func(k)

    def sub(self, p, q):
        return tex(p, q, sep="_")

    def sup(self, p, q):
        return tex(p, q, sep="^")

    def subsup(self, p, q, z):
        return rf"${{{_tex(p)}}}_{{{_tex(q)}}}^{{{_tex(z)}}}$"

    def circledast(self, p):
        return rf"${_tex(p)}^{{\circledast}}$"


class NamingRegistry(AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_pref = [
            "final",
            "initial",
            "cum",
            "lin",
            "scal",
            "abs",
            "dst_to",
            "bearing_to",
            "non",
        ]

        self.k_pairs = AttrDict(
            {
                "vel": "velocity",
                "acc": "acceleration",
                "scal": "scaled",
                "orient": "orientation",
                "unwrap": "unwrapped",
                # 'scal': 'scaled',
            }
        )

        # self.ks = SuperList(self.k_pref+self.k_pairs.keylist+self.k_ops.keylist+['freq', 'chain', 'dur', 'dst']).unique

        # self.tex_symbols = ['bar', 'tilde', 'wave', 'theta_', 'omega_', 'Delta', 'sum', 'delta', 'dot', 'ddot',
        #                     'mathring']

        self.tex = TexNaming()

    def get_kws(self, k):
        noseparator = ["chain"]

        kws = {}
        if k in self.k_pref:
            kws["loc"] = "pref"
        if k in self.k_pairs:
            kws["s"] = self.k_pairs[k]
        else:
            kws["s"] = k
        if k in noseparator:
            kws["c"] = ""
        return kws

    def get_func(self, k):
        kws = self.get_kws(k)

        def func(ps, **kwargs):
            kws.update(kwargs)
            return name(ps=ps, **kws)

        return func

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, k):
        return self.get_func(k)

    def num(self, chunk):
        s = "num"
        temp = name(s, chunk, "pref")
        return name("s", temp, "suf", c="")

    def xy(self, points, flat=False, xsNys=False):
        if type(points) == str:
            if points == "":
                t = ["x", "y"]
            else:
                t = [f"{points}_x", f"{points}_y"]

        elif type(points) == list:
            t = [self.xy(p) for p in points]
            if xsNys:
                t = [np.array(t)[:, i].tolist() for i in [0, 1]]
            if flat:
                t = [item for sublist in t for item in sublist]
        return SuperList(t)

    def chunk_track(self, chunk_name, params):
        return self[chunk_name](params, loc="pref")

    def contour(self, Nc):
        return [f"contour{i}" for i in range(Nc)]

    def midline(self, N, type="point", reverse=False):
        if N >= 2:
            points = ["head"] + [f"{type}{i}" for i in np.arange(2, N, 1)] + ["tail"]
        elif N == 1:
            points = ["body"]
        else:
            points = []
        if reverse:
            points.reverse()
        return points

    def contour_xy(self, Nc, flat=False, xsNys=False):
        return self.xy(self.contour(Nc), flat=flat, xsNys=xsNys)

    def midline_xy(self, N, reverse=False, flat=False, xsNys=False):
        return self.xy(self.midline(N, reverse=reverse), flat=flat, xsNys=xsNys)

    @property
    def centroid_xy(self):
        return self.xy("centroid")

    @property
    def traj_xy(self):
        return self.xy("")

    def at(self, p, t):
        return self[f"{p}_at"](t, loc="pref")

    def atStartStopChunk(self, p, chunk):
        return [
            self.at(p, self.start(chunk)),
            self.at(p, self.stop(chunk)),
            self.chunk_track(chunk, p),
        ]

    @property
    def on_food(self):
        return "on_food"

    @property
    def off_food(self):
        return "off_food"
