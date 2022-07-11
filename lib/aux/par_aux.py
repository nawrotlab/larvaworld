from typing import Tuple, List

import numpy as np




def base(method, input, **kwargs):
    if type(input) == str:
        return method(input, **kwargs)
    elif type(input) == list:
        return [method(i, **kwargs) for i in input]


def bar(p):
    return rf'$\bar{{{p.replace("$", "")}}}$'

def tilde(p):
    return rf'$\tilde{{{p.replace("$", "")}}}$'


def wave(p):
    return rf'$\~{{{p.replace("$", "")}}}$'


def sub(p, q):
    return rf'${{{p.replace("$", "")}}}_{{{q}}}$'


def sup(p, q):
    return rf'${{{p.replace("$", "")}}}^{{{q}}}$'


def subsup(p, q, z):
    return rf'${{{p.replace("$", "")}}}_{{{q}}}^{{{z}}}$'


def hat(p):
    return f'$\hat{{{p.replace("$", "")}}}$'


def ast(p):
    return f'${p.replace("$", "")}^{{*}}$'


def th(p):
    return fr'$\theta_{{{p.replace("$", "")}}}$'

def omega(p):
    return fr'$\omega_{{{p.replace("$", "")}}}$'


def Delta(p):
    return fr'$\Delta{{{p.replace("$", "")}}}$'


def sum(p):
    return fr'$\sum{{{p.replace("$", "")}}}$'


def delta(p):
    return fr'$\delta{{{p.replace("$", "")}}}$'


def hat_th(p):
    return fr'$\hat{{\theta}}_{{{p}}}$'


def dot(p):
    return fr'$\dot{{{p.replace("$", "")}}}$'

def circle(p):
    return fr'$\mathring{{{p.replace("$", "")}}}$'


def circledcirc(p):
    return f'${p.replace("$", "")}^{{\circledcirc}}$'

def mathring(p):
    return fr'$\mathring{{{p.replace("$", "")}}}$'


def circledast(p):
    return f'${p.replace("$", "")}^{{\circledast}}$'


def odot(p):
    return f'${p.replace("$", "")}^{{\odot}}$'
    # return fr'$\odot{{{p.replace("$", "")}}}$'


def paren(p):
    return fr'$({{{p.replace("$", "")}}})$'


def brack(p):
    return fr'$[{{{p.replace("$", "")}}}]$'


def ddot(p):
    return fr'$\ddot{{{p.replace("$", "")}}}$'


def dot_th(p):
    return fr'$\dot{{\theta}}_{{{p.replace("$", "")}}}$'


def ddot_th(p):
    return fr'$\ddot{{\theta}}_{{{p.replace("$", "")}}}$'


def dot_hat_th(p):
    return fr'$\dot{{\hat{{\theta}}}}_{{{p}}}$'


def ddot_hat_th(p):
    return fr'$\ddot{{\hat{{\theta}}}}_{{{p}}}$'


def lin(p):
    return fr'${{{p.replace("$", "")}}}_{{l}}$'

def dtype_name(v) :
    def typing_arg(v):
        return v.__args__[0]
    if v is None :
        n= ' '
    else :
        try :
            n= v.__name__
        except :
            try :
                n= f'{v._name}[{typing_arg(v).__name__}]'
            except :
                try:
                    v0=typing_arg(v)
                    n = f'{v._name}[{v0._name}[{typing_arg(v0).__name__}]]'
                except:
                    n = v
    return n



def base_dtype(t):
    if t in [float, Tuple[float], List[float], List[Tuple[float]]]:
        base_t = float
    elif t in [int, Tuple[int], List[int], List[Tuple[int]]]:
        base_t = int
    else:
        base_t = t
    return base_t


def define_dv(dv, cur_dtype):
    if dv is None:
        if cur_dtype == int:
            dv = 1
        elif cur_dtype == float:
            dv = 0.1
    return dv


def define_vs(vs, dv, lim, cur_dtype):
    if vs is not None:
        return vs
    if dv is not None and lim is not None:
        ar = np.arange(lim[0], lim[1] + dv, dv)
        if cur_dtype == float:
            Ndec = len(str(format(dv, 'f')).split('.')[1])
            ar = np.round(ar, Ndec)
        vs = ar.astype(cur_dtype)

        vs = vs.tolist()
    return vs


def define_lim2(lim, vs, min, max, u, wrap_mode, cur_dtype):
    if lim is not None:
        return lim
    if wrap_mode is not None and u is not None:
        from lib.registry.units import ureg
        if u == ureg.deg:
            if wrap_mode == 'positive':
                lim = (0.0, 360.0)
            elif wrap_mode == 'zero':
                lim = (-180.0, 180.0)
        elif u == ureg.rad:
            if wrap_mode == 'positive':
                lim = (0.0, 2 * np.pi)
            elif wrap_mode == 'zero':
                lim = (-np.pi, np.pi)
    else:
        if cur_dtype in [float, int]:
            if vs is not None:
                lim = (np.min(vs), np.max(vs))
            else:
                if min is None:
                    min = 0
                if max is None:
                    max = 1
                lim = (min, max)
    return lim

def define_lim(lim, vs, u, wrap_mode, cur_dtype):
    if lim is not None:
        return lim
    if wrap_mode is not None and u is not None:
        from lib.registry.units import ureg
        if u == ureg.deg:
            if wrap_mode == 'positive':
                lim = (0.0, 360.0)
            elif wrap_mode == 'zero':
                lim = (-180.0, 180.0)
        elif u == ureg.rad:
            if wrap_mode == 'positive':
                lim = (0.0, 2 * np.pi)
            elif wrap_mode == 'zero':
                lim = (-np.pi, np.pi)
    else:
        if cur_dtype in [float, int]:
            if vs is not None:
                lim = (np.min(vs), np.max(vs))
    return lim

def define_range(dtype, lim, vs, dv, u, wrap_mode):
    cur_dtype = base_dtype(dtype)
    dv = define_dv(dv, cur_dtype)
    lim = define_lim(lim, vs, u, wrap_mode, cur_dtype)
    vs = define_vs(vs, dv, lim, cur_dtype)
    return dv, lim, vs


