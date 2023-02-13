import numpy as np
import typing

from larvaworld.lib import reg, aux


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


def define_lim(lim, vs, u, wrap_mode, cur_dtype):
    if lim is not None:
        return lim
    if wrap_mode is not None and u is not None:
        if u == reg.units.deg:
            if wrap_mode == 'positive':
                lim = (0.0, 360.0)
            elif wrap_mode == 'zero':
                lim = (-180.0, 180.0)
        elif u == reg.units.rad:
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
    cur_dtype = aux.base_dtype(dtype)
    dv = define_dv(dv, cur_dtype)
    lim = define_lim(lim, vs, u, wrap_mode, cur_dtype)
    vs = define_vs(vs, dv, lim, cur_dtype)
    return dv, lim, vs

def maxNdigits(array, Min=None):
    N = len(max(array.astype(str), key=len))
    if Min is not None:
        N = max([N, Min])
    return N


def par(name, dtype=float, v=None, vs=None, lim=None, dv=None, aux_vs=None, disp=None, Ndigits=None,
        h='', k=None, symbol='', u=reg.units.dimensionless, u_name=None, label='', combo=None, entry=None, codename=None,
        **kwargs):
    if dtype == typing.TypedDict:
        return {name: {'initial_value': v, 'dtype': dtype, 'entry': entry, 'disp': disp, 'tooltip': h}}

    dv, lim, vs = define_range(dtype=dtype, lim=lim, vs=vs, dv=dv, u=u, wrap_mode=None)

    if vs not in [None, []]:
        Ndigits = maxNdigits(np.array(vs), 4)
    if aux_vs is not None and vs is not None:
        vs += aux_vs
    d = {'initial_value': v, 'values': vs, 'Ndigits': Ndigits, 'dtype': dtype, 'symbol': symbol, 'unit': u_name,
         'label': label,
         'disp': disp if disp is not None else name, 'combo': combo, 'tooltip': h, 'codename': codename, 'step': dv}

    return {name: d}


def par_dict(d0, **kwargs):
    if d0 is None:
        return None
    d = {}
    for n, v in d0.items():
        if 'dtype' in v.keys() or 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys():
            entry = par(n, **v, **kwargs)
        else:
            entry = {n: {'dtype': dict, 'content': par_dict(d0=v, **kwargs)}}
        d.update(entry)
    return d
