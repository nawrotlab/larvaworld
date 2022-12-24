import numpy as np
from typing import TypedDict

from lib.aux.par_aux import define_range
from lib import reg


def par(name, dtype=float, v=None, vs=None, lim=None, dv=None, aux_vs=None, disp=None, Ndigits=None,
        h='', k=None, symbol='', u=reg.units.dimensionless, u_name=None, label='', combo=None, entry=None, codename=None,
        **kwargs):
    if dtype == TypedDict:
        return {name: {'initial_value': v, 'dtype': dtype, 'entry': entry, 'disp': disp, 'tooltip': h}}

    dv, lim, vs = define_range(dtype=dtype, lim=lim, vs=vs, dv=dv, u=u, wrap_mode=None)

    if vs not in [None, []]:
        from lib.aux.data_aux import maxNdigits
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
