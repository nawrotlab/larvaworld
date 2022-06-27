import numpy as np
from typing import TypedDict

from lib.aux.par_aux import define_range

from lib.registry.units import ureg

def par(name, t=float, v=None, vs=None, lim=None, min=None, max=None, dv=None, aux_vs=None, disp=None, Ndigits=None,
        h='', k=None, symbol='', u=ureg.dimensionless, u_name=None, label='', combo=None, entry=None, codename=None,
        **kwargs):
    if t == TypedDict:
        return {name: {'initial_value': v, 'dtype': t, 'entry': entry, 'disp': disp, 'tooltip': h}}

    dv, lim, vs = define_range(dtype=t, lim=lim, vs=vs, dv=dv, min=min, max=max, u=u, wrap_mode=None)

    if vs not in [None, []]:
        from lib.aux.data_aux import maxNdigits
        Ndigits = maxNdigits(np.array(vs), 4)
    if aux_vs is not None and vs is not None:
        vs += aux_vs
    d = {'initial_value': v, 'values': vs, 'Ndigits': Ndigits, 'dtype': t, 'symbol': symbol, 'unit': u_name,
         'label': label,
         'disp': disp if disp is not None else name, 'combo': combo, 'tooltip': h, 'codename': codename, 'step': dv}

    return {name: d}


def par_dict(d0, **kwargs):
    if d0 is None:
        return None
    d = {}
    for n, v in d0.items():
        if 't' in v.keys() or 'v' in v.keys() or 'k' in v.keys() or 'h' in v.keys() :
        # try:
            entry = par(n, **v, **kwargs)
        else:
        # except:
            entry = {n: {'dtype': dict, 'content': par_dict(d0=v, **kwargs)}}
        d.update(entry)
    return d
