import numpy as np

from lib.aux import dictsNlists as dNl
from lib.registry.dtypes import par_dict


def ga_dict(name=None, suf='', excluded=None, only=None):
    from lib.registry.pars import preg
    d0 = preg.init_dict[name]
    dic = par_dict(d0=d0)
    keys = list(dic.keys())
    if only is not None:
        keys = [k for k in keys if k in only]
    elif excluded is not None:
        keys = [k for k in keys if k not in excluded]
    d = {}
    for k in keys:
        vs = dic[k]
        k0 = f'{suf}{k}'
        kws = {
            'initial_value': vs['initial_value'],
            'tooltip': vs['tooltip'],
            'dtype': vs['dtype'],
            'name': k,
        }
        if vs['dtype'] == str:
            kws['choices'] = vs['values']
        elif vs['dtype'] == bool:
            kws['choices'] = [True, False]
        else:
            kws['min'], kws['max'] = np.min(vs['values']), np.max(vs['values'])
        d[k0] = kws
    return dNl.NestDict(d)


def interference_ga_dict(mID, suf='brain.interference_params.'):
    from lib.registry.pars import preg
    m = preg.loadConf(id=mID, conftype='Model')
    IFmod = m.brain.interference_params.mode

    if IFmod == 'phasic':
        only = ['attenuation', 'attenuation_max', 'max_attenuation_phase']

    elif IFmod == 'square':
        only = ['attenuation', 'attenuation_max', 'crawler_phi_range']

    space_dict = ga_dict(name='interference', suf=suf, only=only)
    return space_dict
