import numpy as np
from typing import TypedDict, List, Tuple
import lib.aux.dictsNlists as dNl


from lib.conf.pars.units import ureg



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


def define_lim(lim, vs, min, max, u, wrap_mode, cur_dtype):
    if lim is not None:
        return lim
    if wrap_mode is not None and u is not None:
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


def define_range(dtype, lim, vs, dv, min, max, u, wrap_mode):
    cur_dtype = base_dtype(dtype)
    dv = define_dv(dv, cur_dtype)
    lim = define_lim(lim, vs, min, max, u, wrap_mode, cur_dtype)
    vs = define_vs(vs, dv, lim, cur_dtype)
    return dv, lim, vs


def par(name, t=float, v=None, vs=None, lim=None, min=None, max=None, dv=None, aux_vs=None, disp=None, Ndigits=None,
        h='', k=None, symbol='',
        u=ureg.dimensionless, u_name=None, label='', combo=None, argparser=False, entry=None, codename=None, vfunc=None,
        vparfunc=None, convert2par=False):
    if argparser:
        from lib.conf.pars.parser_dict import build_ParsArg
        return build_ParsArg(name, k, h, t, v, vs)
    if t == TypedDict:
        return {name: {'initial_value': v, 'dtype': t, 'entry': entry, 'disp': disp, 'tooltip': h}}

    if k is None:
        k = name
    dv, lim, vs = define_range(dtype=t, lim=lim, vs=vs, dv=dv, min=min, max=max, u=u, wrap_mode=None)

    if convert2par:
        # from lib.conf.base.par_dict import preparePar
        p_kws = {
            'p': name,
            'k': k,
            'lim': lim,
            'dv': dv,
            'vs': vs,
            'v0': v,
            'dtype': t,
            'disp': label,
            'h': h,
            'u_name': u_name,
            'u': u,
            'sym': symbol,
            'codename': codename,
            'vfunc': vfunc,
            'vparfunc': vparfunc,
        }
        return p_kws

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
        try:
            entry = par(n, **v, **kwargs)
        except:
            entry = {n: {'dtype': dict, 'content': par_dict(d0=v, **kwargs)}}
        d.update(entry)
    return d


def Box2Djoints(N, **kwargs):
    return {'N': N, 'args': kwargs}


null_Box2D_params = {
    'joint_types': {
        'friction': {'N': 0, 'args': {}},
        'revolute': {'N': 0, 'args': {}},
        'distance': {'N': 0, 'args': {}}
    }
}


def null_dict(name, key='v', **kwargs):
    def v0(d):
        if d is None:
            return None

        null = dNl.NestDict()
        for k, v in d.items():
            if not isinstance(v, dict):
                null[k] = v
            # print(k,v)
            elif 'k' in v.keys() or 'h' in v.keys() or 't' in v.keys():
                null[k] = None if key not in v.keys() else v[key]
            else:
                null[k] = v0(v)
        return null

    if key!='v' :
        raise
    from lib.conf.pars.pars import ParDict
    d0 = ParDict.init_dict[name]
    dic2 = v0(d0)
    if name not in ['visualization', 'enrichment']:
        dic2.update(kwargs)
        return dNl.NestDict(dic2)
    else:
        for k, v in dic2.items():
            if k in list(kwargs.keys()):
                dic2[k] = kwargs[k]
            elif isinstance(v, dict):
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic2[k][k0] = kwargs[k0]
        return dNl.NestDict(dic2)




#
# def null_dict0(name, key='initial_value', **kwargs):
#     from lib.conf.pars.pars import ParDict
#     # from lib.conf.base.init_pars import InitDict
#     def v0(d):
#         null = {}
#         for k, v in d.items():
#             if key in v:
#                 null[k] = v[key]
#             else:
#                 null[k] = v0(v['content'])
#         return null
#
#     dic = par_dict(d0 = ParDict.init_dict[name])
#     dic2 = v0(dic)
#     if name not in ['visualization', 'enrichment']:
#         dic2.update(kwargs)
#         return dNl.NestDict(dic2)
#     else:
#         for k, v in dic2.items():
#             if k in list(kwargs.keys()):
#                 dic2[k] = kwargs[k]
#             elif isinstance(v, dict):
#                 for k0, v0 in v.items():
#                     if k0 in list(kwargs.keys()):
#                         dic2[k][k0] = kwargs[k0]
#         return dNl.NestDict(dic2)




def enr_dict(proc=[], bouts=[], to_keep=[], pre_kws={}, fits=True, on_food=False, def_kws={}, metric_definition=None,
             **kwargs):
    from lib.conf.pars.init_pars import proc_type_keys, bout_keys, to_drop_keys

    if metric_definition is None:
        from lib.conf.stored.data_conf import metric_def
        metric_definition = metric_def(**def_kws)
    pre = null_dict('preprocessing', **pre_kws)
    proc = null_dict('processing', **{k: True if k in proc else False for k in proc_type_keys})
    annot = null_dict('annotation', **{k: True if k in bouts else False for k in bout_keys}, fits=fits,
                      on_food=on_food)
    to_drop = null_dict('to_drop', **{k: True if k not in to_keep else False for k in to_drop_keys})
    dic = null_dict('enrichment', metric_definition=metric_definition, preprocessing=pre, processing=proc,
                    annotation=annot,
                    to_drop=to_drop, **kwargs)
    return dic


def base_enrich(**kwargs):
    return enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                    bouts=['stride', 'pause', 'turn'],
                    to_keep=['midline', 'contour'], **kwargs)


def arena(x, y=None):
    if y is None:
        return null_dict('arena', arena_shape='circular', arena_dims=(x, x))
    else:
        return null_dict('arena', arena_shape='rectangular', arena_dims=(x, y))

def odor(i, s, id='Odor'):
    return null_dict('odor', odor_id=id, odor_intensity=i, odor_spread=s)


def oG(c=1, id='Odor'):
    return odor(i=2.0 * c, s=0.0002 * np.sqrt(c), id=id)


def oD(c=1, id='Odor'):
    return odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)



