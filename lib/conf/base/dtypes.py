import numpy as np

import pandas as pd

from lib.aux.dictsNlists import AttrDict
from lib.conf.base.init_pars import init_pars, proc_type_keys, bout_keys, to_drop_keys
from lib.gui.aux.functions import get_pygame_key


def maxNdigits(array, Min=None):
    N = len(max(array.astype(str), key=len))
    if Min is not None:
        N = max([N, Min])
    return N


def base_dtype(t):
    from typing import List, Tuple
    if t in [float, Tuple[float], List[float], List[Tuple[float]]]:
        base_t = float
    elif t in [int, Tuple[int], List[int], List[Tuple[int]]]:
        base_t = int
    else:
        base_t = t
    return base_t


def par(name, t=float, v=None, vs=None, min=None, max=None, dv=None, aux_vs=None, disp=None, Ndigits=None, h='', s='',
        combo=None, argparser=False, entry=None):

    if not argparser:
        from typing import TypedDict
        if t == TypedDict:
            return {name: {'initial_value': v, 'dtype': t, 'entry': entry, 'disp': disp}}
        cur_dtype = base_dtype(t)
        if cur_dtype in [float, int]:
            if any([arg is not None for arg in [min, max, dv]]):
                if vs is None:
                    if min is None:
                        min = 0
                    if max is None:
                        max = 1
                    if dv is None:
                        if cur_dtype == int:
                            dv = 1
                        elif cur_dtype == float:
                            dv = 0.1

                    ar = np.arange(min, max + dv, dv)
                    if cur_dtype == float:
                        Ndec = len(str(format(dv, 'f')).split('.')[1])
                        ar = np.round(ar, Ndec)
                    vs = ar.astype(cur_dtype)

                    vs = vs.tolist()
        if vs is not None:
            Ndigits = maxNdigits(np.array(vs), 3)
        if aux_vs is not None and vs is not None:
            vs += aux_vs
        d = {'initial_value': v, 'values': vs, 'Ndigits': Ndigits, 'dtype': t,
             'disp': disp if disp is not None else name, 'combo': combo, 'tooltip': h}

        return {name: d}
    else:
        d = {
            'key': name,
            'short': s if s != '' else name,
            'help': h,
        }
        if t == bool:
            d['action'] = 'store_true' if not v else 'store_false'
        else:
            d['type'] = t
            if vs is not None:
                d['choices'] = vs
            if v is not None:
                d['default'] = v
                d['nargs'] = '?'
        return {name: d}


def par_dict(name=None, d0=None, **kwargs):
    if d0 is None:
        d0 = init_pars().get(name, None)
    if d0 is None:
        return None
    d = {}
    for n, v in d0.items():
        try:
            entry = par(n, **v, **kwargs)
        except:
            entry = {n: {'dtype': dict, 'content': par_dict(n, d0=d0[n], **kwargs)}}
        d.update(entry)
    return d


def par_dict_from_df(name, df):
    df = df.where(pd.notnull(df), None)
    d = {}
    for n in df.index:
        entry = par(n, **df.loc[n])
        d.update(entry)
    return {name: d}


def pars_to_df(d):
    df = pd.DataFrame.from_dict(d, orient='index',
                                columns=['dtype', 'initial_value', 'value_list', 'min', 'max', 'interval'])
    df.index.name = 'name'
    df = df.where(pd.notnull(df), None)





col_idx_dict = {
    'LarvaGroup': [[0, 1, 2, 3, 6], [4], [5]],
    'enrichment': [[0], [5, 1, 3], [6, 2, 4]],
    'metric_definition': [[0, 1, 4], [2, 3, 5, 6]],
}


def Box2Djoints(N, **kwargs):
    return {'N': N, 'args': kwargs}


null_Box2D_params = {
    'joint_types': {
        'friction': {'N': 0, 'args': {}},
        'revolute': {'N': 0, 'args': {}},
        'distance': {'N': 0, 'args': {}}
    }
}


def null_dict(n, key='initial_value', **kwargs):
    def v0(d):
        null = {}
        for k, v in d.items():
            if key in v:
                null[k] = v[key]
            else:
                null[k] = v0(v['content'])
        return null

    dic = par_dict(n)
    dic2 = v0(dic)
    if n not in ['visualization', 'enrichment']:
        dic2.update(kwargs)
        return AttrDict.from_nested_dicts(dic2)
        # return dic2
    else:
        for k, v in dic2.items():
            if k in list(kwargs.keys()):
                dic2[k] = kwargs[k]
            elif type(v) == dict:
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic2[k][k0] = kwargs[k0]
        return AttrDict.from_nested_dicts(dic2)
        # return dic2


def ang_def(b='from_angles', fv=(1, 2), rv=(-2, -1), **kwargs):
    return null_dict('ang_definition', bend=b, front_vector=fv, rear_vector=rv, **kwargs)


def metric_def(ang={}, sp={}, **kwargs):
    # def metric_def(ang={}, sp={}, dsp={}, tor={}, str={}, pau={}, tur={}) :
    return null_dict('metric_definition',
                     angular=ang_def(**ang),
                     spatial=null_dict('spatial_definition', **sp),
                     **kwargs
                     )


def enr_dict(proc=[], bouts=[], to_keep=[], pre_kws={}, fits=True, on_food=False, def_kws={}, **kwargs):
    metrdef = metric_def(**def_kws)
    pre = null_dict('preprocessing', **pre_kws)
    proc = null_dict('processing', **{k: True if k in proc else False for k in proc_type_keys})
    annot = null_dict('annotation', **{k: True if k in bouts else False for k in bout_keys}, fits=fits,
                      on_food=on_food)
    to_drop = null_dict('to_drop', **{k: True if k not in to_keep else False for k in to_drop_keys})
    dic = null_dict('enrichment', metric_definition=metrdef, preprocessing=pre, processing=proc, annotation=annot,
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


def border(ps, c='black', w=0.01, id=None):
    b = null_dict('Border', points=ps, default_color=c, width=w)
    if id is not None:
        return {id: b}
    else:
        return b


def hborder(y, xs, **kwargs):
    ps = [(x, y) for x in xs]
    return border(ps, **kwargs)


def vborder(x, ys, **kwargs):
    ps = [(x, y) for y in ys]
    return border(ps, **kwargs)


def prestarved(h=0.0, age=0.0, q=1.0, substrate_type='standard'):
    sub0 = null_dict('substrate', type=substrate_type, quality=q)
    ep0 = {0: null_dict('epoch', start=0.0, stop=age - h, substrate=sub0)}
    if h == 0.0:
        return ep0
    else:
        sub1 = null_dict('substrate', type=substrate_type, quality=0.0)
        ep1 = {1: null_dict('epoch', start=age - h, stop=age, substrate=sub1)}
    return {**ep0, **ep1}


def init_shortcuts():
    draw = {
        'visible trail': 'p',
        '▲ trail duration': '+',
        '▼ trail duration': '-',

        'draw_head': 'h',
        'draw_centroid': 'e',
        'draw_midline': 'm',
        'draw_contour': 'c',
        'draw_sensors': 'j',
    }

    inspect = {
        'focus_mode': 'f',
        'odor gains': 'z',
        'dynamic graph': 'q',
    }

    color = {
        'black_background': 'g',
        'random_colors': 'r',
        'color_behavior': 'b',
    }

    aux = {
        'visible_clock': 't',
        'visible_scale': 'n',
        'visible_state': 's',
        'visible_ids': 'tab',
    }

    screen = {
        'move up': 'UP',
        'move down': 'DOWN',
        'move left': 'LEFT',
        'move right': 'RIGHT',
    }

    sim = {
        'larva_collisions': 'y',
        'pause': 'space',
        'snapshot': 'i',
        'delete item': 'del',

    }

    odorscape = {
        'odor_aura': 'u',
        'windscape': 'w',
        'plot odorscapes': 'o',
        **{f'odorscape {i}': i for i in range(10)},
        # 'move_right': 'RIGHT',
    }

    d = {
        'draw': draw,
        'color': color,
        'aux': aux,
        'screen': screen,
        'simulation': sim,
        'inspect': inspect,
        'landscape': odorscape,
    }

    return d


def init_controls():
    k = init_shortcuts()
    d = {'keys': {}, 'pygame_keys': {}, 'mouse': {
        'select item': 'left click',
        'add item': 'left click',
        'select item mode': 'right click',
        'inspect item': 'right click',
        'screen zoom in': 'scroll up',
        'screen zoom out': 'scroll down',
    }}
    ds = {}
    for title, dic in k.items():
        ds.update(dic)
        d['keys'][title] = dic
    d['pygame_keys'] = {k: get_pygame_key(v) for k, v in ds.items()}
    return d


def store_controls():
    d = init_controls()
    from lib.conf.stored.conf import saveConfDict
    saveConfDict(d, 'Settings')


def store_RefPars():
    from lib.aux.dictsNlists import save_dict
    from lib.conf.base import paths
    import lib.aux.naming as nam
    d = {
        'length': 'body.initial_length',
        nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
        'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_mu',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_std',
        nam.freq('feed'): 'brain.feeder_params.initial_freq',
    }
    save_dict(d, paths.path('ParRef'), use_pickle=False)


def odor(i, s, id='Odor'):
    return null_dict('odor', odor_id=id, odor_intensity=i, odor_spread=s)


def oG(c=1, id='Odor'):
    return odor(i=2.0 * c, s=0.0002 * np.sqrt(c), id=id)


def oD(c=1, id='Odor'):
    return odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)


if __name__ == '__main__':
    store_controls()
    store_RefPars()
    # print(null_dict('Box2D_params'))
