import numpy as np
import pandas as pd
import typing

from lib.aux.dictsNlists import AttrDict, tree_dict, unique_list, flatten_list
from lib.aux.par_aux import dtype_name
from lib.conf.base.init_pars import init_pars, proc_type_keys, bout_keys, to_drop_keys



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
            return {name: {'initial_value': v, 'dtype': t, 'entry': entry, 'disp': disp, 'tooltip': h}}
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
            Ndigits = maxNdigits(np.array(vs), 4)
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


def pars_to_df(names, d0=None):
    from lib.conf.base import paths
    dic={}
    for name in names :
        d=par_dict(name,d0)
        df = pd.DataFrame.from_dict(d, orient='index',
                                    columns=['dtype', 'initial_value', 'tooltip'])
                                    # columns=['dtype', 'initial_value', 'h', 'min', 'max', 'interval'])
        df.index.name = 'parameter'
        df = df.where(pd.notnull(df), None)
        dic[name]=df
    ddf= pd.DataFrame.from_dict(dic, orient='index')
    ddf.index.name = 'group'
    ddf.to_csv(paths.path('ParGlossary'))

def pars_to_tree(name):
    invalid=[]
    valid=[]
    def add_entry(k4,v4, parent) :
        key=f'{parent}.{k4}'
        if 'content' in v4.keys():
            dd=v4['content']
            if key not in valid:
                data.append([parent, key, k4, None, dict, None, k4])
                valid.append(key)
            for k1, v1 in dd.items():
                add_entry(k1, v1, key)
        else:
            entry = [parent, key,k4] + [v4[c] for c in columns[3:]]
            data.append(entry)
            valid.append(key)
    def add_multientry0(d, k0, name):
        key = f'{name}.{k0}'
        if key not in valid:
            data.append([name, key, k0,  None, dict,None, k0])
            valid.append(key)
        for k1, v1 in d.items():
            add_entry(k1, v1, key)
    data=[]
    columns = ['parent', 'key','text','initial_value', 'dtype', 'tooltip', 'disp']
    columns2 = ['parent', 'key','text','default_value', 'dtype', 'description', 'name']
    P=init_pars()[name]
    data.append(['root', name, name, None, dict,None, name])
    valid.append(name)
    for k0,v0 in P.items():
        d0=P.get(k0, None)
        try:
            d = par(k0, **v0)
            add_entry(k0, d[k0], name)
        except:
            d = par_dict(k0, d0)
            add_multientry0(d, k0, name)
    ddf = pd.DataFrame(data, columns=columns2)
    if 'dtype' in columns2 :
        ddf['dtype']=[dtype_name(v) for v in ddf['dtype'] ]
    ddf = ddf.fillna(value=' ')
    ddf =ddf.replace({}, ' ')
    return ddf

def conf_to_tree(conf, id=''):
    from lib.gui.aux.elements import GuiTreeData
    # from lib.conf.stored.conf import expandConf
    # d=expandConf(id, conftype)
    entries=tree_dict(d=conf, parent_key=id, sep='.')
    tree = GuiTreeData(entries=entries, headings=['value'], col_widths=[40,20])
    return tree

def multiconf_to_tree(ids, conftype):
    from lib.gui.aux.elements import GuiTreeData
    from lib.conf.stored.conf import expandConf
    dfs=[]
    for i, id in enumerate(ids) :
        conf=expandConf(id, conftype)
        entries = tree_dict(d=conf, parent_key=id)
        df=pd.DataFrame.from_records(entries, index=['parent', 'key', 'text'])
        dfs.append(df)
    ind0=[]
    for df in dfs :
        for ind in df.index.values :
            if ind not in ind0 :
                ind0.append(ind)
    vs=np.zeros([len(ind0), len(ids)])*np.nan
    df0=pd.DataFrame(vs, index=ind0, columns=ids)
    for id, df in zip(ids, dfs) :
        for key in df.index :
            print(key, key in df0.index)
            df0[id].loc[key]=df['values'].loc[key][0]
    df0.reset_index(inplace=True)
    df0['values']=[df0[id] for id in ids]
    df0.drop(ids, axis=1)
    comp_entries = df0.to_dict(orient='records')
    tree = GuiTreeData(entries=comp_entries, headings=[ids], col_widths=[40]+[20]*len(ids))
    return tree



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
    from lib.gui.aux.functions import get_pygame_key
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
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_mean',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_std',
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
    print(null_dict('crawler'))
    # t=multiconf_to_tree(['explorer', 'nengo_explorer'], 'Model')
    # t=conf_to_tree('explorer', 'Model')
    # t.save(k='text', v='value')
    # t.test()
    raise
    store_controls()
    store_RefPars()
