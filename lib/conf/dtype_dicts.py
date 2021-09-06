import copy
from typing import Tuple

import numpy as np

from lib.conf.init_dtypes import load_dtypes, processing_types, annotation_bouts, init_agent_dtypes

all_null_dicts, all_dtypes = load_dtypes()
dtype_keys=list(all_dtypes.keys())
dict_keys=list(all_null_dicts.keys())





def get_dict(name, class_name=None, basic=True, as_entry=False, **kwargs):
    if name in dict_keys:
        d = all_null_dicts[name]
    elif name == 'distro':
        d = null_distro(class_name=class_name, basic=basic)
    elif name == 'agent':
        d = null_agent(class_name=class_name)

    dic = copy.deepcopy(d)
    if name in ['visualization', 'enrichment']:
        for k, v in dic.items():
            if k in list(kwargs.keys()):
                dic[k] = kwargs[k]
            elif type(v) == dict:
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic[k][k0] = kwargs[k0]
    else:
        dic.update(kwargs)
    if as_entry:
        if name == 'distro' and 'group' in list(dic.keys()):
            id = dic['group']
        elif 'unique_id' in list(dic.keys()):
            id = dic['unique_id']
            dic.pop('unique_id')
        dic = {id: dic}
    return dic


def get_distro(class_name, **kwargs):
    distro = null_distro(class_name)
    distro.update(**kwargs)
    return distro


def null_distro(class_name, basic=True):
    distro = {
        'mode': None,
        'shape': None,
        'N': 0,
        'loc': (0.0, 0.0),
        'scale': (0.0, 0.0),
    }
    if class_name == 'Larva':
        distro = {**distro, 'orientation_range': (0.0, 360.0), 'model': None}
    if not basic:
        distro = {**distro, **get_dict('agent', class_name=class_name)}
        for p in ['unique_id', 'pos']:
            try:
                distro.pop(p)
            except:
                pass
    return distro


def get_dict_dtypes(name, **kwargs):
    if name in dtype_keys:
        return all_dtypes[name]
    elif name == 'distro':
        return get_distro_dtypes(**kwargs)



def null_agent(class_name):
    dic = {
        'unique_id': None,
        # 'default_color': 'black',
        'group': '',
    }
    if class_name in ['Larva', 'LarvaSim', 'LarvaReplay']:
        dic = {**dic, **get_dict('odor'), 'default_color': 'black'}
    elif class_name in ['Source', 'Food']:
        dic = {**dic, **get_dict('odor'), **get_dict('food'), 'default_color': 'green', 'pos': (0.0, 0.0)}
    elif class_name in ['Border']:
        dic = {**dic, 'width': 0.001, 'points': None, 'default_color': 'grey'}
    return dic


def sim_dict(sim_ID=None, duration=3, dt=0.1, path=None, Box2D=False, exp_type=None):
    from lib.conf.conf import next_idx
    if exp_type is not None:
        if sim_ID is None:
            sim_ID = f'{exp_type}_{next_idx(exp_type)}'
        if path is None:
            path = f'single_runs/{exp_type}'
    return {
        'sim_ID': sim_ID,
        'duration': duration,
        'timestep': dt,
        'path': path,
        'Box2D': Box2D,
    }


def brain_dict(modules, nengo=False, odor_dict=None, **kwargs):
    modules = get_dict('modules', **{m: True for m in modules})
    d = {'modules': modules}
    for k, v in modules.items():
        p = f'{k}_params'
        if not v:
            d[p] = None
        elif k in list(kwargs.keys()):
            d[p] = kwargs[k]
        else:
            d[p] = get_dict(k)
        if k == 'olfactor' and d[p] is not None:
            d[p]['odor_dict'] = odor_dict
    d['nengo'] = nengo
    return d


def larva_dict(brain, **kwargs):
    d = {'brain': brain}
    for k in ['energetics', 'physics', 'body', 'odor']:
        if k in list(kwargs.keys()):
            d[k] = kwargs[k]
        elif k == 'energetics':
            d[k] = None
        else:
            d[k] = get_dict(k)
    return d


def new_odor_dict(ids: list, means: list, stds=None) -> dict:
    if stds is None:
        stds = np.array([0.0] * len(means))
    odor_dict = {}
    for id, m, s in zip(ids, means, stds):
        odor_dict[id] = {'mean': m,
                         'std': s}
    return odor_dict


def base_enrich(types=['angular', 'spatial','dispersion', 'tortuosity'],bouts=['stride', 'pause', 'turn'],  **kwargs):
    d = {
        'types': processing_types(types),
        # 'types': ['angular', 'spatial', 'source', 'dispersion', 'tortuosity'],
        'dsp_starts': [0, 20], 'dsp_stops': [40, 80], 'tor_durs': [2, 5, 10, 20],
        'min_ang': 5.0, 'bouts': annotation_bouts(bouts)
    }

    d.update(**kwargs)
    return get_dict('enrichment', **d)


def get_distro_dtypes(class_name, basic=True):
    from lib.conf.conf import loadConfDict
    dtypes = {
        'mode': ['normal', 'periphery', 'uniform'],
        'shape': ['circle', 'rect', 'oval'],
        'N': int,
        'loc': Tuple[float, float],
        'scale': Tuple[float, float],
    }
    if class_name == 'Larva':
        dtypes = {**dtypes, 'orientation_range': Tuple[float, float], 'model': list(loadConfDict('Model').keys())}
    if not basic:
        dtypes = {**dtypes, **get_dict_dtypes(class_name)}
        for p in ['unique_id', 'pos']:
            try:
                dtypes.pop(p)
            except:
                pass
    return dtypes