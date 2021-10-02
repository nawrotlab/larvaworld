import copy

from lib.conf.init_dtypes import load_dtypes, processing_types, annotation_bouts

all_null_dicts, all_dtypes = load_dtypes()
dtype_keys=list(all_dtypes.keys())
dict_keys=list(all_null_dicts.keys())





def get_dict(name, class_name=None, basic=True, as_entry=False, **kwargs):
    if name in dict_keys:
        d = all_null_dicts[name]
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


# def null_distro(class_name, basic=True):
#     distro = {
#         'mode': None,
#         'shape': None,
#         'N': 0,
#         'loc': (0.0, 0.0),
#         'scale': (0.0, 0.0),
#     }
#     if class_name == 'Larva':
#         distro = {**distro, 'orientation_range': (0.0, 360.0), 'model': None}
#     if not basic:
#         distro = {**distro, **get_dict('agent', class_name=class_name)}
#         for p in ['unique_id', 'pos']:
#             try:
#                 distro.pop(p)
#             except:
#                 pass
#     return distro


def get_dict_dtypes(name, **kwargs):
    if name in dtype_keys:
        return all_dtypes[name]
    # elif name == 'distro':
    #     return get_distro_dtypes(**kwargs)



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


def base_enrich(types=['angular', 'spatial','dispersion', 'tortuosity'],bouts=['stride', 'pause', 'turn'],  **kwargs):
    d = {
        'types': processing_types(types),
        # 'types': ['angular', 'spatial', 'source', 'dispersion', 'tortuosity'],
        'dsp_starts': [0, 20], 'dsp_stops': [40, 80], 'tor_durs': [2, 5, 10, 20],
        'min_ang': 30.0, 'bouts': annotation_bouts(bouts)
    }

    d.update(**kwargs)
    return get_dict('enrichment', **d)



if __name__ == '__main__':
    pass