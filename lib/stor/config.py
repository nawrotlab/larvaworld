import os

import numpy as np
from lib.aux import dictsNlists as dNl, xy_aux,data_aux, naming as nam
def update_metric_definition(md=None, mdconf=None):
    if mdconf is None :

        if md is None:
            from lib.registry.pars import preg
            md = preg.get_null('metric_definition')
        from lib.aux import dictsNlists as dNl
        mdconf = dNl.NestDict({
            'spatial': {
                'hardcoded': md['spatial'],
                'fitted': None,
            },
            'angular': {
                'hardcoded': md['angular'],
                'fitted': None
            }
        })

    else :
        if md is not None:
            mdconf.angular.hardcoded.update(**md['angular'])
            mdconf.spatial.hardcoded.update(**md['spatial'])
    return mdconf


def dataset_config(dir=None, id='unnamed', fr=16, Npoints=3, Ncontour=0, metric_definition=None, env_params={},
                   larva_groups={}, source_xy={}, **kwargs):
    from lib.aux import dictsNlists as dNl

    group_ids = list(larva_groups.keys())
    samples = dNl.unique_list([larva_groups[k]['sample'] for k in group_ids])
    if len(group_ids) == 1:
        group_id = group_ids[0]
        color = larva_groups[group_id]['default_color']
        sample = larva_groups[group_id]['sample']
        life_history = larva_groups[group_id]['life_history']
    else:
        group_id = None
        color = None
        sample = samples[0] if len(samples) == 1 else None
        life_history = None

    return dNl.NestDict({'id': id,
                         'group_id': group_id,
                         'group_ids': group_ids,
                         'refID': None,
                         'dir': dir,
                         # 'parent_plot_dir': f'{dir}/plots',
                         'fr': fr,
                         'dt': 1 / fr,
                         'Npoints': Npoints,
                         'Ncontour': Ncontour,
                         'sample': sample,
                         'color': color,

                         'metric_definition': update_metric_definition(md=metric_definition),
                         'env_params': env_params,
                         'larva_groups': larva_groups,
                         'source_xy': source_xy,
                         'life_history': life_history,
                        **kwargs
                         })



def retrieve_config(dir=None, **kwargs):
    new_config = dataset_config(dir=dir, **kwargs)
    if dir is not None :
        from lib.registry.pars import preg
        from lib.aux.stor_aux import loadDic
        os.makedirs(dir, exist_ok=True)
        os.makedirs(preg.datapath('data', dir), exist_ok=True)
        try :
            oldconfig=loadDic(path=preg.datapath('conf',dir), use_pickle=True)
            print('Config with pickle True')
        except :
            oldconfig = loadDic(path=preg.datapath('conf', dir), use_pickle=False)
            print('Config with pickle False')
        if oldconfig is not None :
            return oldconfig
    return new_config



def update_config(obj, c) :
    c.dt = 1 / obj.fr
    # if 'agent_ids' not in c.keys():
    try:
        ids = obj.agent_ids
    except:
        try:
            ids = obj.endpoint_data.index.values
        except:
            ids = obj.read('end').index.values

    c.agent_ids = list(ids)
    c.N = len(ids)
    if 't0' not in c.keys():
        try:
            c.t0 = int(obj.step_data.index.unique('Step')[0])
        except:
            c.t0 = 0
    if 'Nticks' not in c.keys():
        try:
            c.Nticks = obj.step_data.index.unique('Step').size
        except:
            try:
                c.Nticks = obj.endpoint_data['num_ticks'].max()
            except:
                pass
    if 'duration' not in c.keys():
        try:
            c.duration = int(obj.endpoint_data['cum_dur'].max())
        except:
            c.duration = c.dt * c.Nticks
    if 'quality' not in c.keys():
        try:
            df = obj.step_data[nam.xy(obj.point)[0]].values.flatten()
            valid = np.count_nonzero(~np.isnan(df))
            c.quality = np.round(valid / df.shape[0], 2)
        except:
            pass

    for k, v in c.items():
        if isinstance(v, np.ndarray):
            c[k] = v.tolist()
    return c


# if __name__ == '__main__':
#     c=retrieve_config()
#
#     raise