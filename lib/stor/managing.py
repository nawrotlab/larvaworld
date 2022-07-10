import os
import shutil
import warnings
from itertools import product

from lib.aux import dictsNlists as dNl
from lib.registry.pars import preg

from lib.stor.building import build_Jovanic, build_Schleyer, build_Berni, build_Arguello
from lib.stor.larva_dataset import LarvaDataset


def import_Jovanic_datasets(parent_dir, source_ids=['Fed', 'Deprived', 'Starved'], **kwargs):
    datagroup_id = 'Jovanic lab'
    group_id = parent_dir
    merged = False
    N = None
    ds = {}
    for source_id in source_ids:
        # try:
        id = source_id
        # id = f'{source_id}_0_60'
        d = import_dataset(N=N, id=id, datagroup_id=datagroup_id, group_id=group_id, parent_dir=parent_dir,
                           source_id=source_id,
                           merged=merged, **kwargs)
        ds[d.id] = d
    return ds


def import_dataset(N, datagroup_id='Schleyer lab', id=None, group_id='exploration', min_duration_in_sec=180,
                   age=96.0, parent_dir='no_odor', merged=True, enrich=True, add_reference=True, **kwargs):
    # N = 150
    group = preg.get_null('LarvaGroup', sample=None, model=None, life_history={'age': age, 'epochs': {}})
    group.distribution.N = N

    if id is None:
        id = f'{N}controls'

    g = preg.loadConf(id=datagroup_id, conftype='Group')
    group_dir = f'{preg.path_dict["DATA"]}/{g["path"]}'
    raw_folder = f'{group_dir}/raw'
    proc_folder = f'{group_dir}/processed'
    # parent_dir = 'no_odor'
    # target_dir = f'{proc_folder}/{group_id}/{id}'
    source_dir = f'{raw_folder}/{parent_dir}'

    if merged:
        source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    kws = {
        'datagroup_id': datagroup_id,
        'larva_groups': {group_id: group},
        'target_dir': f'{proc_folder}/{group_id}/{id}',
        'source_dir': source_dir,
        'max_Nagents': N,
        'min_duration_in_sec': min_duration_in_sec,
        # 'build_conf':g.tracker
        **kwargs
    }
    # print(kws['Ncontour'])
    # raise
    d = build_dataset(id=id, **kws)
    d.save(add_reference=add_reference)
    if enrich:
        d.enrich(**g.enrichment, store=True, add_reference=add_reference)
    else:
        d.save(add_reference=add_reference)

    return d


def build_dataset(datagroup_id, id, target_dir, source_dir=None, source_files=None, larva_groups={}, **kwargs):
    warnings.filterwarnings('ignore')
    g = preg.loadConf(id=datagroup_id, conftype='Group')
    build_conf = g.tracker.filesystem
    data_conf = g.tracker.resolution
    metric_definition = g.enrichment.metric_definition
    env_params = preg.get_null('env_conf', arena=g.tracker.arena)
    data_conf.Ncontour = 0

    try:
        shutil.rmtree(target_dir)
    except:
        pass

    d = LarvaDataset(dir=target_dir, id=id, metric_definition=metric_definition, env_params=env_params,
                     load_data=False, larva_groups=larva_groups, **data_conf)

    print(f'Initializing {datagroup_id} format-specific dataset import...')
    if datagroup_id in ['Jovanic lab']:
        step, end = build_Jovanic(d, build_conf, source_dir=source_dir, **kwargs)
    elif datagroup_id in ['Berni lab']:
        step, end = build_Berni(d, build_conf, source_files=source_files, **kwargs)
    elif datagroup_id in ['Schleyer lab']:
        step, end = build_Schleyer(d, build_conf, raw_folders=source_dir, **kwargs)
    elif datagroup_id in ['Arguello lab']:
        step, end = build_Arguello(d, build_conf, source_files=source_files, **kwargs)

    else:
        raise ValueError(f'Configuration for {datagroup_id} is not supported for building new datasets')
    if step is not None:
        step.sort_index(level=['Step', 'AgentID'], inplace=True)
        end.sort_index(inplace=True)
        d.set_data(step=step, end=end)
        d.save(food=False)
        d.agent_ids = d.step_data.index.unique('AgentID').values
        d.num_ticks = d.step_data.index.unique('Step').size
        # d.starting_tick = d.step_data.index.unique('Step')[0]
        print(f'--- Dataset {d.id} created with {len(d.agent_ids)} larvae! ---')
    else:
        print(f'--- Failed to create dataset {d.id}! ---')
        d.delete()
    return d


def get_datasets(datagroup_id, names, last_common='processed', folders=None, suffixes=None,
                 mode='load', load_data=True, ids=None, **kwargs):
    g = preg.loadConf(id=datagroup_id, conftype='Group')
    data_conf = g.tracker.resolution
    spatial_def = g.enrichment.metric_definition.spatial
    arena_pars = g.tracker.arena
    par_conf = g['parameterization']
    group_dir = f'{preg.path_dict["DATA"]}/{g["path"]}'

    last_common = f'{group_dir}/{last_common}'
    if folders is None:
        new_ids = ['']
        folders = [last_common]
    else:
        new_ids = folders
        folders = [f'{last_common}/{f}' for f in folders]
    if suffixes is not None:
        names = [f'{n}_{s}' for (n, s) in list(product(names, suffixes))]
    new_ids = [f'{id}{n}' for (id, n) in list(product(new_ids, names))]
    if ids is None:
        ids = new_ids
    dirs = [f'{f}/{n}' for (f, n) in list(product(folders, names))]
    ds = []
    for dir, id in zip(dirs, ids):
        if mode == 'load':
            if not os.path.exists(dir):
                print(f'No dataset found at {dir}')
                continue
            d = LarvaDataset(dir=dir, load_data=load_data)
        elif mode == 'initialize':
            try:
                shutil.rmtree(dir)
            except:
                pass

            d = LarvaDataset(dir=dir, id=id, par_conf=par_conf, arena_pars=arena_pars,
                             load_data=False, **data_conf)
        ds.append(d)
    return ds
