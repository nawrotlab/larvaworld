import os
import shutil
import warnings
from itertools import product

from lib.aux import dictsNlists as dNl
from lib.registry.pars import preg

from lib.stor.building import build_Jovanic, build_Schleyer, build_Berni, build_Arguello
from lib.stor.larva_dataset import LarvaDataset


def import_Jovanic_datasets(parent_dir, source_ids=['Fed', 'Deprived', 'Starved'], **kwargs):
    # datagroup_id = 'Jovanic lab'
    # group_id = parent_dir
    # merged = False
    # N = None
    ds = {}
    for source_id in source_ids:
        id = source_id
        d = import_dataset(N=None, id=id, datagroup_id='Jovanic lab', group_id=parent_dir, parent_dir=parent_dir,
                           source_id=source_id,
                           merged=False, **kwargs)
        ds[d.id] = d
    return ds


def import_dataset(datagroup_id,  parent_dir, group_id=None, N=None,  id=None, age=0.0,  merged=True,enrich=True,add_reference=True,   **kwargs):
    # N = 150
    group = preg.get_null('LarvaGroup', sample=None, model=None, life_history={'age': age, 'epochs': {}})
    group.distribution.N = N

    if id is None:
        id = f'{N}controls'
    if group_id is None:
        group_id = parent_dir

    g = preg.loadConf(id=datagroup_id, conftype='Group')
    group_dir = f'{preg.path_dict["DATA"]}/{g["path"]}'
    raw_folder = f'{group_dir}/raw'
    proc_folder = f'{group_dir}/processed'
    source_dir = f'{raw_folder}/{parent_dir}'

    if merged:
        source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    kws = {
        'datagroup_id': datagroup_id,
        'larva_groups': {group_id: group},
        'target_dir': f'{proc_folder}/{group_id}/{id}',
        'source_dir': source_dir,
        'max_Nagents': N,
        **kwargs
    }
    d = build_dataset(id=id, **kws)
    if d is not None:
        d.save(food=False, add_reference=add_reference)
        if enrich :
            d.enrich(**g.enrichment, store=True, add_reference=add_reference)
    return d


def build_dataset(datagroup_id, id, target_dir, larva_groups={},**kwargs):
    func_dict = {
        'Jovanic lab': build_Jovanic,
        'Berni lab': build_Berni,
        'Schleyer lab': build_Schleyer,
        'Arguello lab': build_Arguello,
    }

    warnings.filterwarnings('ignore')
    print(f'Initializing {datagroup_id} format-specific dataset import...')
    shutil.rmtree(target_dir, ignore_errors=True)
    g = preg.loadConf(id=datagroup_id, conftype='Group')
    d = LarvaDataset(dir=target_dir, id=id, metric_definition=g.enrichment.metric_definition,
                     env_params=preg.get_null('env_conf', arena=g.tracker.arena),
                     load_data=False, larva_groups=larva_groups, **g.tracker.resolution)
    kws0 = {
        'dataset': d,
        'build_conf': g.tracker.filesystem,
        **kwargs
    }
    try:


        step, end = func_dict[datagroup_id](**kws0)
        d.set_data(step=step, end=end)


        print(f'--- Dataset {d.id} created with {len(d.agent_ids)} larvae! ---')
        return d
    except:
        print(f'--- Failed to create dataset {id}! ---')
        # d.delete()
        return None



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
