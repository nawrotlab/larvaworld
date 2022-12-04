import os
import shutil
import warnings
from itertools import product

from lib.aux import dictsNlists as dNl, colsNstr as cNs
from lib.registry import reg

from lib.stor.building import build_Jovanic, build_Schleyer, build_Berni, build_Arguello
from lib.stor.building import build_Schleyer
from lib.stor.larva_dataset import LarvaDataset


def import_datasets(source_ids, ids=None, colors=None, refIDs=None, **kwargs):
    if colors is None:
        colors = cNs.N_colors(len(source_ids))
    if ids is None:
        ids = source_ids
    ds = []
    for i, source_id in enumerate(source_ids):
        refID = None if refIDs is None else refIDs[i]

        d = import_dataset(id=ids[i], color=colors[i], source_id=source_id, refID=refID, **kwargs)
        ds.append(d)

    return ds


def import_dataset(datagroup_id, parent_dir, group_id=None, N=None, id=None, merged=False, enrich=True,
                   add_reference=True, refID=None, enrich_conf=None, **kwargs):
    print()
    print(f'----- Initializing {datagroup_id} format-specific dataset import. -----')
    # N = 150

    if id is None:
        id = f'{N}controls'
    if group_id is None:
        group_id = parent_dir

    g = reg.loadConf(id=datagroup_id, conftype='Group')
    group_dir = g.path
    # group_dir = f'{preg.path_dict["DATA"]}/{g.path}'
    raw_folder = f'{group_dir}/raw'
    proc_folder = f'{group_dir}/processed'
    source_dir = f'{raw_folder}/{parent_dir}'

    if merged:
        source_dir = [f'{source_dir}/{f}' for f in os.listdir(source_dir)]
    kws = {
        'datagroup_id': datagroup_id,
        'group_id': group_id,
        'Ν': N,
        # 'larva_groups': {group_id: group},
        'target_dir': f'{proc_folder}/{group_id}/{id}',
        'source_dir': source_dir,
        'max_Nagents': N,
        **kwargs
    }
    d = build_dataset(id=id, **kws)
    if d is not None:
        print(f'***-- Dataset {d.id} created with {len(d.agent_ids)} larvae! -----')
        if enrich:
            print(f'****- Processing dataset {d.id} to derive secondary metrics -----')
            if enrich_conf is None:
                enrich_conf = g.enrichment

            d = d.enrich(**enrich_conf, store=True, is_last=False)
        d.save(food=False, add_reference=add_reference, refID=refID)
    else:
        print(f'xxxxx Failed to create dataset {id}! -----')
    return d


def build_dataset(datagroup_id, id, target_dir, group_id, N=None, sample=None,
                  color='black', epochs={},age=0.0, **kwargs):
    print(f'*---- Building dataset {id} under the {datagroup_id} format. -----')

    func_dict = {
        'Jovanic lab': build_Jovanic,
        'Berni lab': build_Berni,
        'Schleyer lab': build_Schleyer,
        'Arguello lab': build_Arguello,
    }

    warnings.filterwarnings('ignore')

    shutil.rmtree(target_dir, ignore_errors=True)
    g = reg.loadConf(id=datagroup_id, conftype='Group')

    conf = {
        'load_data': False,
        'dir': target_dir,
        'id': id,
        'metric_definition': g.enrichment.metric_definition,
        'larva_groups': reg.lg(id=group_id, c=color, sample=sample, mID= None, N=N,epochs={},age=0.0),
        'env_params': reg.get_null('env_conf', arena=g.tracker.arena),
        **g.tracker.resolution
    }

    d = LarvaDataset(**conf)
    kws0 = {
        'dataset': d,
        'build_conf': g.tracker.filesystem,
        **kwargs
    }
    try:

        step, end = func_dict[datagroup_id](**kws0)
        d.set_data(step=step, end=end)
        # print(f'***-- Dataset {d.id} created with {len(d.agent_ids)} larvae! -----')
        return d
    except:
        # print(f'xxxxx Failed to create dataset {id}! -----')
        # d.delete()
        return None


def get_datasets(datagroup_id, names, last_common='processed', folders=None, suffixes=None,
                 mode='load', load_data=True, ids=None, **kwargs):
    g = reg.loadConf(id=datagroup_id, conftype='Group')
    data_conf = g.tracker.resolution
    spatial_def = g.enrichment.metric_definition.spatial
    arena_pars = g.tracker.arena
    par_conf = g['parameterization']
    group_dir = f'{reg.Path["DATA"]}/{g["path"]}'

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
