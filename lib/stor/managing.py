import copy
import warnings
from itertools import product
import pandas as pd

from distutils.dir_util import copy_tree

from lib.anal.plotting import comparative_analysis, plot_marked_turns, plot_marked_strides
from lib.stor.building import build_Jovanic, build_Schleyer
from lib.conf.conf import *
from lib.stor.datagroup import LarvaDataGroup
from lib.stor.larva_dataset import LarvaDataset
import lib.aux.functions as fun


def build_datasets(datagroup_id, raw_folders='each', folders=None, suffixes=None,
                   ids=None, arena_pars=None, names=['raw'], group_ids=None, **kwargs):
    warnings.filterwarnings('ignore')
    g = LarvaDataGroup(datagroup_id)
    build_conf = g.tracker['filesystem']
    # build_conf = g.get_conf()['build']
    if raw_folders == 'all':
        raw_folders = [np.sort(os.listdir(g.raw_dir))]
        names = ['merged']
    elif raw_folders == 'each':
        raw_folders = [[f] for f in np.sort(os.listdir(g.raw_dir))]
        names = [f'{f[0]}' for f in raw_folders]

    ds = get_datasets(datagroup_id=datagroup_id, last_common='processed', names=names,
                      folders=folders, suffixes=suffixes, mode='initialize', ids=ids, arena_pars=arena_pars)
    # print()
    # print(f'------ Building {len(ds)} datasets ------')
    # print()
    if group_ids in [None, '']:
        group_ids =[d.id for d in ds]
    elif type(group_ids)==str:
        group_ids=[group_ids]*len(ds)
    elif len(group_ids)!=len(ds) :
        raise ValueError (f'Number of datasets ({len(ds)}) does not match number of provided group-IDs ({len(group_ids)})')
    for d, raw, group_id in zip(ds, raw_folders, group_ids):
        if datagroup_id in ['JovanicGroup', 'JovanicFormat', 'Jovanic lab']:
            step, end = build_Jovanic(d, build_conf, source_dir=f'{g.raw_dir}/{raw}',**kwargs)
        elif datagroup_id in ['SchleyerGroup', 'SchleyerFormat', 'Schleyer lab']:
            if type(raw) == str:
                temp = [f'{g.raw_dir}/{raw}']
            elif type(raw) == list:
                temp = [f'{g.raw_dir}/{r}' for r in raw]
            step, end = build_Schleyer(d, build_conf, raw_folders=temp,**kwargs)
        else:
            raise ValueError(f'Configuration for {datagroup_id} is not supported for building new datasets')
        if step is not None :
            step.sort_index(level=['Step', 'AgentID'], inplace=True)
            end.sort_index(inplace=True)
            d.set_data(step=step, end=end)
            d.config['group_id']=group_id
            d.save(food=False)
            d.agent_ids = d.step_data.index.unique('AgentID').values
            d.num_ticks = d.step_data.index.unique('Step').size
            d.starting_tick = d.step_data.index.unique('Step')[0]
            print(f'--- Dataset {d.id} created with {len(d.agent_ids)} larvae! ---')
        else :
            print(f'--- Failed to create dataset {d.id}! ---')
            # ds.remove(d)
            d.delete()
            d=None
    # print()
    # print(f'------ {len(ds)} datasets built------')
    # print()
    return ds


def get_datasets(datagroup_id, names, last_common='processed', folders=None, suffixes=None,
                 mode='load', load_data=True, ids=None, arena_pars=None, **kwargs):
    datagroup = LarvaDataGroup(datagroup_id)
    data_conf = datagroup.tracker['resolution']
    # data_conf = datagroup.get_conf()['data']
    par_conf = datagroup.parameterization
    # par_conf = datagroup.get_par_conf()
    last_common = f'{datagroup.get_path()}/{last_common}'
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
            if arena_pars is None:
                arena_pars = datagroup.tracker['arena']
            d = LarvaDataset(dir=dir, id=id, par_conf=par_conf, arena_pars=arena_pars,
                             load_data=False, **data_conf)
        ds.append(d)
    return ds


def enrich_datasets(datagroup_id, datasets=None, names=None, keep_raw=False, enrich_conf=None, **kwargs):
    warnings.filterwarnings('ignore')
    if datasets is None and names is not None :
        datasets = get_datasets(datagroup_id, last_common='processed', names=names, mode='load', **kwargs)
    if keep_raw:
        raw_names = [f'raw_{n}' for n in names]
        raw_ds = get_datasets(datagroup_id, last_common='processed', names=raw_names, mode='initialize', **kwargs)
        for raw, new in zip(raw_ds, datasets):
            copy_tree(new.dir, raw.dir)
    if enrich_conf is None:
        # print('ffffff')
        enrich_conf = LarvaDataGroup(datagroup_id).enrichment
        # enrich_conf = LarvaDataGroup(datagroup_id).get_conf()['enrich']
    print()
    print(f'------ Enriching {len(datasets)} datasets ------')
    print()
    ds = [d.enrich(**enrich_conf, **kwargs) for d in datasets]
    # print()
    print(f'------ {len(ds)} datasets enriched ------')
    print()
    return ds


def analyse_datasets(datagroup_id, save_to=None, sample_individuals=False, **kwargs):
    ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    if save_to is None and len(ds) > 1:
        save_to = LarvaDataGroup(datagroup_id).plot_dir
    if sample_individuals:
        for d in ds:
            plot_marked_strides(datasets=[d], agent_idx=0, slice=[0, 180])
            try:
                plot_marked_turns(dataset=d, agent_ids=d.agent_ids[:1])
            except:
                pass
    fig_dict = comparative_analysis(datasets=ds, labels=[d.id for d in ds], save_to=save_to)
    return fig_dict


def visualize_datasets(datagroup_id, save_to=None, save_as=None, vis_kwargs={}, replay_kwargs={}, **kwargs):
    warnings.filterwarnings('ignore')
    ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    if save_to is None and len(ds) > 1:
        save_to = LarvaDataGroup(datagroup_id).vis_dir
    if save_as is None:
        save_as = [d.id for d in ds]
    for d, n in zip(ds, save_as):
        vis_kwargs['media_name'] = n
        d.visualize(save_to=save_to, vis_kwargs=vis_kwargs, **replay_kwargs)


def compute_PIs(datagroup_id=None, save_to=None,ds=None,save_as='PIs.csv', **kwargs):
    # filename = 'PIs.csv'
    if ds is None :
        ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    ids = [d.id for d in ds]
    if save_to is None and len(ds) > 1 and datagroup_id is not None:
        save_to = f'{LarvaDataGroup(datagroup_id).plot_dir}/PIs'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    PIs = []
    Ns = []
    for j, d in enumerate(ds):
        PI, N = d.compute_preference_index(return_num=True)
        PIs.append(PI)
        Ns.append(N)
    df = pd.DataFrame({'PI': PIs, 'N': Ns}, index=ids)
    df.to_csv(f'{save_to}/{save_as}', header=True, index=True)
    print(f'PIs saved as {save_as}')
    print(df)


def detect_dataset(datagroup_id=None, folder_path=None,raw=True, **kwargs):
    dic={}
    if folder_path in ['', None]:
        return dic
    if raw :
        conf = loadConf(datagroup_id, 'Group')['tracker']['filesystem']
        if 'detect' in conf.keys():
            d = conf['detect']
            dF, df = d['folder'], d['file']
            dFp, dFs = dF['pref'], dF['suf']
            dfp, dfs, df_ = df['pref'], df['suf'], df['sep']

            fn = folder_path.split('/')[-1]
            if dFp is not None:
                if fn.startswith(dFp):
                    dic[fn] = folder_path
                else:
                    ids, dirs = detect_dataset_in_subdirs(datagroup_id, folder_path, fn, **kwargs)
                    for id, dr in zip(ids,dirs) :
                        dic[id]=dr
            elif dFs is not None:
                if fn.startswith(dFs):
                    dic[fn] = folder_path
                else:
                    ids, dirs = detect_dataset_in_subdirs(datagroup_id, folder_path, fn, **kwargs)
                    for id, dr in zip(ids,dirs) :
                        dic[id]=dr
            elif dfp is not None:
                fs = os.listdir(folder_path)
                ids, dirs = [f.split(df_)[1:][0] for f in fs if f.startswith(dfp)], [folder_path]
                for id, dr in zip(ids, dirs):
                    dic[id] = dr
            elif dfs is not None:
                fs = os.listdir(folder_path)
                ids = [f.split(df_)[:-1][0] for f in fs if f.endswith(dfs)]
                for id in ids:
                    dic[id] = folder_path
        return dic
    else :
        if os.path.exists(f'{folder_path}/data'):
            dd = LarvaDataset(dir=folder_path)
            dic[dd.id]=dd
        else:
            for ddr in [x[0] for x in os.walk(folder_path)]:
                if os.path.exists(f'{ddr}/data'):
                    dd = LarvaDataset(dir=ddr)
                    dic[dd.id]=dd
        return dic

def detect_dataset_in_subdirs(datagroup_id, folder_path, last_dir, full_ID=False) :
    fn=last_dir
    ids, dirs = [], []
    if os.path.isdir(folder_path) :
        fs = os.listdir(folder_path)
        for f in fs:
            dic = detect_dataset(datagroup_id, f'{folder_path}/{f}', full_ID=full_ID, raw=True)
            # id, dir = detect_dataset(datagroup_id, f'{folder_path}/{f}', full_ID=full_ID)
            for id, dr in dic.items():
                if full_ID :
                    ids += [f'{fn}/{id0}' for id0 in id]
                else :
                    ids.append(id)
                dirs.append(dr)
    return ids, dirs

if __name__ == '__main__':
    import lib.conf.dtype_dicts as dtypes
    vis_kwargs=dtypes.get_dict('visualization', mode='video', draw_head=True, video_speed=100)
    d=LarvaDataset('/home/panos/nawrot_larvaworld/larvaworld/data/JovanicGroup/processed/food_line/SHAM/Starved')
    d.visualize(vis_kwargs=vis_kwargs)
    # s,e=d.step_data,d.endpoint_data
    # a=s.xs(d.agent_ids[0], level='AgentID', drop_level=True)
    # print(a[list(d.contour_xy[0])])
    # print(a[list(d.contour_xy[1])])
    # print(s['point3_x'].min(), s['point3_x'].max())
    # print(s.columns.values[20:100])
    # print(d.config)
#     # print(s['head_x'].min(), s['head_x'].max())
#     # print(s['head_y'].min(), s['head_y'].max())
#     # print(d.config)
#     # print(d.config['point'])