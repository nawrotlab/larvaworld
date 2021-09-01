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
                   ids=None, arena_pars=None, names=['raw'],**kwargs):
    warnings.filterwarnings('ignore')
    g = LarvaDataGroup(datagroup_id)
    build_conf = g.get_conf()['build']
    conf_id = g.get_conf()['id']
    if raw_folders == 'all':
        raw_folders = [np.sort(os.listdir(g.raw_dir))]
        names = ['merged']
    elif raw_folders == 'each':
        raw_folders = [[f] for f in np.sort(os.listdir(g.raw_dir))]
        names = [f'{f[0]}' for f in raw_folders]

    ds = get_datasets(datagroup_id=datagroup_id, last_common='processed', names=names,
                      folders=folders, suffixes=suffixes, mode='initialize', ids=ids, arena_pars=arena_pars)
    print()
    print(f'------ Building {len(ds)} datasets ------')
    print()
    for d, raw in zip(ds, raw_folders):
        if conf_id == 'JovanicConf':
            step, end = build_Jovanic(d, build_conf, source_dir=f'{g.raw_dir}/{raw}',**kwargs)
        elif conf_id == 'SchleyerConf':
            if type(raw) == str:
                temp = [f'{g.raw_dir}/{raw}']
            elif type(raw) == list:
                temp = [f'{g.raw_dir}/{r}' for r in raw]
            step, end = build_Schleyer(d, build_conf, raw_folders=temp,**kwargs)
        else:
            raise ValueError(f'Configuration {conf_id} is not supported for building new datasets')
        step.sort_index(level=['Step', 'AgentID'], inplace=True)
        end.sort_index(inplace=True)
        d.set_data(step=step, end=end)
        d.save(food=False)
        d.agent_ids = d.step_data.index.unique('AgentID').values
        d.num_ticks = d.step_data.index.unique('Step').size
        d.starting_tick = d.step_data.index.unique('Step')[0]
        print(f'--- Dataset {d.id} created with {len(d.agent_ids)} larvae! ---')
    print()
    print(f'------ {len(ds)} datasets built------')
    print()
    return ds


def get_datasets(datagroup_id, names, last_common='processed', folders=None, suffixes=None,
                 mode='load', load_data=True, ids=None, arena_pars=None, **kwargs):
    datagroup = LarvaDataGroup(datagroup_id)
    data_conf = datagroup.get_conf()['data']
    par_conf = datagroup.get_par_conf()
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
                arena_pars = datagroup.arena_pars
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
        enrich_conf = LarvaDataGroup(datagroup_id).get_conf()['enrich']
    print()
    print(f'------ Enriching {len(datasets)} datasets ------')
    print()
    ds = [d.enrich(**enrich_conf, **kwargs) for d in datasets]
    print()
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


def compute_PIs(datagroup_id, save_to=None, **kwargs):
    filename = 'PIs.csv'
    ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    ids = [d.id for d in ds]
    if save_to is None and len(ds) > 1:
        save_to = f'{LarvaDataGroup(datagroup_id).plot_dir}/PIs'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    PIs = []
    Ns = []
    for j, d in enumerate(ds):
        PI, N = d.compute_preference_index(return_num=True)
        PIs.append(PI)
        Ns.append(N)
        # print(j, PI, N)
    # for i in range(len(PIs)):
    #     print(exp_labels[i], PIs[i], Ns[i])
    df = pd.DataFrame({'PI': PIs, 'N': Ns}, index=ids)
    df.to_csv(f'{save_to}/{filename}', header=True, index=True)
    print(f'PIs saved as {filename}')


def detect_dataset(datagroup_id, folder_path,raw=True, **kwargs):
    dic={}
    if raw :
        # ids, dirs = [], []
        conf = loadConf(datagroup_id, 'Group')
        if 'detect' in conf.keys():
            d = conf['detect']
            dF, df = d['folder'], d['file']
            dFp, dFs = dF['pref'], dF['suf']
            dfp, dfs, df_ = df['pref'], df['suf'], df['sep']

            fn = folder_path.split('/')[-1]
            if dFp is not None:
                if fn.startswith(dFp):
                    dic[fn] = folder_path
                    # ids, dirs = [fn], [folder_path]
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
                ids, dirs = [f.split(df_)[:-1][0] for f in fs if f.endswith(dfs)], [folder_path]
                for id, dr in zip(ids, dirs):
                    dic[id] = dr
        return dic
    else :
        # ids, dds = [], []
        if os.path.exists(f'{folder_path}/data'):
            dd = LarvaDataset(dir=folder_path)
            dic[dd.id]=dd
            # ids, dds = [dd.id], [dd]
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

# def merge_datasets(datasets, id, dir) :
#     d0=LarvaDataset(dir, id=id)
#     N=sum([d.Nagents for d in datasets])
#     s0,e0=[], []
#     for i, d in enumerate(datasets) :
#         sigma = copy.deepcopy(d.step)
#         e = copy.deepcopy(d.end)
#         sigma.index['AgentID']= f'D{i}_' + sigma.index['AgentID'].astype('str')
#         e.index['AgentID']= f'D{i}_' + e.index['AgentID'].astype('str')
#         # e = copy.deepcopy(d.end)
#         s0.append(sigma)
#         e0.append(e)
#     s0=
#
#     dd.config=datasets[0].config
# k=get_datasets(datagroup_id='JovanicGroup', last_common='processed/AttP2@UAS_TNT', names = ['enriched_dataset'],
#                 folders=['Fed', 'ProteinDeprived', 'Starved'], suffixes=None, load_data=True)

# ds = get_datasets(datagroup_id='SimGroup', last_common='single_runs', names=['dish', 'chemorbit'],
#                   folders=['dish', 'chemorbit'], suffixes=[1, 2, 3, 4], load_data=True)

# raw_ds = get_datasets(datagroup_id='TestGroup', last_common='processed', names=['raw_dish'],
#                       folders=None, suffixes=[0, 1, 2], mode='initialize')

# ds = get_datasets(datagroup_id='TestGroup', last_common='processed', names=['enriched_dish'],
#                   folders=None, suffixes=[0,1,2], mode='create', load_data=True)
#
# for raw, new in zip(raw_ds, ds) :
#     copy_tree(raw.dir, new.dir)
# ds=[d.enrich() for d in ds]
# cs=['Fed', 'Starved']
# k=build_datasets('JovanicGroup', names=['raw' for c in cs], raw_folders=[f'raw/AttP240@UAS_TNT/{c}' for c in cs],
#                  folders=[f'AttP240@UAS_TNT/{c}' for c in cs],
#                suffixes=None, max_Nagents=None, min_Nids=200)

# k=build_datasets('TestGroup', names=['raw_dish'], raw_folders=[['dish_0'],['dish_1'], ['dish_2']],
#                  folders=None, suffixes=[0,1,2], ids=None)

# k=build_datasets('TestGroup', names=['raw_merged'], raw_folders=[['dish_0','dish_1','dish_2']],
#                  folders=None, ids=None)

if __name__ == "__main__":
    # folder_path = '/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/raw'
    # folder_path = '/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/raw/FRUconc'
    # folder_path = '/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/raw/FRUconc/High'
    # folder_path = '/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/raw/FRUconc/High/AM+'
    # folder_path='/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/processed/FRUconc/High/EM+/full_dish'
    # ids=detect_dataset(datagroup_id='SchleyerGroup', folder_path='/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/raw/FRUconc/High/box1-2016-05-23_12_41_17')
    # ids, dirs = detect_dataset(datagroup_id='SchleyerGroup', folder_path=folder_path, full_ID=True)
    # print(os.listdir(folder_path))
    # print(ids)
    print()
    # print(dirs)
    # dr = '/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/processed/FRUconc/High/AM+/box1-2016-05-23_12_41_17'
    # d = LarvaDataset(dir=dr)
    # s,e=d.step_data, d.endpoint_data
    # k=os.path.exists(d.dir_dict['conf'])
    # print(k)
    #
    # with open(d.dir_dict['conf']) as tfp:
    #     c = json.load(tfp)
    # # print(os.listdir(d.data_dir))
    # print(s['point5_x'].dropna().max())
    # print(s['point5_x'].dropna().min())
    # print(s['point5_y'].dropna().max())
    # print(s['point5_y'].dropna().min())
    # print(d.config)
    # d.visualize()

