import copy
import warnings
from itertools import product
import pandas as pd

from distutils.dir_util import copy_tree

from lib.anal.plotting import comparative_analysis, plot_marked_strides, plot_marked_turns
from lib.stor.building import build_Jovanic, build_Schleyer
from lib.conf.conf import *
from lib.stor.datagroup import LarvaDataGroup
from lib.stor.larva_dataset import LarvaDataset


def build_datasets(datagroup_id, raw_folders='each', folders=None, suffixes=None,
                   ids=None, arena_pars=None,names=['raw'], **kwargs):
    warnings.filterwarnings('ignore')
    datagroup = LarvaDataGroup(datagroup_id)
    build_conf = datagroup.get_conf()['build']
    conf_id = datagroup.get_conf()['id']
    if raw_folders == 'all':
        raw_folders = [np.sort(os.listdir(datagroup.raw_dir))]
        names = ['merged']
    elif raw_folders == 'each':
        raw_folders = [[f] for f in np.sort(os.listdir(datagroup.raw_dir))]
        names = [f'{f[0]}' for f in raw_folders]
    # elif len(raw_folders)>0 and len(names)==1 :
    #     names=names*len(raw_folders)
    # else :
    #     raise ValueError('Raw folders must be set to all or each')
    ds = get_datasets(datagroup_id=datagroup_id, last_common='processed', names=names,
                      folders=folders, suffixes=suffixes, mode='initialize', ids=ids, arena_pars=arena_pars)
    for d, raw in zip(ds, raw_folders):
        if conf_id == 'JovanicConf':
            # with fun.suppress_stdout():
            step_data, endpoint_data = build_Jovanic(d, build_conf, source_dir=f'{datagroup.raw_dir}/{raw}', **kwargs)
            # if step_data is None and endpoint_data is None :
            #     print(f'Temporarily saved {d.id} dataset')
            #     continue
        elif conf_id == 'SchleyerConf':
            step_data, endpoint_data = build_Schleyer(d, build_conf,
                                                      raw_folders=[f'{datagroup.raw_dir}/{r}' for r in raw], **kwargs)
        else:
            raise ValueError(f'Configuration {conf_id} is not supported for building new datasets')

        step_data.sort_index(level=['Step', 'AgentID'], inplace=True)
        endpoint_data.sort_index(inplace=True)
        d.set_step_data(step_data)
        d.set_end_data(endpoint_data)
        d.save(food_endpoint_data=False)
        d.agent_ids = d.step_data.index.unique('AgentID').values
        d.num_ticks = d.step_data.index.unique('Step').size
        d.starting_tick = d.step_data.index.unique('Step')[0]
        print(f'Dataset {d.id} created with {len(d.agent_ids)} larvae!')
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
        # print(dir)
        if mode == 'load':
            if not os.path.exists(dir):
                print(f'No dataset found at {dir}')
                continue
            d = LarvaDataset(dir=dir, load_data=load_data)
            # d = LarvaDataset(dir=dir, load_data=load_data, id=id)
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
    print(f'{len(ds)} datasets loaded.')
    return ds


def enrich_datasets(datagroup_id, names, keep_raw=False, enrich_conf=None, **kwargs):
    warnings.filterwarnings('ignore')
    ds = get_datasets(datagroup_id, last_common='processed', names=names, mode='load', **kwargs)
    if keep_raw:
        raw_names=[f'raw_{n}' for n in names]
        raw_ds = get_datasets(datagroup_id, last_common='processed', names=raw_names, mode='initialize', **kwargs)
        for raw, new in zip(raw_ds, ds):
            copy_tree(new.dir, raw.dir)
    if enrich_conf is None :
        enrich_conf = LarvaDataGroup(datagroup_id).get_conf()['enrich']

    # with fun.suppress_stdout():

    ds = [d.enrich(**enrich_conf, **kwargs) for d in ds]
    return ds


def analyse_datasets(datagroup_id, save_to=None, sample_individuals=False, **kwargs):
    ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    if save_to is None and len(ds) > 1:
        save_to = LarvaDataGroup(datagroup_id).plot_dir
    if sample_individuals:
        for d in ds:
            plot_marked_strides(dataset=d, agent_ids=d.agent_ids[:1], title=' ')
            try:
                plot_marked_turns(dataset=d, agent_ids=d.agent_ids[:1])
            except:
                pass
    fig_dict = comparative_analysis(datasets=ds, labels=[d.id for d in ds], save_to=save_to)
    return fig_dict


def visualize_datasets(datagroup_id, save_to=None, save_as=None, vis_kwargs={},replay_kwargs={}, **kwargs):
    warnings.filterwarnings('ignore')
    ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    if save_to is None and len(ds) > 1:
        save_to = LarvaDataGroup(datagroup_id).vis_dir
    if save_as is None :
        save_as=[d.id for d in ds]
    for d,n in zip(ds, save_as):
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

# def merge_datasets(datasets, id, dir) :
#     d0=LarvaDataset(dir, id=id)
#     N=sum([d.Nagents for d in datasets])
#     s0,e0=[], []
#     for i, d in enumerate(datasets) :
#         s = copy.deepcopy(d.step_data)
#         e = copy.deepcopy(d.endpoint_data)
#         s.index['AgentID']= f'D{i}_' + s.index['AgentID'].astype('str')
#         e.index['AgentID']= f'D{i}_' + e.index['AgentID'].astype('str')
#         # e = copy.deepcopy(d.endpoint_data)
#         s0.append(s)
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
