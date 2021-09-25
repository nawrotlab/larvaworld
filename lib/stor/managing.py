import copy
import time
import warnings
from itertools import product
import pandas as pd


from lib.stor.building import build_Jovanic, build_Schleyer, build_Berni
from lib.conf.conf import *
from lib.stor.larva_dataset import LarvaDataset
import lib.aux.functions as fun
import lib.aux.naming as nam

def build_dataset(datagroup_id,id,group_id, target_dir, source_dir=None,source_files=None, **kwargs):
    warnings.filterwarnings('ignore')
    g = loadConf(datagroup_id, 'Group')
    build_conf = g['tracker']['filesystem']
    data_conf = g['tracker']['resolution']
    par_conf = g['parameterization']
    arena_pars = g['tracker']['arena']
    env_params={'arena' : arena_pars}


    try:
        shutil.rmtree(target_dir)
    except:
        pass

    d = LarvaDataset(dir=target_dir, id=id, group_id=group_id, par_conf=par_conf, env_params=env_params,
                     load_data=False, **data_conf)


    if datagroup_id in [ 'Jovanic lab']:
        step, end = build_Jovanic(d, build_conf, source_dir=source_dir, **kwargs)
    elif datagroup_id in [ 'Berni lab']:
        step, end = build_Berni(d, build_conf, source_files=source_files, **kwargs)
    elif datagroup_id in ['Schleyer lab']:
        step, end = build_Schleyer(d, build_conf, raw_folders=source_dir, **kwargs)
    else:
        raise ValueError(f'Configuration for {datagroup_id} is not supported for building new datasets')
    if step is not None:
        step.sort_index(level=['Step', 'AgentID'], inplace=True)
        end.sort_index(inplace=True)
        d.set_data(step=step, end=end)
        d.save(food=False)
        d.agent_ids = d.step_data.index.unique('AgentID').values
        d.num_ticks = d.step_data.index.unique('Step').size
        d.starting_tick = d.step_data.index.unique('Step')[0]
        print(f'--- Dataset {d.id} created with {len(d.agent_ids)} larvae! ---')
    else:
        print(f'--- Failed to create dataset {d.id}! ---')
        d.delete()
    return d


def build_datasets_old(datagroup_id, raw_folders, folders=None, suffixes=None,
                       ids=None, names=['raw'], group_ids=None, **kwargs):
    warnings.filterwarnings('ignore')
    g = loadConf(datagroup_id, 'Group')
    build_conf = g['tracker']['filesystem']
    group_dir=f'{paths.DataFolder}/{g["path"]}'
    raw_dir=f'{group_dir}/raw'

    ds = get_datasets(datagroup_id=datagroup_id, last_common='processed', names=names,
                      folders=folders, suffixes=suffixes, mode='initialize', ids=ids)
    if group_ids in [None, '']:
        group_ids = [d.id for d in ds]
    elif type(group_ids) == str:
        group_ids = [group_ids] * len(ds)
    elif len(group_ids) != len(ds):
        raise ValueError(
            f'Number of datasets ({len(ds)}) does not match number of provided group-IDs ({len(group_ids)})')
    for d, raw, group_id in zip(ds, raw_folders, group_ids):
        if datagroup_id in [ 'Jovanic lab']:
            step, end = build_Jovanic(d, build_conf, source_dir=f'{raw_dir}/{raw}', **kwargs)
        elif datagroup_id in [ 'Berni lab']:
            step, end = build_Berni(d, build_conf, source_dir=f'{raw_dir}/{raw}', **kwargs)
        elif datagroup_id in ['Schleyer lab']:

            if type(raw) == str:
                temp = [f'{raw_dir}/{raw}']
            elif type(raw) == list:
                temp = [f'{raw_dir}/{r}' for r in raw]
            step, end = build_Schleyer(d, build_conf, raw_folders=temp, **kwargs)
        else:
            raise ValueError(f'Configuration for {datagroup_id} is not supported for building new datasets')
        if step is not None:
            step.sort_index(level=['Step', 'AgentID'], inplace=True)
            end.sort_index(inplace=True)
            d.set_data(step=step, end=end)
            d.config['group_id'] = group_id
            d.save(food=False)
            d.agent_ids = d.step_data.index.unique('AgentID').values
            d.num_ticks = d.step_data.index.unique('Step').size
            d.starting_tick = d.step_data.index.unique('Step')[0]
            print(f'--- Dataset {d.id} created with {len(d.agent_ids)} larvae! ---')
        else:
            print(f'--- Failed to create dataset {d.id}! ---')
            d.delete()
    return ds


def get_datasets(datagroup_id, names, last_common='processed', folders=None, suffixes=None,
                 mode='load', load_data=True, ids=None, **kwargs):
    g = loadConf(datagroup_id, 'Group')
    data_conf = g['tracker']['resolution']
    par_conf = g['parameterization']
    arena_pars = g['tracker']['arena']
    group_dir = f'{paths.DataFolder}/{g["path"]}'

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


def enrich_datasets(datagroup_id, datasets=None, names=None, enrich_conf=None, **kwargs):
    warnings.filterwarnings('ignore')

    if datasets is None and names is not None:
        datasets = get_datasets(datagroup_id, last_common='processed', names=names, mode='load', **kwargs)
    if enrich_conf is None:
        g = loadConf(datagroup_id, 'Group')
        enrich_conf = g['enrichment']
    print()
    print(f'------ Enriching {len(datasets)} datasets ------')
    print()
    ds = [d.enrich(**enrich_conf, **kwargs) for d in datasets]
    # print()
    print(f'------ {len(ds)} datasets enriched ------')
    print()
    return ds


def analyse_datasets(datagroup_id, save_to=None, **kwargs):
    from lib.anal.plotting import comparative_analysis
    ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    if save_to is None and len(ds) > 1:
        g = loadConf(datagroup_id, 'Group')
        save_to = f'{paths.DataFolder}/{g["path"]}/plots'
    fig_dict = comparative_analysis(datasets=ds, labels=[d.id for d in ds], save_to=save_to)
    return fig_dict


def visualize_datasets(datagroup_id, save_to=None, save_as=None, vis_kwargs={}, replay_kwargs={}, **kwargs):
    warnings.filterwarnings('ignore')
    ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
    if save_to is None and len(ds) > 1:
        g = loadConf(datagroup_id, 'Group')
        save_to = f'{paths.DataFolder}/{g["path"]}/visuals'
    if save_as is None:
        save_as = [d.id for d in ds]
    for d, n in zip(ds, save_as):
        vis_kwargs['media_name'] = n
        d.visualize(save_to=save_to, vis_kwargs=vis_kwargs, **replay_kwargs)


# def compute_PIs(datagroup_id=None, save_to=None, ds=None, save_as='PIs.csv', **kwargs):
#     if ds is None:
#         ds = get_datasets(datagroup_id=datagroup_id, **kwargs)
#     ids = [d.id for d in ds]
#     if save_to is None and len(ds) > 1 and datagroup_id is not None:
#         g = loadConf(datagroup_id, 'Group')
#         save_to = f'{paths.DataFolder}/{g["path"]}/plots/PIs'
#     if not os.path.exists(save_to):
#         os.makedirs(save_to)
#     PIs = []
#     Ns = []
#     for j, d in enumerate(ds):
#         PI, N = d.compute_preference_index(return_num=True)
#         PIs.append(PI)
#         Ns.append(N)
#     df = pd.DataFrame({'PI': PIs, 'N': Ns}, index=ids)
#     df.to_csv(f'{save_to}/{save_as}', header=True, index=True)
#     print(f'PIs saved as {save_as}')
#     print(df)


def detect_dataset(datagroup_id=None, folder_path=None, raw=True, **kwargs):
    dic = {}
    if folder_path in ['', None]:
        return dic
    if raw:
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
                    for id, dr in zip(ids, dirs):
                        dic[id] = dr
            elif dFs is not None:
                if fn.startswith(dFs):
                    dic[fn] = folder_path
                else:
                    ids, dirs = detect_dataset_in_subdirs(datagroup_id, folder_path, fn, **kwargs)
                    for id, dr in zip(ids, dirs):
                        dic[id] = dr
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
            elif df_ is not None:
                fs = os.listdir(folder_path)
                ids = fun.unique_list([f.split(df_)[0] for f in fs if df_ in f])
                for id in ids:
                    dic[id] = folder_path
        return dic
    else:
        if os.path.exists(f'{folder_path}/data'):
            dd = LarvaDataset(dir=folder_path)
            dic[dd.id] = dd
        else:
            for ddr in [x[0] for x in os.walk(folder_path)]:
                if os.path.exists(f'{ddr}/data'):
                    dd = LarvaDataset(dir=ddr)
                    dic[dd.id] = dd
        return dic


def detect_dataset_in_subdirs(datagroup_id, folder_path, last_dir, full_ID=False):
    fn = last_dir
    ids, dirs = [], []
    if os.path.isdir(folder_path):
        fs = os.listdir(folder_path)
        for f in fs:
            dic = detect_dataset(datagroup_id, f'{folder_path}/{f}', full_ID=full_ID, raw=True)
            # id, dir = detect_dataset(datagroup_id, f'{folder_path}/{f}', full_ID=full_ID)
            for id, dr in dic.items():
                if full_ID:
                    ids += [f'{fn}/{id0}' for id0 in id]
                else:
                    ids.append(id)
                dirs.append(dr)
    return ids, dirs


if __name__ == '__main__':
    # dr = '/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/FRUvsQUI/Naive->PUR/EM/control'
    # dr1='/home/panos/nawrot_larvaworld/larvaworld/data/SimGroup/single_runs/nengo_dish/nengo_dish_9'
    dr1='/home/panos/nawrot_larvaworld/larvaworld/data/SimGroup/single_runs/imitation/exp_13l_imitation_0'
    # ddr='/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/processed/FRUvsQUI/Naive->PUR/EM/controls_20l'
    # c1='/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/processed/FRUvsQUI/FRU->FRU/AM/test_10l'
    # c2='/home/panos/nawrot_larvaworld/larvaworld/data/pairs/controls_20l_imitation_0'
    # c3='/home/panos/nawrot_larvaworld/larvaworld/data/SimGroup/single_runs/chemotaxis_local/test2'
    # c3='/home/panos/nawrot_larvaworld/larvaworld/data/SimGroup/single_runs/chemotaxis_local/test2'
    # pref = '/home/panos/nawrot_larvaworld/larvaworld/data/JovanicGroup/raw/3_conditions/AttP240@UAS_TNT/Starved'
    # x_contour_file = f'{pref}_x_contour.txt'
    # y_contour_file = f'{pref}_y_contour.txt'
    # xcs = pd.read_csv(x_contour_file, header=None, sep='\t')
    # ycs = pd.read_csv(y_contour_file, header=None, sep='\t')
    # xcs, ycs = fun.convex_hull(xs=xcs.values, ys=ycs.values, N=30)
    #
    # contour_xy = nam.xy(nam.contour(30))
    # xc_pars = [x for x, y in contour_xy]
    # yc_pars = [y for x, y in contour_xy]
    # xcs = pd.DataFrame(xcs, columns=xc_pars, index=None)
    # ycs = pd.DataFrame(ycs, columns=yc_pars, index=None)

    # yys=vertices[:,:][1]
    import lib.conf.dtype_dicts as dtypes
    from lib.sim.single_run import run_sim
    from lib.conf.conf import imitation_exp
    from lib.anal.comparing import ExpFitter
    vis_kwargs = dtypes.get_dict('visualization', mode='video', video_speed=60)
    dr2=paths.RefDatasetPath
    # print(paths.RefConf)
    # raise
    # dr='/home/panos/nawrot_larvaworld/larvaworld/data/JovanicGroup/processed/3_conditions/AttP240@UAS_TNT/Starved'
    for dddr in [dr2,dr1] :
        d = LarvaDataset(dddr)
        print(d.step_data['bend'].dropna().abs().groupby('AgentID').mean().mean())
        # print(d.step_data['velocity'].dropna().abs().groupby('AgentID').mean().mean())
        # print(d.endpoint_data['tortuosity_2_mean'].mean())
        # print(d.endpoint_data['tortuosity_5_mean'].mean())
        # print(d.endpoint_data['tortuosity_10_mean'].mean())
        # print(d.endpoint_data['dispersion'].mean())
        # print(d.endpoint_data['cum_dst'].mean())
        print(d.endpoint_data['dispersion_max'].mean())
        print(d.endpoint_data['final_dispersion'].mean())
        print(d.endpoint_data['dispersion_mean'].mean())
        # print(d.endpoint_data.columns[10:20])
    # # f = ExpFitter(paths.RefConf)
    # # print(f.df_st)
    raise

    exp_conf=imitation_exp(paths.RefConf, model='explorer', idx=0, save_data_flag=True)
    # exp_conf['experiment'] = 'dish'
    # exp_conf['save_data_flag'] = True
    f = ExpFitter(paths.RefConf)
    print(exp_conf['save_data_flag'])
    dd= run_sim(**exp_conf)
    print(dd.step_data['front_orientation_velocity'].dropna().abs().groupby('AgentID').mean())
    # dd= run_sim(**exp_conf, vis_kwargs=vis_kwargs)
    print(dd.endpoint_data['tortuosity_10_mean'].mean())
    fit=f.compare(dd)
    print(fit)
    # dd= run_sim(**exp_conf, vis_kwargs=vis_kwargs)
    # from lib.sim.analysis import sim_analysis
    #
    # fig_dict, results = sim_analysis(dd, 'imitation')
    # # print(results)
    # # # res=mimic_dataset(d)
    # # # d.visualize(vis_kwargs=vis_kwargs, space_in_mm=False, draw_Nsegs=2)
    # # # s,e=d.step_data,d.endpoint_data
    #
    # c=dd.config
    # print(c['sample_fit'])
        # print(c.keys())
        # print(e.columns.values[0:20])
        # print(e['tortuosity_2_mean'].mean())
        # print(e['tortuosity_5_mean'].mean())
        # print(e['tortuosity_10_mean'].mean())
        # print(e['tortuosity_20_mean'].mean())
        # print(e['tortuosity_5_std'])
        # print(e['cum_dst'])
        # print(s['x'].max(), s['x'].min())
        # print(s['y'].max(), s['y'].min())
        # print(e['cum_dur'].mean())
        # print(e['cum_dur'].std())
    # ss={'a' :[2], 'b':[3]}
    # s=pd.DataFrame.from_dict(ss)
    # bar = pd.DataFrame(np.random.randn(10, 4))
    # s0=time.time()
    # store = pd.HDFStore('test.h5')
    # store['foo'] = s  # write to HDF5
    # bar = store['foo']  # retrieve
    # store.close()

