import copy
import json
import sys
import shutil
import os

import numpy as np

import lib.aux.dictsNlists


def get_input(message, itype, default='', accepted=None, range=None):
    while True:
        t = input(f'{message} (default : {default}) :')
        if t == '':
            if default != '':
                v = default
                break
            else:
                print(f"Input not given and default not set! Try again.")
                continue
        if itype == type:
            if t == 'str':
                v = str
                break
            if t == 'int':
                v = int
                break
            if t == 'float':
                v = float
                break
            if t == 'bool':
                v = bool
                break
            else:
                print("Data mode must be one of ['str','int','float','bool']")
        if itype == bool:
            if t in ['n', 'N', 'False', False]:
                v = False
                break
            elif t in ['y', 'Y', 'True', True]:
                v = True
                break
            else:
                print(f"Input must be converted to boolean! Type 'n'/'N' or 'y'/'Y'.")
                continue
        try:
            t = itype(t)
        except:
            print(f"Input must be of mode {itype}! Try again.")
            continue
        if accepted is None and range is None:
            v = t
            break
        elif accepted is not None:
            if t in accepted:
                v = t
                break
            else:
                print(f"Input must be one of {accepted}")
                continue
        elif range is not None:
            if range[0] <= t <= range[1]:
                v = t
                break
            else:
                print(f"Input must be within {range}")
                continue
    print(f'Input stored : {v}')
    return v


def setDataConf():
    Npoints = get_input("Enter the number of midline points tracked per larva", itype=int, range=[0, +np.inf],
                        default=12)
    Ncontour = get_input("Enter the number of contour points tracked per larva", itype=int, range=[0, +np.inf],
                         default=22)
    fr = get_input("Enter the framerate of the tracking", itype=float, range=[0, +np.inf], default=16.0)

    data_conf = {'fr': fr,
                 'Npoints': Npoints,
                 'Ncontour': Ncontour}
    return data_conf


def setParConf(N):
    bend_mode = get_input("How is the body bend computed? Enter 1 if computed from angles, 2 if computed from vectors",
                          itype=int, accepted=[1, 2], default=1)
    if bend_mode == 1:
        bend = 'from_angles'
    elif bend_mode == 2:
        bend = 'from_vectors'
    f0 = get_input("Enter midline point where front vector starts", itype=int, range=[1, N - 3], default=1)
    f1 = get_input("Enter midline point where front vector ends", itype=int, range=[f0 + 1, N - 2], default=2)
    r0 = get_input("Enter midline point where rear vector starts", itype=int, range=[f1, N - 1], default=7)
    r1 = get_input("Enter midline point where rear vector ends", itype=int, range=[r0 + 1, N], default=11)
    fr_ratio = get_input("Enter ratio between front and rear vectors", itype=float, range=[0, 1], default=0.5)
    point_idx = get_input("Enter midline point used for linear velocity computation (nan defines centroid)", itype=int,
                          range=[1, +np.inf], default=np.nan)
    use_component_vel = get_input("Use component linear velocity?", itype=bool, default=False)
    scaled_vel_threshold = get_input("Enter scal linear velocity threshold for stride detection", itype=float,
                                     range=[0, +np.inf], default=0.2)
    par_conf = {'bend': bend,
                'front_vector_start': f0,
                'front_vector_stop': f1,
                'rear_vector_start': r0,
                'rear_vector_stop': r1,
                'front_body_ratio': fr_ratio,
                'point_idx': point_idx,
                'use_component_vel': use_component_vel,
                'scaled_vel_threshold': scaled_vel_threshold}
    return par_conf


def setEnrichConf():
    rescale_by = get_input("Rescale xy coordinates by a scalar :", itype=float, default=None)
    drop_collisions = get_input("Drop timesteps where collisions are detected?", itype=bool, default=True)
    filter_f = get_input("Enter cut-off frequency for xy filtering", itype=float, default=2.0)
    interpolate_nans = get_input("Interpolate missing track xy coordinates?", itype=bool, default=False)
    length_and_centroid = get_input("Compute length and centroid?", itype=bool, default=True)
    drop_contour = get_input("Drop contour xy coordinates?", itype=bool, default=False)
    drop_non_simulated_parameters = get_input("Drop parameters not used in simulation?", itype=bool, default=False)
    drop_immobile = get_input("Drop larvae that are immobile during tracking?", itype=bool, default=False)
    ang_analysis = get_input("Perform angular analysis?", itype=bool, default=True)
    lin_analysis = get_input("Perform linear analysis?", itype=bool, default=True)
    N_dispersion_starts = get_input("Enter number of starting timepoints for dispersion computation", itype=int,
                                    range=[1, 10], default=1)
    dispersion_starts = []
    for i in range(N_dispersion_starts):
        dispersion_starts.append(
            get_input(f"Enter starting timepoints for dispersion computation {i} of {N_dispersion_starts}", itype=float,
                      range=[0.0, +np.inf], default=0.0))
    dispersion_starts = list(set(dispersion_starts))
    bout_annotation = []
    for ep in ['turn', 'stride', 'pause']:
        if get_input(f"Detect epochs of {ep}?", itype=bool, default=True):
            bout_annotation.append(ep)
    mode = get_input("Perform minimal or full enrichment?", itype=str, accepted=['full', 'minimal'], default='minimal')

    enrich_config = {'rescale_by': rescale_by,
                     'drop_collisions': drop_collisions,
                     'interpolate_nans': interpolate_nans,
                     'filter_f': filter_f,
                     'length_and_centroid': length_and_centroid,
                     'drop_contour': drop_contour,
                     'drop_unused_pars': drop_non_simulated_parameters,
                     'drop_immobile': drop_immobile,
                     'ang_analysis': ang_analysis,
                     'lin_analysis': lin_analysis,
                     'dispersion_starts': dispersion_starts,
                     'bouts': bout_annotation,
                     'mode': mode}
    return enrich_config


def setConf(id):
    import lib.conf.data_conf as dat

    print(f' --- Definition of Configuration : {id} --- ')
    print(f' - Step 1 : Data configuration')
    data_conf = setDataConf()

    print(f' - Step 2 : Raw data configuration')
    Ncols = get_input("Enter sequence of columns in raw data", itype=int, range=[0, +np.inf],
                      default=len(dat.Schleyer_raw_cols))
    read_sequence = []
    for i in range(Ncols):
        read_sequence.append(
            get_input(f"Enter raw column name {i} of {Ncols}", itype=str, default=dat.Schleyer_raw_cols[i]))
    read_metadata = get_input("Read dataset metadata?", itype=bool, default=True)

    print(f' - Step 3 : Enrichment configuration')
    enrich_conf = setEnrichConf()

    print(f' - Step 4 : Parameter configuration')
    if get_input("Use an existing parameter configuration?", itype=bool, default=False):
        par_conf_id = get_input("Enter the id of the parameter configuration to use", itype=str,
                                accepted=list(loadConfDict('Par').keys()))
    else:
        par_conf_id = get_input("Enter the id of the new parameter configuration", itype=str, default=f'{id}Par')
        par_conf = setParConf(data_conf['Npoints'])
        saveConf(par_conf, 'Par', par_conf_id)

    conf = {'id': id,
            'data': data_conf,
            'par': par_conf_id,
            'build': {'read_sequence': read_sequence,
                      'read_metadata': read_metadata},
            'enrich': enrich_conf}
    saveConf(conf, 'Data')
    return conf


def setDataGroup(id=None):
    import lib.conf.env_conf as env
    if id is None:
        id = get_input("Enter DataGroup id", itype=str)
    print(f' ----- Registration of new DataGroup : {id} ----- ')

    print(f' -- Step 1 : DataGroup Configuration')
    if get_input("Use an existing configuration? ", itype=bool, default=False):
        conf_id = get_input("Enter the id of the DataGroup Configuration to use", itype=str,
                            accepted=list(loadConfDict('Data').keys()))
    else:
        conf_id = get_input("Enter the id of the new DataGroup Configuration", itype=str, default=f'{id}Conf')
        conf = setConf(conf_id)

    print(f' -- Step 2 : DataGroup Path')
    path = get_input("Enter DataGroup relative path", itype=str, default=id)

    print(f' -- Step 3 : DataGroup Arena')
    shape = get_input("Enter arena shape", itype=str, accepted=['circular', 'rectangular'], default='circular')
    if shape == 'circular':
        r = get_input("Enter arena radius in m", itype=float, range=[0, +np.inf], default=0.15)
        arena_pars = env.dish(r)
    elif shape == 'rectangular':
        x = get_input("Enter arena x dimension in m", itype=float, range=[0, +np.inf], default=0.15)
        y = get_input("Enter arena y dimension in m", itype=float, range=[0, +np.inf], default=0.15)
        arena_pars = env.arena(x, y)

    print(f' -- Step 4 : DataGroup Subgroups')
    Nsubgroups = get_input("How many subgroups are there?", itype=int, default=0)
    subgroups = []
    for i in range(Nsubgroups):
        subgroups.append(get_input(f"Set id of subgroup {i} of {Nsubgroups}", itype=str, default=f'subgroup_{i}'))

    DataGroup = {
        'id': id,
        'conf': conf_id,
        'path': path,
        'arena_pars': arena_pars,
        'subgroups': subgroups
    }
    print(f' -- Step 5 : DataGroup additional parameters')
    while get_input("Add additional parameter?", itype=bool, default=False):
        key = get_input("Enter additional parameter name", itype=str)
        itype = get_input(f"Enter data mode for {key}", itype=type)
        value = get_input(f"Enter value for {key}", itype=itype)
        DataGroup[key] = value
    saveConf(DataGroup, 'Group')
    return DataGroup


def loadConf(id, conf_type):
    try:
        conf_dict = loadConfDict(conf_type)
        conf = conf_dict[id]
        return conf
    except:
        raise ValueError(f'{conf_type} Configuration {id} does not exist')

def expandConf(id, conf_type):
    conf = loadConf(id, conf_type)
    # print(conf.keys(), id)
    try:
        if conf_type=='Batch' :
            conf['exp'] = expandConf(conf['exp'], 'Exp')
        elif conf_type=='Exp' :
            conf['env_params']=expandConf(conf['env_params'], 'Env')
            conf['life_params'] = loadConf(conf['life_params'], 'Life')
        elif conf_type=='Env' :
            for k, v in conf['larva_groups'].items():
                if type(v['model']) == str:
                    v['model'] = loadConf(v['model'], 'Model')
    except :
        pass
    return conf


def loadConfDict(conf_type):
    from lib.stor.paths import conf_paths
    try :
        with open(conf_paths[conf_type]) as tfp:
            Conf_dict = json.load(tfp)
        return Conf_dict
    except :
        return {}


def saveConf(conf, conf_type, id=None, mode='overwrite'):
    try:
        conf_dict = loadConfDict(conf_type)
    except:
        conf_dict = {}
    if id is None:
        id = conf['id']

    if id in list(conf_dict.keys()):
        for k, v in conf.items():
            if type(k) == dict and k in list(conf_dict[id].keys()) and mode == 'update':
                conf_dict[id][k].update(conf[k])
            else:
                conf_dict[id][k] = v
    else:
        conf_dict[id] = conf
    saveConfDict(conf_dict, conf_type)
    print(f'{conf_type} Configuration saved under the id : {id}')


def saveConfDict(ConfDict, conf_type):
    from lib.stor.paths import conf_paths
    with open(conf_paths[conf_type], "w") as fp:
        json.dump(ConfDict, fp)


def deleteConf(id, conf_type):
    if conf_type == 'Data':
        DataGroup = loadConf(id, conf_type)
        path = DataGroup['path']
        try:
            shutil.rmtree(path)
        except:
            pass
    conf_dict = loadConfDict(conf_type)
    try:
        conf_dict.pop(id, None)
        saveConfDict(conf_dict, conf_type)
        print(f'Deleted {conf_type} configuration under the id : {id}')
    except:
        pass


def initializeDataGroup(id):
    from lib.stor.paths import DataFolder
    DataGroup = loadConf(id, 'Group')
    path = DataGroup['path']
    raw_path = f'{path}/raw'
    processed_path = f'{path}/processed'
    plot_path = f'{path}/plots'
    visuals_path = f'{path}/visuals'
    subgroups = DataGroup['subgroups']
    dirs = [f'{DataFolder}/{i}' for i in [path, raw_path, processed_path, plot_path, visuals_path]]
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)


def next_idx(exp, type='single'):
    from lib.stor.paths import SimIdx_path
    try:
        with open(SimIdx_path) as tfp:
            idx_dict = json.load(tfp)
    except:
        exp_names = list(loadConfDict('Exp').keys())
        batch_names = list(loadConfDict('Batch').keys())
        essay_names = list(loadConfDict('Essay').keys())
        exp_idx_dict = dict(zip(exp_names, [0] * len(exp_names)))
        batch_idx_dict = dict(zip(batch_names, [0] * len(batch_names)))
        essay_idx_dict = dict(zip(essay_names, [0] * len(essay_names)))
        # batch_idx_dict.update(loadConfDict('Batch'))
        idx_dict = {'single': exp_idx_dict,
                    'batch': batch_idx_dict,
                    'essay' : essay_idx_dict}
    if not exp in idx_dict[type].keys():
        idx_dict[type][exp] = 0
    idx_dict[type][exp] += 1
    with open(SimIdx_path, "w") as fp:
        json.dump(idx_dict, fp)
    return idx_dict[type][exp]


def store_reference_data_confs() :
    from lib.stor.larva_dataset import LarvaDataset
    from lib.stor.paths import DataFolder
    import lib.aux.colsNstr as fun
    dds = [
        [f'{DataFolder}/JovanicGroup/processed/3_conditions/AttP{g}@UAS_TNT/{c}' for g
         in ['2', '240']] for c in ['Fed', 'Deprived', 'Starved']]
    dds = lib.aux.dictsNlists.flatten_list(dds)
    dds.append(f'{DataFolder}/SchleyerGroup/processed/FRUvsQUI/Naive->PUR/EM/exploration')
    for dr in dds:
        d = LarvaDataset(dr, load_data=False)
        # # c = d.config
        # del d.config['agent_ids']
        # d.config['bout_distros']['stride']=d.config['bout_distros']['stride']['best']
        # d.config['bout_distros']['pause']=d.config['bout_distros']['pause']['best']
        d.save_config(add_reference=True)

def store_confs(keys=None) :
    if keys is None :
        keys=['Ref','Data', 'Model', 'Env', 'Exp']

    if 'Ref' in keys :
        store_reference_data_confs()
    if 'Data' in keys :
        import lib.conf.data_conf as dat
        dat_list = [
            dat.SchleyerConf,
            dat.JovanicConf,
            dat.SimConf,
        ]
        for d in dat_list:
            saveConf(d, 'Data')

        par_conf_dict = {
            'SchleyerParConf': dat.SchleyerParConf,
            'JovanicParConf': dat.JovanicParConf,
            'PaisiosParConf': dat.PaisiosParConf,
            'SinglepointParConf': dat.SinglepointParConf,
            'SimParConf': dat.SimParConf,
        }
        for k, v in par_conf_dict.items():
            saveConf(v, 'Par', k)
        group_list = [
            dat.SchleyerFormat,
            dat.JovanicFormat,
            dat.BerniFormat,
        ]
        for g in group_list:
            saveConf(g, 'Group')
    if 'Model' in keys:
        from lib.conf.larva_conf import mod_dict
        for k, v in mod_dict.items():
            saveConf(v, 'Model', k)
    if 'Env' in keys :
        from lib.conf.env_conf import env_dict
        for k, v in env_dict.items():
            saveConf(v, 'Env', k)
    if 'Exp' in keys :
        import lib.conf.exp_conf as exp
        import lib.conf.essay_conf as essay
        import lib.conf.batch_conf as bat
        from lib.aux.dictsNlists import merge_dicts

        d = exp.grouped_exp_dict
        exp_dict = merge_dicts(list(d.values()))
        exp_group_dict = {k: {'simulations': list(v.keys())} for k, v in d.items()}
        for k, v in exp_dict.items():
            saveConf(v, 'Exp', k)
        for k, v in exp_group_dict.items():
            saveConf(v, 'ExpGroup', k)

        for k, v in essay.essay_dict.items():
            saveConf(v, 'Essay', k)

        for k, v in bat.batch_dict.items():
            saveConf(v, 'Batch', k)






























# if __name__ == '__main__':
#     init_confs()

def imitation_exp(config, model='explorer', idx=0, **kwargs):
    from lib.conf.dtype_dicts import base_enrich
    from lib.conf.init_dtypes import null_dict
    if type(config)==str :
        config=loadConf(config, 'Ref')
    from lib.anal.comparing import ExpFitter
    # f = ExpFitter(config)

    id = config['id']
    base_larva = expandConf(model, 'Model')

    sim_params = {
        'timestep': 1/config['fr'],
        'duration': config['duration'] / 60,
        'path': 'single_runs/imitation',
        'sim_ID': f'{id}_imitation_{idx}',
        # 'sample': id,
        'Box2D': False
    }
    env_params =null_dict('env_conf', arena=config['env_params']['arena'], larva_groups={'ImitationGroup': null_dict('LarvaGroup', sample= config, model= base_larva, default_color = 'blue', imitation=True, distribution=None)})

    exp_conf=null_dict('exp_conf', sim_params=sim_params, env_params=env_params, life_params=null_dict('life'), enrichment=base_enrich())
    # print(config)
    # exp_conf = expandConf(exp, 'Exp')
    # exp_conf['env_params']['larva_groups'] = {'ImitationGroup': null_dict('LarvaGroup', sample= config, model= base_larva, default_color = 'blue', imitation=True, distribution=None)}
    # exp_conf['env_params']['arena'] = config['env_params']['arena']
    # exp_conf['sim_params'] = sim_params
    exp_conf['experiment'] = 'imitation'
    exp_conf.update(**kwargs)
    # print(exp_conf.keys())
    return exp_conf


def get_exp_conf(exp_type, sim_params, life_params=None, N=None, larva_model=None):
    conf = copy.deepcopy(expandConf(exp_type, 'Exp'))
    # print(conf['sample'])
    for k in list(conf['env_params']['larva_groups'].keys()):
        if N is not None:
            conf['env_params']['larva_groups'][k]['N'] = N
        if larva_model is not None:
            conf['env_params']['larva_groups'][k]['model'] = loadConf(larva_model, 'Model')
    if life_params is not None:
        conf['life_params'] = life_params

    if sim_params['sim_ID'] is None:
        idx = next_idx(exp_type)
        sim_params['sim_ID'] = f'{exp_type}_{idx}'
    if sim_params['path'] is None:
        sim_params['path'] = f'single_runs/{exp_type}'
    if sim_params['duration'] is None:
        sim_params['duration'] = conf['sim_params']['duration']
    conf['sim_params'] = sim_params
    conf['experiment'] = exp_type

    return conf