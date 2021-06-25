import json
import sys
import shutil
import os
import numpy as np

from lib.stor import paths as paths

sys.path.insert(0, paths.get_parent_dir())

import lib.conf.larva_conf as mod
import lib.conf.data_conf as dat
import lib.conf.batch_conf as bat
import lib.conf.exp_conf as exp


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


def loadConfDict(conf_type):
    with open(paths.conf_paths[conf_type]) as tfp:
        Conf_dict = json.load(tfp)
    return Conf_dict


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
    with open(paths.conf_paths[conf_type], "w") as fp:
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
    DataGroup = loadConf(id, 'Group')
    path = DataGroup['path']
    raw_path = f'{path}/raw'
    processed_path = f'{path}/processed'
    plot_path = f'{path}/plots'
    visuals_path = f'{path}/visuals'
    subgroups = DataGroup['subgroups']
    dirs = [f'{paths.DataFolder}/{i}' for i in [path, raw_path, processed_path, plot_path, visuals_path]]
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)


if __name__ == '__main__':
    import lib.conf.env_conf as env

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
        dat.SchleyerGroup,
        dat.JovanicGroup,
        dat.TestGroup,
        dat.SimGroup,
    ]
    for g in group_list:
        saveConf(g, 'Group')

    env_dict = {
        'focus': env.focus_env,
        'dish': env.dish_env,
        'dispersion': env.dispersion_env,
        'chemotaxis_approach': env.chemotax_env,
        'chemotaxis_local': env.chemorbit_env,
        'chemotaxis_diffusion': env.chemorbit_diffusion_env,
        'odor_pref_test': env.pref_test_env,
        'odor_pref_test_on_food': env.pref_test_env_on_food,
        'odor_pref_train': env.pref_train_env,
        'odor_preference_RL': env.pref_env_RL,
        'patchy_food': env.patchy_food_env,
        'uniform_food': env.uniform_food_env,
        'food_grid': env.food_grid_env,
        'growth': env.growth_env,
        'rovers_sitters': env.rovers_sitters_env,
        'reorientation': env.reorientation_env,
        'realistic_imitation': env.imitation_env_p,
        'maze': env.maze_env,
        'keep_the_flag': env.king_env,
        'capture_the_flag': env.flag_env,
        'catch_me': env.catch_me_env,
        'chemotaxis_RL': env.RL_chemorbit_env,
        'food_at_bottom': env.food_at_bottom_env,
        '4corners': env.RL_4corners_env,
    }
    for k, v in env_dict.items():
        saveConf(v, 'Env', k)

    mod_dict = {
        'explorer': mod.exploring_larva,
        'navigator': mod.odor_larva,
        'navigator-x2': mod.odor_larva_x2,
        'immobile': mod.immobile_odor_larva,
        'feeder-explorer': mod.feeding_larva,
        'feeder-navigator': mod.feeding_odor_larva,
        'feeder-navigator-x2': mod.feeding_odor_larva_x2,
        'rover': mod.growing_rover,
        'mock_rover': mod.mock_growing_rover,
        'sitter': mod.growing_sitter,
        'mock_sitter': mod.mock_growing_sitter,
        'imitation': mod.imitation_larva,
        'gamer': mod.flag_larva,
        'gamer-L': mod.king_larva_L,
        'gamer-R': mod.king_larva_R,
        'follower-R': mod.follower_R,
        'follower-L': mod.follower_L,
        'RL-learner': mod.RL_odor_larva,
        'RL-feeder': mod.RL_feed_odor_larva,
        'basic_navigator': mod.basic_larva,
        'explorer_3con': mod.exploring_3c_larva,
    }
    for k, v in mod_dict.items():
        saveConf(v, 'Model', k)

    batch_dict = {
        # 'focused view': env.focus_env,
        # 'dish': env.dish_env,
        # 'dispersion': env.dispersion_env,
        'chemotaxis_approach': bat.chemotax_batch,
        'chemotaxis_local': bat.chemorbit_batch,
        # 'chemotaxis local diffusion': env.chemorbit_diffusion_env,
        'odor_preference': bat.odor_pref_batch,
        'patchy_food': bat.uniform_food_batch,
        # 'uniform food': env.uniform_food_env,
        'food_grid': bat.food_grid_batch,
        'growth': bat.growth_batch,
        'rovers_sitters': bat.rovers_sitters_batch,
        # 'reorientation': env.reorientation_env,
        # 'realistic imitation': env.imitation_env_p,
        # 'maze': env.maze_env,
        # 'keep the flag': env.king_env,
        # 'flag to base': env.flag_env,
        # 'RL chemotaxis local': env.RL_chemorbit_env,
    }
    for k, v in batch_dict.items():
        saveConf(v, 'Batch', k)

    exp_dict = {
        'focus': exp.focus,
        'dish': exp.dish,
        'dispersion': exp.dispersion,
        'chemotaxis_approach': exp.chemotax,
        'chemotaxis_local': exp.chemorbit,
        'chemotaxis_diffusion': exp.chemorbit_diffusion,
        'odor_pref_test': exp.odor_pref_test,
        'odor_pref_test_on_food': exp.odor_pref_test_on_food,
        'odor_pref_train': exp.odor_pref_train,
        'odor_pref_RL': exp.odor_pref_RL,
        'patchy_food': exp.patchy_food,
        'uniform_food': exp.uniform_food,
        'food_grid': exp.food_grid,
        'growth': exp.growth,
        'rovers_sitters': exp.rovers_sitters,
        'reorientation': exp.reorientation,
        'realistic_imitation': exp.imitation,
        'maze': exp.maze,
        'keep_the_flag': exp.keep_the_flag,
        'capture_the_flag': exp.capture_the_flag,
        'catch_me': exp.catch_me,
        'chemotaxis_RL': exp.chemotaxis_RL,
        'food_at_bottom': exp.food_at_bottom,
        '4corners': exp.RL_4corners,
    }
    for k, v in exp_dict.items():
        saveConf(v, 'Exp', k)


def next_idx(exp, type='single'):
    try:
        with open(paths.SimIdx_path) as tfp:
            idx_dict = json.load(tfp)
    except:
        exp_names = list(loadConfDict('Exp').keys())
        batch_names = list(loadConfDict('Batch').keys())
        exp_idx_dict = dict(zip(exp_names, [0] * len(exp_names)))
        batch_idx_dict = dict(zip(batch_names, [0] * len(batch_names)))
        # batch_idx_dict.update(loadConfDict('Batch'))
        idx_dict = {'single': exp_idx_dict,
                    'batch': batch_idx_dict}
    if not exp in idx_dict[type].keys():
        idx_dict[type][exp] = 0
    idx_dict[type][exp] += 1
    with open(paths.SimIdx_path, "w") as fp:
        json.dump(idx_dict, fp)
    return idx_dict[type][exp]


