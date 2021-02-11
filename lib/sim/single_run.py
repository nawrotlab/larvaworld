""" Run a simulation and save the parameters and data to files."""
import datetime
import json
import time

import lib.aux.functions as fun
import lib.conf.data_modes as conf
import lib.stor.paths as paths
from lib.anal.plotting import *
from lib.aux.collecting import effector_collection
from lib.conf import exp_types, default_sim
from lib.model.envs._larvaworld import LarvaWorldSim
from lib.model.larva.deb import deb_dict, deb_default
from lib.stor.larva_dataset import LarvaDataset


def run_sim(sim_id,
            sim_params,
            env_params,
            fly_params,
            save_to=None,
            common_folder=None,
            media_name=None,
            save_data_flag=True,
            par_config=conf.SimParConf,
            **kwargs):
    print(f'Initializing simulation {sim_id}!')
    if save_to is None:
        save_to = paths.SimFolder
    # current_date = date.today()
    # Store the parameters so that we can save them in the results folder
    sim_date = datetime.datetime.now()
    param_dict = locals()
    start = time.time()
    dt = sim_params['dt']
    Nsec = sim_params['sim_time_in_min'] * 60
    Nsteps = int(Nsec / dt)

    if common_folder:
        parentdir = os.path.join(save_to, common_folder)
    else:
        parentdir = save_to
    dir_path = os.path.join(parentdir, sim_id)
    d = LarvaDataset(dir=dir_path, id=sim_id, fr=int(1 / dt),
                     Npoints=fly_params['body_params']['Nsegs'] + 1, Ncontour=0,
                     arena_pars=env_params['arena_params'],
                     par_conf=par_config, save_data_flag=save_data_flag, load_data=False)

    collected_pars = data_collection_config(dataset=d, sim_params=sim_params)

    # Build the environment
    env = LarvaWorldSim(fly_params, id=sim_id, env_params=env_params, dt=dt, Nsteps=Nsteps,
                        collected_pars=collected_pars,
                        media_name=media_name,
                        save_to=d.vis_dir,
                        **kwargs)

    # Prepare the odor layer for a number of timesteps
    env.prepare_odor_layer(int(sim_params['odor_prep_time_in_min'] * 60 / env.dt))
    # Prepare the flies for a number of timesteps
    env.prepare_flies(int(sim_params['fly_prep_time_in_min'] * 60 / env.dt))

    # Run the simulation
    env.run()

    # Read the data collected during the simulation
    larva_step_data = env.larva_step_collector.get_agent_vars_dataframe()
    # Collect and read the endpoint data
    env.larva_endpoint_collector.collect(env)
    env.food_endpoint_collector.collect(env)
    larva_endpoint_data = env.larva_endpoint_collector.get_agent_vars_dataframe()
    food_endpoint_data = env.food_endpoint_collector.get_agent_vars_dataframe()
    larva_endpoint_data = larva_endpoint_data.droplevel('Step')
    food_endpoint_data = food_endpoint_data.droplevel('Step')
    if 'cum_dur' in sim_params['end_pars']:
        larva_endpoint_data['cum_dur'] = Nsec
    if 'num_ticks' in sim_params['end_pars']:
        larva_endpoint_data['num_ticks'] = Nsteps
    env.close()
    end = time.time()
    dur = end - start
    param_dict['duration'] = np.round(dur, 2)
    d.set_step_data(larva_step_data)
    d.set_endpoint_data(larva_endpoint_data)
    d.set_food_endpoint_data(food_endpoint_data)

    # Save simulation data and parameters
    if save_data_flag:
        d.save()
        fun.dict_to_file(param_dict, d.sim_pars_file_path)
        # Show the odor layer
        if env.Nodors>0:
            env.plot_odorscape(save_to=d.plot_dir)
    print(f'Simulation complete in {dur} seconds!')
    return d


def data_collection_config(dataset, sim_params):
    d = dataset
    effectors = [e for e in sim_params['collect_effectors']]
    # effectors = [e for e in sim_params['collect_effectors'] if component_params[e]]
    step_pars = list(set(fun.flatten_list([effector_collection[e]['step'] for e in effectors])))
    end_pars = list(set(fun.flatten_list([effector_collection[e]['endpoint'] for e in effectors])))

    if sim_params['collect_midline']:
        step_pars += fun.flatten_list(d.points_xy)
    if sim_params['collect_contour']:
        step_pars += fun.flatten_list(d.contour_xy)
    collected_pars = {'step': sim_params['step_pars'] + step_pars,
                      'endpoint': sim_params['end_pars'] + end_pars}
    return collected_pars


def load_reference_dataset():
    reference_dataset = LarvaDataset(dir=paths.RefFolder, load_data=False)
    reference_dataset.load(step_data=False)
    return reference_dataset


def next_idx(exp, type='single'):
    from lib.conf.batch_modes import batch_types
    try:
        with open(paths.SimIdx_path) as tfp:
            idx_dict = json.load(tfp)
    except:
        exp_names = exp_types.keys()
        batch_names = batch_types.keys()
        exp_idx_dict = dict(zip(exp_names, [0] * len(exp_names)))
        batch_idx_dict = dict(zip(batch_names, [0] * len(batch_names)))
        idx_dict = {'single' : exp_idx_dict,
                    'batch' : batch_idx_dict}
    if not exp in idx_dict[type].keys():
        idx_dict[type][exp] = 0
    idx_dict[type][exp] += 1
    with open(paths.SimIdx_path, "w") as fp:
        json.dump(idx_dict, fp)
    return idx_dict[type][exp]


def generate_config(exp, Nagents=None, sim_time=None, sim_id=None):
    config = exp_types[exp]
    if sim_id is None:
        idx = next_idx(exp)
        sim_id = f'{exp}_{idx}'
    config['sim_id'] = sim_id
    if 'sim_params' not in config.keys():
        config['sim_params'] = default_sim

    if 'component_params' in config.keys():
        config['fly_params']['neural_params']['component_params'] = config['component_params']
        del config['component_params']
    if 'traj_mode' in config.keys():
        config['sim_params']['traj_mode'] = config['traj_mode']
        del config['traj_mode']

    if 'collect_effectors' in config.keys():
        config['sim_params']['collect_effectors'] = config['collect_effectors']
        del config['collect_effectors']
    if 'end_pars' in config.keys():
        config['sim_params']['end_pars'] += config['end_pars']
        del config['end_pars']
    if 'step_pars' in config.keys():
        config['sim_params']['step_pars'] += config['step_pars']
        del config['step_pars']
    if 'draw_mode' in config.keys():
        config['sim_params']['draw_mode'] = config['draw_mode']
        del config['draw_mode']

    if Nagents is not None:
        config['env_params']['place_params']['initial_num_flies'] = Nagents
    if sim_time is not None:
        config['sim_params']['sim_time_in_min'] = sim_time

    return config


def sim_analysis(d, experiment):
    s, e = d.step_data, d.endpoint_data
    if experiment in ['feed_patchy', 'feed_scatter', 'feed_grid']:
        # am = e['amount_eaten'].values
        # print(am)
        # cr,pr,fr=e['stride_dur_ratio'].values, e['pause_dur_ratio'].values, e['feed_dur_ratio'].values
        # print(cr+pr+fr)
        # cN, pN, fN = e['num_strides'].values, e['num_pauses'].values, e['num_feeds'].values
        # print(cN, pN, fN)
        # cum_sd, f_success=e['cum_scaled_dst'].values, e['feed_success_rate'].values
        # print(cum_sd, f_success)
        plot_endpoint_scatter(datasets=[d], labels=[d.id], par_shorts=['cum_sd', 'f_am', 'str_tr', 'fee_tr'])
        plot_endpoint_scatter(datasets=[d], labels=[d.id], par_shorts=['cum_sd', 'f_am'])

    elif experiment == 'growth':
        d.deb_analysis()
        # print(d.endpoint_data['deb_f_mean'])
        deb_dicts= [deb_dict(d, id) for id in d.agent_ids]+[deb_default()]
        plot_debs(deb_dicts=deb_dicts,save_to=d.plot_dir, save_as='comparative_deb.pdf')
        plot_debs(deb_dicts=deb_dicts,save_to=d.plot_dir, save_as='comparative_deb_minimal.pdf', mode='minimal')
        plot_debs(deb_dicts=deb_dicts[:-1], save_to=d.plot_dir, save_as='deb_f.pdf', mode='f')
        plot_debs(deb_dicts=deb_dicts[:-1],save_to=d.plot_dir, save_as='deb.pdf')
        plot_debs(deb_dicts=deb_dicts[:-1],save_to=d.plot_dir, save_as='deb_minimal.pdf', mode='minimal')

        # plot_growth(d, default_deb)
        # try:
        #     plot_deb(d)
        # except:
        #     pass
    elif experiment == 'focus':
        d.angular_analysis(is_last=False)
        d.detect_turns()
    elif experiment == 'dispersion':
        d.enrich(length_and_centroid=False)
        target_dataset = load_reference_dataset()
        datasets = [d, target_dataset]
        labels = ['simulated', 'empirical']
        comparative_analysis(datasets=datasets, labels=labels, simVSexp=True, save_to=None)
        plot_marked_strides(dataset=d, agent_ids=d.agent_ids[:3], title=' ', slices=[[10,50], [60,100]])
        plot_marked_turns(dataset=d, agent_ids=d.agent_ids[:3], min_turn_angle=20)
    elif experiment in ['chemorbit', 'chemotax']:
        plot_distance_to_source(dataset=d, experiment=experiment)
        plot_odor_concentration(dataset=d)
        d.visualize(agent_ids=[d.agent_ids[0]], mode='image', image_mode='final',
                    contours=False, centroid=False, spinepoints=False,
                    random_larva_colors=True, trajectories=True, trail_decay_in_sec=None,
                    save_as='single_trajectory')
    elif experiment == 'odor_pref':
        ind = d.compute_preference_index(arena_diameter_in_mm=100)
        print(ind)
        return ind
    elif experiment == 'imitation':
        d.save_agent(pars=fun.flatten_list(d.points_xy) + fun.flatten_list(d.contour_xy), header=True)
