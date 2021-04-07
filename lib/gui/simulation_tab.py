import copy

import PySimpleGUI as sg

from lib.aux.collecting import effector_collection
from lib.conf import exp_types, test_env, agent_pars, distro_pars, arena_pars_dict
from lib.gui.gui_lib import CollapsibleDict, named_list_layout, button_kwargs, Collapsible, text_kwargs, \
    named_bool_button, header_kwargs, set_agent_dict, build_table_window, buttonM_kwargs
from lib.sim.single_run import next_idx, run_sim, configure_sim
from lib.stor.datagroup import loadConfDict, loadConf, saveConf, deleteConf
import lib.aux.functions as fun

def save_env(window, env):
    l = [[sg.Text('Store new environment', size=(20, 1)), sg.In(k='ENV_ID', size=(10, 1))],
         [sg.Ok(), sg.Cancel()]]
    e, v = sg.Window('Environment configuration', l).read(close=True)
    if e == 'Ok':
        env_id = v['ENV_ID']
        saveConf(env, 'Env', env_id)
        window['ENV_CONF'].update(values=list(loadConfDict('Env').keys()))
        window['ENV_CONF'].update(value=env_id)


def init_env(env_params, collapsibles={}):
    collapsibles['ARENA'] = CollapsibleDict('ARENA', True, dict=env_params['arena_params'], type_dict=arena_pars_dict)
    collapsibles['FOOD_GRID'] = CollapsibleDict('FOOD_GRID', True, dict=env_params['food_params']['food_grid'],
                                                disp_name='FOOD GRID', toggle=True, disabled=False)

    food_conf = [
        [sg.Button('SOURCE GROUPS', **buttonM_kwargs)],
        [sg.Button('SOURCE UNITS', **buttonM_kwargs)],
        collapsibles['FOOD_GRID'].get_section()
    ]

    odor_conf = [

    ]

    collapsibles['SOURCES'] = Collapsible('SOURCES', True, food_conf)
    collapsibles['ODORSCAPE'] = Collapsible('ODORSCAPE', True, odor_conf)

    env_layout = [
        collapsibles['ARENA'].get_section(),
        [sg.Button('BORDERS', **buttonM_kwargs)],
        collapsibles['SOURCES'].get_section(),
        [sg.Button('LARVA GROUPS', **buttonM_kwargs)],
        collapsibles['ODORSCAPE'].get_section()
    ]

    collapsibles['ENVIRONMENT'] = Collapsible('ENVIRONMENT', True, env_layout)

    return collapsibles['ENVIRONMENT'].get_section()


def update_env(env_params, window, collapsibles):
    # arena_params = env_params['arena_params']
    # for k, v in arena_params.items():
    #     window.Element(k).Update(value=v)
    print(env_params)
    food_params = env_params['food_params']
    source_units = food_params['source_units']
    source_groups = food_params['source_groups']
    collapsibles['FOOD_GRID'].update(window, food_params['food_grid'])

    collapsibles['ARENA'].update(window, env_params['arena_params'])

    # collapsibles['LARVA_DISTRIBUTION'].update(window, env_params['larva_params'])

    larva_groups = env_params['larva_params']
    if 'border_list' in env_params.keys():
        border_list = env_params['border_list']
    else:
        border_list = {}
    return source_units, border_list, larva_groups, source_groups


def get_env(window, values, collapsibles, source_units, border_list, larva_groups, source_groups):
    env = {}
    env['larva_params'] = larva_groups
    env['food_params'] = {}
    env['food_params']['source_groups'] = source_groups
    env['food_params']['food_grid'] = collapsibles['FOOD_GRID'].get_dict(values, window)
    env['food_params']['source_units'] = source_units
    env['border_list'] = border_list
    env['arena_params'] = collapsibles['ARENA'].get_dict(values, window)

    for k, v in env['larva_params'].items():
        if type(v['model']) == str:
            v['model'] = loadConf(v['model'], 'Model')
    return copy.deepcopy(env)


def build_sim_tab(collapsibles):
    sim_datasets = []

    env_params = copy.deepcopy(test_env)
    larva_groups = env_params['larva_params']
    source_units = env_params['food_params']['source_units']
    source_groups = env_params['food_params']['source_groups']
    border_list = {}

    l_exp = [sg.Col([
        named_list_layout(text='Experiment:', key='EXP', choices=list(exp_types.keys())),
        [sg.Button('Load', key='LOAD_EXP', **button_kwargs), sg.Button('Run', **button_kwargs)]
    ])]
    output_keys = list(effector_collection.keys())
    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    collapsibles['OUTPUT'] = CollapsibleDict('OUTPUT', False, dict=output_dict)

    sim_conf = [[sg.Text('Sim id:', **text_kwargs), sg.In('unnamed_sim', key='sim_id', **text_kwargs)],
                [sg.Text('Path:', **text_kwargs), sg.In('single_runs', key='path', **text_kwargs)],
                [sg.Text('Duration (min):', **text_kwargs), sg.In(3, key='sim_dur', **text_kwargs)],
                [sg.Text('Timestep (sec):', **text_kwargs), sg.In(0.1, key='dt', **text_kwargs)],
                named_bool_button('Box2D', False),
                collapsibles['OUTPUT'].get_section()
                ]

    collapsibles['CONFIGURATION'] = Collapsible('CONFIGURATION', True, sim_conf)

    l_conf1 = collapsibles['CONFIGURATION'].get_section()

    l_conf = [[sg.Col([l_exp, l_conf1])]]

    l_env0 = [sg.Col([
        [sg.Text('Environment:', **header_kwargs),
         sg.Combo(list(loadConfDict('Env').keys()), key='ENV_CONF', enable_events=True, readonly=True, **text_kwargs)],
        [sg.Button('Load', key='LOAD_ENV', **button_kwargs),
         sg.Button('Configure', key='CONF_ENV', **button_kwargs),
         sg.Button('Save', key='SAVE_ENV', **button_kwargs),
         sg.Button('Delete', key='DELETE_ENV', **button_kwargs)]
    ])]
    l_env1 = init_env(env_params, collapsibles)



    l_env = [[sg.Col([l_env0, l_env1])]]

    l_sim = [[sg.Col(l_conf), sg.Col(l_env)]]

    return l_sim, sim_datasets, collapsibles, output_keys, source_units, border_list, larva_groups, source_groups


def eval_sim(event, values, window, sim_datasets, collapsibles, output_keys,
             source_units, border_list, larva_groups, source_groups):
    if event == 'LOAD_EXP' and values['EXP'] != '':
        source_units, border_list, larva_groups, source_groups = update_sim(window, values, collapsibles, output_keys)


    elif event == 'LOAD_ENV' and values['ENV_CONF'] != '':
        conf = loadConf(values['ENV_CONF'], 'Env')
        source_units, border_list, larva_groups, source_groups = update_env(conf, window, collapsibles)

    elif event == 'SAVE_ENV':
        env = get_env(window, values, collapsibles, source_units, border_list, larva_groups, source_groups)
        save_env(window, env)



    elif event == 'DELETE_ENV' and values['ENV_CONF'] != '':
        deleteConf(values['ENV_CONF'], 'Env')
        window['ENV_CONF'].update(values=list(loadConfDict('Env').keys()))
        window['ENV_CONF'].update(value='')


    elif event == 'LARVA GROUPS':
        larva_groups = set_agent_dict(larva_groups, distro_pars('Larva'), header='group', title='Larva distribution')

    elif event == 'SOURCE UNITS':
        source_units = set_agent_dict(source_units, agent_pars['Food'], title='Food distribution')

    elif event == 'SOURCE GROUPS':
        source_groups = set_agent_dict(source_groups, distro_pars('Food'), header='group', title='Food distribution')

    elif event == 'BORDERS':
        border_list = set_agent_dict(border_list, agent_pars['Border'], title='Impassable borders')


    elif event == 'CONF_ENV':
        env = get_env(window, values, collapsibles, source_units, border_list, larva_groups, source_groups)
        new_source_units, new_border_list = configure_sim(env_params=env)
        l = [
            [sg.Text('Food agents and borders have been individually stored.', size=(70, 1))],
            [sg.Text('If you choose to continue the existing group distributions will be erased.', size=(70, 1))],
            [sg.Text('Continue?', size=(70, 1))],
            [sg.Ok(), sg.Cancel()]]
        e, v = sg.Window('Environment configuration', l).read(close=True)
        if e == 'Ok':
            source_units = new_source_units
            border_list = new_border_list
            source_groups = {}


    elif event == 'Run' and values['EXP'] != '':
        sim_config = get_sim(window, values, collapsibles, output_keys, source_units, border_list, larva_groups,
                             source_groups)
        vis_kwargs = {'mode': 'video'}
        d = run_sim(**sim_config, **vis_kwargs)
        if d is not None:
            sim_datasets.append(d)
    return source_units, border_list, larva_groups, source_groups


def update_sim(window, values, collapsibles, output_keys):
    exp = values['EXP']
    exp_conf = copy.deepcopy(exp_types[exp])
    env=exp_conf['env_params']
    if type(env) == str:
        window.Element('ENV_CONF').Update(value=env)
        env = loadConf(env, 'Env')
    source_units, border_list, larva_groups, source_groups = update_env(env, window, collapsibles)

    output_dict = dict(zip(output_keys, [True if k in exp_conf['collections'] else False for k in output_keys]))
    collapsibles['OUTPUT'].update(window, output_dict)

    window.Element('sim_id').Update(value=f'{exp}_{next_idx(exp)}')
    window.Element('path').Update(value=f'single_runs/{exp}')
    return source_units, border_list, larva_groups, source_groups


def get_sim(window, values, collapsibles, output_keys, source_units, border_list, larva_groups, source_groups):
    exp = values['EXP']
    sim = {}
    sim['sim_id'] = str(values['sim_id'])
    sim['sim_dur'] = float(values['sim_dur'])
    sim['dt'] = float(values['dt'])
    sim['path'] = str(values['path'])
    sim['Box2D'] = window['TOGGLE_Box2D'].metadata.state

    temp = collapsibles['OUTPUT'].get_dict(values, window)
    collections = [k for k in output_keys if temp[k]]

    env = get_env(window, values, collapsibles, source_units, border_list, larva_groups, source_groups)


    sim_config = {
        'enrich': True,
        'experiment': exp,
        'sim_params': sim,
        'env_params': env,
        'collections': collections,
    }
    return sim_config
