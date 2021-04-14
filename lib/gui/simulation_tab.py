import copy

import PySimpleGUI as sg
from lib.conf.dtype_dicts import agent_pars, distro_pars, arena_pars_dict, life_pars_dict, odorscape_pars_dict, \
    get_vis_kwargs_dict, vis_pars_dict, get_replay_kwargs_dict, replay_pars_dict
from lib.aux.collecting import output_keys
from lib.conf import test_env
from lib.conf import env_conf

from lib.gui.gui_lib import CollapsibleDict, named_list_layout, button_kwargs, Collapsible, text_kwargs, \
    named_bool_button, header_kwargs, set_agent_dict, buttonM_kwargs, save_gui_conf, delete_gui_conf, GraphList
from lib.sim.single_run import run_sim, configure_sim, sim_analysis
from lib.conf.conf import loadConfDict, loadConf, next_idx


def init_env(env_params, collapsibles={}):
    collapsibles['ARENA'] = CollapsibleDict('ARENA', True, dict=env_params['arena_params'], type_dict=arena_pars_dict)
    collapsibles['FOOD_GRID'] = CollapsibleDict('FOOD_GRID', True, dict=env_params['food_params']['food_grid'],
                                                disp_name='FOOD GRID', toggle=True, disabled=False)

    food_conf = [
        [sg.Button('SOURCE GROUPS', **buttonM_kwargs)],
        [sg.Button('SOURCE UNITS', **buttonM_kwargs)],
        collapsibles['FOOD_GRID'].get_section()
    ]

    collapsibles['SOURCES'] = Collapsible('SOURCES', True, food_conf)
    collapsibles['ODORSCAPE'] = CollapsibleDict('ODORSCAPE', True, dict=env_params['odorscape'],
                                                type_dict=odorscape_pars_dict)

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
    food_params = env_params['food_params']
    source_units = food_params['source_units']
    source_groups = food_params['source_groups']
    collapsibles['FOOD_GRID'].update(window, food_params['food_grid'])

    collapsibles['ARENA'].update(window, env_params['arena_params'])
    collapsibles['ODORSCAPE'].update(window, env_params['odorscape'])

    larva_groups = env_params['larva_params']
    if 'border_list' in env_params.keys():
        border_list = env_params['border_list']
    else:
        border_list = {}
    return source_units, border_list, larva_groups, source_groups


def get_env(window, values, collapsibles, dicts):
    env = {}
    env['larva_params'] = dicts['larva_groups']
    env['food_params'] = {}
    env['food_params']['source_groups'] = dicts['source_groups']
    env['food_params']['food_grid'] = collapsibles['FOOD_GRID'].get_dict(values, window)
    env['food_params']['source_units'] = dicts['source_units']
    env['border_list'] = dicts['border_list']
    env['arena_params'] = collapsibles['ARENA'].get_dict(values, window)
    env['odorscape'] = collapsibles['ODORSCAPE'].get_dict(values, window)

    for k, v in env['larva_params'].items():
        if type(v['model']) == str:
            v['model'] = loadConf(v['model'], 'Model')
    return copy.deepcopy(env)


def build_sim_tab(collapsibles, graph_lists, dicts):
    dicts['sim_results'] = {}
    dicts['sim_results']['datasets'] = []

    env_params = copy.deepcopy(test_env)
    dicts['larva_groups'] = env_params['larva_params']
    dicts['source_units'] = env_params['food_params']['source_units']
    dicts['source_groups'] = env_params['food_params']['source_groups']
    dicts['border_list'] = {}

    l_exp = [sg.Col([
        named_list_layout(text='Experiment:', key='EXP', choices=list(loadConfDict('Exp').keys())),
        [sg.Button('Load', key='LOAD_EXP', **button_kwargs), sg.Button('Run', **button_kwargs)]
    ])]

    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    collapsibles['OUTPUT'] = CollapsibleDict('OUTPUT', False, dict=output_dict)

    s = CollapsibleDict('VISUALIZATION', False, dict=get_vis_kwargs_dict(video_speed=60), type_dict=vis_pars_dict, toggled_subsections=None)
    collapsibles.update(s.get_subdicts())

    sim_conf = [[sg.Text('Sim id:', **text_kwargs), sg.In('unnamed_sim', key='sim_id', **text_kwargs)],
                [sg.Text('Path:', **text_kwargs), sg.In('single_runs', key='path', **text_kwargs)],
                [sg.Text('Duration (min):', **text_kwargs), sg.In(3, key='sim_dur', **text_kwargs)],
                [sg.Text('Timestep (sec):', **text_kwargs), sg.In(0.1, key='dt', **text_kwargs)],
                named_bool_button('Box2D', False),

                ]

    collapsibles['CONFIGURATION'] = Collapsible('CONFIGURATION', True, sim_conf)

    # l_conf1 = collapsibles['CONFIGURATION'].get_section()

    life_dict = {'starvation_hours': None,
                 'hours_as_larva': 0.0,
                 'deb_base_f': 1.0}
    collapsibles['LIFE'] = CollapsibleDict('LIFE', False, dict=life_dict, type_dict=life_pars_dict)
    # l_life = collapsibles['LIFE'].get_section()
    s = CollapsibleDict('REPLAY', False, dict=get_replay_kwargs_dict(arena_pars=env_conf.dish(0.15)), type_dict=replay_pars_dict)
    collapsibles.update(s.get_subdicts())

    graph_lists['EXP'] = GraphList('EXP')

    l_conf = [[sg.Col([
        l_exp,
        collapsibles['CONFIGURATION'].get_section(),
        collapsibles['OUTPUT'].get_section(),
        collapsibles['VISUALIZATION'].get_section(),
        collapsibles['REPLAY'].get_section(),
        collapsibles['LIFE'].get_section(),
        [graph_lists['EXP'].get_layout()]
    ])]]

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

    l_sim = [[sg.Col(l_conf), sg.Col(l_env), graph_lists['EXP'].canvas]]

    return l_sim, collapsibles, graph_lists, dicts


def eval_sim(event, values, window, collapsibles, dicts, graph_lists):
    source_units = dicts['source_units']
    border_list = dicts['border_list']
    larva_groups = dicts['larva_groups']
    source_groups = dicts['source_groups']

    if event == 'LOAD_EXP' and values['EXP'] != '':
        exp_id = values['EXP']
        source_units, border_list, larva_groups, source_groups = update_sim(window, exp_id, collapsibles)


    elif event == 'LOAD_ENV' and values['ENV_CONF'] != '':
        conf = loadConf(values['ENV_CONF'], 'Env')
        source_units, border_list, larva_groups, source_groups = update_env(conf, window, collapsibles)

    elif event == 'SAVE_ENV':
        env = get_env(window, values, collapsibles, dicts)
        save_gui_conf(window, env, 'Env')



    elif event == 'DELETE_ENV':
        delete_gui_conf(window, values, 'Env')

    elif event == 'LARVA GROUPS':
        larva_groups = set_agent_dict(larva_groups, distro_pars('Larva'), header='group', title='Larva distribution')

    elif event == 'SOURCE UNITS':
        source_units = set_agent_dict(source_units, agent_pars['Food'], title='Food distribution')

    elif event == 'SOURCE GROUPS':
        source_groups = set_agent_dict(source_groups, distro_pars('Food'), header='group', title='Food distribution')

    elif event == 'BORDERS':
        border_list = set_agent_dict(border_list, agent_pars['Border'], title='Impassable borders')


    elif event == 'CONF_ENV':
        env = get_env(window, values, collapsibles, dicts)
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
        exp_conf = get_exp(window, values, collapsibles, dicts)
        exp_conf['enrich'] = True
        vis_kwargs = collapsibles['VISUALIZATION'].get_dict(values, window)
        d = run_sim(**exp_conf, vis_kwargs=vis_kwargs)
        if d is not None:
            from lib.gui.analysis_tab import update_data_list
            dicts['analysis_data'][d.id] = d
            update_data_list(window, dicts['analysis_data'])
            dicts['sim_results']['datasets'].append(d)
            fig_dict, results = sim_analysis(d, exp_conf['experiment'])
            # fig_keys = list(fig_dict.keys())
            dicts['sim_results']['fig_dict'] = fig_dict
            graph_lists['EXP'].update(window, fig_dict)
            # window.Element('EXP_GRAPH_LIST').Update(values=fig_keys)

    # elif event == 'EXP_GRAPH_LIST':
    # # elif event == 'DRAW_EXP_FIG':
    #     if len(values['EXP_GRAPH_LIST']) > 0:
    #         choice = values['EXP_GRAPH_LIST'][0]
    #         fig=dicts['sim_results']['fig_dict'][choice]
    #         exp_fig_agg = draw_exp_canvas(window, fig, exp_fig_agg)

    dicts['source_units'] = source_units
    dicts['border_list'] = border_list
    dicts['larva_groups'] = larva_groups
    dicts['source_groups'] = source_groups

    return dicts, graph_lists


def update_sim(window, exp_id, collapsibles):
    # exp = values['EXP']
    # exp_conf = copy.deepcopy(exp_types[exp_id])
    exp_conf = loadConf(exp_id, 'Exp')
    env = exp_conf['env_params']
    if type(env) == str:
        window.Element('ENV_CONF').Update(value=env)
        env = loadConf(env, 'Env')
    source_units, border_list, larva_groups, source_groups = update_env(env, window, collapsibles)

    output_dict = dict(zip(output_keys, [True if k in exp_conf['collections'] else False for k in output_keys]))
    collapsibles['OUTPUT'].update(window, output_dict)

    window.Element('sim_id').Update(value=f'{exp_id}_{next_idx(exp_id)}')
    window.Element('path').Update(value=f'single_runs/{exp_id}')
    return source_units, border_list, larva_groups, source_groups


def get_sim_conf(window, values):
    sim = {}
    sim['sim_id'] = str(values['sim_id'])
    sim['sim_dur'] = float(values['sim_dur'])
    sim['dt'] = float(values['dt'])
    sim['path'] = str(values['path'])
    sim['Box2D'] = window['TOGGLE_Box2D'].metadata.state
    return sim


def get_exp(window, values, collapsibles, dicts):
    exp_id = values['EXP']

    sim = get_sim_conf(window, values)

    temp = collapsibles['OUTPUT'].get_dict(values, window)
    collections = [k for k in output_keys if temp[k]]

    env = get_env(window, values, collapsibles, dicts)

    life = collapsibles['LIFE'].get_dict(values, window)

    exp_conf = {
        # 'enrich': True,
        'experiment': exp_id,
        'sim_params': sim,
        'env_params': env,
        'life_params': life,
        'collections': collections,
    }
    return exp_conf
