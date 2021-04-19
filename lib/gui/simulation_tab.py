import copy
import threading
import PySimpleGUI as sg
import lib.conf.dtype_dicts as dtypes

from lib.aux.collecting import output_keys
from lib.conf import test_env
from lib.conf import env_conf

from lib.gui.gui_lib import CollapsibleDict, named_list_layout, t8_kws, Collapsible, t14_kws, \
    named_bool_button, t14_kws, set_agent_dict, t14_kws, save_gui_conf, delete_gui_conf, GraphList, b12_kws, b6_kws
from lib.gui.draw_env import draw_env
from lib.sim.single_run import run_sim, configure_sim, sim_analysis
from lib.conf.conf import loadConfDict, loadConf, next_idx


def init_env(env_params, collapsibles={}):
    collapsibles['ARENA'] = CollapsibleDict('ARENA', True, dict=env_params['arena_params'], type_dict=dtypes.get_dict_dtypes('arena'))
    collapsibles['FOOD_GRID'] = CollapsibleDict('FOOD_GRID', True, dict=env_params['food_params']['food_grid'],
                                                disp_name='FOOD GRID', toggle=True, disabled=False)

    food_conf = [
        [sg.B('SOURCE GROUPS', **b12_kws)],
        [sg.B('SOURCE UNITS', **b12_kws)],
        collapsibles['FOOD_GRID'].get_section()
    ]

    collapsibles['SOURCES'] = Collapsible('SOURCES', True, food_conf)
    collapsibles['ODORSCAPE'] = CollapsibleDict('ODORSCAPE', True,
                                                dict=env_params['odorscape'],
                                                type_dict=dtypes.get_dict_dtypes('odorscape'))

    env_layout = [
        collapsibles['ARENA'].get_section(),
        [sg.B('BORDERS', **b12_kws)],
        collapsibles['SOURCES'].get_section(),
        [sg.B('LARVA GROUPS', **b12_kws)],
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
        [sg.B('Load', key='LOAD_EXP', **b6_kws), sg.B('Run', **b6_kws)]
    ])]

    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    collapsibles['OUTPUT'] = CollapsibleDict('OUTPUT', False, dict=output_dict)

    s = CollapsibleDict('VISUALIZATION', False,
                        dict=dtypes.get_dict_dtypes('visualization', video_speed=60),
                        # dict=get_vis_kwargs_dict(video_speed=60),
                        type_dict=dtypes.get_dict_dtypes('visualization'),
                        toggled_subsections=None)
    collapsibles.update(s.get_subdicts())

    sim_conf = [[sg.Text('Sim id:', **t14_kws), sg.In('unnamed_sim', key='sim_id', **t14_kws)],
                [sg.Text('Path:', **t14_kws), sg.In('single_runs', key='path', **t14_kws)],
                [sg.Text('Duration (min):', **t14_kws), sg.In(3, key='sim_dur', **t14_kws)],
                [sg.Text('Timestep (sec):', **t14_kws), sg.In(0.1, key='dt', **t14_kws)],
                named_bool_button('Box2D', False),

                ]

    collapsibles['CONFIGURATION'] = Collapsible('CONFIGURATION', True, sim_conf)


    collapsibles['LIFE'] = CollapsibleDict('LIFE', False,
                                           dict=dtypes.get_dict('life'),
                                           type_dict=dtypes.get_dict_dtypes('life'))
    s = CollapsibleDict('REPLAY', False,
                        dict=dtypes.get_dict('replay'),
                        type_dict=dtypes.get_dict_dtypes('replay'))
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
        [sg.Text('Environment:', **t14_kws),
         sg.Combo(list(loadConfDict('Env').keys()), key='ENV_CONF', enable_events=True, readonly=True, **t14_kws)],
        [sg.B('Load', key='LOAD_ENV', **b6_kws),
         sg.B('Configure', key='CONF_ENV', **b6_kws),
         sg.B('Draw', key='DRAW_ENV', **b6_kws),
         sg.B('Save', key='SAVE_ENV', **b6_kws),
         sg.B('Delete', key='DELETE_ENV', **b6_kws)]
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
        larva_groups = set_agent_dict(larva_groups, dtypes.get_dict_dtypes('distro', class_name='Larva', basic=False), header='group', title='Larva distribution')
        # larva_groups = set_agent_dict(larva_groups, distro_dtypes('Larva'), header='group', title='Larva distribution')

    elif event == 'SOURCE UNITS':
        source_units = set_agent_dict(source_units, dtypes.get_dict_dtypes('agent', class_name='Source'), title='Source distribution')
        # source_units = set_agent_dict(source_units, agent_dtypes['Source'], title='Source distribution')

    elif event == 'SOURCE GROUPS':
        source_groups = set_agent_dict(source_groups, dtypes.get_dict_dtypes('distro', class_name='Source', basic=False), header='group', title='Source distribution')
        # source_groups = set_agent_dict(source_groups, distro_dtypes('Source'), header='group', title='Source distribution')

    elif event == 'BORDERS':
        border_list = set_agent_dict(border_list, dtypes.get_dict_dtypes('agent', class_name='Border'), title='Impassable borders')
        # border_list = set_agent_dict(border_list, agent_dtypes['Border'], title='Impassable borders')

    elif event == 'DRAW_ENV':
        env = get_env(window, values, collapsibles, dicts)
        new_env = draw_env(env)
        source_units = new_env['food_params']['source_units']
        border_list = new_env['border_list']

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
