import copy
import threading
import PySimpleGUI as sg
import lib.conf.dtype_dicts as dtypes

from lib.aux.collecting import output_keys
from lib.gui.gui_lib import CollapsibleDict, Collapsible, \
    named_bool_button, save_gui_conf, delete_gui_conf, GraphList, CollapsibleTable, \
    graphic_button, t10_kws, t18_kws, w_kws, default_run_window, col_kws, col_size
from lib.gui.draw_env import draw_env
from lib.gui.life_conf import life_conf
from lib.sim.single_run import run_sim, sim_analysis
from lib.conf.conf import loadConfDict, loadConf, next_idx


def init_env(collapsibles):
    s1 = CollapsibleDict('Arena', False, dict=dtypes.get_dict('arena'), type_dict=dtypes.get_dict_dtypes('arena'))
    s2 = CollapsibleDict('Food_grid', False, dict=dtypes.get_dict('food_grid'),disp_name='Food grid',
                         type_dict=dtypes.get_dict_dtypes('food_grid'), toggle=True, disabled=False)
    s3 = CollapsibleDict('Odorscape', False, dict=dtypes.get_dict('odorscape'),
                         type_dict=dtypes.get_dict_dtypes('odorscape'))
    for s in [s1, s2, s3]:
        collapsibles.update(s.get_subdicts())
    food_conf = [
        collapsibles['source_groups'].get_section(),
        collapsibles['source_units'].get_section(),
        collapsibles['Food_grid'].get_section()
    ]
    collapsibles['Sources'] = Collapsible('Sources', True, food_conf)

    env_layout = [
        collapsibles['Arena'].get_section(),
        collapsibles['border_list'].get_section(),
        collapsibles['Sources'].get_section(),
        collapsibles['larva_groups'].get_section(),
        collapsibles['Odorscape'].get_section()
    ]
    collapsibles['Environment'] = Collapsible('Environment', True, env_layout)
    return collapsibles['Environment'].get_section()


def update_env(env_params, window, collapsibles):
    food_params = env_params['food_params']
    collapsibles['Food_grid'].update(window, food_params['food_grid'])
    collapsibles['Arena'].update(window, env_params['arena_params'])
    collapsibles['Odorscape'].update(window, env_params['odorscape'])
    border_list = env_params['border_list'] if 'border_list' in env_params.keys() else {}
    collapsibles['border_list'].update_table(window, border_list)
    collapsibles['source_units'].update_table(window, food_params['source_units'])
    collapsibles['source_groups'].update_table(window, food_params['source_groups'])
    collapsibles['larva_groups'].update_table(window, env_params['larva_params'])


def get_env(window, values, collapsibles, extend=True):
    d = collapsibles
    env = {
        'larva_params': d['larva_groups'].dict,
        'food_params': {
            'source_groups': d['source_groups'].dict,
            'food_grid': d['Food_grid'].get_dict(values, window),
            'source_units': d['source_units'].dict,
        },
        'border_list': d['border_list'].dict,
        'arena_params': d['Arena'].get_dict(values, window),
        'odorscape': d['Odorscape'].get_dict(values, window),
    }
    env0 = copy.deepcopy(env)
    if extend:
        for k, v in env0['larva_params'].items():
            if type(v['model']) == str:
                v['model'] = loadConf(v['model'], 'Model')
    return env0


def build_sim_tab():
    dicts={}
    collapsibles={}
    graph_lists={}


    s1 = CollapsibleTable('larva_groups', True, headings=['group', 'color', 'N', 'model'], dict={},
                          disp_name='Larva groups',
                          type_dict=dtypes.get_dict_dtypes('distro', class_name='Larva', basic=False))
    s2 = CollapsibleTable('source_groups', True, headings=['group', 'color', 'N', 'amount', 'odor_id'], dict={},
                          disp_name='Source groups',
                          type_dict=dtypes.get_dict_dtypes('distro', class_name='Source', basic=False))
    s3 = CollapsibleTable('source_units', True, headings=['id', 'color', 'amount', 'odor_id'], dict={},
                          disp_name='Single sources',
                          type_dict=dtypes.get_dict_dtypes('agent', class_name='Source'))
    s4 = CollapsibleTable('border_list', True, headings=['id', 'color', 'points'], dict={}, disp_name='Borders',
                          type_dict=dtypes.get_dict_dtypes('agent', class_name='Border'))
    for s in [s1, s2, s3, s4]:
        collapsibles.update(**s.get_subdicts())
    l_exp = [sg.Col([
        [sg.Text('Experiment', **t10_kws, tooltip='The currently selected simulation experiment.'),
         graphic_button('load', 'LOAD_EXP', tooltip='Load the configuration for a simulation experiment.'),
         graphic_button('play', 'RUN_EXP', tooltip='Run the selected simulation experiment.')],
        [sg.Combo(list(loadConfDict('Exp').keys()), key='EXP', enable_events=True, readonly=True, **t18_kws)]
    ],**col_kws)]
    sim_conf = [[sg.Text('Sim id:'), sg.In('unnamed_sim', key='sim_id')],
                [sg.Text('Path:'), sg.In('single_runs', key='path')],
                [sg.Text('Duration (min):'), sg.In(3, key='sim_dur')],
                [sg.Text('Timestep (sec):'), sg.In(0.1, key='dt')],
                named_bool_button('Box2D', False)]
    collapsibles['Configuration'] = Collapsible('Configuration', True, sim_conf)
    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    s1 = CollapsibleDict('Output', False, dict=output_dict, auto_open=False)
    s2 = CollapsibleDict('Visualization', False, dict=dtypes.get_dict('visualization', mode='video', video_speed=60),
                         type_dict=dtypes.get_dict_dtypes('visualization'), toggled_subsections=None)
    s3 = CollapsibleDict('Life', False, dict=dtypes.get_dict('life'), type_dict=dtypes.get_dict_dtypes('life'),
                         next_to_header=[graphic_button('edit', 'CONF_LIFE',
                                                        tooltip='Configure the life history of the simulated larvae.')])
    s4 = CollapsibleDict('Replay', False, dict=dtypes.get_dict('replay'), type_dict=dtypes.get_dict_dtypes('replay'))
    for s in [s1, s2, s3, s4]:
        collapsibles.update(s.get_subdicts())
    graph_lists['EXP'] = GraphList('EXP')
    l_conf = [[sg.Col([
        l_exp,
        collapsibles['Configuration'].get_section(),
        collapsibles['Output'].get_section(),
        collapsibles['Visualization'].get_section(),
        collapsibles['Replay'].get_section(),
        collapsibles['Life'].get_section(),
        [graph_lists['EXP'].get_layout()]
    ])]]
    l_env0 = [sg.Col([
        [sg.Text('Environment', **t10_kws, tooltip='The currently selected environment configuration'),
         graphic_button('load', 'LOAD_ENV', tooltip='Load a stored environment configuration.'),
         graphic_button('edit', 'CONF_ENV', tooltip='Configure an existing or draw an entirely new environment.'),
         graphic_button('data_add', 'SAVE_ENV', tooltip='Save a new environment configuration.'),
         graphic_button('data_remove', 'DELETE_ENV', tooltip='Delete an existing environment configuration.'),
         ],
        [sg.Combo(list(loadConfDict('Env').keys()), key='ENV_CONF', enable_events=True, readonly=True, **t18_kws)],
    ])]
    l_env1 = init_env(collapsibles)
    # l_env = [[sg.Col([l_env0, l_env1], **col_kws)]]
    l_sim = [[sg.Col(l_conf,**col_kws, size=col_size(0.25)), sg.Col([l_env0, l_env1],**col_kws, size=col_size(0.25)), graph_lists['EXP'].canvas]]
    return l_sim, collapsibles, graph_lists, dicts


def eval_sim(event, values, window, collapsibles, dicts, graph_lists):
    if event == 'LOAD_EXP' and values['EXP'] != '':
        exp_id = values['EXP']
        update_sim(window, exp_id, collapsibles)
    elif event == 'LOAD_ENV' and values['ENV_CONF'] != '':
        conf = loadConf(values['ENV_CONF'], 'Env')
        update_env(conf, window, collapsibles)
    elif event == 'SAVE_ENV':
        env = get_env(window, values, collapsibles)
        save_gui_conf(window, env, 'Env')
    elif event == 'DELETE_ENV':
        delete_gui_conf(window, values, 'Env')
    elif event == 'CONF_LIFE':
        collapsibles['Life'].update(window, life_conf())

    elif event == 'CONF_ENV':
        env = get_env(window, values, collapsibles, extend=False)
        new_env = draw_env(env)
        update_env(new_env, window, collapsibles)

    # elif event == 'CONF_ENV':
    #     env = get_env(window, values, collapsibles, dicts)
    #     new_source_units, new_border_list = configure_sim(env_params=env)
    #     l = [
    #         [sg.Text('Food agents and borders have been individually stored.', size=(70, 1))],
    #         [sg.Text('If you choose to continue the existing group distributions will be erased.', size=(70, 1))],
    #         [sg.Text('Continue?', size=(70, 1))],
    #         [sg.Ok(), sg.Cancel()]]
    #     e, v = sg.Window('Environment configuration', l).read(close=True)
    #     if e == 'Ok':
    #         source_units = new_source_units
    #         border_list = new_border_list
    #         source_groups = {}

    elif event == 'RUN_EXP' and values['EXP'] != '':
        exp_conf = get_exp(window, values, collapsibles)
        exp_conf['enrich'] = True
        d = run_sim(**exp_conf, vis_kwargs=collapsibles['Visualization'].get_dict(values, window))
        if d is not None:
            dicts['analysis_data'][d.id] = d
            if 'DATASET_IDS' in window.element_list() :
                window.Element('DATASET_IDS').Update(values=list(dicts['analysis_data'].keys()))
            dicts['sim_results']['datasets'].append(d)
            fig_dict, results = sim_analysis(d, exp_conf['experiment'])
            dicts['sim_results']['fig_dict'] = fig_dict
            graph_lists['EXP'].update(window, fig_dict)
    return dicts, graph_lists


def update_sim(window, exp_id, collapsibles):
    exp_conf = loadConf(exp_id, 'Exp')
    env = exp_conf['env_params']
    if type(env) == str:
        window.Element('ENV_CONF').Update(value=env)
        env = loadConf(env, 'Env')
    update_env(env, window, collapsibles)
    output_dict = dict(zip(output_keys, [True if k in exp_conf['collections'] else False for k in output_keys]))
    collapsibles['Output'].update(window, output_dict)
    window.Element('sim_id').Update(value=f'{exp_id}_{next_idx(exp_id)}')
    window.Element('path').Update(value=f'single_runs/{exp_id}')


def get_sim_conf(window, values):
    sim = {
        'sim_id': str(values['sim_id']),
        'sim_dur': float(values['sim_dur']),
        'dt': float(values['dt']),
        'path': str(values['path']),
        'Box2D': window['TOGGLE_Box2D'].get_state(),
    }
    return sim


def get_exp(window, values, collapsibles):
    exp_conf = {
        'experiment': values['EXP'],
        'sim_params': get_sim_conf(window, values),
        'env_params': get_env(window, values, collapsibles),
        'life_params': collapsibles['Life'].get_dict(values, window),
        'collections': [k for k in output_keys if collapsibles['Output'].get_dict(values, window)[k]],
    }
    return exp_conf

if __name__ == "__main__":
    sg.theme('LightGreen')
    dicts = {
        'sim_results': {'datasets': []},
        'batch_kwargs': None,
        'batch_results': {},
        'analysis_data': {},
    }
    l, col, graphs = build_sim_tab()
    w = sg.Window('Simulation gui', l, size=(1800, 1200), **w_kws, location=(300, 100))
    # print(graphs)
    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, col, graphs)
        dicts, graphs = eval_sim(e, v, w, col, dicts, graphs)
    w.close()