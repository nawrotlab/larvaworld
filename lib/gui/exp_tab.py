import copy
import threading
import PySimpleGUI as sg
import numpy as np

import lib.conf.dtype_dicts as dtypes

from lib.aux.collecting import output_keys
from lib.gui.env_tab import update_env, get_env
from lib.gui.gui_lib import CollapsibleDict, Collapsible, \
    named_bool_button, save_gui_conf, delete_gui_conf, GraphList, CollapsibleTable, \
    graphic_button, t10_kws, t18_kws, w_kws, default_run_window, col_kws, col_size, window_size, t24_kws, t8_kws, \
    t16_kws, t2_kws, t14_kws, t5_kws, t11_kws, t1_kws, t15_kws, t6_kws
from lib.gui.draw_env import draw_env
from lib.gui.life_conf import life_conf
from lib.sim.single_run import run_sim, sim_analysis
from lib.conf.conf import loadConfDict, loadConf, next_idx


def build_sim_tab():
    dicts={}
    collapsibles={}
    graph_lists={}

    l_exp = [sg.Col([
        [sg.Text('Experiment', **t10_kws, tooltip='The currently selected simulation experiment.'),
         graphic_button('load', 'LOAD_EXP', tooltip='Load the configuration for a simulation experiment.'),
         graphic_button('play', 'RUN_EXP', tooltip='Run the selected simulation experiment.')],
        [sg.Combo(list(loadConfDict('Exp').keys()), key='EXP', enable_events=True, readonly=True, **t24_kws)],
        [sg.Text('Progress :', **t8_kws), sg.ProgressBar(100, orientation='h', size=(8.8, 20), key='EXP_PROGRESSBAR',
                                                         bar_color=('green', 'lightgrey'), border_width=3)]
    ],**col_kws)]
    sim_conf = [[sg.Text('Sim id :', **t8_kws), sg.In('unnamed_sim', key='sim_id', **t16_kws)],
                [sg.Text('Path :', **t8_kws), sg.In('single_runs', key='path', **t16_kws)],
                [sg.Text('Duration :', **t8_kws), sg.Spin(values=np.round(np.arange(0.0,100.1, 0.1),1).tolist(),initial_value=3.0, key='sim_dur', **t6_kws), sg.Text('minutes', **t8_kws, justification='center')],
                [sg.Text('Timestep :', **t8_kws), sg.Spin(values=np.round(np.arange(0.01,1.01, 0.01),2).tolist(),initial_value=0.1, key='dt', **t6_kws), sg.Text('seconds', **t8_kws, justification='center')],
                named_bool_button('Box2D', False)]
    collapsibles['Configuration'] = Collapsible('Configuration', True, sim_conf)
    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    s1 = CollapsibleDict('Output', False, dict=output_dict, auto_open=False)
    s3 = CollapsibleDict('Life', False, dict=dtypes.get_dict('life'), type_dict=dtypes.get_dict_dtypes('life'),
                         next_to_header=[graphic_button('edit', 'CONF_LIFE',
                                                        tooltip='Configure the life history of the simulated larvae.')])
    for s in [s1, s3]:
        collapsibles.update(s.get_subdicts())
    graph_lists['EXP'] = GraphList('EXP')
    l_conf = [[sg.Col([
        l_exp,
        collapsibles['Configuration'].get_section(),
        collapsibles['Output'].get_section(),
        collapsibles['Life'].get_section(),
        [graph_lists['EXP'].get_layout()]
    ])]]
    # col1=sg.Col(l_conf,**col_kws, size=col_size(0.25,0.9))
    # col2 = sg.Col([[sg.Output(key='EXP_OUTPUT', size=col_size(0.25, 0.1))]])
    # l_sim0 = sg.Col([[col1, graph_lists['EXP'].canvas]])

    # l_sim= [[sg.Col([[sg.Pane([col1, col2])],[sg.Text('Grab square above and slide upwards to view console output')]],
    #                size=col_size(0.25)), graph_lists['EXP'].canvas]]
    l_sim=[[sg.Col(l_conf,**col_kws, size=col_size(0.25)), graph_lists['EXP'].canvas]]
    return l_sim, collapsibles, graph_lists, dicts


def eval_sim(event, values, window, collapsibles, dicts, graph_lists):
    if event == 'LOAD_EXP' and values['EXP'] != '':
        exp_id = values['EXP']
        update_sim(window, exp_id, collapsibles)
    elif event == 'CONF_LIFE':
        collapsibles['Life'].update(window, life_conf())
    elif event == 'RUN_EXP' and values['EXP'] != '':
        exp_conf = get_exp(window, values, collapsibles)
        window['EXP_PROGRESSBAR'].update(0,max=exp_conf['sim_params']['sim_dur']*60/exp_conf['sim_params']['dt'])
        exp_conf['enrich'] = True
        vis_kwargs = collapsibles['Visualization'].get_dict(values, window) if 'Visualization' in list(collapsibles.keys()) else dtypes.get_dict('visualization', mode='video', video_speed=60)
        d = run_sim(**exp_conf, vis_kwargs=vis_kwargs, progress_bar=window['EXP_PROGRESSBAR'])
        if d is not None:
            dicts['analysis_data'][d.id] = d
            if 'DATASET_IDS' in window.element_list() :
                window.Element('DATASET_IDS').Update(values=list(dicts['analysis_data'].keys()))
            dicts['sim_results']['datasets'].append(d)
            fig_dict, results = sim_analysis(d, exp_conf['experiment'])
            dicts['sim_results']['fig_dict'] = fig_dict
            graph_lists['EXP'].update(window, fig_dict)
        else :
            window['EXP_PROGRESSBAR'].update(0)
    return dicts, graph_lists


def update_sim(window, exp_id, collapsibles):
    exp_conf = loadConf(exp_id, 'Exp')
    env = exp_conf['env_params']
    if 'ENV_CONF' in window.element_list():
        if type(env) == str:
            window.Element('ENV_CONF').Update(value=env)
            env = loadConf(env, 'Env')
        update_env(env, window, collapsibles)
    output_dict = dict(zip(output_keys, [True if k in exp_conf['collections'] else False for k in output_keys]))
    collapsibles['Output'].update(window, output_dict)
    window.Element('sim_id').Update(value=f'{exp_id}_{next_idx(exp_id)}')
    window.Element('path').Update(value=f'single_runs/{exp_id}')
    window['EXP_PROGRESSBAR'].update(0)

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
    exp=values['EXP']
    if 'ENV_CONF' in window.element_list() :
        env = get_env(window, values, collapsibles)
    else :
        env = loadConf(exp, 'Exp')['env_params']
        if type(env) == str:
            env = loadConf(env, 'Env')
            for k, v in env['larva_params'].items():
                if type(v['model']) == str:
                    v['model'] = loadConf(v['model'], 'Model')
    exp_conf = {
        'experiment': exp,
        'sim_params': get_sim_conf(window, values),
        'env_params': env,
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
    l, col, graphs, d = build_sim_tab()
    dicts.update(d)
    w = sg.Window('Simulation gui', l, size=(1800, 1200), **w_kws, location=(300, 100))
    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, col, graphs)
        dicts, graphs = eval_sim(e, v, w, col, dicts, graphs)
    w.close()