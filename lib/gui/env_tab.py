import copy
import threading
import PySimpleGUI as sg
import lib.conf.dtype_dicts as dtypes

from lib.aux.collecting import output_keys
from lib.gui.gui_lib import CollapsibleDict, Collapsible, \
    named_bool_button, save_gui_conf, delete_gui_conf, GraphList, CollapsibleTable, \
    graphic_button, t10_kws, t18_kws, w_kws, default_run_window, col_kws, col_size, t24_kws
from lib.gui.draw_env import draw_env
from lib.gui.life_conf import life_conf
from lib.sim.single_run import run_sim, sim_analysis
from lib.conf.conf import loadConfDict, loadConf, next_idx


def init_env(collapsibles):
    s1 = CollapsibleDict('Arena', True, dict=dtypes.get_dict('arena'), type_dict=dtypes.get_dict_dtypes('arena'))
    s2 = CollapsibleDict('Food_grid', True, dict=dtypes.get_dict('food_grid'),disp_name='Food grid',
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
        collapsibles['larva_groups'].get_section(),
        collapsibles['Sources'].get_section(),
        collapsibles['border_list'].get_section(),
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


def build_env_tab():
    dicts={}
    collapsibles={}
    graph_lists={}


    s1 = CollapsibleTable('larva_groups', False, headings=['group', 'N', 'color', 'model'], dict={},
                          disp_name='Larva groups',
                          type_dict=dtypes.get_dict_dtypes('distro', class_name='Larva', basic=False))
    s2 = CollapsibleTable('source_groups', False, headings=['group', 'N', 'color', 'amount', 'odor_id'], dict={},
                          disp_name='Source groups',
                          type_dict=dtypes.get_dict_dtypes('distro', class_name='Source', basic=False))
    s3 = CollapsibleTable('source_units', False, headings=['id', 'color', 'amount', 'odor_id'], dict={},
                          disp_name='Single sources',
                          type_dict=dtypes.get_dict_dtypes('agent', class_name='Source'))
    s4 = CollapsibleTable('border_list', False, headings=['id', 'color', 'points'], dict={}, disp_name='Borders',
                          type_dict=dtypes.get_dict_dtypes('agent', class_name='Border'))
    for s in [s1, s2, s3, s4]:
        collapsibles.update(**s.get_subdicts())

    l_env0 = [sg.Col([
        [sg.Text('Environment', **t10_kws, tooltip='The currently selected environment configuration'),
         graphic_button('load', 'LOAD_ENV', tooltip='Load a stored environment configuration.'),
         graphic_button('edit', 'CONF_ENV', tooltip='Configure an existing or draw an entirely new environment.'),
         graphic_button('data_add', 'SAVE_ENV', tooltip='Save a new environment configuration.'),
         graphic_button('data_remove', 'DELETE_ENV', tooltip='Delete an existing environment configuration.'),
         ],
        [sg.Combo(list(loadConfDict('Env').keys()), key='ENV_CONF', enable_events=True, readonly=True, **t24_kws)],
    ])]
    l_env1 = init_env(collapsibles)
    # l_env = [[sg.Col([l_env0, l_env1], **col_kws)]]
    l_env = [[sg.Col([l_env0, l_env1],**col_kws, size=col_size(0.5))]]
    return l_env, collapsibles, graph_lists, dicts


def eval_env(event, values, window, collapsibles, dicts, graph_lists):

    if event == 'LOAD_ENV' and values['ENV_CONF'] != '':
        conf = loadConf(values['ENV_CONF'], 'Env')
        update_env(conf, window, collapsibles)
    elif event == 'SAVE_ENV':
        env = get_env(window, values, collapsibles)
        save_gui_conf(window, env, 'Env')
    elif event == 'DELETE_ENV':
        delete_gui_conf(window, values, 'Env')
    elif event == 'CONF_ENV':
        env = get_env(window, values, collapsibles, extend=False)
        new_env = draw_env(env)
        update_env(new_env, window, collapsibles)

    return dicts, graph_lists


if __name__ == "__main__":
    sg.theme('LightGreen')
    dicts = {
        'sim_results': {'datasets': []},
        'batch_kwargs': None,
        'batch_results': {},
        'analysis_data': {},
    }
    l, col, graphs, d = build_env_tab()
    dicts.update(d)
    w = sg.Window('Environment gui', l, size=(1800, 1200), **w_kws, location=(300, 100))
    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, col, graphs)
        dicts, graphs = eval_env(e, v, w, col, dicts, graphs)
    w.close()