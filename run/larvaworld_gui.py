#!/usr/bin/env python
# !/usr/bin/env python
import copy
import json

import PySimpleGUI as sg
import matplotlib
import inspect
from tkinter import *
from typing import List, Tuple
from lib.stor import paths
from lib.stor.datagroup import saveSimConf, loadSimConfDict, loadSimConf, deleteSimConf

sys.path.insert(0, '..')
from lib.aux.collecting import effector_collection
from lib.conf import exp_types, default_sim, mock_larva, box2d_space, larva_place_modes, \
    food_place_modes, pref_exp_np, agent_pars, mock_env, mesa_space
from lib.sim.gui_lib import gui_table, SectionDict, bool_button, Collapsible, \
    set_kwargs, on_image, off_image, SYMBOL_UP, SYMBOL_DOWN, button_kwargs, header_kwargs, \
    text_kwargs, on_image_disabled, retrieve_value, draw_canvas, delete_figure_agg, CollapsibleDict
from lib.sim.single_run import run_sim, next_idx, configure_sim
from lib.anal.plotting import *
from lib.stor.larva_dataset import LarvaDataset
from lib.stor.paths import SingleRunFolder, RefFolder, get_parent_dir

matplotlib.use('TkAgg')


# Class holding the button graphic info. At this time only the state is kept


def save_plot(fig, save_to, save_as):
    if fig is not None:
        layout = [
            [sg.Text('Filename', size=(10, 1)), sg.In(default_text=save_as, k='SAVE_AS', size=(80, 1))],
            [sg.Text('Directory', size=(10, 1)), sg.In(save_to, k='SAVE_TO', size=(80, 1)),
             sg.FolderBrowse(initial_folder=get_parent_dir(), key='SAVE_TO', change_submits=True)],
            [sg.Ok(), sg.Cancel()]]

        event, values = sg.Window('Save figure', layout).read(close=True)
        if event == 'Ok':
            save_as = values['SAVE_AS']
            save_to = values['SAVE_TO']
            filepath = os.path.join(save_to, save_as)
            fig.savefig(filepath, dpi=300)
            # save_canvas(window['GRAPH_CANVAS'].TKCanvas, filepath)
            # figure_agg.print_figure(filepath)
            print(f'Plot saved as {save_as}')


def get_graph_kwargs(func):
    signature = inspect.getfullargspec(func)
    kwargs = dict(zip(signature.args[-len(signature.defaults):], signature.defaults))
    for k in ['datasets', 'labels', 'save_to', 'save_as', 'return_fig', 'deb_dicts']:
        if k in kwargs.keys():
            del kwargs[k]
    return kwargs


def update_data_list(window, data):
    window.Element('DATASET_IDS').Update(values=list(data.keys()))


def change_dataset_id(window, values, data):
    if len(values['DATASET_IDS']) > 0:
        old_id = values['DATASET_IDS'][0]
        l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k='NEW_ID', size=(10, 1))],
             [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
        e, v = sg.Window('Change dataset ID', l).read(close=True)
        if e == 'Ok':
            data[v['NEW_ID']] = data.pop(old_id)
            update_data_list(window, data)
        elif e == 'Store':
            d = data[old_id]
            d.set_id(v['NEW_ID'])
            data[v['NEW_ID']] = data.pop(old_id)
            update_data_list(window, data)
    return data


def draw_figure(window, func, func_kwargs, data, figure_agg):
    if func is not None and len(list(data.keys())) > 0:
        if figure_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(figure_agg)
        try:
            fig, save_to, save_as = func(datasets=list(data.values()), labels=list(data.keys()),
                                         return_fig=True, **func_kwargs)  # call function to get the figure
            figure_agg = draw_canvas(window['GRAPH_CANVAS'].TKCanvas, fig)  # draw the figure
            return figure_agg, fig, save_to, save_as
        except:
            print('Plot not available for these datasets')
            return None, None, None, None


def update_func(window, values, func, func_kwargs, graph_dict):
    if len(values['GRAPH_LIST']) > 0:
        choice = values['GRAPH_LIST'][0]
        if graph_dict[choice] != func:
            func = graph_dict[choice]
            func_kwargs = get_graph_kwargs(func)
        window['GRAPH_CODE'].update(inspect.getsource(func))
    return func, func_kwargs


def update_model(larva_model, window, collapsibles, sectiondicts):
    for name, dict in zip(['PHYSICS', 'ENERGETICS', 'BODY'],
                          [larva_model['sensorimotor_params'], larva_model['energetics_params'],
                           larva_model['body_params']]):
        collapsibles[name].update(window, dict)
    module_dict = larva_model['neural_params']['modules']
    for k, v in module_dict.items():
        collapsibles[k.upper()].update(window, larva_model['neural_params'][f'{k}_params'])
    module_dict_upper = copy.deepcopy(module_dict)
    for k in list(module_dict_upper.keys()):
        module_dict_upper[k.upper()] = module_dict_upper.pop(k)
    collapsibles['BRAIN'].update(window, module_dict_upper, use_prefix=False)


def update_environment(env_params, window, collapsibles, sectiondicts, food_list, border_list):
    # arena_params = env_params['arena_params']
    # for k, v in arena_params.items():
    #     window.Element(k).Update(value=v)

    food_params = env_params['food_params']
    food_list = food_params['food_list']
    collapsibles['FOOD_DISTRIBUTION'].update(window, food_params['food_distro'])
    collapsibles['FOOD_GRID'].update(window, food_params['food_grid'])

    collapsibles['ARENA'].update(window, env_params['arena_params'])

    collapsibles['LARVA_DISTRIBUTION'].update(window, env_params['place_params'])
    if 'border_list' in env_params.keys():
        border_list = env_params['border_list']
    else:
        border_list = {}
    return food_list, border_list

def init_model(larva_model, collapsibles={}, sectiondicts={}):
    # update_window_from_dict(model['sensorimotor_params'], window)
    # window = collapsibles['ENERGETICS'].update(model['energetics_params'], window)
    # window = collapsibles['BODY'].update(model['body_params'], window)

    for name, dict, kwargs in zip(['PHYSICS', 'ENERGETICS', 'BODY'],[larva_model['sensorimotor_params'], larva_model['energetics_params'],
                                   larva_model['body_params']],
                                  [{}, {'toggle': True, 'disabled': True}, {}]):
        collapsibles[name] = CollapsibleDict(name, True, dict=dict, type_dict=None, **kwargs)

    module_conf = []
    for k, v in larva_model['neural_params']['modules'].items():
        s = CollapsibleDict(k.upper(), False, dict=larva_model['neural_params'][f'{k}_params'],
                            dict_name=k.upper(), toggle=v)
        # s = CollapsibleDict(k.upper(), False, dict=larva_model['neural_params'][f'{k}_params'],
        #                     dict_name=k, toggle=v)
        collapsibles[s.name] = s
        # print('k')
        # sectiondicts[d.name] = d
        module_conf.append(s.get_section())
    collapsibles['BRAIN'] = Collapsible('BRAIN', True, module_conf)
    brain_layout = sg.Col([collapsibles['BRAIN'].get_section()])
    non_brain_layout = sg.Col([collapsibles['PHYSICS'].get_section(),
                               collapsibles['BODY'].get_section(),
                               collapsibles['ENERGETICS'].get_section()])

    model_layout = [[brain_layout, non_brain_layout]]

    collapsibles['MODEL'] = Collapsible('MODEL', True, model_layout)
    return [collapsibles['MODEL'].get_section()]


def init_environment(env_params, collapsibles={}, sectiondicts={}):
    collapsibles['ARENA'] = CollapsibleDict('ARENA', True, dict=env_params['arena_params'])

    # larva_place_conf = [
    #     [sg.Text('# larvae:', **text_kwargs), sg.In(1, key='Nagents', **text_kwargs)],
    #     [sg.Text('distribution:', **text_kwargs),
    #      sg.Combo(larva_place_modes, key='larva_place_mode', enable_events=True, readonly=True, **text_kwargs)],
    #     [sg.Text('loc:', **text_kwargs), sg.In(None, key='larva_loc', **text_kwargs)],
    #     [sg.Text('scale:', **text_kwargs), sg.In(None, key='larva_scale', **text_kwargs)],
    #     # [sg.Text('orients:', size=(12, 1)), sg.In(None, key='larva_orientations', **text_kwargs)],
    # ]
    temp = CollapsibleDict('LARVA_DISTRIBUTION', True, dict=env_params['place_params'], toggle=True,
                                                    disabled=False)
    collapsibles.update(temp.get_subdicts())

    food_pars=env_params['food_params']

    temp = CollapsibleDict('FOOD_DISTRIBUTION', True, dict=food_pars['food_distro'], toggle=True,
                                                    disabled=False)
    collapsibles.update(temp.get_subdicts())


    collapsibles['FOOD_GRID'] = CollapsibleDict('FOOD_GRID', True, dict=food_pars['food_grid'], toggle=True,
                                                    disabled=False)

    food_conf = [
        collapsibles['FOOD_DISTRIBUTION'].get_section(),
        collapsibles['FOOD_GRID'].get_section(),
        [sg.Button('Food list', **button_kwargs)]
    ]

    odor_conf = [

    ]

    collapsibles['FOOD'] = Collapsible('FOOD', True, food_conf)
    collapsibles['ODORS'] = Collapsible('ODORS', True, odor_conf)

    env_layout = [
        collapsibles['ARENA'].get_section(),
        collapsibles['LARVA_DISTRIBUTION'].get_section(),
        collapsibles['FOOD'].get_section(),
        collapsibles['ODORS'].get_section()
    ]

    collapsibles['ENVIRONMENT'] = Collapsible('ENVIRONMENT', True, env_layout)

    return collapsibles['ENVIRONMENT'].get_section()


def update_sim(window, values, collapsibles, sectiondicts, output_keys, food_list, border_list):
    if values['EXP'] != '':
        exp = values['EXP']
        exp_conf = copy.deepcopy(exp_types[exp])
        update_model(exp_conf['fly_params'], window, collapsibles, sectiondicts)

        food_list, border_list = update_environment(exp_conf['env_params'], window, collapsibles, sectiondicts,
                                                    food_list, border_list)


        output_dict = {}
        for k in output_keys:

            if k in exp_conf['collections']:
                output_dict[k] = True
            else:
                output_dict[k] = False
        collapsibles['OUTPUT'].update(window, output_dict)
        window.Element('sim_id').Update(value=f'{exp}_{next_idx(exp)}')
        window.Element('path').Update(value=f'single_runs/{exp}')
        return food_list, border_list


def build_analysis_tab():
    fig, save_to, save_as, figure_agg = None, '', '', None
    func, func_kwargs = None, {}
    data = {}
    data_list = [
        [sg.Text('DATASETS', **header_kwargs)],
        [sg.Listbox(values=[], change_submits=False, size=(20, len(data.keys())), key='DATASET_IDS',
                    enable_events=True)],
        [sg.FolderBrowse(button_text='Add', initial_folder=SingleRunFolder, key='DATASET_DIR', change_submits=True,
                         enable_events=True, **button_kwargs)],
        [sg.Button('Remove', **button_kwargs), sg.Button('Add ref', **button_kwargs),
         sg.Button('Change ID', **button_kwargs)],
        # [sg.Text(' ' * 12)]
    ]

    dim = 2000
    figure_w, figure_h = dim, dim
    graph_dict = {
        'crawl_pars': plot_crawl_pars,
        'angular_pars': plot_ang_pars,
        'endpoint_params': plot_endpoint_params,
        'stride_Dbend': plot_stride_Dbend,
        'stride_Dorient': plot_stride_Dorient,
        'interference': plot_interference,
        'dispersion': plot_dispersion,
        'stridesNpauses': plot_stridesNpauses,
        'turn_duration': plot_turn_duration,
        'turns': plot_turns,
        'odor_concentration': plot_odor_concentration,
        'pathlength': plot_pathlength,
        'food_amount': plot_food_amount,
        'gut': plot_gut,
        'barplot': barplot,
        'deb': plot_debs,
    }
    graph_list = [
        [sg.Text('GRAPHS', **header_kwargs)],
        [sg.Listbox(values=list(graph_dict), change_submits=True, size=(20, len(list(graph_dict))), key='GRAPH_LIST')],
        [sg.Button('Graph args', **button_kwargs), sg.Button('Draw', **button_kwargs),
         sg.Button('Save', **button_kwargs)]]

    graph_code = sg.Col([[sg.MLine(size=(70, 30), key='GRAPH_CODE')]])
    graph_canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='GRAPH_CANVAS')]])
    graph_instructions = sg.Col([[sg.Pane([graph_canvas, graph_code], size=(figure_w, figure_h))],
                                 [sg.Text('Grab square above and slide upwards to view source code for graph')]])

    analysis_layout = [
        [sg.Col(data_list)],
        [sg.Col(graph_list), graph_instructions]
    ]
    return analysis_layout, graph_dict, data, func, func_kwargs, fig, save_to, save_as, figure_agg


def build_simulation_tab():
    sim_datasets = []

    larva_model = copy.deepcopy(mock_larva)
    env_params = copy.deepcopy(mock_env)
    food_list = env_params['food_params']['food_list']
    border_list = []

    module_dict = larva_model['neural_params']['modules']
    module_keys = list(module_dict.keys())

    collapsibles = {}
    sectiondicts = {}

    exp_layout = [sg.Col([
        [sg.Text('Experiment:', **header_kwargs),
         sg.Combo(list(exp_types.keys()), key='EXP', enable_events=True, readonly=True, **text_kwargs)],
        [sg.Button('Load', **button_kwargs), sg.Button('Configure', **button_kwargs),
         sg.Button('Run', **button_kwargs)]
    ])]

    saved_conf_layout = [sg.Col([
        [sg.Text('Environment:', **header_kwargs),
         sg.Combo(list(loadSimConfDict().keys()), key='SAVED_CONF', enable_events=True, readonly=True,**text_kwargs)],
        [sg.Button('Use', **button_kwargs), sg.Button('Delete', **button_kwargs)]
    ])]

    output_keys = list(effector_collection.keys())
    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    # output_dict=dict(zip([f'collect_{k}' for k in output_keys], [False]*len(output_keys)))
    # sectiondicts['OUTPUT'] = SectionDict('OUTPUT', output_dict)
    collapsibles['OUTPUT'] = CollapsibleDict('OUTPUT', True, dict=output_dict)

    sim_conf = [[sg.Text('Sim id:', **text_kwargs), sg.In('unnamed_sim', key='sim_id', **text_kwargs)],
                [sg.Text('Path:', **text_kwargs), sg.In('single_runs', key='path', **text_kwargs)],
                [sg.Text('Duration (min):', **text_kwargs), sg.In(3, key='sim_dur', **text_kwargs)],
                [sg.Text('Timestep (sec):', **text_kwargs), sg.In(0.1, key='dt', **text_kwargs)],
                bool_button('Box2D', False),
                collapsibles['OUTPUT'].get_section()
                ]

    collapsibles['CONFIGURATION'] = Collapsible('CONFIGURATION', True, sim_conf)

    global_conf_layout = collapsibles['CONFIGURATION'].get_section()

    conf_layout = [[sg.Col([exp_layout, global_conf_layout])]]
    # conf_layout = [[sg.Col([exp_layout, saved_conf_layout, global_conf_layout])]]

    model_layout = init_model(larva_model, collapsibles, sectiondicts)

    env_layout = init_environment(env_params, collapsibles, sectiondicts)
    env_layout = [[sg.Col([saved_conf_layout, env_layout])]]
    # env_layout = init_environment(env_params, collapsibles, sectiondicts)

    simulation_layout = [
        [
            sg.Col(conf_layout),
            sg.Col(env_layout),
            sg.Col(model_layout)

        ]
    ]
    return simulation_layout, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys, food_list, border_list


def build_model_tab():
    model_layout = []
    return model_layout


def eval_analysis(event, values, window, func, func_kwargs, data, figure_agg, fig, save_to, save_as, graph_dict):
    if event == 'DATASET_DIR':
        if values['DATASET_DIR'] != '':
            d = LarvaDataset(dir=values['DATASET_DIR'])
            data[d.id] = d
            update_data_list(window, data)

            # window['DATASET_DIR'] = ''
    elif event == 'Add ref':
        d = LarvaDataset(dir=RefFolder)
        data[d.id] = d
        window.Element('DATASET_IDS').Update(values=list(data.keys()))
    elif event == 'Remove':
        if len(values['DATASET_IDS']) > 0:
            id = values['DATASET_IDS'][0]
            data.pop(id, None)
            update_data_list(window, data)
    elif event == 'Change ID':
        data = change_dataset_id(window, values, data)
    elif event == 'Save':
        save_plot(fig, save_to, save_as)
    elif event == 'Graph args':
        func_kwargs = set_kwargs(func_kwargs, title='Graph arguments')
    elif event == 'Draw':
        figure_agg, fig, save_to, save_as = draw_figure(window, func, func_kwargs, data, figure_agg)
    func, func_kwargs = update_func(window, values, func, func_kwargs, graph_dict)
    # print(values['DATASET_DIR'], type(values['DATASET_DIR']))
    # print(window.FindElement('DATASET_DIR').values)
    return window, func, func_kwargs, data, figure_agg, fig, save_to, save_as


def eval_model(event, values, window):
    return window


def get_model(window, values, module_keys, sectiondicts, collapsibles, base_model):
    module_dict = dict(zip(module_keys, [window[f'TOGGLE_{k.upper()}'].metadata.state for k in module_keys]))
    base_model['neural_params']['modules'] = module_dict

    for name, pars in zip(['PHYSICS', 'ENERGETICS', 'BODY'],
                          ['sensorimotor_params', 'energetics_params', 'body_params']):
        if collapsibles[name].state is None:
            base_model[pars] = None
        else:
            base_model[pars] = collapsibles[name].get_dict(values, window)
        # collapsibles[name].update(window,dict)

    # module_conf = []
    for k, v in module_dict.items():
        base_model['neural_params'][f'{k}_params'] = collapsibles[k.upper()].get_dict(values, window)
        # collapsibles[k.upper()].update(window,larva_model['neural_params'][f'{k}_params'])
    return base_model


def get_environment(window, values, module_keys, sectiondicts, collapsibles, base_environment, food_list, border_list):
    base_environment['place_params']=collapsibles['LARVA_DISTRIBUTION'].get_dict(values, window)
    base_environment['food_params']['food_distro']=collapsibles['FOOD_DISTRIBUTION'].get_dict(values, window)
    base_environment['food_params']['food_grid']=collapsibles['FOOD_GRID'].get_dict(values, window)
    base_environment['food_params']['food_list'] = food_list
    print(base_environment['food_params']['food_grid'])
    base_environment['border_list'] = border_list
    base_environment['arena_params'] = collapsibles['ARENA'].get_dict(values, window)
    return base_environment


def get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list, border_list):
    exp = values['EXP']
    exp_conf = copy.deepcopy(exp_types[exp])

    sim_params ={}
    sim_params['sim_id'] = str(values['sim_id'])
    sim_params['sim_dur'] = float(values['sim_dur'])
    sim_params['dt'] = float(values['dt'])
    sim_params['path'] = str(values['path'])
    sim_params['Box2D'] = window['TOGGLE_Box2D'].metadata.state

    temp=collapsibles['OUTPUT'].get_dict(values,window)
    exp_conf['collections'] = [k for k in output_keys if temp[k]]

    env_params = get_environment(window, values, module_keys, sectiondicts, collapsibles, exp_conf['env_params'],
                                 food_list, border_list)

    fly_params = get_model(window, values, module_keys, sectiondicts, collapsibles, exp_conf['fly_params'])

    sim_config = {
                  'enrich': True,
                  'experiment': exp,
                  'sim_params': sim_params,
                  'env_params': env_params,
                  'fly_params': fly_params,
                  }
    return sim_config


def eval_simulation(event, values, window, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys,
                    food_list, border_list):
    # print(list(collapsibles.keys()))
    if event.startswith('OPEN SEC'):
        sec_name = event.split()[-1]
        if collapsibles[sec_name].state is not None:
            collapsibles[sec_name].state = not collapsibles[sec_name].state
            window[event].update(SYMBOL_DOWN if collapsibles[sec_name].state else SYMBOL_UP)
            window[f'SEC {sec_name}'].update(visible=collapsibles[sec_name].state)
    elif event == 'Load':
        food_list, border_list = update_sim(window, values, collapsibles, sectiondicts, output_keys, food_list,
                                            border_list)
    elif event == 'Delete':
        if values['SAVED_CONF'] != '':
            deleteSimConf(values['SAVED_CONF'])
            window['SAVED_CONF'].update(values=list(loadSimConfDict().keys()))

    elif event == 'Use':
        if values['SAVED_CONF'] != '':
            conf = loadSimConf(values['SAVED_CONF'])
            food_list = conf['food_list']
            border_list = conf['border_list']
            collapsibles['ARENA'].update(window, conf['arena_params'])
            window['TOGGLE_Box2D'].metadata.state = conf['Box2D']
            window['TOGGLE_Box2D'].update(image_data=on_image if window['TOGGLE_Box2D'].metadata.state else off_image)
    elif 'TOGGLE' in event:
        # print(event)
        if window[event].metadata.state is not None:
            window[event].metadata.state = not window[event].metadata.state
            window[event].update(image_data=on_image if window[event].metadata.state else off_image)
    elif event == 'Food list':
        food_list = gui_table(food_list, agent_pars['Food'])
        # update_food_placement(window, food_list, collapsibles, place_params=None)

    elif event == 'Configure':
        if values['EXP'] != '':
            sim_config = get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list,
                                        border_list)
            new_food_list, new_border_list = configure_sim(
                fly_params=sim_config['fly_params'],
                env_params=sim_config['env_params'])
            l = [[sg.Text('Store new configuration', size=(20, 1)), sg.In(k='CONF_ID', size=(10, 1))],
                 [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
            e, v = sg.Window('Sim configuration', l).read(close=True)
            if e == 'Ok':
                food_list = new_food_list
                border_list = new_border_list
                # update_food_placement(window, food_list, collapsibles, place_params=None)
            elif e == 'Store' and v['CONF_ID'] != '':
                food_list = new_food_list
                border_list = new_border_list
                # update_food_placement(window, food_list, collapsibles, place_params=None)
                conf = {
                    'food_list': food_list,
                    'border_list': border_list,
                    'arena_params': sectiondicts['ARENA'].get_dict(values, window),
                    'Box2D': window['TOGGLE_Box2D'].metadata.state,
                }
                conf_id = v['CONF_ID']
                saveSimConf(conf, conf_id)
                window['SAVED_CONF'].update(values=list(loadSimConfDict().keys()))


    elif event == 'Run':
        if values['EXP'] != '':
            sim_config = get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list,
                                        border_list)
            vis_kwargs = {'mode': 'video'}
            d = run_sim(**sim_config, **vis_kwargs)
            if d is not None:
                sim_datasets.append(d)
    print(values)
    return food_list, border_list


# -------------------------------- GUI Starts Here -------------------------------#
# fig = your figure you want to display.  Assumption is that 'fig' holds the      #
#       information to display.                                                   #
# --------------------------------------------------------------------------------#
sg.theme('LightGreen')


def run_gui():
    analysis_layout, graph_dict, data, func, func_kwargs, fig, save_to, save_as, figure_agg = build_analysis_tab()
    # fig, save_to, save_as, figure_agg = None, '', '', None
    # func, func_kwargs = None, {}
    # data = {}
    # data_list = [
    #     [sg.Text('DATASETS', **header_kwargs)],
    #     [sg.Listbox(values=[], change_submits=True, size=(20, len(data.keys())), key='DATASET_IDS')],
    #     [sg.FolderBrowse(button_text='Add', initial_folder=SingleRunFolder, key='DATASET_DIR', change_submits=True,
    #                      **button_kwargs)],
    #     [sg.Button('Remove', **button_kwargs), sg.Button('Add ref', **button_kwargs),
    #      sg.Button('Change ID', **button_kwargs)],
    #     # [sg.Text(' ' * 12)]
    # ]
    #
    # dim = 1000
    # figure_w, figure_h = dim, dim
    # graph_dict = {
    #     'crawl_pars': plot_crawl_pars,
    #     'angular_pars': plot_ang_pars,
    #     'endpoint_params': plot_endpoint_params,
    #     'stride_Dbend': plot_stride_Dbend,
    #     'stride_Dorient': plot_stride_Dorient,
    #     'interference': plot_interference,
    #     'dispersion': plot_dispersion,
    #     'stridesNpauses': plot_stridesNpauses,
    #     'turn_duration': plot_turn_duration,
    #     'turns': plot_turns,
    #     'pathlength': plot_pathlength,
    #     'food_amount': plot_food_amount,
    #     'gut': plot_gut,
    #     'barplot': barplot,
    #     'deb': plot_debs,
    # }
    # graph_list = [
    #     [sg.Text('GRAPHS', **header_kwargs)],
    #     [sg.Listbox(values=list(graph_dict), change_submits=True, size=(20, len(list(graph_dict))), key='GRAPH_LIST')],
    #     [sg.Button('Set args', **button_kwargs), sg.Button('Draw', **button_kwargs), sg.Button('Save', **button_kwargs)]]
    #
    # graph_code = sg.Col([[sg.MLine(size=(70, 30), key='GRAPH_CODE')]])
    # graph_canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='GRAPH_CANVAS')]])
    # graph_instructions = sg.Col([[sg.Pane([graph_canvas, graph_code], size=(figure_w, figure_h))],
    #                              [sg.Text('Grab square above and slide upwards to view source code for graph')]])
    #
    # analysis_layout = [
    #     [sg.Col(data_list)],
    #     [sg.Col(graph_list), graph_instructions]
    # ]
    # fig, save_to, save_as, figure_agg = None, '', '', None
    # func, func_kwargs = None, {}
    # data = {}
    # data_list = [
    #     [sg.Text('DATASETS', **header_kwargs)],
    #     [sg.Listbox(values=[], change_submits=True, size=(20, len(data.keys())), key='DATASET_IDS')],
    #     [sg.FolderBrowse(button_text='Add', initial_folder=SingleRunFolder, key='DATASET_DIR', change_submits=True,
    #                      **button_kwargs)],
    #     [sg.Button('Remove', **button_kwargs), sg.Button('Add ref', **button_kwargs),
    #      sg.Button('Change ID', **button_kwargs)],
    #     # [sg.Text(' ' * 12)]
    # ]
    #
    # dim = 1000
    # figure_w, figure_h = dim, dim
    # graph_dict = {
    #     'crawl_pars': plot_crawl_pars,
    #     'angular_pars': plot_ang_pars,
    #     'endpoint_params': plot_endpoint_params,
    #     'stride_Dbend': plot_stride_Dbend,
    #     'stride_Dorient': plot_stride_Dorient,
    #     'interference': plot_interference,
    #     'dispersion': plot_dispersion,
    #     'stridesNpauses': plot_stridesNpauses,
    #     'turn_duration': plot_turn_duration,
    #     'turns': plot_turns,
    #     'pathlength': plot_pathlength,
    #     'food_amount': plot_food_amount,
    #     'gut': plot_gut,
    #     'barplot': barplot,
    #     'deb': plot_debs,
    # }
    # graph_list = [
    #     [sg.Text('GRAPHS', **header_kwargs)],
    #     [sg.Listbox(values=list(graph_dict), change_submits=True, size=(20, len(list(graph_dict))), key='GRAPH_LIST')],
    #     [sg.Button('Set args', **button_kwargs), sg.Button('Draw', **button_kwargs), sg.Button('Save', **button_kwargs)]]
    #
    # graph_code = sg.Col([[sg.MLine(size=(70, 30), key='GRAPH_CODE')]])
    # graph_canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='GRAPH_CANVAS')]])
    # graph_instructions = sg.Col([[sg.Pane([graph_canvas, graph_code], size=(figure_w, figure_h))],
    #                              [sg.Text('Grab square above and slide upwards to view source code for graph')]])
    #
    # analysis_layout = [
    #     [sg.Col(data_list)],
    #     [sg.Col(graph_list), graph_instructions]
    # ]
    simulation_layout, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys, food_list, border_list = build_simulation_tab()
    model_layout = build_model_tab()

    layout = [
        [sg.TabGroup([[
            sg.Tab('Model', model_layout, background_color='darkseagreen', key='MODEL_TAB'),
            sg.Tab('Simulation', simulation_layout, background_color='darkseagreen', key='SIMULATION_TAB'),
            sg.Tab('Analysis', analysis_layout, background_color='darkseagreen', key='ANALYSIS_TAB')]],
            key='ACTIVE_TAB', tab_location='top', selected_title_color='purple')]
    ]

    window = sg.Window('Larvaworld gui', layout, resizable=True, finalize=True, size=(2000, 1200))

    while True:
        event, values = window.read()
        if event in (None, 'Exit'):
            break
        tab = values['ACTIVE_TAB']
        if tab == 'ANALYSIS_TAB':
            window, func, func_kwargs, data, figure_agg, fig, save_to, save_as = eval_analysis(event, values, window,
                                                                                               func, func_kwargs, data,
                                                                                               figure_agg, fig, save_to,
                                                                                               save_as, graph_dict)
        elif tab == 'MODEL_TAB':
            window = eval_model(event, values, window)
        elif tab == 'SIMULATION_TAB':
            food_list, border_list = eval_simulation(event, values, window, sim_datasets, collapsibles, module_keys,
                                                     sectiondicts,
                                                     output_keys, food_list, border_list)
    window.close()


if __name__ == "__main__":
    run_gui()
