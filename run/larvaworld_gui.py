#!/usr/bin/env python
# !/usr/bin/env python
import copy

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
import inspect

from tkinter import *

from lib.aux.collecting import effector_collection
from lib.conf import exp_types, default_sim, mock_larva, box2d_space, larva_place_modes, \
    food_place_modes, pref_exp_np, agent_pars

from lib.sim.gui_lib import gui_table, SectionDict, bool_button, Collapsible, \
    set_kwargs, on_image, off_image, SYMBOL_UP, SYMBOL_DOWN, button_kwargs, header_kwargs, \
    text_kwargs, on_image_disabled, retrieve_value
from lib.sim.single_run import run_sim, next_idx, configure_sim

sys.path.insert(0, '..')
from lib.anal.plotting import *
from lib.stor.larva_dataset import LarvaDataset
from lib.stor.paths import SingleRunFolder, RefFolder, get_parent_dir

matplotlib.use('TkAgg')


# Class holding the button graphic info. At this time only the state is kept


def draw_canvas(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


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
             [sg.Ok(), sg.Cancel()]]
        e, v = sg.Window('Change dataset ID', l).read(close=True)
        if e == 'Ok':
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
        except:
            print('Plot not available for these datasets')
    return figure_agg, fig, save_to, save_as


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
    collapsibles['BRAIN'].update(window, module_dict_upper)


def update_environment(env_params, window, collapsibles, sectiondicts, food_list):
    arena_params = env_params['arena_params']
    for k, v in arena_params.items():
        window.Element(k).Update(value=v)

    if env_params['food_params'] is None :
        food_list=[]
    else :
        food_list = env_params['food_params']['food_list']
    place_params = env_params['place_params']
    update_placement(place_params, window, collapsibles, sectiondicts, food_list)
    return food_list


def update_placement(place_params, window, collapsibles, sectiondicts, food_list):
    window.Element('Nagents').Update(value=place_params['initial_num_flies'])
    window.Element('larva_place_mode').Update(value=place_params['initial_fly_positions']['mode'])
    window.Element('larva_positions').Update(value=place_params['initial_fly_positions']['loc'])
    update_food_placement(window, food_list, collapsibles, place_params=place_params)


def update_food_placement(window, food_list, collapsibles, place_params=None):
    if len(food_list) > 0:
        Nfood = len(food_list)
        food_place_mode = None
        food_loc = None
        food_scale = None
        collapsibles['FOOD_DISTRIBUTION'].update(window, dict=None)
    else:
        if place_params is None:
            return
        Nfood = place_params['initial_num_food']
        if Nfood>0 :
            food_place_mode = place_params['initial_food_positions']['mode']
            food_loc = place_params['initial_food_positions']['loc']
            food_scale = place_params['initial_food_positions']['scale']
            window[f'TOGGLE_FOOD_DISTRIBUTION'].update(image_data=on_image_disabled)
            if collapsibles['FOOD_DISTRIBUTION'].state is None:
                collapsibles['FOOD_DISTRIBUTION'].state = False
        else :
            food_place_mode = None
            food_loc = None
            food_scale = None
            window[f'TOGGLE_FOOD_DISTRIBUTION'].update(image_data=off_image)
    window.Element('Nfood').Update(value=Nfood)
    window.Element('food_place_mode').Update(value=food_place_mode)
    window.Element('food_loc').Update(value=food_loc)
    window.Element('food_scale').Update(value=food_scale)


def init_model(larva_model, collapsibles={}, sectiondicts={}):
    # update_window_from_dict(model['sensorimotor_params'], window)
    # window = collapsibles['ENERGETICS'].update(model['energetics_params'], window)
    # window = collapsibles['BODY'].update(model['body_params'], window)

    for name, dict, kwargs in zip(['PHYSICS', 'ENERGETICS', 'BODY'],
                                  [larva_model['sensorimotor_params'], larva_model['energetics_params'],
                                   larva_model['body_params']],
                                  [{}, {'toggle': True, 'disabled': True}, {}]):
        sectiondicts[name] = SectionDict(name, dict)
        collapsibles[name] = Collapsible(name, False, sectiondicts[name].init_section(), **kwargs)

    module_conf = []
    for k, v in larva_model['neural_params']['modules'].items():
        d = SectionDict(k, larva_model['neural_params'][f'{k}_params'])
        s = Collapsible(k.upper(), False, d.init_section(), toggle=v)
        collapsibles[s.name] = s
        sectiondicts[d.name] = d
        module_conf.append(s.get_section())
    collapsibles['BRAIN'] = Collapsible('BRAIN', False, module_conf)

    model_layout = [
        collapsibles['PHYSICS'].get_section(),
        collapsibles['BODY'].get_section(),
        collapsibles['ENERGETICS'].get_section(),
        collapsibles['BRAIN'].get_section()
    ]

    collapsibles['MODEL'] = Collapsible('MODEL', True, model_layout)

    return [collapsibles['MODEL'].get_section()]


def init_environment(env_params, collapsibles={}, sectiondicts={}):
    sectiondicts['ARENA'] = SectionDict('ARENA', env_params['arena_params'])
    collapsibles['ARENA'] = Collapsible('ARENA', False, sectiondicts['ARENA'].init_section())

    larva_place_conf = [
        [sg.Text('# larvae:', size=(12, 1)), sg.In(1, key='Nagents', **text_kwargs)],
        [sg.Text('placement:', size=(12, 1)),
         sg.Combo(larva_place_modes, key='larva_place_mode', enable_events=True, readonly=True, font=('size', 10),
                  size=(15, 1))],
        [sg.Text('positions:', size=(12, 1)), sg.In(None, key='larva_positions', **text_kwargs)],
        # [sg.Text('orientations:', size=(12, 1)), sg.In(None, key='larva_orientations', **text_kwargs)],
    ]

    # food_list_conf = []
    # for f in env_params['food_params']['food_list']:
    #     d = SectionDict(k, larva_model['neural_params'][f'{k}_params'])
    #     s = Collapsible(k.upper(), False, d.init_section(), toggle=v)
    #     collapsibles[s.name] = s
    #     sectiondicts[d.name] = d
    #     food_list_conf.append(s.get_section())
    # collapsibles['FOOD_LIST'] = Collapsible('FOOD_LIST', False, food_list_conf)

    food_place_distro = [
        [sg.Text('distribution:', size=(12, 1)),
         sg.Combo(food_place_modes, key='food_place_mode', enable_events=True, readonly=True, font=('size', 10),
                  size=(15, 1))],
        [sg.Text('loc:', size=(12, 1)), sg.In(None, key='food_loc', **text_kwargs)],
        [sg.Text('scale:', size=(12, 1)), sg.In(None, key='food_scale', **text_kwargs)],
    ]
    collapsibles['FOOD_DISTRIBUTION'] = Collapsible('FOOD_DISTRIBUTION', False, food_place_distro, toggle=True, disabled=False)

    food_place_conf = [
        [sg.Text('# food:', size=(12, 1)), sg.In(1, key='Nfood', **text_kwargs)],
        collapsibles['FOOD_DISTRIBUTION'].get_section(),
        [sg.Button('Food list', **button_kwargs)]
    ]

    odor_conf = [

    ]

    collapsibles['LARVA_PLACEMENT'] = Collapsible('LARVA_PLACEMENT', False, larva_place_conf)
    collapsibles['FOOD_PLACEMENT'] = Collapsible('FOOD_PLACEMENT', False, food_place_conf)
    collapsibles['ODORS'] = Collapsible('ODORS', False, odor_conf)

    env_layout = [
        collapsibles['ARENA'].get_section(),
        collapsibles['LARVA_PLACEMENT'].get_section(),
        collapsibles['FOOD_PLACEMENT'].get_section(),
        collapsibles['ODORS'].get_section()
    ]

    collapsibles['ENVIRONMENT'] = Collapsible('ENVIRONMENT', True, env_layout)

    return [collapsibles['ENVIRONMENT'].get_section()]


def update_sim(window, values, collapsibles, sectiondicts, output_keys, food_list):
    if values['EXP'] != '':
        exp = values['EXP']
        exp_conf = copy.deepcopy(exp_types[exp])
        update_model(exp_conf['fly_params'], window, collapsibles, sectiondicts)

        food_list = update_environment(exp_conf['env_params'], window, collapsibles, sectiondicts, food_list)

        if 'sim_params' not in exp_conf.keys():
            exp_conf['sim_params'] = default_sim.copy()
        sim_params = exp_conf['sim_params']
        window.Element('sim_time_in_min').Update(value=sim_params['sim_time_in_min'])
        output_dict = {}
        for k in output_keys:
            # if k in sim_params['collect_effectors'] :
            #     output_dict[f'collect_{k}']=True
            # else :
            #     output_dict[f'collect_{k}'] = False
            if k in exp_conf['collect_effectors']:
                output_dict[k] = True
            else:
                output_dict[k] = False
        collapsibles['OUTPUT'].update(window, output_dict)
        sim_id = f'{exp}_{next_idx(exp)}'
        window.Element('sim_id').Update(value=sim_id)
        common_folder = f'single_runs/{exp}'
        window.Element('common_folder').Update(value=common_folder)
        return food_list


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
    env_params = pref_exp_np
    food_list = env_params['food_params']['food_list']

    module_dict = larva_model['neural_params']['modules']
    module_keys = list(module_dict.keys())

    collapsibles = {}
    sectiondicts = {}

    exp_layout = [
        [sg.Text('Experiment:', size=(10, 1)),
         sg.Combo(list(exp_types.keys()), key='EXP', enable_events=True, readonly=True, font=('size', 10),
                  size=(15, 1))],
        [sg.Button('Load', **button_kwargs), sg.Button('Configure', **button_kwargs),
         sg.Button('Run', **button_kwargs)]
    ]

    output_keys = list(effector_collection.keys())
    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    # output_dict=dict(zip([f'collect_{k}' for k in output_keys], [False]*len(output_keys)))
    sectiondicts['OUTPUT'] = SectionDict('OUTPUT', output_dict)
    collapsibles['OUTPUT'] = Collapsible('OUTPUT', False, sectiondicts['OUTPUT'].init_section())

    sim_conf = [[sg.Text('Sim id:', size=(12, 1)), sg.In('unnamed_sim', key='sim_id', **text_kwargs)],
                [sg.Text('Path:', size=(12, 1)), sg.In('single_runs', key='common_folder', **text_kwargs)],
                [sg.Text('Duration (min):', size=(12, 1)), sg.In(3, key='sim_time_in_min', **text_kwargs)],
                [sg.Text('Timestep (sec):', size=(12, 1)), sg.In(0.1, key='dt', **text_kwargs)],
                bool_button('Box2D', False),
                collapsibles['OUTPUT'].get_section()
                ]

    collapsibles['CONFIGURATION'] = Collapsible('CONFIGURATION', False, sim_conf)

    conf_layout = [
        collapsibles['CONFIGURATION'].get_section()
    ]

    model_layout = init_model(larva_model, collapsibles, sectiondicts)

    env_layout = init_environment(env_params, collapsibles, sectiondicts)

    simulation_layout = [
        [sg.Col(exp_layout)],
        [sg.Col(conf_layout)],
        [sg.Col(model_layout)],
        [sg.Col(env_layout)]
    ]
    return simulation_layout, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys, food_list


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
            base_model[pars] = sectiondicts[name].get_dict(values, window)
        # collapsibles[name].update(window,dict)

    # module_conf = []
    for k, v in module_dict.items():
        base_model['neural_params'][f'{k}_params'] = sectiondicts[k].get_dict(values, window)
        # collapsibles[k.upper()].update(window,larva_model['neural_params'][f'{k}_params'])
    return base_model


def get_environment(window, values, module_keys, sectiondicts, collapsibles, base_environment, food_list):
    base_environment['place_params']['initial_num_flies'] = retrieve_value(values['Nagents'], int)
    base_environment['place_params']['initial_fly_positions']['mode'] = retrieve_value(values['larva_place_mode'], str)
    # larva_loc=values['larva_positions']
    #
    # base_environment['place_params']['initial_fly_positions']['loc'] = np.array(literal_eval(larva_loc))

    base_environment['place_params']['initial_num_food'] = retrieve_value(values['Nfood'], int)
    base_environment['place_params']['initial_food_positions']['mode'] = retrieve_value(values['food_place_mode'], str)
    # base_environment['place_params']['initial_food_positions']['loc'] = retrieve_value(values['food_loc'], tuple)
    # base_environment['place_params']['initial_food_positions']['scale'] = retrieve_value(values['food_scale'], float)

    if base_environment['food_params'] is not None :
        base_environment['food_params']['food_list'] = food_list

    if window['TOGGLE_Box2D'].metadata.state:
        base_environment['space_params'] = box2d_space
    base_environment['arena_params'] = sectiondicts['ARENA'].get_dict(values, window)
    return base_environment


def get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list):
    exp = values['EXP']
    exp_conf = copy.deepcopy(exp_types[exp])

    sim_params = copy.deepcopy(default_sim)
    sim_params['sim_time_in_min'] = float(values['sim_time_in_min'])
    sim_params['dt'] = float(values['dt'])

    sim_params['collect_effectors'] = [k for k in output_keys if window[f'TOGGLE_{k}'].metadata.state == True]
    # sim_params['collect_effectors'] = dict(zip(output_keys, [window[f'TOGGLE_{f"collect_{k}"}'].metadata.state for k in output_keys]))

    env_params = get_environment(window, values, module_keys, sectiondicts, collapsibles, exp_conf['env_params'],
                                 food_list)
    # print(env_params['place_params'])

    fly_params = get_model(window, values, module_keys, sectiondicts, collapsibles, exp_conf['fly_params'])

    sim_config = {'sim_id': str(values['sim_id']),
                  'common_folder': str(values['common_folder']),
                  'enrich': True,
                  'experiment': exp,
                  'sim_params': sim_params,
                  'env_params': env_params,
                  'fly_params': fly_params,
                  }
    return sim_config


def eval_simulation(event, values, window, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys,
                    food_list):
    if event.startswith('OPEN SEC'):
        sec_name = event.split()[-1]
        if collapsibles[sec_name].state is not None:
            collapsibles[sec_name].state = not collapsibles[sec_name].state
            window[event].update(SYMBOL_DOWN if collapsibles[sec_name].state else SYMBOL_UP)
            window[f'SEC {sec_name}'].update(visible=collapsibles[sec_name].state)
    # elif event == 'EXP':
    elif event == 'Load':
        food_list = update_sim(window, values, collapsibles, sectiondicts, output_keys, food_list)
    elif 'TOGGLE' in event:
        if window[event].metadata.state is not None:
            window[event].metadata.state = not window[event].metadata.state
            window[event].update(image_data=on_image if window[event].metadata.state else off_image)
    elif event == 'Food list':
        food_list = gui_table(food_list, agent_pars['Food'])
        update_food_placement(window, food_list, collapsibles, place_params=None)

    elif event == 'Configure':
        if values['EXP'] != '':
            sim_config = get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list)
            food_list = configure_sim(
                fly_params=sim_config['fly_params'],
                env_params=sim_config['env_params'])
            update_food_placement(window, food_list, collapsibles, place_params=None)

    elif event == 'Run':
        if values['EXP'] != '':
            sim_config = get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list)
            vis_kwargs = {'mode': 'video'}
            d = run_sim(**sim_config, **vis_kwargs)
            if d is not None:
                sim_datasets.append(d)
    return food_list


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

    simulation_layout, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys, food_list = build_simulation_tab()
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
                                                                                               func,
                                                                                               func_kwargs, data,
                                                                                               figure_agg, fig, save_to,
                                                                                               save_as, graph_dict)
        elif tab == 'MODEL_TAB':
            window = eval_model(event, values, window)
        elif tab == 'SIMULATION_TAB':

            food_list = eval_simulation(event, values, window, sim_datasets, collapsibles, module_keys, sectiondicts,
                                        output_keys,
                                        food_list)
    window.close()


if __name__ == "__main__":
    run_gui()
