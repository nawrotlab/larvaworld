import PySimpleGUI as sg
import matplotlib
import inspect
from tkinter import *

from lib.anal.plotting import *
from lib.gui.gui_lib import header_kwargs, button_kwargs, delete_figure_agg, draw_canvas, set_kwargs
from lib.stor.larva_dataset import LarvaDataset
from lib.stor.paths import SingleRunFolder, get_parent_dir, RefFolder


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