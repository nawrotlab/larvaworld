import os

import PySimpleGUI as sg
import matplotlib
import inspect
from tkinter import *

from PySimpleGUI import BUTTON_TYPE_BROWSE_FOLDER

from lib.gui import graphics
from lib.gui.gui_lib import t8_kws, ButtonGraphList, b6_kws, graphic_button, t10_kws, t16_kws
from lib.stor.paths import SingleRunFolder, RefFolder
from lib.anal.plotting import graph_dict
from lib.stor.larva_dataset import LarvaDataset


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

def build_analysis_tab(collapsibles, graph_lists):

    data_list = [
        [sg.Text('Datasets', **t8_kws),
            graphic_button('remove', 'Remove'),
            graphic_button('play', 'Replay'),
            graphic_button('box_add', 'Add ref'),
            graphic_button('edit', 'Change ID'),
            # sg.B('Remove', **b6_kws),
            # sg.B('Add ref', **b6_kws),
         # sg.B('Change ID', **b6_kws),
            # sg.B('Replay', **b6_kws)
        ],
        [sg.Col([[sg.Listbox(values=[], **t16_kws, change_submits=False, key='DATASET_IDS',enable_events=True),
                  graphic_button('search_add', 'DATASET_DIR',initial_folder=SingleRunFolder, change_submits=True,
                                  enable_events=True,target=(0, -1), button_type=BUTTON_TYPE_BROWSE_FOLDER)]])]]

    graph_lists['ANALYSIS'] = ButtonGraphList(name='ANALYSIS', fig_dict=graph_dict)
    analysis_layout = [[sg.Col(data_list)], [graph_lists['ANALYSIS'].get_layout(), graph_lists['ANALYSIS'].canvas]]
    return analysis_layout, collapsibles, graph_lists


def eval_analysis(event, values, window, collapsibles, graph_lists, dicts):
    if event == 'DATASET_DIR':
        if values['DATASET_DIR'] != '':
            d = LarvaDataset(dir=values['DATASET_DIR'])
            dicts['analysis_data'][d.id] = d
            update_data_list(window, dicts['analysis_data'])

    elif event == 'Add ref':
        d = LarvaDataset(dir=RefFolder)
        dicts['analysis_data'][d.id] = d
        window.Element('DATASET_IDS').Update(values=list(dicts['analysis_data'].keys()))
    elif event == 'Remove':
        if len(values['DATASET_IDS']) > 0:
            id = values['DATASET_IDS'][0]
            dicts['analysis_data'].pop(id, None)
            update_data_list(window, dicts['analysis_data'])
    elif event == 'Change ID':
        dicts['analysis_data'] = change_dataset_id(window, values, dicts['analysis_data'])
    elif event == 'Replay':
        if len(values['DATASET_IDS']) > 0:
            id = values['DATASET_IDS'][0]
            d=dicts['analysis_data'][id]
            vis_kwargs=collapsibles['VISUALIZATION'].get_dict(values, window)
            replay_kwargs = collapsibles['REPLAY'].get_dict(values, window)
            d.visualize(vis_kwargs=vis_kwargs, **replay_kwargs)

    elif event == 'ANALYSIS_SAVE_FIG':
        graph_lists['ANALYSIS'].save_fig()
    elif event == 'ANALYSIS_FIG_ARGS':
        graph_lists['ANALYSIS'].set_fig_args()
    elif event == 'ANALYSIS_DRAW_FIG':
        graph_lists['ANALYSIS'].generate(window,dicts['analysis_data'])
    return graph_lists,dicts