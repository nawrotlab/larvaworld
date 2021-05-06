import os
import PySimpleGUI as sg
# from tkinter import *

from lib.gui.gui_lib import t8_kws, ButtonGraphList, b6_kws, graphic_button, t10_kws, t16_kws, default_run_window, w_kws
from lib.stor import paths
from lib.anal.plotting import graph_dict
from lib.stor.larva_dataset import LarvaDataset
import lib.conf.dtype_dicts as dtypes

# def update_data_list(window, data):
#     window.Element('DATASET_IDS').Update(values=list(data.keys()))



def change_dataset_id(window, values, data):
    if len(values['DATASET_IDS']) > 0:
        old_id = values['DATASET_IDS'][0]
        l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k='NEW_ID', size=(10, 1))],
             [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
        e, v = sg.Window('Change dataset ID', l).read(close=True)
        if e == 'Ok':
            data[v['NEW_ID']] = data.pop(old_id)
            window.Element('DATASET_IDS').Update(values=list(data.keys()))
        elif e == 'Store':
            d = data[old_id]
            d.set_id(v['NEW_ID'])
            data[v['NEW_ID']] = data.pop(old_id)
            window.Element('DATASET_IDS').Update(values=list(data.keys()))
    return data


def build_analysis_tab():
    collapsibles = {}
    graph_lists = {}
    dicts={'analysis_data' : {}}
    data = dicts['analysis_data']
    data_list = [
        [sg.Text('Datasets', **t8_kws),
         graphic_button('remove', 'Remove', tooltip='Remove a dataset from the analysis list.'),
         graphic_button('play', 'Replay', tooltip='Replay/Visualize the dataset.'),
         graphic_button('box_add', 'Add ref', tooltip='Add the reference experimental dataset to the analysis list.'),
         graphic_button('edit', 'Change ID', tooltip='Change the dataset ID transiently or permanently.')],
        [sg.Col([[sg.Listbox(values=list(data.keys()), size=(16, len(data)), change_submits=False, key='DATASET_IDS',
                             enable_events=True),
                  graphic_button('search_add', 'DATASET_DIR', initial_folder=paths.SingleRunFolder, change_submits=True,
                                 enable_events=True, target=(0, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
                                 tooltip='Browse to add datasets to the analysis list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.')]])]]

    graph_lists['ANALYSIS'] = ButtonGraphList(name='ANALYSIS', fig_dict=graph_dict)
    analysis_layout = [[sg.Col(data_list + graph_lists['ANALYSIS'].get_layout(as_col=False),vertical_alignment='t'),graph_lists['ANALYSIS'].canvas]]
    return analysis_layout, collapsibles, graph_lists, dicts


def eval_analysis(event, values, window, collapsibles, dicts, graph_lists):
    if event == 'DATASET_DIR':
        dr=values['DATASET_DIR']
        if dr != '':
            if os.path.exists(f'{dr}/data'):
                d = LarvaDataset(dir=dr)
                dicts['analysis_data'][d.id] = d
            else:
                for ddr in [x[0] for x in os.walk(dr)]:
                    if os.path.exists(f'{ddr}/data'):
                        d = LarvaDataset(dir=ddr)
                        dicts['analysis_data'][d.id] = d
            window.Element('DATASET_IDS').Update(values=list(dicts['analysis_data'].keys()))

    elif event == 'Add ref':
        d = LarvaDataset(dir=paths.RefFolder)
        dicts['analysis_data'][d.id] = d
        window.Element('DATASET_IDS').Update(values=list(dicts['analysis_data'].keys()))
    elif event == 'Remove':
        if len(values['DATASET_IDS']) > 0:
            id = values['DATASET_IDS'][0]
            dicts['analysis_data'].pop(id, None)
            window.Element('DATASET_IDS').Update(values=list(dicts['analysis_data'].keys()))
    elif event == 'Change ID':
        dicts['analysis_data'] = change_dataset_id(window, values, dicts['analysis_data'])
    elif event == 'Replay':
        if len(values['DATASET_IDS']) > 0:
            id = values['DATASET_IDS'][0]
            d = dicts['analysis_data'][id]
            if 'Visualization' in list(collapsibles.keys()) :
                vis_kwargs = collapsibles['Visualization'].get_dict(values, window)
            else :
                vis_kwargs = dtypes.get_dict('visualization', mode='video', video_speed=60)

            if 'Replay' in list(collapsibles.keys()) :
                replay_kwargs = collapsibles['Replay'].get_dict(values, window)
            else :
                replay_kwargs = dtypes.get_dict('replay', arena_pars=None)
            d.visualize(vis_kwargs=vis_kwargs, **replay_kwargs)

    elif event == 'ANALYSIS_SAVE_FIG':
        graph_lists['ANALYSIS'].save_fig()
    elif event == 'ANALYSIS_FIG_ARGS':
        graph_lists['ANALYSIS'].set_fig_args()
    elif event == 'ANALYSIS_DRAW_FIG':
        graph_lists['ANALYSIS'].generate(window, dicts['analysis_data'])
    return dicts, graph_lists

if __name__ == "__main__":
    sg.theme('LightGreen')
    l, col, graphs, dicts = build_analysis_tab()
    w = sg.Window('Analysis gui', l, size=(1800, 1200), **w_kws, location=(300, 100))
    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, col, graphs)
        dicts, graphs = eval_analysis(e, v, w, col, dicts, graphs)
    w.close()