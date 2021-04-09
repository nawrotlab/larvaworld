import copy

import PySimpleGUI as sg

from lib.conf import test_larva, odor_gain_pars, opt_pars_dict, space_pars_dict
from lib.conf.batch_modes import test_batch, batch_types
from lib.gui.gui_lib import CollapsibleDict, button_kwargs, Collapsible, text_kwargs, header_kwargs, set_agent_dict, \
    buttonM_kwargs, named_list_layout, gui_table
from lib.gui.simulation_tab import get_sim_conf
from lib.sim.single_run import generate_config, next_idx
from lib.stor.datagroup import loadConfDict, loadConf, deleteConf, saveConf
import numpy as np


def init_batch(batch, collapsibles={}):
    # collapsibles['METHODS'] = CollapsibleDict('METHODS', True,
    #                                                dict=batch['methods'], type_dict=method_pars_dict)

    # collapsibles['SPACE_SEARCH'] = CollapsibleDict('SPACE_SEARCH', True,
    #                                                dict=batch['space_search'], type_dict=space_pars_dict)
    collapsibles['OPTIMIZATION'] = CollapsibleDict('OPTIMIZATION', True,
                                                   dict=batch['optimization'], type_dict=opt_pars_dict,
                                                   toggle=True, disabled=True)
    batch_layout = [
        # collapsibles['SPACE_SEARCH'].get_section(),
        collapsibles['OPTIMIZATION'].get_section(),
    ]
    collapsibles['BATCH'] = Collapsible('BATCH', True, batch_layout)
    return collapsibles['BATCH'].get_section()


def update_batch(batch, window, collapsibles, space_search):
    window.Element('EXP').Update(value=batch['exp'])
    collapsibles['OPTIMIZATION'].update(window, batch['optimization'])
    space_search = batch['space_search']
    return space_search


def get_batch(window, values, collapsibles, space_search):
    batch = {}
    batch['optimization'] = collapsibles['OPTIMIZATION'].get_dict(values, window)
    batch['space_search'] = space_search
    batch['exp'] = values['EXP']
    return copy.deepcopy(batch)


def build_batch_tab(collapsibles):
    batch = copy.deepcopy(test_batch)
    space_search = batch['space_search']
    l_exp = [sg.Col([
        named_list_layout(text='Batch:', key='BATCH_CONF', choices=list(loadConfDict('Batch').keys())),
        [sg.Button('Load', key='LOAD_BATCH', **button_kwargs),
         sg.Button('Save', key='SAVE_BATCH', **button_kwargs),
         sg.Button('Delete', key='DELETE_BATCH', **button_kwargs),
         sg.Button('Run', key='RUN_BATCH', **button_kwargs)]
    ])]
    batch_conf = [[sg.Text('Batch id:', **text_kwargs), sg.In('unnamed_batch_0', key='batch_id', **text_kwargs)],
                  [sg.Text('Path:', **text_kwargs), sg.In('unnamed_batch', key='batch_path', **text_kwargs)],
                  # [sg.Text('Duration (min):', **text_kwargs), sg.In(3, key='batch_dur', **text_kwargs)],
                  # [sg.Text('Timestep (sec):', **text_kwargs), sg.In(0.1, key='dt', **text_kwargs)],
                  # collapsibles['OUTPUT'].get_section()
                  ]

    collapsibles['BATCH_CONFIGURATION'] = Collapsible('BATCH_CONFIGURATION', True, batch_conf)
    l_conf = collapsibles['BATCH_CONFIGURATION'].get_section()
    l_batch0 = [[sg.Col([l_exp,
                         l_conf,
                         [sg.Button('SPACE_SEARCH', **buttonM_kwargs)]])]]
    l_batch1 = [init_batch(batch, collapsibles)]
    l_batch = [[sg.Col(l_batch0), sg.Col(l_batch1)]]
    return l_batch, collapsibles, space_search


def set_space_table(space_search):
    N = len(space_search['pars'])
    t0 = []
    for i in range(N):
        d = {}
        for k, v in space_search.items():
            d[k] = v[i]
        t0.append(d)

    t1 = gui_table(t0, space_pars_dict, title='space search')
    dic = {}
    for k in list(space_pars_dict.keys()):
        dic[k] = [l[k] for l in t1]
        # if k == 'ranges':
        #     dic[k] = np.array(dic[k])
    return dic





def eval_batch(event, values, window, collapsibles, space_search):
    if event == 'LOAD_BATCH':
        if values['BATCH_CONF'] != '':
            # batch = copy.deepcopy(batch_types[values['BATCH_CONF']])
            batch=values['BATCH_CONF']
            window.Element('batch_id').Update(value=f'{batch}_{next_idx(batch, type="batch")}')
            window.Element('batch_path').Update(value=batch)
            conf = loadConf(batch, 'Batch')
            space_search = update_batch(conf, window, collapsibles, space_search)

    elif event == 'SAVE_BATCH':
        l = [[sg.Text('Store new batch', size=(20, 1)), sg.In(k='BATCH_ID', size=(10, 1))],
             [sg.Ok(), sg.Cancel()]]
        e, v = sg.Window('Batch configuration', l).read(close=True)
        if e == 'Ok':
            batch = get_batch(window, values, collapsibles)
            batch_id = v['BATCH_ID']
            saveConf(batch, 'Batch', batch_id)
            window['BATCH_CONF'].update(values=list(loadConfDict('Batch').keys()))

    elif event == 'DELETE_BATCH':
        if values['BATCH_CONF'] != '':
            deleteConf(values['BATCH_CONF'], 'Batch')
            window['BATCH_CONF'].update(values=list(loadConfDict('Batch').keys()))
            window['BATCH_CONF'].update(value='')

    elif event == 'RUN_BATCH':
        if values['BATCH_CONF'] != '' and values['EXP'] != '':
            from lib.sim.batch_lib import prepare_batch, batch_run
            batch = get_batch(window, values, collapsibles, space_search)
            batch_id = str(values['batch_id'])
            batch_path = str(values['batch_path'])
            # dir = f'{batch_path}/{batch["exp"]}'
            sim_params = get_sim_conf(window, values)
            life_params = collapsibles['LIFE'].get_dict(values, window)
            sim_config = generate_config(exp=batch["exp"], sim_params=sim_params, life_params=life_params)
            batch_kwargs = prepare_batch(batch, batch_id, sim_config)
            df=batch_run(**batch_kwargs)

            # run_batch(batch)

    elif event == 'SPACE_SEARCH':
        space_search = set_space_table(space_search)

    return space_search
