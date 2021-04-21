import copy
import threading

import PySimpleGUI as sg

from lib.anal.combining import render_mpl_table
from lib.conf.batch_conf import test_batch
from lib.gui.gui_lib import CollapsibleDict, Collapsible, named_list_layout, \
    gui_table, save_gui_conf, delete_gui_conf, named_bool_button, on_image, off_image, GraphList, b12_kws, b_kws
from lib.gui.simulation_tab import update_sim, get_exp
from lib.conf.conf import loadConfDict, loadConf, next_idx
import lib.conf.dtype_dicts as dtypes


def update_batch(batch, window, collapsibles):
    collapsibles['Methods'].update(window, batch['methods'])
    collapsibles['Optimization'].update(window, batch['optimization'])
    window['TOGGLE_save_data_flag'].metadata.state = batch['run_kwargs']['save_data_flag']
    window['TOGGLE_save_data_flag'].update(
        image_data=on_image if window['TOGGLE_save_data_flag'].metadata.state else off_image)
    return batch['space_search']


def get_batch(window, values, collapsibles, space_search):
    batch = {
        'methods': collapsibles['Methods'].get_dict(values, window),
        'optimization': collapsibles['Optimization'].get_dict(values, window),
        'space_search': space_search,
        'exp': values['EXP'],
        'run_kwargs': {'save_data_flag': window['TOGGLE_save_data_flag'].metadata.state},
    }
    return copy.deepcopy(batch)


def build_batch_tab(collapsibles, graph_lists, dicts):
    batch = copy.deepcopy(test_batch)
    dicts['batch_results'] = {}
    dicts['space_search'] = batch['space_search']
    l_exp = [sg.Col([
        named_list_layout(text='Batch:', key='BATCH_CONF', choices=list(loadConfDict('Batch').keys())),
        [sg.B('Load', key='LOAD_BATCH', **b_kws),
         sg.B('Save', key='SAVE_BATCH', **b_kws),
         sg.B('Delete', key='DELETE_BATCH', **b_kws),
         sg.B('Run', key='RUN_BATCH', **b_kws)]
    ])]
    batch_conf = [[sg.Text('Batch id:'), sg.In('unnamed_batch_0', key='batch_id')],
                  [sg.Text('Path:'), sg.In('unnamed_batch', key='batch_path')],
                  named_bool_button('Save data', False, toggle_name='save_data_flag'),
                  ]

    collapsibles['BATCH_CONFIGURATION'] = Collapsible('BATCH_CONFIGURATION', True, batch_conf,
                                                      disp_name='Configuration')
    s1 = CollapsibleDict('Methods', True, dict=dtypes.get_dict('batch_methods'),
                         type_dict=dtypes.get_dict_dtypes('batch_methods'))
    s2 = CollapsibleDict('Optimization', True, dict=batch['optimization'],
                         type_dict=dtypes.get_dict_dtypes('optimization'),
                         toggle=True, disabled=True, toggled_subsections=None)
    for s in [s1, s2]:
        collapsibles.update(s.get_subdicts())
    collapsibles.update(s.get_subdicts())

    graph_lists['BATCH'] = GraphList('BATCH')

    l_batch0 = sg.Col([l_exp,
                       collapsibles['BATCH_CONFIGURATION'].get_section(),
                       collapsibles['Methods'].get_section(),
                       [sg.B('Space search', **b12_kws)],
                       collapsibles['Optimization'].get_section(),
                       [graph_lists['BATCH'].get_layout()]
                       ])

    l_batch = [[l_batch0, graph_lists['BATCH'].canvas]]
    return l_batch, collapsibles, graph_lists, dicts


def set_space_table(space_search):
    N = len(space_search['pars'])
    t0 = []
    for i in range(N):
        d = {}
        for k, v in space_search.items():
            d[k] = v[i]
        t0.append(d)

    t1 = gui_table(t0, dtypes.get_dict_dtypes('space_search'), title='space search')
    dic = {}
    for k in list(dtypes.get_dict_dtypes('space_search').keys()):
        dic[k] = [l[k] for l in t1]
        # if k == 'ranges':
        #     dic[k] = np.array(dic[k])
    return dic


def eval_batch(event, values, window, collapsibles, dicts, graph_lists):
    if event == 'LOAD_BATCH':
        if values['BATCH_CONF'] != '':
            batch_type = values['BATCH_CONF']
            window.Element('batch_id').Update(value=f'{batch_type}_{next_idx(batch_type, type="batch")}')
            window.Element('batch_path').Update(value=batch_type)
            conf = loadConf(batch_type, 'Batch')
            dicts['space_search'] = update_batch(conf, window, collapsibles)

            window.Element('EXP').Update(value=conf['exp'])
            source_units, border_list, larva_groups, source_groups = update_sim(window, conf['exp'], collapsibles)
            dicts['source_units'] = source_units
            dicts['border_list'] = border_list
            dicts['larva_groups'] = larva_groups
            dicts['source_groups'] = source_groups



    elif event == 'SAVE_BATCH':
        batch = get_batch(window, values, collapsibles, dicts['space_search'])
        save_gui_conf(window, batch, 'Batch')


    elif event == 'DELETE_BATCH':
        delete_gui_conf(window, values, 'Batch')


    elif event == 'RUN_BATCH':
        if values['BATCH_CONF'] != '' and values['EXP'] != '':
            from lib.sim.batch_lib import prepare_batch, batch_run
            batch = get_batch(window, values, collapsibles, dicts['space_search'])
            batch_id = str(values['batch_id'])
            batch_path = str(values['batch_path'])
            exp_conf = get_exp(window, values, collapsibles, dicts)
            batch_kwargs = prepare_batch(batch, batch_id, exp_conf)
            # dicts['batch_kwargs']=batch_kwargs
            #
            # thread = threading.Thread(target=batch_run, kwargs=batch_kwargs, daemon=True)
            # thread.start()

            df, fig_dict = batch_run(**batch_kwargs)
            df_ax, df_fig = render_mpl_table(df)
            fig_dict['dataframe'] = df_fig
            dicts['batch_results']['df'] = df
            dicts['batch_results']['fig_dict'] = fig_dict
            graph_lists['BATCH'].update(window, dicts['batch_results']['fig_dict'])

    elif event == 'Space search':
        dicts['space_search'] = set_space_table(dicts['space_search'])

    return dicts, graph_lists
