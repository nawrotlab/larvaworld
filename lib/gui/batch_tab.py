import copy
import threading

import PySimpleGUI as sg

from lib.anal.combining import render_mpl_table
from lib.gui.gui_lib import CollapsibleDict, Collapsible, \
    save_gui_conf, delete_gui_conf, named_bool_button, GraphList, b12_kws, b_kws, \
    graphic_button, t10_kws, t12_kws, t18_kws, t8_kws, t6_kws, CollapsibleTable, w_kws, default_run_window, col_kws, \
    col_size
from lib.gui.simulation_tab import update_sim, get_exp
from lib.conf.conf import loadConfDict, loadConf, next_idx
import lib.conf.dtype_dicts as dtypes
from lib.sim.single_run import get_exp_conf


def update_batch(batch, window, collapsibles):
    collapsibles['Methods'].update(window, batch['methods'])
    collapsibles['Optimization'].update(window, batch['optimization'])
    window['TOGGLE_save_data_flag'].set_state(state=batch['run_kwargs']['save_data_flag'])
    collapsibles['space_search'].update_table(window, batch['space_search'])


def get_batch(window, values, collapsibles, exp=None):
    if exp is None:
        exp = values['EXP']
    batch = {
        'methods': collapsibles['Methods'].get_dict(values, window),
        'optimization': collapsibles['Optimization'].get_dict(values, window),
        'space_search': collapsibles['space_search'].dict,
        'exp': exp,
        'run_kwargs': {'save_data_flag': window['TOGGLE_save_data_flag'].metadata.state},
    }
    return copy.deepcopy(batch)


def build_batch_tab():
    dicts={}
    collapsibles = {}
    graph_lists = {}

    l_exp = [sg.Col([
        [sg.Text('Batch', **t6_kws),
         graphic_button('load', 'LOAD_BATCH'),
         graphic_button('data_add', 'SAVE_BATCH'),
         graphic_button('data_remove', 'DELETE_BATCH'),
         graphic_button('play', 'RUN_BATCH')],
        [sg.Combo(list(loadConfDict('Batch').keys()), key='BATCH_CONF',
                  enable_events=True, readonly=True, **t18_kws)],

    ])]
    batch_conf = [[sg.Text('Batch id:', **t10_kws), sg.In('unnamed_batch_0', key='batch_id', **t18_kws)],
                  [sg.Text('Path:', **t10_kws), sg.In('unnamed_batch', key='batch_path', **t18_kws)],
                  named_bool_button('Save data', False, toggle_name='save_data_flag'),
                  ]

    collapsibles['BATCH_CONFIGURATION'] = Collapsible('BATCH_CONFIGURATION', True, batch_conf,
                                                      disp_name='Configuration')
    s1 = CollapsibleDict('Methods', False, dict=dtypes.get_dict('batch_methods'),
                         type_dict=dtypes.get_dict_dtypes('batch_methods'))
    s2 = CollapsibleTable('space_search', False, headings=['pars', 'ranges', 'Ngrid'], dict={},
                          disp_name='Space search',
                          type_dict=dtypes.get_dict_dtypes('space_search'))
    s3 = CollapsibleDict('Optimization', False, dict=dtypes.get_dict('optimization'),
                         type_dict=dtypes.get_dict_dtypes('optimization'),
                         toggle=True, disabled=True, toggled_subsections=None)
    for s in [s1, s2, s3]:
        collapsibles.update(s.get_subdicts())
    graph_lists['BATCH'] = GraphList('BATCH')

    l_batch0 = sg.Col([l_exp,
                       collapsibles['BATCH_CONFIGURATION'].get_section(),
                       collapsibles['Methods'].get_section(),
                       collapsibles['space_search'].get_section(),
                       collapsibles['Optimization'].get_section(),
                       [graph_lists['BATCH'].get_layout()]
                       ],**col_kws, size=col_size(0.3))

    l_batch = [[l_batch0, graph_lists['BATCH'].canvas]]
    return l_batch, collapsibles, graph_lists, dicts


# def set_space_table(space_search):
#     if space_search['pars'] is None :
#         t0=[]
#     else :
#         N = len(space_search['pars'])
#         t0 = []
#         for i in range(N):
#             d = {}
#             for k, v in space_search.items():
#                 d[k] = v[i]
#             t0.append(d)
#
#     t1 = gui_table(t0, dtypes.get_dict_dtypes('space_search'), title='space search')
#     dic = {}
#     for k in list(dtypes.get_dict_dtypes('space_search').keys()):
#         dic[k] = [l[k] for l in t1]
#         # if k == 'ranges':
#         #     dic[k] = np.array(dic[k])
#     return dic


def eval_batch(event, values, window, collapsibles, dicts, graph_lists):
    if event == 'LOAD_BATCH':
        if values['BATCH_CONF'] != '':
            batch_type = values['BATCH_CONF']
            window.Element('batch_id').Update(value=f'{batch_type}_{next_idx(batch_type, type="batch")}')
            window.Element('batch_path').Update(value=batch_type)
            conf = loadConf(batch_type, 'Batch')
            update_batch(conf, window, collapsibles)
            if 'EXP' in window.element_list():
                window.Element('EXP').Update(value=conf['exp'])
                update_sim(window, conf['exp'], collapsibles)
            else:
                dicts['batch_exp'] = conf['exp']

    elif event == 'SAVE_BATCH':
        batch = get_batch(window, values, collapsibles)
        save_gui_conf(window, batch, 'Batch')

    elif event == 'DELETE_BATCH':
        delete_gui_conf(window, values, 'Batch')

    elif event == 'RUN_BATCH':
        if values['BATCH_CONF'] != '':
            from lib.sim.batch_lib import prepare_batch, batch_run
            batch_id = str(values['batch_id'])
            batch_path = str(values['batch_path'])
            if 'EXP' not in window.element_list() or values['EXP'] == '':
                exp = dicts['batch_exp']
                batch = get_batch(window, values, collapsibles, exp=exp)
                sim_params = {
                    'sim_id': None,
                    'sim_dur': 1.0,
                    'dt': 0.1,
                    'path': None,
                    'Box2D': False,
                }
                exp_conf = get_exp_conf(exp, sim_params)
            else:
                batch = get_batch(window, values, collapsibles)
                exp_conf = get_exp(window, values, collapsibles)
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

    return dicts, graph_lists


if __name__ == "__main__":
    sg.theme('LightGreen')
    dicts = {
        'sim_results': {'datasets': []},
        'batch_kwargs': None,
        'batch_results': {},
        'analysis_data': {},
    }
    l, col, graphs,d = build_batch_tab()
    dicts.update(d)
    w = sg.Window('Batch gui', l, size=(1800, 1200), **w_kws, location=(300, 100))
    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, col, graphs)
        dicts, graphs = eval_batch(e, v, w, col, dicts, graphs)
    w.close()
