import copy

import PySimpleGUI as sg

from lib.anal.combining import render_mpl_table
from lib.conf.dtype_dicts import opt_pars_dict, space_pars_dict, batch_methods_dict
from lib.conf.batch_conf import test_batch
from lib.gui.gui_lib import CollapsibleDict, button_kwargs, Collapsible, text_kwargs, buttonM_kwargs, named_list_layout, \
    gui_table, save_gui_conf, delete_gui_conf, draw_canvas, delete_figure_agg, header_kwargs, named_bool_button, \
    on_image, off_image
from lib.gui.simulation_tab import update_sim, get_exp
from lib.conf.conf import loadConfDict, loadConf, next_idx


def update_batch(batch, window, collapsibles, space_search):

    collapsibles['METHODS'].update(window, batch['methods'])
    collapsibles['OPTIMIZATION'].update(window, batch['optimization'])
    space_search = batch['space_search']
    window['TOGGLE_save_data_flag'].metadata.state = batch['run_kwargs']['save_data_flag']
    window['TOGGLE_save_data_flag'].update(image_data=on_image if window['TOGGLE_save_data_flag'].metadata.state else off_image)
    return space_search


def get_batch(window, values, collapsibles, space_search):
    batch = {}
    batch['methods'] = collapsibles['METHODS'].get_dict(values, window)
    batch['optimization'] = collapsibles['OPTIMIZATION'].get_dict(values, window)
    batch['space_search'] = space_search
    batch['exp'] = values['EXP']
    batch['run_kwargs']={}
    batch['run_kwargs']['save_data_flag'] = window['TOGGLE_save_data_flag'].metadata.state
    return copy.deepcopy(batch)


def build_batch_tab(collapsibles):
    batch_results={}
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
                  named_bool_button('Save data', False, toggle_name='save_data_flag'),
                  ]

    collapsibles['BATCH_CONFIGURATION'] = Collapsible('BATCH_CONFIGURATION', True, batch_conf, disp_name='CONFIGURATION')
    collapsibles['METHODS'] = CollapsibleDict('METHODS', True, dict=batch['methods'], type_dict=batch_methods_dict)
    collapsibles['OPTIMIZATION'] = CollapsibleDict('OPTIMIZATION', True, dict=batch['optimization'], type_dict=opt_pars_dict,
                                                   toggle=True, disabled=True)

    batch_graph_list = [
        [sg.Text('GRAPHS', **header_kwargs)],
        [sg.Listbox(values=[], change_submits=True, size=(20, 5), key='BATCH_GRAPH_LIST')],
        [sg.Button('Draw', **button_kwargs, k='DRAW_BATCH_FIG')]]

    l_batch0 = sg.Col([l_exp,
                       collapsibles['BATCH_CONFIGURATION'].get_section(),
                       collapsibles['METHODS'].get_section(),
                       [sg.Button('SPACE_SEARCH', **buttonM_kwargs)],
                       collapsibles['OPTIMIZATION'].get_section(),
                       [sg.Col(batch_graph_list)]
                       # [sg.Button('RESULTS', **buttonM_kwargs, k='BATCH_DF')],
                       ])

    # dim = 500
    figure_w, figure_h = 800, 800

    batch_canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='BATCH_CANVAS')]])
    # batch_canvas_0 = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='BATCH_CANVAS')]])
    # batch_canvas_1 = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='BATCH_DF')]])
    #
    # batch_canvas = sg.Col([[sg.Pane([batch_canvas_0, batch_canvas_1], size=(figure_w, 2*figure_h))],
    #                              [sg.Text('Grab square above and slide upwards to view batch results')]])


    l_batch = [[l_batch0, batch_canvas]]
    batch_fig_agg= None
    # batch_fig_agg= {'figs' : [], 'df' : []}
    return l_batch, collapsibles, space_search, batch_results, batch_fig_agg


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


# def draw_batch_canvas(window, figs, df, batch_fig_agg):
#     # if len(figs)>1 :
#     #     fig=combine_figures(figs)
#     # else :
#     #     fig = figs[0]
#     for f in batch_fig_agg['figs']:
#         delete_figure_agg(f)
#         batch_fig_agg['figs'].remove(f)
#     for f in figs :
#         try :
#             batch_fig_agg['figs'].append(draw_canvas(window['BATCH_CANVAS'].TKCanvas, f))
#         except :
#             pass
#     for f in batch_fig_agg['df']:
#         delete_figure_agg(f)
#         batch_fig_agg['df'].remove(f)
#     df_ax, df_fig = render_mpl_table(df)
#     batch_fig_agg['df'].append(draw_canvas(window['BATCH_DF'].TKCanvas, df_fig))
#     return batch_fig_agg


def draw_batch_canvas(window, fig, batch_fig_agg):
    if batch_fig_agg:
        delete_figure_agg(batch_fig_agg)
    figure_agg = draw_canvas(window['BATCH_CANVAS'].TKCanvas, fig)  # draw the figure
    return figure_agg


def eval_batch(event, values, window, collapsibles, dicts, batch_fig_agg):
    space_search = dicts['space_search']
    if event == 'LOAD_BATCH':
        if values['BATCH_CONF'] != '':
            batch=values['BATCH_CONF']
            window.Element('batch_id').Update(value=f'{batch}_{next_idx(batch, type="batch")}')
            window.Element('batch_path').Update(value=batch)
            conf = loadConf(batch, 'Batch')
            space_search = update_batch(conf, window, collapsibles, space_search)

            window.Element('EXP').Update(value=conf['exp'])
            source_units, border_list, larva_groups, source_groups = update_sim(window, conf['exp'], collapsibles)
            dicts['source_units'] = source_units
            dicts['border_list'] = border_list
            dicts['larva_groups'] = larva_groups
            dicts['source_groups'] = source_groups



    elif event == 'SAVE_BATCH':
        batch = get_batch(window, values, collapsibles, space_search)
        save_gui_conf(window, batch, 'Batch')


    elif event == 'DELETE_BATCH':
        delete_gui_conf(window, values, 'Batch')


    elif event == 'RUN_BATCH':
        if values['BATCH_CONF'] != '' and values['EXP'] != '':
            from lib.sim.batch_lib import prepare_batch, batch_run
            batch = get_batch(window, values, collapsibles, space_search)
            batch_id = str(values['batch_id'])
            batch_path = str(values['batch_path'])
            # sim_params = get_sim_conf(window, values)
            # life_params = collapsibles['LIFE'].get_dict(values, window)
            exp_conf = get_exp(window, values, collapsibles, dicts)
            # sim_config = generate_config(exp=batch["exp"], sim_params=sim_params, life_params=life_params)
            batch_kwargs = prepare_batch(batch, batch_id, exp_conf)
            df, fig_dict=batch_run(**batch_kwargs)
            df_ax, df_fig = render_mpl_table(df)
            fig_dict['dataframe'] = df_fig
            dicts['batch_results']['df'] = df
            dicts['batch_results']['fig_dict'] = fig_dict
            window.Element('BATCH_GRAPH_LIST').Update(values=list(fig_dict.keys()))
            # batch_fig_agg = draw_batch_canvas(window, figs, df, batch_fig_agg)

    elif event == 'SPACE_SEARCH':
        space_search = set_space_table(space_search)

    elif event == 'DRAW_BATCH_FIG':
        if len(values['BATCH_GRAPH_LIST']) > 0:
            choice = values['BATCH_GRAPH_LIST'][0]
            fig=dicts['batch_results']['fig_dict'][choice]
            batch_fig_agg = draw_batch_canvas(window, fig, batch_fig_agg)

    dicts['space_search'] = space_search

    return dicts, batch_fig_agg
