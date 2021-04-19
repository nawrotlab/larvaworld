import random

import PySimpleGUI as sg
import numpy as np

import lib.aux.functions as fun
import lib.conf.dtype_dicts as dtypes
from lib.gui.gui_lib import CollapsibleDict, t8_kws, t14_kws, check_collapsibles, check_toggles, \
    retrieve_dict, t5_kws, t2_kws, color_pick_layout, popup_color_chooser, b6_kws, b12_kws

"""
    Demo - Drawing and moving demo

    This demo shows how to use a Graph Element to (optionally) display an image and then use the
    mouse to "drag" and draw rectangles and circles.
"""

W, H = 800, 800


def update_window_distro(values, window, name, start_point, end_point, s):
    p1, p2 = scale_xy(start_point, s), scale_xy(end_point, s)
    shape=values[f'{name}_DISTRO_shape']
    scale = np.abs(np.array(p2) - np.array(p1))
    if shape=='circle' :
        scale = tuple([np.max(scale), np.max(scale)])
    else :
        scale=tuple(scale/2)
    window[f'{name}_DISTRO_scale'].update(value=scale)
    window[f'{name}_DISTRO_loc'].update(value=p1)


def draw_shape(graph, p1, p2, shape, **kwargs):
    if p2 == p1:
        return None
    pp1, pp2 = np.array(p1), np.array(p2)
    dpp = np.abs(pp2 - pp1)
    if shape in ['rect', 'oval']:
        p1 = tuple(pp1 - dpp / 2)
        p2 = tuple(pp1 + dpp / 2)
        if shape == 'rect':
            fig = graph.draw_rectangle(p1, p2, **kwargs)
        elif shape == 'oval':
            fig = graph.draw_oval(p1, p2, **kwargs)
    elif shape == 'circle':
        fig = graph.draw_circle(p1, np.max(dpp), **kwargs)
    else:
        fig = None
    return fig








def add_agent_layout(name0, color, collapsibles):
    name = name0.upper()

    collapsibles[f'{name}_DISTRO'] = CollapsibleDict(f'{name}_DISTRO', False, dict=dtypes.get_dict('distro', class_name=name),
                                                     type_dict=dtypes.get_dict_dtypes('distro', class_name=name),
                                                     toggle=False, disabled=True, disp_name='distribution')

    collapsibles[f'{name}_ODOR'] = CollapsibleDict(f'{name}_ODOR', False, dict=dtypes.get_dict('odor'),
                                                   type_dict=dtypes.get_dict_dtypes('odor'),
                                                   toggle=False, disp_name='odor')
    l = [[sg.R(f'Add {name0}', 1, k=name, enable_events=True)],
         [sg.T('', **t2_kws),
          sg.R('single id', 2, disabled=True, k=f'{name}_single', enable_events=True, **t5_kws),
          sg.In(f'{name}_0', k=f'{name}_id', **t14_kws)],
         [sg.T('', **t2_kws), sg.R('group id', 2, disabled=True, k=f'{name}_group', enable_events=True, **t5_kws),
          sg.In(k=f'{name}_group_id', **t14_kws)],
         color_pick_layout(name, color),
         [sg.T('', **t5_kws), *collapsibles[f'{name}_DISTRO'].get_section()],
         [sg.T('', **t5_kws), *collapsibles[f'{name}_ODOR'].get_section()]]
    return l, collapsibles


def draw_arena(graph, arena_pars):
    graph.erase()
    shape = arena_pars['arena_shape']
    X, Y = arena_pars['arena_xdim'], arena_pars['arena_ydim']
    if shape == 'circular' and X is not None:
        arena = graph.draw_circle((int(W / 2), int(H / 2)), int(W / 2), fill_color='white', line_color='black',
                                  line_width=5)
        s = W / X
    elif shape == 'rectangular' and not None in (X, Y):
        if X >= Y:
            dif = (X - Y) / X
            arena = graph.draw_rectangle((0, int(H * dif / 2)), (W, H - int(H * dif / 2)), fill_color='white',
                                         line_color='black', line_width=5)
            s = W / X
        else:
            dif = (Y - X) / Y
            arena = graph.draw_rectangle((int(W * dif / 2), 0), (W - int(W * dif / 2), H), fill_color='white',
                                         line_color='black')
            s = H / Y
    return s, arena
    # pass


def scale_xy(xy, s):
    return (xy[0] - W / 2) / s, (xy[1] - H / 2) / s


def unscale_xy(xy, s):
    return xy[0] * s + W / 2, xy[1] * s + H / 2


def out_of_bounds(xy, arena_pars):
    shape = arena_pars['arena_shape']
    X, Y = arena_pars['arena_xdim'], arena_pars['arena_ydim']
    x, y = xy
    if shape == 'circular':
        return np.sqrt(x ** 2 + y ** 2) > X / 2
    elif shape == 'rectangular':
        return not (-X / 2 < x < X / 2 and -Y / 2 < y < Y / 2)


def delete_prior(prior_rect, graph):
    # print('xx')
    if type(prior_rect) == list:
        for pr in prior_rect:
            graph.delete_figure(pr)
    else:
        graph.delete_figure(prior_rect)


def inspect_distro(default_color, mode, shape, N, loc, scale, graph, s, id=None, item='LARVA', **kwargs):
    Ps = fun.generate_xy_distro(mode, shape, N, loc=unscale_xy(loc, s), scale=np.array(scale) * s)
    group_figs = []
    for i, P0 in enumerate(Ps):
        if item == 'SOURCE':
            temp = draw_source(P0, default_color, graph, s, **kwargs)
        elif item == 'LARVA':
            temp = draw_larva(P0, default_color, graph, s, **kwargs)
        group_figs.append(temp)
    return group_figs


def draw_source(P0, color, graph, s, amount, radius, **kwargs):
    fill_color = color if amount > 0 else None
    temp = graph.draw_circle(P0, radius * s, line_width=3, line_color=color, fill_color=fill_color)
    return temp


def draw_larva(P0, color, graph, s, orientation_range, **kwargs):
    points = np.array([[0.9, 0.1], [0.05, 0.1]])
    xy0 = fun.body(points)-np.array([0.5,0.0])
    a1, a2 = orientation_range
    a1, a2 = np.deg2rad(a1), np.deg2rad(a2)
    xy0 = fun.rotate_multiple_points(xy0, random.uniform(a1, a2), origin=[0, 0])
    xy0 /= 250
    xy0 *= s
    xy0 += np.array(P0)
    temp = graph.draw_polygon(xy0, line_width=3, line_color=color, fill_color=color)
    return temp


def check_abort(name, w, v, units, groups):
    o = name
    info = w['info']
    abort = True
    odor_on = w[f'TOGGLE_{o}_ODOR'].metadata.state

    if not odor_on:
        w[f'{o}_ODOR_odor_id'].update(value=None)
        w[f'{o}_ODOR_odor_intensity'].update(value=0.0)

    if o == 'SOURCE':
        food_on = w[f'TOGGLE_{o}_FOOD'].metadata.state
        if not odor_on and not food_on:
            info.update(value=f"Assign food and/or odor to the drawn source")
            return True
        else:
            if not food_on and float(v[f'{o}_FOOD_amount']) != 0.0:
                w[f'{o}_FOOD_amount'].update(value=0.0)
                info.update(value=f"Source food amount set to 0")
            elif food_on and float(v[f'{o}_FOOD_amount']) == 0.0:
                w[f'{o}_FOOD_amount'].update(value=10 ** -3)
                info.update(value=f"Source food amount set to default")
    if v[f'{o}_group_id'] == '' and v[f'{o}_id'] == '':
        info.update(value=f"Both {o.lower()} single id and group id are empty")
    elif not v[f'{o}_group'] and not v[f'{o}_single']:
        info.update(value=f"Select to add a single or a group of {o.lower()}s")
    elif v[f'{o}_single'] and (
            v[f'{o}_id'] in list(units.keys()) or v[f'{o}_id'] == ''):
        info.update(value=f"{o.lower()} id {v[f'{o}_id']} already exists or is empty")
    elif odor_on and v[f'{o}_ODOR_odor_id'] == '':
        info.update(value=f"Default odor id automatically assigned to the odor")
        id = v[f'{o}_group_id'] if v[f'{o}_group_id'] != '' else v[f'{o}_id']
        w[f'{o}_ODOR_odor_id'].update(value=f'{id}_odor')
    elif odor_on and not float(v[f'{o}_ODOR_odor_intensity']) > 0:
        info.update(value=f"Assign positive odor intensity to the drawn odor source")
    elif odor_on and (
            v[f'{o}_ODOR_odor_spread'] == '' or not float(v[f'{o}_ODOR_odor_spread']) > 0):
        info.update(value=f"Assign positive spread to the odor")
    elif v[f'{o}_group'] and (
            v[f'{o}_group_id'] in list(groups.keys()) or v[f'{o}_group_id'] == ''):
        info.update(value=f"{o.lower()} group id {v[f'{o}_group_id']} already exists or is empty")
    elif v[f'{o}_group'] and v[f'{o}_DISTRO_mode'] in ['', None]:
        info.update(value=f"Define a distribution mode")
    elif v[f'{o}_group'] and v[f'{o}_DISTRO_shape'] in ['', None]:
        info.update(value=f"Define a distribution shape")
    elif v[f'{o}_group'] and not int(v[f'{o}_DISTRO_N']) > 0:
        info.update(value=f"Assign a positive integer number of items for the distribution")
    else:
        abort = False
    # print(abort, o)
    return abort


def draw_env(env=None):
    sg.theme('LightGreen')
    # sg.theme('Dark Blue 3')
    collapsibles = {}
    if env is None:
        env = {'border_list': {},
               'arena_params': {'arena_xdim': 0.1,
                                'arena_ydim': 0.1,
                                'arena_shape': 'circular'},
               'food_params': {'source_units': {}, 'source_groups': {}, 'food_grid': None},
               'larva_params': {}
               }

    borders = env['border_list']
    arena_pars = env['arena_params']
    source_units = env['food_params']['source_units']
    source_groups = env['food_params']['source_groups']
    larva_groups = env['larva_params']
    larva_units = {}
    borders_f, source_units_f, source_groups_f, larva_units_f, larva_groups_f = {}, {}, {}, {}, {}
    inspect_figs = {}
    sample_fig, sample_pars = None, {}

    collapsibles['ARENA'] = CollapsibleDict('ARENA', True,
                                            dict=arena_pars, type_dict=dtypes.get_dict_dtypes('arena'),
                                            next_to_header=sg.B('Reset', k='RESET_ARENA', **b6_kws))

    source_l, collapsibles = add_agent_layout('Source', 'green', collapsibles)
    larva_l, collapsibles = add_agent_layout('Larva', 'black', collapsibles)

    collapsibles['SOURCE_FOOD'] = CollapsibleDict('SOURCE_FOOD', False, dict=dtypes.get_dict('food'), type_dict=dtypes.get_dict_dtypes('food'),
                                                  toggle=False, disp_name='food')

    s = None
    arena = None

    col2 = [

        *larva_l,

        *source_l,

        [sg.T('', **t5_kws), *collapsibles['SOURCE_FOOD'].get_section()],

        [sg.T('', **t5_kws), sg.T('shape', **t5_kws),
         sg.Combo(['rect', 'circle'], default_value='circle', k='SOURCE_shape', enable_events=True, readonly=True,
                  **t14_kws)],

        [sg.R('Add Border', 1, k='BORDER', enable_events=True)],
        [sg.T('', **t5_kws), sg.T('id', **t5_kws),
         sg.In(f'BORDER_{len(borders.keys())}', k='BORDER_id', **t14_kws)],
        [sg.T('', **t5_kws), sg.T('width', **t5_kws), sg.In(0.001, k='BORDER_width', **t14_kws)],
        color_pick_layout('BORDER', 'black'),

        [sg.R('Erase item', 1, k='-ERASE-', enable_events=True)],
        [sg.R('Move item', 1, True, k='-MOVE-', enable_events=True)],
        [sg.R('Inspect item', 1, True, k='-INSPECT-', enable_events=True)],
    ]

    col1 = [
        collapsibles['ARENA'].get_section(),
        [sg.Graph(
            canvas_size=(W, H),
            graph_bottom_left=(0, 0),
            graph_top_right=(W, H),
            key="-GRAPH-",
            change_submits=True,  # mouse click events
            background_color='black',
            drag_submits=True)],
        [sg.T('Instructions : ', k='info', size=(60, 3))],
        [sg.B('Ok', **t8_kws), sg.B('Cancel', **t8_kws)]
    ]
    layout = [[sg.Col(col1), sg.Col(col2)]]

    w = sg.Window("Environment configuration", layout, finalize=True)

    graph = w["-GRAPH-"]  # type: sg.Graph

    dragging, current = False, {}
    start_point = end_point = prior_rect = None

    graph.bind('<Button-3>', '+RIGHT+')
    while True:
        e, v = w.read()
        info = w["info"]
        if e in [None, 'Cancel']:
            break
        elif e == 'Ok':
            env['arena_params'] = collapsibles['ARENA'].get_dict(v, w)
            env['border_list'] = borders
            env['food_params']['source_units'] = source_units
            env['food_params']['source_groups'] = source_groups
            env['larva_params'] = larva_groups
            break  # exit
        check_collapsibles(w, e, collapsibles)
        check_toggles(w, e)
        if e == 'RESET_ARENA':
            info.update(value='Arena has been reset. All items erased.')
            s, arena = draw_arena(graph, collapsibles['ARENA'].get_dict(v, w))
            borders, source_units, source_groups, larva_units, larva_groups = {}, {}, {}, {}, {}
            borders_f, source_units_f, source_groups_f, larva_units_f, larva_groups_f = {}, {}, {}, {}, {}

            for ii in ['BORDER', 'SOURCE', 'LARVA']:
                w[f'{ii}_id'].update(value=f'{ii}_0')

        if arena is None:
            continue
        if e == '-MOVE-':
            graph.Widget.config(cursor='fleur')
            # graph.set_cursor(cursor='fleur')          # not yet released method... coming soon!
        elif not e.startswith('-GRAPH-'):
            # graph.set_cursor(cursor='left_ptr')       # not yet released method... coming soon!
            graph.Widget.config(cursor='left_ptr')
        if e.startswith('PICK'):
            target = e.split()[-1]
            choice = popup_color_chooser('Dark Blue 3')
            w[target].update(choice)
        if e == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = v["-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
                drag_figures = graph.get_figures_at_location((x, y))
                lastxy = x, y
            else:
                end_point = (x, y)
            if prior_rect:
                delete_prior(prior_rect, graph)
            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = x, y
            if None not in (start_point, end_point):
                if v['-MOVE-']:
                    delta_X, delta_Y = delta_x / s, delta_y / s
                    for fig in drag_figures:
                        if fig != arena:
                            for dic, f_dic in zip([borders, source_units, source_groups, larva_units, larva_groups],
                                                  [borders_f, source_units_f, source_groups_f, larva_units_f,
                                                   larva_groups_f]):
                                if fig in list(f_dic.keys()):
                                    w["info"].update(value=f"Item {f_dic[fig]} moved by ({delta_X}, {delta_Y})")
                                    if dic == source_units:
                                        X0, Y0 = dic[f_dic[fig]]['pos']
                                        dic[f_dic[fig]]['pos'] = (X0 + delta_X, Y0 + delta_Y)
                                        print(dic[f_dic[fig]]['pos'])
                                    elif dic in [source_groups, larva_groups]:
                                        X0, Y0 = dic[f_dic[fig]]['loc']
                                        dic[f_dic[fig]]['loc'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif dic == borders:
                                        dic[f_dic[fig]]['points'] = [(X0 + delta_X, Y0 + delta_Y) for X0, Y0 in
                                                                     dic[f_dic[fig]]['points']]
                            graph.move_figure(fig, delta_x, delta_y)
                            graph.update()
                elif v['-ERASE-']:
                    for figure in drag_figures:
                        if figure != arena:
                            # print(figure)
                            for dic, f_dic in zip([borders, source_units, source_groups, larva_groups],
                                                  [borders_f, source_units_f, source_groups_f, larva_groups_f]):
                                if figure in list(f_dic.keys()):
                                    w["info"].update(value=f"Item {f_dic[figure]} erased")
                                    dic.pop(f_dic[figure])
                                    f_dic.pop(figure)
                            graph.delete_figure(figure)
                elif v['-INSPECT-']:
                    for figure in drag_figures:
                        if figure != arena:
                            for dic, f_dic in zip([borders, source_units, source_groups, larva_groups],
                                                  [borders_f, source_units_f, source_groups_f, larva_groups_f]):

                                if figure in list(f_dic.keys()):
                                    if f_dic[figure] in list(inspect_figs.keys()):
                                        for f in inspect_figs[f_dic[figure]]:
                                            graph.delete_figure(f)
                                        inspect_figs.pop(f_dic[figure])
                                    else:
                                        w["info"].update(value=f"Inspecting item {f_dic[figure]} ")
                                        if dic in [source_groups, larva_groups]:
                                            inspect_figs[f_dic[figure]] = inspect_distro(id=f_dic[figure], s=s,
                                                                                         graph=graph,
                                                                                         **dic[f_dic[figure]])

                elif v['SOURCE'] or v['BORDER'] or v['LARVA']:
                    P1, P2 = scale_xy(start_point, s), scale_xy(end_point, s)
                    if any([out_of_bounds(P, collapsibles['ARENA'].get_dict(v, w)) for P in [P1, P2]]):
                        current = {}
                    else:
                        if v['SOURCE'] and not check_abort('SOURCE', w, v, source_units, source_groups):
                            o = 'SOURCE'
                            color = v[f'{o}_color']
                            if v['SOURCE_single'] or (v['SOURCE_group'] and sample_fig is None):

                                c = {'fill_color': color if w['TOGGLE_SOURCE_FOOD'].metadata.state else None,
                                     'line_color': color,
                                     'line_width': 5,
                                     }
                                prior_rect = draw_shape(graph, shape=v[f'{o}_shape'], p1=start_point,
                                                        p2=end_point, **c)
                                temp=np.max(np.abs(np.array(end_point)-np.array(start_point)))
                                w['SOURCE_FOOD_radius'].update(value=temp / s)
                                food_pars = collapsibles['SOURCE_FOOD'].get_dict(v, w)
                                odor_pars = collapsibles['SOURCE_ODOR'].get_dict(v, w)
                                sample_pars = {'default_color': color,
                                               **food_pars,
                                               **odor_pars,
                                               }

                                if v['SOURCE_single']:

                                    current = {v['SOURCE_id']: {
                                        'group': v['SOURCE_group_id'],
                                        'pos': P1,
                                        **sample_pars
                                    }}
                                    sample_fig, sample_pars = None, {}
                                else:
                                    info.update(value=f"Draw a sample item for the distribution")
                            elif v[f'{o}_group']:
                                update_window_distro(v, w, o, start_point, end_point, s)
                                distro_pars = collapsibles['SOURCE_DISTRO'].get_dict(v, w)
                                current = {v['SOURCE_group_id']: {
                                    **distro_pars,
                                    **sample_pars
                                }}
                                c = {'fill_color': None,
                                     'line_color': color,
                                     'line_width': 5,
                                     }
                                prior_rect = draw_shape(graph, shape=v[f'{o}_DISTRO_shape'], p1=start_point,
                                                        p2=end_point, **c)
                        elif v['LARVA'] and not check_abort('LARVA', w, v, larva_units, larva_groups):
                            o = 'LARVA'
                            color = v[f'{o}_color']
                            odor_pars = collapsibles[f'{o}_ODOR'].get_dict(v, w)
                            sample_larva_pars = {'default_color': color,
                                                 **odor_pars,
                                                 }
                            if v[f'{o}_group']:
                                update_window_distro(v, w, o, start_point, end_point, s)
                                distro_pars = collapsibles[f'{o}_DISTRO'].get_dict(v, w)
                                current = {v[f'{o}_group_id']: {
                                    **distro_pars,
                                    **sample_larva_pars
                                }}
                                c = {'fill_color': None,
                                     'line_color': color,
                                     'line_width': 5,
                                     }
                                prior_rect = draw_shape(graph, shape=v[f'{o}_DISTRO_shape'], p1=start_point,
                                                        p2=end_point, **c)

                        elif v['BORDER']:
                            id = v['BORDER_id']
                            if id in list(borders.keys()) or id == '':
                                info.update(value=f"Border id {id} already exists or is empty")
                            else:
                                dic = {'unique_id': id,
                                       'default_color': v['BORDER_color'],
                                       'width': v['BORDER_width'],
                                       'points': [P1, P2]}
                                current = fun.agent_list2dict([retrieve_dict(dic, dtypes.get_dict_dtypes('agent', class_name='Border'))])

                                prior_rect = graph.draw_line(start_point, end_point, color=v['BORDER_color'],
                                                             width=int(float(v['BORDER_width']) * s))



        elif e.endswith('+UP'):  # The drawing has ended because mouse up
            if v['BORDER'] and current != {}:
                info.update(value=f"Border {v['BORDER_id']} placed from {P1} to {P2}")
                borders_f[prior_rect] = id
                borders.update(current)
                w['BORDER_id'].update(value=f'BORDER_{len(borders.keys())}')
            elif v['SOURCE']:
                if v['SOURCE_single'] and current != {}:
                    info.update(value=f"Source {v['SOURCE_id']} placed at {P1}")
                    source_units_f[prior_rect] = v['SOURCE_id']
                    source_units.update(current)
                    w['SOURCE_id'].update(value=f'SOURCE_{len(source_units.keys())}')
                    w['SOURCE_ODOR_odor_id'].update(value='')
                elif v['SOURCE_group'] and sample_pars != {}:
                    if current == {}:
                        info.update(value=f"Sample item for source group {v['SOURCE_group_id']} detected." \
                                          "Now draw the distribution's space")

                        sample_fig = prior_rect
                    else:
                        id = v['SOURCE_group_id']
                        info.update(value=f"Source group {id} placed at {P1}")
                        source_groups_f[prior_rect] = id
                        source_groups.update(current)
                        w['SOURCE_group_id'].update(value=f'SOURCE_GROUP_{len(source_groups.keys())}')
                        w['SOURCE_ODOR_odor_id'].update(value='')
                        inspect_distro(id=id, **source_groups[id], graph=graph, s=s, item='SOURCE')
                        delete_prior(sample_fig, graph)
                        sample_fig, sample_pars = None, {}
            elif v['LARVA'] and current != {}:
                o = 'LARVA'
                units, groups = larva_units, larva_groups
                units_f, groups_f = larva_units_f, larva_groups_f
                if v[f'{o}_single']:
                    pass
                elif v[f'{o}_group']:
                    id = v[f'{o}_group_id']
                    info.update(value=f"{o} group {id} placed at {P1}")
                    groups_f[prior_rect] = id
                    groups.update(current)
                    w[f'{o}_group_id'].update(value=f'{o}_GROUP_{len(groups.keys())}')
                    w[f'{o}_ODOR_odor_id'].update(value='')
                    inspect_distro(id=id, **groups[id], graph=graph, s=s, item=o)
                    # delete_prior(sample_fig, graph)
                    sample_larva_pars = {}
            else:
                delete_prior(prior_rect, graph)

            dragging, current = False, {}
            start_point = end_point = prior_rect = None

        for o in ['SOURCE', 'LARVA']:
            w[f'{o}_single'].update(disabled=not v[o])
            w[f'{o}_group'].update(disabled=not v[o])
            collapsibles[f'{o}_DISTRO'].disable(w) if not v[f'{o}_group'] else collapsibles[f'{o}_DISTRO'].enable(w)
            if v[f'{o}_group']:
                w[f'{o}_id'].update(value='')
            # print(o, v[f'{o}_group'], int(v[f'{o}_DISTRO_N']))
            # elif v['SOURCE_id'] == '':
            #     w['SOURCE_id'].update(value=f'SOURCE_{len(source_units.keys())}')
        # print(list(borders.keys()))
    #
    w.close()
    return env


if __name__ == '__main__':
    draw_env()
