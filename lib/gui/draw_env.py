import copy
import random
import PySimpleGUI as sg
import numpy as np

import lib.aux.functions as fun
import lib.conf.dtype_dicts as dtypes
from lib.gui.gui_lib import CollapsibleDict, check_collapsibles, check_toggles, \
    retrieve_dict, t5_kws, t2_kws, color_pick_layout, b_kws, t40_kws, b_kws, w_kws, graphic_button, \
    check_togglesNcollapsibles

"""
    Demo - Drawing and moving demo

    This demo shows how to use a Graph Element to (optionally) display an image and then use the
    mouse to "drag" and draw rectangles and circles.
"""

W, H = 800, 800


def update_window_distro(values, window, name, start_point, end_point, s):
    p1, p2 = scale_xy(start_point, s), scale_xy(end_point, s)
    shape = values[f'{name}_DISTRO_shape']
    scale = np.abs(np.array(p2) - np.array(p1))
    if shape == 'circle':
        scale = tuple([np.max(scale), np.max(scale)])
    else:
        scale = tuple(scale / 2)
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
            fig = graph.draw_rectangle(p1, p2, line_width=5, **kwargs)
        elif shape == 'oval':
            fig = graph.draw_oval(p1, p2, line_width=5, **kwargs)
    elif shape == 'circle':
        fig = graph.draw_circle(p1, np.max(dpp), line_width=5, **kwargs)
    else:
        fig = None
    return fig


def add_agent_layout(name0, color, collapsibles):
    name = name0.upper()

    collapsibles[f'{name}_DISTRO'] = CollapsibleDict(f'{name}_DISTRO', False,
                                                     dict=dtypes.get_dict('distro', class_name=name0),
                                                     type_dict=dtypes.get_dict_dtypes('distro', class_name=name0),
                                                     toggle=False, disabled=True, disp_name='distribution')

    collapsibles[f'{name}_ODOR'] = CollapsibleDict(f'{name}_ODOR', False, dict=dtypes.get_dict('odor'),
                                                   type_dict=dtypes.get_dict_dtypes('odor'),
                                                   toggle=False, disp_name='odor')


    l = [[sg.R(f'Add {name0}', 1, k=name, enable_events=True)],
         [sg.T('', **t2_kws),
          sg.R('single id', 2, disabled=True, k=f'{name}_single', enable_events=True, **t5_kws),
          sg.In(f'{name}_0', k=f'{name}_id')],
         [sg.T('', **t2_kws), sg.R('group id', 2, disabled=True, k=f'{name}_group', enable_events=True, **t5_kws),
          sg.In(k=f'{name}_group_id')],
         color_pick_layout(name, color),
         [sg.T('', **t5_kws), *collapsibles[f'{name}_DISTRO'].get_layout()],
         [sg.T('', **t5_kws), *collapsibles[f'{name}_ODOR'].get_layout()]]
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


def reset_arena(w, graph, arena_pars, env_db):
    db = copy.deepcopy(env_db)

    s, arena = draw_arena(graph, arena_pars)
    c = {'graph': graph, 'sigma': s}
    for id, pars in db['s_u']['items'].items():
        temp = draw_source(P0=unscale_xy(pars['pos'], s), **c, **pars)
        db['s_u']['figs'][temp] = id
    for id, pars in db['s_g']['items'].items():
        figs = inspect_distro(**c, id=id, item='SOURCE', **pars)
        for f in figs:
            db['s_g']['figs'][f] = id
    for id, pars in db['l_g']['items'].items():
        figs = inspect_distro(**c, id=id, item='LARVA', **pars)
        for f in figs:
            db['l_g']['figs'][f] = id
    for id, pars in db['b']['items'].items():
        points = [scale_xy(p, s) for p in pars['points']]
        temp = graph.draw_lines(points=points, color=pars['default_color'],
                               width=int(pars['width'] * s))
        db['b']['figs'][temp] = id
    w['out'].update(value='Arena has been reset.')
    return s, arena, db


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
    if type(prior_rect) == list:
        for pr in prior_rect:
            graph.delete_figure(pr)
    else:
        graph.delete_figure(prior_rect)


def inspect_distro(default_color, mode, shape, N, loc, scale, graph, s, id=None, item='LARVA', **kwargs):
    # Ps = fun.generate_xy_distro(mode, shape, N, loc=loc, scale=scale)
    Ps = fun.generate_xy_distro(mode, shape, N, loc=unscale_xy(loc, s), scale=np.array(scale) * s)
    group_figs = []
    for i, P0 in enumerate(Ps):
        if item == 'SOURCE':
            temp = draw_source(P0, default_color, graph, s, **kwargs)
        elif item == 'LARVA':
            temp = draw_larva(P0, default_color, graph, s, **kwargs)
        group_figs.append(temp)
    return group_figs


def draw_source(P0, default_color, graph, s, amount, radius, **kwargs):
    # P0=scale_xy(pos,sigma)
    fill_color = default_color if amount > 0 else None
    temp = graph.draw_circle(P0, radius * s, line_width=3, line_color=default_color, fill_color=fill_color)
    return temp


def draw_larva(P0, color, graph, s, orientation_range, **kwargs):
    points = np.array([[0.9, 0.1], [0.05, 0.1]])
    xy0 = fun.body(points) - np.array([0.5, 0.0])
    xy0 = fun.rotate_multiple_points(xy0, random.uniform(*np.deg2rad(orientation_range)), origin=[0, 0])
    xy0 =xy0*s/ 250+np.array(P0)
    temp = graph.draw_polygon(xy0, line_width=3, line_color=color, fill_color=color)
    return temp

def check_abort(name, w, v, units, groups):
    o = name
    info = w['info']
    abort = True
    odor_on = w[f'TOGGLE_{o}_ODOR'].get_state()

    if not odor_on:
        w[f'{o}_ODOR_odor_id'].update(value=None)
        w[f'{o}_ODOR_odor_intensity'].update(value=0.0)

    if o == 'SOURCE':
        food_on = w[f'TOGGLE_{o}_FOOD'].get_state()
        if not odor_on and not food_on:
            info.update(value=f"Assign food and/or odor to the drawn source")
            return True
        elif food_on and float(v[f'{o}_FOOD_amount']) == 0.0:
            w[f'{o}_FOOD_amount'].update(value=10 ** -3)
            info.update(value=f"Source food amount set to default")
            return True
        elif not food_on and float(v[f'{o}_FOOD_amount']) != 0.0:
            w[f'{o}_FOOD_amount'].update(value=0.0)
            info.update(value=f"Source food amount set to 0")

    if v[f'{o}_group_id'] == '' and v[f'{o}_id'] == '':
        info.update(value=f"Both {o.lower()} single id and group id are empty")
    elif not v[f'{o}_group'] and not v[f'{o}_single']:
        info.update(value=f"Select to add a single or a group of {o.lower()}sigma")
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
    return abort


def draw_env(env=None):
    sg.theme('LightGreen')
    collapsibles = {}
    if env is None:
        env = {'border_list': {},
               'arena': dtypes.get_dict('arena'),
               'food_params': {'source_units': {}, 'source_groups': {}, 'food_grid': None},
               'larva_groups': {}
               }
    arena_pars = env['arena']
    items = [env['border_list'],
             env['food_params']['source_units'], env['food_params']['source_groups'],
             {}, env['larva_groups']]
    env_db = {k: {'items': ii, 'figs': {}} for k, ii in zip(['b', 's_u', 's_g', 'l_u', 'l_g'], items)}
    sample_fig, sample_pars = None, {}

    collapsibles['arena'] = CollapsibleDict('arena', True,
                                            dict=arena_pars, type_dict=dtypes.get_dict_dtypes('arena'),
                                            next_to_header=[
                                                graphic_button('burn', 'RESET_ARENA', tooltip='Reset to the initial arena. All drawn items will be erased.'),
                                                graphic_button('globe_active', 'NEW_ARENA', tooltip='Create a new arena.All drawn items will be erased.'),
                                            ])

    source_l, collapsibles = add_agent_layout('Source', 'green', collapsibles)
    larva_l, collapsibles = add_agent_layout('Larva', 'black', collapsibles)

    collapsibles['SOURCE_FOOD'] = CollapsibleDict('SOURCE_FOOD', False, dict=dtypes.get_dict('food'),
                                                  type_dict=dtypes.get_dict_dtypes('food'),
                                                  toggle=False, disp_name='food')

    s = None
    arena = None

    col2 = [
        *larva_l, *source_l,

        [sg.T('', **t5_kws), *collapsibles['SOURCE_FOOD'].get_layout()],
        [sg.T('', **t5_kws), sg.T('shape', **t5_kws),
         sg.Combo(['rect', 'circle'], default_value='circle', k='SOURCE_shape', enable_events=True, readonly=True)],

        [sg.R('Add Border', 1, k='BORDER', enable_events=True)],
        [sg.T('', **t5_kws), sg.T('id', **t5_kws),
         sg.In('', k='BORDER_id')],
        [sg.T('', **t5_kws), sg.T('width', **t5_kws), sg.In(0.001, k='BORDER_width')],
        color_pick_layout('BORDER', 'black'),

        [sg.R('Erase item', 1, k='-ERASE-', enable_events=True)],
        [sg.R('Move item', 1, True, k='-MOVE-', enable_events=True)],
        [sg.R('Inspect item', 1, True, k='-INSPECT-', enable_events=True)],
    ]

    col1 = [
        collapsibles['arena'].get_layout(),
        [sg.Graph(
            canvas_size=(W, H),
            graph_bottom_left=(0, 0),
            graph_top_right=(W, H),
            key="-GRAPH-",
            change_submits=True,  # mouse click events
            background_color='black',
            drag_submits=True)],
        [sg.T('Hints : '), sg.T('', k='info', **t40_kws)],
        [sg.T('Actions : '), sg.T('', k='out', **t40_kws)],
        [sg.B('Ok', **b_kws), sg.B('Cancel', **b_kws)]
    ]
    layout = [[sg.Col(col1), sg.Col(col2)]]

    w = sg.Window("Environment configuration", layout, **w_kws,location=(600, 200))
    graph = w["-GRAPH-"]  # mode: sg.Graph
    graph.bind('<Button-3>', '+RIGHT+')

    dragging, current = False, {}
    start_point = end_point = prior_rect = None
    w.write_event_value('RESET_ARENA', 'Draw the initial arena')

    while True:
        e, v = w.read()
        # print(e)
        info = w["info"]
        if e in [None, 'Cancel']:
            break
        elif e == 'Ok':
            env['arena'] = collapsibles['arena'].get_dict(v, w)
            env['border_list'] = db['b']['items']
            env['food_params']['source_units'] = db['s_u']['items']
            env['food_params']['source_groups'] = db['s_g']['items']
            env['larva_groups'] = db['l_g']['items']
            break  # exit
        # check_collapsibles(w, e, collapsibles)
        # check_toggles(w, e)
        check_togglesNcollapsibles(w, e, v,collapsibles)
        if e == 'RESET_ARENA':
            s, arena, db = reset_arena(w, graph, arena_pars, env_db)
        elif e == 'NEW_ARENA':
            w['out'].update(value='New arena initialized. All items erased.')
            s, arena = draw_arena(graph, collapsibles['arena'].get_dict(v, w))
            db = {k: {'items': {}, 'figs': {}} for k in ['b', 's_u', 's_g', 'l_u', 'l_g']}

        if arena is None:
            continue
        if e == '-MOVE-':
            graph.Widget.config(cursor='fleur')
            # graph.set_cursor(cursor='fleur')          # not yet released method... coming soon!
        elif not e.startswith('-GRAPH-'):
            # graph.set_cursor(cursor='left_ptr')       # not yet released method... coming soon!
            graph.Widget.config(cursor='left_ptr')
        # if e.startswith('PICK'):
        #     target = e.split()[-1]
        #     choice = popup_color_chooser('Dark Blue 3')
        #     w[target].update(choice)
        if e == "-GRAPH-":  # if there'sigma a "Graph" event, then it'sigma a mouse
            x, y = v["-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
                drag_figures = [f for f in graph.get_figures_at_location((x, y)) if f != arena]
                lastxy = x, y
            else:
                end_point = (x, y)
            if prior_rect:
                delete_prior(prior_rect, graph)
            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = x, y
            if None not in (start_point, end_point):
                if v['-MOVE-']:
                    # delta_X, delta_Y = scale_xy((delta_x, delta_y),sigma)
                    delta_X, delta_Y = delta_x / s, delta_y / s
                    for fig in drag_figures:
                        for k in list(db.keys()):
                            if fig in list(db[k]['figs'].keys()):
                                id = db[k]['figs'][fig]
                                w['out'].update(value=f"Item {id} moved by ({delta_X}, {delta_Y})")
                                figs = [k for k, v in db[k]['figs'].items() if v == id]
                                for f in figs:
                                    if k == 's_u':
                                        X0, Y0 = db[k]['items'][id]['pos']
                                        db[k]['items'][id]['pos'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif k in ['s_g', 'l_g']:
                                        X0, Y0 = db[k]['items'][id]['loc']
                                        db[k]['items'][id]['loc'] = (X0 + delta_X, Y0 + delta_Y)
                                    elif k == 'b':
                                        db[k]['items'][id]['points'] = [(X0 + delta_X, Y0 + delta_Y) for X0, Y0 in
                                                                        db[k]['items'][id]['points']]
                                    graph.move_figure(f, delta_x, delta_y)
                        graph.update()
                elif v['-ERASE-']:
                    for fig in drag_figures:
                        for k in list(db.keys()):
                            if fig in list(db[k]['figs'].keys()):
                                id = db[k]['figs'][fig]
                                w['out'].update(value=f"Item {id} erased")
                                figs = [k for k, v in db[k]['figs'].items() if v == id]
                                for f in figs:
                                    graph.delete_figure(f)
                                    db[k]['figs'].pop(f)
                                db[k]['items'].pop(id)
                elif v['-INSPECT-']:
                    for fig in drag_figures:
                        for k in list(db.keys()):
                            if fig in list(db[k]['figs'].keys()):
                                id = db[k]['figs'][fig]
                                w['out'].update(value=f"Inspecting item {id} ")
                                if k in ['s_g', 'l_g']:
                                    figs = inspect_distro(id=id, s=s, graph=graph, **db[k]['items'][id])
                                    for f in figs:
                                        db[k]['figs'][f] = id
                elif v['SOURCE'] or v['BORDER'] or v['LARVA']:
                    P1, P2 = scale_xy(start_point, s), scale_xy(end_point, s)
                    if any([out_of_bounds(P, collapsibles['arena'].get_dict(v, w)) for P in [P1, P2]]):
                        current = {}
                    else:
                        if v['SOURCE'] and not check_abort('SOURCE', w, v, db['s_u']['items'], db['s_g']['items']):
                            o = 'SOURCE'
                            color = v[f'{o}_color']
                            if v['SOURCE_single'] or (v['SOURCE_group'] and sample_fig is None):
                                fill_color = color if w['TOGGLE_SOURCE_FOOD'].get_state() else None
                                prior_rect = draw_shape(graph, shape=v[f'{o}_shape'], p1=start_point, p2=end_point,
                                                        line_color=color, fill_color=fill_color)
                                temp = np.max(np.abs(np.array(end_point) - np.array(start_point)))
                                w['SOURCE_FOOD_radius'].update(value=temp / s)


                                sample_pars = {'default_color': color,
                                               **collapsibles['SOURCE_FOOD'].get_dict(v, w, check_toggle=False),
                                               **collapsibles['SOURCE_ODOR'].get_dict(v, w, check_toggle=False),
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
                                current = {v['SOURCE_group_id']: {
                                    **collapsibles['SOURCE_DISTRO'].get_dict(v, w, check_toggle=False),
                                    **sample_pars
                                }}
                                prior_rect = draw_shape(graph, shape=v[f'{o}_DISTRO_shape'], p1=start_point,
                                                        p2=end_point, line_color=color)
                        elif v['LARVA'] and not check_abort('LARVA', w, v, db['l_u']['items'], db['l_g']['items']):
                            o = 'LARVA'
                            color = v[f'{o}_color']
                            sample_larva_pars = {'default_color': color,
                                                 **collapsibles[f'{o}_ODOR'].get_dict(v, w, check_toggle=False),
                                                 }
                            if v[f'{o}_group']:
                                update_window_distro(v, w, o, start_point, end_point, s)
                                current = {v[f'{o}_group_id']: {
                                    **collapsibles[f'{o}_DISTRO'].get_dict(v, w, check_toggle=False),
                                    **sample_larva_pars
                                }}
                                prior_rect = draw_shape(graph, shape=v[f'{o}_DISTRO_shape'], p1=start_point,
                                                        p2=end_point, line_color=color)

                        elif v['BORDER']:
                            id = v['BORDER_id']
                            if id in list(db['b']['items'].keys()) or id == '':
                                info.update(value=f"Border id {id} already exists or is empty")
                            else:
                                dic = {'unique_id': id,
                                       'default_color': v['BORDER_color'],
                                       'width': v['BORDER_width'],
                                       'points': [P1, P2]}
                                current = fun.agent_list2dict(
                                    [retrieve_dict(dic, dtypes.get_dict_dtypes('agent', class_name='Border'))])

                                prior_rect = graph.draw_line(start_point, end_point, color=v['BORDER_color'],
                                                             width=int(float(v['BORDER_width']) * s))



        elif e.endswith('+UP'):  # The drawing has ended because mouse up
            if v['BORDER'] and current != {}:
                o = 'BORDER'
                units = db['b']
                id = v[f'{o}_id']
                w['out'].update(value=f"Border {id} placed from {P1} to {P2}")
                units['figs'][prior_rect] = id
                units['items'].update(current)
                w[f'{o}_id'].update(value=f"BORDER_{len(units['items'].keys())}")
            elif v['SOURCE']:
                o = 'SOURCE'
                units, groups = db['s_u'], db['s_g']
                if v[f'{o}_single'] and current != {}:
                    id=v[f'{o}_id']
                    w['out'].update(value=f"Source {id} placed at {P1}")
                    units['figs'][prior_rect] = id
                    units['items'].update(current)
                    w[f'{o}_id'].update(value=f"SOURCE_{len(units['items'].keys())}")
                    w[f'{o}_ODOR_odor_id'].update(value='')
                elif v[f'{o}_group'] and sample_pars != {}:
                    id = v[f'{o}_group_id']
                    if current == {}:
                        info.update(value=f"Sample item for source group {id} detected." \
                                          "Now draw the distribution'sigma space")

                        sample_fig = prior_rect
                    else:
                        w['out'].update(value=f"Source group {id} placed at {P1}")
                        groups['items'].update(current)
                        w[f'{o}_group_id'].update(value=f"SOURCE_GROUP_{len(groups['items'].keys())}")
                        w[f'{o}_ODOR_odor_id'].update(value='')
                        figs = inspect_distro(id=id, **groups['items'][id], graph=graph, s=s, item=o)
                        for f in figs:
                            groups['figs'][f] = id
                        delete_prior(prior_rect, graph)
                        delete_prior(sample_fig, graph)
                        sample_fig, sample_pars = None, {}
            elif v['LARVA'] and current != {}:
                o = 'LARVA'
                units, groups = db['l_u'], db['l_g']
                if v[f'{o}_single']:
                    pass
                elif v[f'{o}_group']:
                    id = v[f'{o}_group_id']
                    w['out'].update(value=f"{o} group {id} placed at {P1}")
                    groups['items'].update(current)
                    w[f'{o}_group_id'].update(value=f"{o}_GROUP_{len(groups['items'].keys())}")
                    w[f'{o}_ODOR_odor_id'].update(value='')
                    figs = inspect_distro(id=id, **groups['items'][id], graph=graph, s=s, item=o)
                    for f in figs:
                        groups['figs'][f] = id
                    delete_prior(prior_rect, graph)
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
    w.close()
    return env


if __name__ == '__main__':
    draw_env()
