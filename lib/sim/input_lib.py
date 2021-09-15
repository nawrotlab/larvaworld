import os

import pygame
import numpy as np
import lib.gui.gui_lib as gui
import lib.gui.windows
from lib.conf.conf import loadConfDict
from lib.model.agents._larva import Larva, LarvaSim

# shortcuts = {
#     # 'trajectory_dt' : ['MINUS', 'PLUS'],
#     'trajectories': 'p',
#     'focus_mode': 'f',
#     'draw_centroid': 'e',
#     'draw_head': 'h',
#     'draw_midline': 'm',
#     'draw_contour': 'c',
#     'visible_clock': 't',
#     'visible_ids': 'TAB',
#     'visible_state': 'sigma',
#     'color_behavior': 'b',
#     'random_colors': 'r',
#     'black_background': 'g',
#     'larva_collisions': 'y',
#     # 'zoom' : ,
#     'snapshot #': 'i',
#     'sim_paused': 'SPACE',
#     # 'odorscape #' : 'o'
# }

# shortcuts = gui.load_shortcuts()
shortcuts = loadConfDict('Settings')


def evaluate_input(model, screen):
    d_zoom = 0.01
    ev = pygame.event.get()
    for event in ev:
        if event.type == pygame.QUIT:
            screen.close_requested()
        if event.type == pygame.KEYDOWN:
            for k, v in shortcuts['pygame_keys'].items():
                if event.key == getattr(pygame, v):
                    eval_keypress(k, screen, model)

        if model.allow_clicks:
            if event.type == pygame.MOUSEBUTTONDOWN:
                model.mousebuttondown_pos = screen.get_mouse_position()
                # model.mousebuttondown_time = time.time()
            elif event.type == pygame.MOUSEBUTTONUP:
                # model.mousebuttonup_time = time.time()
                # dt = model.mousebuttonup_time - model.mousebuttondown_time
                p = screen.get_mouse_position()
                if event.button == 1:
                    if not eval_selection(model, p, ctrl=pygame.key.get_mods() & pygame.KMOD_CTRL):
                        model.add_agent(agent_class=model.selected_type, p0=tuple(p),
                                        p1=tuple(model.mousebuttondown_pos))

                elif event.button == 3:
                    loc = tuple(np.array(screen.w_loc) + np.array(pygame.mouse.get_pos()))
                    if len(model.selected_agents) > 0:
                        for sel in model.selected_agents:
                            # sel = model.selected_agents[0]
                            sel = lib.gui.windows.set_agent_kwargs(sel, location=loc)
                    else:
                        model.selected_type = lib.gui.windows.object_menu(model.selected_type, location=loc)
                elif event.button in [4, 5]:
                    screen.zoom_screen(d_zoom=-d_zoom if event.button == 4 else d_zoom)
                    model.toggle(name='zoom', value=screen.zoom)
            model.input_box.get_input(event)
    if model.focus_mode and len(model.selected_agents) > 0:
        try:
            sel = model.selected_agents[0]
            screen.move_center(pos=sel.get_position())
        except:
            pass


def eval_keypress(k, screen, model):
    if k == '▲ trail duration':
        model.toggle('trajectory_dt', plus=True, disp='trail duration')
    elif k == '▼ trail duration':
        model.toggle('trajectory_dt', minus=True, disp='trail duration')
    elif k == 'visible trail':
        model.toggle('trails')
    elif k == 'pause':
        model.toggle('sim_paused')
    elif k == 'move left':
        screen.move_center(-0.05, 0)
    elif k == 'move right':
        screen.move_center(+0.05, 0)
    elif k == 'move up':
        screen.move_center(0, +0.05)
    elif k == 'move down':
        screen.move_center(0, -0.05)
    elif k == 'plot odorscapes':
        model.toggle('odorscape #', show=pygame.key.get_mods() & pygame.KMOD_CTRL)
    elif 'odorscape' in k:
        idx = int(k.split(' ')[-1])
        try :
            layer_id = list(model.odor_layers.keys())[idx]
            layer = model.odor_layers[layer_id]
            layer.visible = not layer.visible
            model.toggle(layer_id, 'ON' if layer.visible else 'OFF')
        except :
            pass
    elif k == 'snapshot':
        model.toggle('snapshot #')
    elif k == 'delete item':
        if lib.gui.windows.delete_objects_window(model.selected_agents):
            for f in model.selected_agents:
                model.selected_agents.remove(f)
                model.delete_agent(f)
    elif k == 'dynamic graph':
        if len(model.selected_agents) > 0:
            sel = model.selected_agents[0]
            if isinstance(sel, Larva):
                model.dynamic_graphs.append(gui.DynamicGraph(agent=sel))
    elif k == 'odor gains':
        if len(model.selected_agents) > 0:
            sel = model.selected_agents[0]
            if isinstance(sel, LarvaSim):
                if sel.brain.olfactor is not None:
                    odor_gains = sel.brain.olfactor.gain
                    odor_gains = lib.gui.windows.set_kwargs(odor_gains, title='Odor gains')
                    sel.brain.olfactor.gain = odor_gains
    else:
        model.toggle(k)


def evaluate_graphs(model):
    for g in model.dynamic_graphs:
        running = g.evaluate()
        if not running:
            model.dynamic_graphs.remove(g)
            del g


def eval_selection(model, p, ctrl):
    res = False if len(model.selected_agents) == 0 else True
    for f in model.get_food() + model.get_flies() + model.borders:
        if f.contained(p):
            if not f.selected:
                f.selected = True
                model.selected_agents.append(f)
            elif ctrl:
                f.selected = False
                model.selected_agents.remove(f)
            res = True
        elif f.selected and not ctrl:
            f.selected = False
            model.selected_agents.remove(f)
    return res
