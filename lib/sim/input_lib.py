import pygame
import numpy as np
import lib.gui.gui_lib as gui
from lib.model import Larva, LarvaSim, Border

shortcuts = {
    # 'trajectory_dt' : ['MINUS', 'PLUS'],
    'trajectories': 'p',
    'focus_mode': 'f',
    'draw_centroid': 'e',
    'draw_head': 'h',
    'draw_midline': 'm',
    'draw_contour': 'c',
    'visible_clock': 't',
    'visible_ids': 'TAB',
    'visible_state': 's',
    'color_behavior': 'b',
    'random_colors': 'r',
    'black_background': 'g',
    'larva_collisions': 'y',
    # 'zoom' : ,
    'snapshot #': 'i',
    # 'odorscape #' : 'o'
}


def evaluate_input(model, screen):
    d_zoom = 0.01
    ev = pygame.event.get()
    for event in ev:
        if event.type == pygame.QUIT:
            screen.close_requested()
        if event.type == pygame.KEYDOWN:
            for k,v in shortcuts.items() :
                if event.key==getattr(pygame, f'K_{v}'):
                    toggle(model, k)

            if event.key == pygame.K_MINUS:
                toggle(model, 'trajectory_dt', minus=True)
            elif event.key == pygame.K_PLUS:
                toggle(model, 'trajectory_dt', plus=True)
            elif event.key == pygame.K_o:
                toggle(model, 'odorscape #', show=pygame.key.get_mods() & pygame.KMOD_CTRL)

            elif event.key == pygame.K_LEFT:
                screen.move_center(-0.05, 0)
            elif event.key == pygame.K_RIGHT:
                screen.move_center(+0.05, 0)
            elif event.key == pygame.K_UP:
                screen.move_center(0, +0.05)
            elif event.key == pygame.K_DOWN:
                screen.move_center(0, -0.05)
            elif event.key == pygame.K_DELETE:
                if gui.delete_objects_window(model.selected_agents):
                    for f in model.selected_agents:
                        model.selected_agents.remove(f)
                        model.delete_agent(f)
            elif event.key == pygame.K_q:
                if len(model.selected_agents) > 0:
                    sel = model.selected_agents[0]
                    if isinstance(sel, Larva):
                        model.dynamic_graphs.append(gui.DynamicGraph(agent=sel, available_pars=model.available_pars))
            elif event.key == pygame.K_w:
                if len(model.selected_agents) > 0:
                    sel = model.selected_agents[0]
                    if isinstance(sel, LarvaSim):
                        if sel.brain.olfactor is not None:
                            odor_gains = sel.brain.olfactor.gain
                            odor_gains = gui.set_kwargs(odor_gains, title='Odor gains')
                            sel.brain.olfactor.gain = odor_gains
            else:
                for i in range(model.Nodors):
                    if event.key == getattr(pygame, f'K_{i}'):
                        layer_id = list(model.odor_layers.keys())[i]
                        layer = model.odor_layers[layer_id]
                        layer.visible = not layer.visible
                        toggle(model, layer_id, 'ON' if layer.visible else 'OFF')

        if model.allow_clicks:
            if event.type == pygame.MOUSEBUTTONDOWN:
                model.mousebuttondown_pos = screen.get_mouse_position()
                # model.mousebuttondown_time = time.time()
            elif event.type == pygame.MOUSEBUTTONUP:
                # model.mousebuttonup_time = time.time()
                # dt = model.mousebuttonup_time - model.mousebuttondown_time
                p = screen.get_mouse_position()
                if event.button == 1:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        ctrl = True
                    else:
                        ctrl = False
                    eval_selection(model, p, ctrl)
                    # model.mousebuttondown_time = time.time()
                    if len(model.selected_agents) == 0:
                        # if len(model.selected_agents) == 0 and isinstance(model, LarvaWorldSim):
                        try:
                            p = tuple(p)
                            if model.selected_type == 'Food':
                                f = model.add_food(p)
                            elif model.selected_type == 'Larva':
                                f = model.add_larva(p)
                            elif model.selected_type == 'Border':
                                b = Border(model=model, points=[tuple(model.mousebuttondown_pos), p],
                                           from_screen=True)
                                model.add_border(b)
                        except:
                            pass
                elif event.button == 3:
                    if len(model.selected_agents) > 0:
                        sel = model.selected_agents[0]
                        sel = gui.set_agent_kwargs(sel)
                    else:
                        model.selected_type = gui.object_menu(model.selected_type)
                elif event.button == 4:
                    screen.zoom_screen(d_zoom=-d_zoom)
                    toggle(model, name='zoom', value=screen.zoom)
                elif event.button == 5:
                    screen.zoom_screen(d_zoom=+d_zoom)
                    toggle(model, name='zoom', value=screen.zoom)
            model.input_box.get_input(event)
    if model.focus_mode and len(model.selected_agents) > 0:
        try:
            sel = model.selected_agents[0]
            screen.move_center(pos=sel.get_position())
        except:
            pass


def evaluate_graphs(model):
    for g in model.dynamic_graphs:
        running = g.evaluate()
        if not running:
            model.dynamic_graphs.remove(g)
            del g


def eval_selection(model, p, ctrl):
    for f in model.get_food() + model.get_flies() + model.borders:
        if f.contained(p):
            if not f.selected:
                f.selected = True
                model.selected_agents.append(f)
        else:
            if f.selected and not ctrl:
                f.selected = False
                model.selected_agents.remove(f)


def toggle(model, name, value=None, show=False, minus=False, plus=False):
    if name == 'visible_ids':
        for a in model.get_flies() + model.get_food():
            a.id_box.visible = not a.id_box.visible
    elif name == 'random_colors':
        for f in model.get_flies():
            f.set_default_color(model.generate_larva_color())
    elif name == 'black_background':
        model.update_default_colors()
    elif name == 'larva_collisions':
        model.eliminate_overlap()
    elif name == 'snapshot #':
        import imageio
        record_image_to = f'{model.media_name}_{model.snapshot_counter}.png'
        model._screen._image_writer = imageio.get_writer(record_image_to, mode='i')
        value = model.snapshot_counter
        model.snapshot_counter += 1
    elif name == 'odorscape #':
        model.plot_odorscape(save_to=model.save_to, show=show)
        value = model.odorscape_counter
        model.odorscape_counter += 1
    elif name == 'trajectory_dt':
        if minus:
            dt = -1
        elif plus:
            dt = +1
        model.trajectory_dt = np.clip(model.trajectory_dt + 5 * dt, a_min=0, a_max=np.inf)
        value = model.trajectory_dt
    # elif name=='black_background' :
    # elif name=='black_background' :

    if value is None:
        setattr(model, name, not getattr(model, name))
        value = 'ON' if getattr(model, name) else 'OFF'
    model.screen_texts[name].text = f'{name} {value}'
    model.screen_texts[name].end_time = pygame.time.get_ticks() + 3000
