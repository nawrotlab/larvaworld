import pygame
import numpy as np


def evaluate_input(m, screen):

    if m.pygame_keys is None :
        from lib.registry.controls import load_controls
        m.pygame_keys = load_controls()['pygame_keys']

    d_zoom = 0.01
    ev = pygame.event.get()
    for e in ev:
        if e.type == pygame.QUIT:
            screen.close()
        if e.type == pygame.KEYDOWN:
            for k, v in m.pygame_keys.items():
                if e.key == getattr(pygame, v):
                    eval_keypress(k, screen, m)

        if m.allow_clicks:
            if e.type == pygame.MOUSEBUTTONDOWN:
                m.mousebuttondown_pos = screen.mouse_position
            elif e.type == pygame.MOUSEBUTTONUP:
                p = screen.mouse_position
                if e.button == 1:
                    if not eval_selection(m, p, ctrl=pygame.key.get_mods() & pygame.KMOD_CTRL):
                        m.add_agent(agent_class=m.selected_type, p0=tuple(p),
                                        p1=tuple(m.mousebuttondown_pos))

                elif e.button == 3:
                    import lib.gui.aux.windows
                    loc = tuple(np.array(screen.w_loc) + np.array(pygame.mouse.get_pos()))
                    if len(m.selected_agents) > 0:
                        for sel in m.selected_agents:
                            sel = lib.gui.aux.windows.set_agent_kwargs(sel, location=loc)
                    else:
                        m.selected_type = lib.gui.aux.windows.object_menu(m.selected_type, location=loc)
                elif e.button in [4, 5]:
                    m.apply_screen_zoom(screen, d_zoom=-d_zoom if e.button == 4 else d_zoom)
                    m.toggle(name='zoom', value=screen.zoom)
    if m.focus_mode and len(m.selected_agents) > 0:
        try:
            sel = m.selected_agents[0]
            screen.move_center(pos=sel.get_position())
        except:
            pass


def eval_keypress(k, screen, model):
    from lib.model.agents._larva_sim import LarvaSim
    from lib.model.agents._larva import Larva
    # print(k)
    if k == '▲ trail duration':
        model.toggle('trajectory_dt', plus=True, disp='trail duration')
    elif k == '▼ trail duration':
        model.toggle('trajectory_dt', minus=True, disp='trail duration')
    elif k == 'visible trail':
        model.toggle('trails')
    elif k == 'pause':
        model.toggle('is_paused')
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
    elif k == 'windscape' :
        try :
            model.windscape.visible = not model.windscape.visible
            model.toggle('windscape', 'ON' if model.windscape.visible else 'OFF')
        except :
            pass
    elif k == 'delete item':
        from lib.gui.aux.windows import delete_objects_window
        if delete_objects_window(model.selected_agents):
            for f in model.selected_agents:
                model.selected_agents.remove(f)
                model.delete_agent(f)
    elif k == 'dynamic graph':
        if len(model.selected_agents) > 0:
            sel = model.selected_agents[0]
            if isinstance(sel, Larva):
                from lib.gui.aux.elements import DynamicGraph
                model.dynamic_graphs.append(DynamicGraph(agent=sel))
    elif k == 'odor gains':
        if len(model.selected_agents) > 0:
            sel = model.selected_agents[0]
            if isinstance(sel, LarvaSim) and sel.brain.olfactor is not None:
                from lib.gui.aux.windows import set_kwargs
                sel.brain.olfactor.gain = set_kwargs(sel.brain.olfactor.gain, title='Odor gains')
    else:
        model.toggle(k)


def evaluate_graphs(m):
    for g in m.dynamic_graphs:
        running = g.evaluate()
        if not running:
            m.dynamic_graphs.remove(g)
            del g


def eval_selection(m, p, ctrl):
    res = False if len(m.selected_agents) == 0 else True
    for f in m.get_all_objects():
        if f.contained(p):
            if not f.selected:
                f.selected = True
                m.selected_agents.append(f)
            elif ctrl:
                f.selected = False
                m.selected_agents.remove(f)
            res = True
        elif f.selected and not ctrl:
            f.selected = False
            m.selected_agents.remove(f)
    return res
