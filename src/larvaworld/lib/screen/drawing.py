import os
import sys
import numpy as np
import param
from param import Boolean, String

from larvaworld.lib.param import NestedConf, PositiveNumber, OptionalSelector, PositiveInteger, Area2DPixel

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from larvaworld.lib import reg, aux, screen
from larvaworld.lib.screen import Viewer, SidePanel, ScreenMsgText, SimulationClock, SimulationScale, \
    SimulationState, ScreenTextBoxRect


class MediaDrawOps(NestedConf):
    image_mode = OptionalSelector(objects=['final', 'snapshots', 'overlap'],doc='When to save images.')
    image_file = String(None,doc='Filename for the saved image. File extension png sutomatically added.')
    snapshot_interval_in_sec= PositiveInteger(60, softmax=100,doc='Sec between snapshots')
    video_file = String(None,doc='Filename for the saved video. File extension mp4 sutomatically added.')
    fps = PositiveInteger(60, softmax=100,doc='Video speed')
    save_video= Boolean(False, doc='Whether to save a video.')
    mode=OptionalSelector(objects=['video', 'image'],doc='Screen mode.')
    show_display = Boolean(False, doc='Whether to launch the pygame-visualization.')

class AgentDrawOps(NestedConf):
    visible_trails = Boolean(True,doc='Draw the larva trajectories')
    trail_dt = PositiveNumber(20,step=0.2,doc='Duration of the drawn trajectories')
    trail_color = param.Selector(objects=['normal', 'linear', 'angular'], doc='Whether to display larva tracks according to the instantaneous forward or angular velocity.')
    draw_sensors = Boolean(False,doc='Draw the larva sensors')
    draw_contour = Boolean(True,doc='Draw the larva contour')
    draw_segs = Boolean(True,doc='Draw the larva body segments')
    draw_midline = Boolean(True,doc='Draw the larva midline')
    draw_centroid = Boolean(False,doc='Draw the larva centroid')
    draw_head = Boolean(False,doc='Draw the larva head')
    draw_orientations = Boolean(False,doc='Draw the larva body vector orientations')


class ColorDrawOps(NestedConf):
    intro_text = Boolean(True, doc='Show the introductory configuration screen')
    odor_aura = Boolean(False, doc='Draw the aura around odor sources')
    allow_clicks = Boolean(True, doc='Whether to allow input from display')
    black_background = Boolean(False, doc='Set the background color to black')
    random_colors = Boolean(False,doc='Color each larva with a random color')
    color_behavior = Boolean(False,doc='Color the larvae according to their instantaneous behavior')
    panel_width = PositiveInteger(0, doc='The width of the side panel in pixels')

# class VisOps(NestedConf):
    # visible_clock = Boolean(True, doc='Whether clock is visible')
    # visible_scale = Boolean(True, doc='Whether scale is visible')
    # visible_state = Boolean(False, doc='Whether state is visible')
    # visible_ids = Boolean(False, doc='Whether the agent IDs are visible')

class ScreenOps(ColorDrawOps, AgentDrawOps, MediaDrawOps):pass

class BaseScreenManager(Area2DPixel,ScreenOps) :

    def __init__(self, model, background_motion=None,vis_kwargs=None, video=None, **kwargs):
        m = self.model = model
        super().__init__(dims=aux.get_window_dims(m.space.dims), **kwargs)
        if self.model.offline:
            self.show_display = False
        if self.video_file is None:
            self.video_file = str(m.id)
        if self.image_file is None:
            self.image_file = str(m.id)
        self._fps= int(self.fps / m.dt)
        if vis_kwargs is not None:
            self.mode=vis_kwargs.render.mode
            # self.__dict__.update(vis_kwargs.aux)
            if self.mode=='video' and not self.save_video:
                self.show_display=True


        self.bg = background_motion

        self.active = self.save_video or self.image_mode or self.show_display or (self.mode is not None)
        self.v = None

        self.selected_type = ''
        self.selected_agents = []
        self.selection_color = 'red'

        self.dynamic_graphs = []
        self.focus_mode = False

        self.snapshot_interval = int(self.snapshot_interval_in_sec / m.dt)

        self.snapshot_counter = 0
        self.odorscape_counter = 0

        self.pygame_keys = None
        self.screen_kws = aux.AttrDict({
            'manager': self,


        })



    def draw_agents(self, v):

        for o in self.model.sources:
            o._draw(v=v)
        for g in self.model.agents:
            g._draw(v=v)




    def check(self,**kwargs):
        if self.v is None:
            self.v = self.initialize(**kwargs)
        elif self.v.close_requested():
            self.close()


    def close(self):
        self.v.close()
        self.v = None
        self.model.running = False
        reg.vprint('Terminated by the user', 3)
        return


    def render(self,**kwargs):

        if self.active:
            self.check(**kwargs)
            if self.image_mode != 'overlap':
                self.draw_arena(self.v)

            self.draw_agents(self.v)
            if self.show_display:
                self.evaluate_input()
                self.evaluate_graphs()
            if self.image_mode != 'overlap':
                self.draw_aux(self.v)
                self.v.render()

    def initialize(self, **kwargs):
        return None

    def evaluate_input(self):
        pass

    def evaluate_graphs(self):
        pass

    def draw_arena(self, v):
        pass

    def draw_aux(self, v,**kwargs):
        pass

    @property
    def screen_color(self):
        return (200, 200, 200) if not self.black_background else (50, 50, 50)

    @property
    def tank_color(self):
        return aux.Color.WHITE if not self.black_background else aux.Color.BLACK

    @property
    def snapshot_tick(self):
        return (self.model.Nticks - 1) % self.snapshot_interval == 0

class GA_ScreenManager(BaseScreenManager):
    def __init__(self, panel_width=600,scene='no_boxes',**kwargs):
        super().__init__(black_background=True,panel_width=panel_width,**kwargs)
        self.screen_kws.caption = f'GA {self.model.experiment} : {self.model.id}'
        self.screen_kws.file_path = f'{reg.ROOT_DIR}/lib/sim/ga_scenes/{scene}.txt'


    def evaluate_input(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.close()
                sys.exit()
            elif e.type == pygame.KEYDOWN and (e.key == pygame.K_PLUS or e.key == 93 or e.key == 270):
                self.v.increase_fps()
            elif e.type == pygame.KEYDOWN and (e.key == pygame.K_MINUS or e.key == 47 or e.key == 269):
                self.v.decrease_fps()




    def initialize(self):
        v = Viewer.load_from_file(**self.screen_kws)
        self.side_panel = SidePanel(v)
        print('Screen opened')
        return v

    def draw_arena(self, v):
        v._window.fill(aux.Color.BLACK)

    def draw_aux(self, v,**kwargs):
        self.side_panel.draw(v)

    def finalize(self):
        if self.v:
            self.v.close()




class ScreenManager(BaseScreenManager):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.screen_kws.caption = str(self.model.id)




    def initialize(self):

        v = Viewer(**self.screen_kws)
        self.build_aux(v)

        self.draw_arena(v)

        print('Screen opened')
        return v





    def build_aux(self,v):
        m=self.model
        self.input_box = ScreenTextBoxRect(text_color='lightgreen', default_color='white',
                                              frame_rect=v.get_rect_at_pos(),
                                         font_type = "comicsansms",font_size = 40,
        )
        if self.intro_text:
            box = ScreenTextBoxRect(
                text=m.configuration_text,
                text_color='lightgreen', default_color='white',
                visible=True, frame_rect=v.get_rect_at_pos(),
                font_type="comicsansms", font_size=30)
            box.draw(v)
            v.render()
            pygame.time.wait(2000)
            box.visible = False


        kws={
            'reference_area':v,
            'default_color':'black',
        }
        self.screen_items = aux.AttrDict({
            'clock': SimulationClock(sim_step_in_sec=m.dt, **kws),
            'scale': SimulationScale(**kws),
            'state': SimulationState(model=m,**kws),
        })
        self.screen_texts = aux.AttrDict({name: ScreenMsgText(text=name, **kws) for name in [
            'trail_dt',
            'trail_color',
            'visible_trails',
            'focus_mode',
            'draw_centroid',
            'draw_head',
            'draw_midline',
            'draw_contour',
            'draw_sensors',
            'draw_orientations',
            'draw_segs',
            'visible_clock',
            'visible_ids',
            'visible_state',
            'visible_scale',
            'odor_aura',
            'color_behavior',
            'random_colors',
            'black_background',
            'larva_collisions',
            'zoom',
            'snapshot #',
            'odorscape #',
            'windscape',
            'is_paused',
        ] + list(m.odor_layers.keys())
                              })




    def step(self):
        self.check()
        if self.active :

            self.screen_items.clock.tick_clock()
            if self.mode == 'video':
                if self.image_mode != 'snapshots' or self.snapshot_tick:
                    self.render()
            elif self.mode == 'image':
                if self.image_mode == 'overlap':
                    self.render()
                elif self.image_mode == 'snapshots' and self.snapshot_tick:
                    self.capture_snapshot()

    def finalize(self):
        if self.active:
            if self.image_mode == 'overlap':
                self.v.render()
                pygame.time.wait(5000)
                # raise
            elif self.image_mode == 'final':
                self.capture_snapshot()
            if self.v:
                self.v.close()

    def capture_snapshot(self):
        self.render()
        self.toggle('snapshot #')
        self.v.render()













    def draw_aux(self, v, **kwargs):
        v.draw_arena(self.tank_color, self.screen_color)
        for t in list(self.screen_items.values()):
            t._draw(v)
        for t in list(self.screen_texts.values()) + [self.input_box]:
            t.visible = t.start_time < pygame.time.get_ticks() < t.end_time
            t._draw(v)


    def draw_arena(self, v):
        m = self.model
        arena_drawn = False
        for id, layer in m.odor_layers.items():
            if layer.visible:
                layer.draw(v)
                arena_drawn = True
                break


        if not arena_drawn and m.food_grid is not None:
            m.food_grid._draw(v=v)
            # print(self.model.food_grid.visible)
            arena_drawn = True

        if not arena_drawn:
            v.draw_polygon(m.space.vertices, color=self.tank_color)
            v.draw_background()



        if m.windscape is not None:
            m.windscape._draw(v=v)

        for b in m.borders:
            b._draw(v=v)
        # self.model.borders._draw(v=v)




    def toggle(self, name, value=None, show=False, minus=False, plus=False, disp=None):
        m=self.model
        if disp is None:
            disp = name

        if name == 'snapshot #':
            self.v.snapshot_requested = int(m.Nticks * m.dt)
            value = self.snapshot_counter
            self.snapshot_counter += 1
        elif name == 'odorscape #':
            reg.graphs.dict['odorscape'](odor_layers = m.odor_layers,save_to=m.plot_dir,
                                         show=show, scale=m.scaling_factor, idx=self.odorscape_counter)
            value = self.odorscape_counter
            self.odorscape_counter += 1
        elif name == 'trail_dt':
            if minus:
                dt = -1
            elif plus:
                dt = +1
            self.trail_dt = np.clip(self.trail_dt + 5 * dt, a_min=0, a_max=np.inf)
            value = self.trail_dt
        elif name == 'trail_color':
            obs=self.param.trail_color.objects
            self.trail_color = obs[(obs.index(self.trail_color) + 1) % len(obs)]
            value = self.trail_color

        if value is None:
            setattr(self, name, not getattr(self, name))
            value = 'ON' if getattr(self, name) else 'OFF'

        self.screen_texts[name].flash_text(f'{disp} {value}')


        if name == 'random_colors':
            for f in m.agents:
                color = aux.random_colors(1)[0] if self.random_colors else f.default_color
                f.set_default_color(color)
        elif name == 'black_background':
            for a in self.model.get_all_objects() + list(
                    self.screen_items.values()) + list(
                    self.screen_texts.values()):
                a.invert_default_color()
        elif name == 'larva_collisions':

            self.model.eliminate_overlap()









    def evaluate_input(self):

        if self.pygame_keys is None:
            self.pygame_keys = reg.controls.load()['pygame_keys']

        ev = pygame.event.get()
        for e in ev:
            if e.type == pygame.QUIT:
                self.close()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                for k, v in self.pygame_keys.items():
                    if e.key == getattr(pygame, v):
                        self.eval_keypress(k)

            if self.allow_clicks:
                if e.type == pygame.MOUSEWHEEL:
                    self.v.zoom_screen(e.y, pos=self.v.mouse_position)
                    self.toggle(name='zoom', value=self.v.zoom)
                elif e.type == pygame.MOUSEBUTTONUP:
                    if e.button == 1:
                        if not self.eval_selection(p=self.v.mouse_position, ctrl=pygame.key.get_mods() & pygame.KMOD_CTRL):
                        #     self.model.add_agent(agent_class=self.selected_type, p0=tuple(p),
                        #                 p1=tuple(self.mousebuttondown_pos))
                            pass

                    elif e.button == 3:
                        from larvaworld.gui.gui_aux.windows import set_agent_kwargs, object_menu
                        loc = tuple(np.array([int(x) for x in os.environ['SDL_VIDEO_WINDOW_POS'].split(',')]) + np.array(pygame.mouse.get_pos()))
                        if len(self.selected_agents) > 0:
                            for sel in self.selected_agents:
                                sel = set_agent_kwargs(sel, location=loc)
                        else:
                            self.selected_type = object_menu(self.selected_type, location=loc)

        if self.focus_mode and len(self.selected_agents) > 0:
            try:
                sel = self.selected_agents[0]
                self.v.move_center(pos=sel.get_position())
            except:
                pass
        # print(self.selected_agents)

    def eval_keypress(self, k):

        # print(k)
        if k == 'visible_ids':
            for a in self.model.agents + self.model.sources:
                a.id_box.toggle_vis()
                temp=a.id_box.visible
        # elif name == 'color_behavior':
            # if not self.color_behavior:
            #     for f in self.model.agents:
            #         f.set_color(f.default_color)
            self.toggle(k, 'ON' if temp else 'OFF', disp='IDs')
        elif k == 'visible_clock':
            self.screen_items.clock.toggle_vis()
            self.toggle(k,'ON' if self.screen_items.clock.visible else 'OFF', disp='clock')
        elif k == 'visible_scale':
            self.screen_items.scale.toggle_vis()
            self.toggle(k, 'ON' if self.screen_items.scale.visible else 'OFF', disp='scale')
        elif k == 'visible_state':
            self.screen_items.state.toggle_vis()
            self.toggle(k, 'ON' if self.screen_items.state.visible else 'OFF', disp='state')
        elif k == '▲ trail duration':
            self.toggle('trail_dt', plus=True, disp='trail duration')
        elif k == '▼ trail duration':
            self.toggle('trail_dt', minus=True, disp='trail duration')
        elif k == 'visible_trails':
            self.toggle(k, disp='trails')
        elif k == 'pause':
            self.toggle('is_paused')
        elif k == 'move left':
            self.v.move_center(-0.05, 0)
        elif k == 'move right':
            self.v.move_center(+0.05, 0)
        elif k == 'move up':
            self.v.move_center(0, +0.05)
        elif k == 'move down':
            self.v.move_center(0, -0.05)
        elif k == 'plot odorscapes':
            self.toggle('odorscape #', show=pygame.key.get_mods() & pygame.KMOD_CTRL)
        elif 'odorscape' in k:
            idx = int(k.split(' ')[-1])
            try:
                layer_id = list(self.model.odor_layers.keys())[idx]
                layer = self.model.odor_layers[layer_id]
                layer.toggle_vis()
                self.toggle(layer_id, 'ON' if layer.visible else 'OFF')
            except:
                pass
        elif k == 'snapshot':
            self.toggle('snapshot #')
        elif k == 'windscape':
            try:
                self.model.windscape.toggle_vis()
                self.toggle('windscape', 'ON' if self.model.windscape.visible else 'OFF')
            except:
                pass
        elif k == 'delete item':
            from larvaworld.gui.gui_aux.windows import delete_objects_window
            if delete_objects_window(self.selected_agents):
                for f in self.selected_agents:
                    self.selected_agents.remove(f)
                    self.model.delete_agent(f)
        elif k == 'dynamic graph':
            from larvaworld.lib.model.agents._larva import Larva
            if len(self.selected_agents) > 0:
                sel = self.selected_agents[0]
                if isinstance(sel, Larva):
                    from larvaworld.gui.gui_aux import DynamicGraph
                    self.dynamic_graphs.append(DynamicGraph(agent=sel))
        elif k == 'odor gains':
            if len(self.selected_agents) > 0:
                sel = self.selected_agents[0]
                from larvaworld.lib.model.agents._larva_sim import LarvaSim
                if isinstance(sel, LarvaSim) and sel.brain.olfactor is not None:
                    from larvaworld.gui.gui_aux.windows import set_kwargs
                    sel.brain.olfactor.gain = set_kwargs(sel.brain.olfactor.gain, title='Odor gains')
        else:
            self.toggle(k)

    def evaluate_graphs(self):
        for g in self.dynamic_graphs:
            running = g.evaluate()
            if not running:
                self.dynamic_graphs.remove(g)
                del g


    def eval_selection(self, p, ctrl):
        res = False if len(self.selected_agents) == 0 else True
        for f in self.model.get_all_objects():
            if f.contained(p):
                if not f.selected:
                    f.selected = True
                    self.selected_agents.append(f)
                elif ctrl:
                    f.selected = False
                    self.selected_agents.remove(f)
                res = True
            elif f.selected and not ctrl:
                f.selected = False
                self.selected_agents.remove(f)
        return res




