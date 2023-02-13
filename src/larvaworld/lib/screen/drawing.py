import math
import os
import sys

import numpy as np

from larvaworld.lib.screen import SimulationScale

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from larvaworld.lib import reg, aux, screen


class BaseScreenManager :
    def __init__(self, model,  mode=None, show_display = True,black_background=False, **kwargs):

        self.model = model
        self.s = self.model.scaling_factor
        self.space_bounds = aux.get_arena_bounds(self.model.space.dims, self.s)
        self.window_dims = aux.get_window_dims(self.model.space.dims)

        self.black_background = black_background
        self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
            self.black_background)

        self.sim_clock = screen.SimulationClock(self.model.dt, color=self.scale_clock_color)

        self.mode =mode
        self.show_display =show_display
        if self.mode is None and not self.show_display:
            reg.vprint('Storage of media or visualization not requested.')
            self.active=False
        else :
            self.active=True
            # self.screen_kws = self.define_screen_kws()

            self.build(**kwargs)


        self.v = None


    def build(self,**kwargs):
        pass


    def space2screen_pos(self, pos):
        if pos is None or any(np.isnan(pos)):
            return None
        try:
            return self.v._transform(pos)
        except:
            X, Y = self.model.space.dims * self.s
            X0, Y0 = self.window_dims

            p = pos[0] * 2 / X, pos[1] * 2 / Y
            pp = ((p[0] + 1) * X0 / 2, (-p[1] + 1) * Y0)
            return pp

    # def define_screen_kws(self):
    #     pass

    def set_default_colors(self, black_background):
        if black_background:
            tank_color = (0, 0, 0)
            screen_color = (50, 50, 50)
            scale_clock_color = (255, 255, 255)
            default_larva_color = np.array([255, 255, 255])
        else:
            tank_color = (255, 255, 255)
            screen_color = (200, 200, 200)
            scale_clock_color = (0, 0, 0)
            default_larva_color = np.array([0, 0, 0])
        return tank_color, screen_color, scale_clock_color, default_larva_color

class GA_ScreenManager(BaseScreenManager):
    def __init__(self, **kwargs):
        super().__init__(mode=None, black_background=True,**kwargs)

    def build(self,panel_width=600,fps=10,scene='no_boxes', **kwargs):
        self.screen_kws = {
            'file_path': f'{reg.ROOT_DIR}/lib/sim/ga_scenes/{scene}.txt',
            'show_display': self.show_display,
            'panel_width': panel_width,
            'caption': f'GA {self.model.experiment} : {self.model.id}',
            'window_dims': self.window_dims,
            'space_bounds': self.space_bounds,
            'dt': self.model.dt,
            'fps': fps,
        }

    def render(self, tick=None):
        if self.v is None:
            self.v = self.initialize()
        elif self.v.close_requested():
            self.v.close()
            self.v = None
            self.model.running = False
            return

        if self.v.show_display:
            from pygame import KEYDOWN, K_ESCAPE, K_r, K_MINUS, K_PLUS, K_s, QUIT, event, Rect, draw, display
            for e in event.get():
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    sys.exit()
                elif e.type == KEYDOWN and e.key == K_r:
                    self.model.run()
                elif e.type == KEYDOWN and (e.key == K_PLUS or e.key == 93 or e.key == 270):
                    self.v.increase_fps()
                elif e.type == KEYDOWN and (e.key == K_MINUS or e.key == 47 or e.key == 269):
                    self.v.decrease_fps()
                elif e.type == KEYDOWN and e.key == K_s:
                    pass
                    # self.engine.save_genomes()
                # elif e.type == KEYDOWN and e.key == K_e:
                #     self.engine.evaluation_mode = 'preparing'

            if self.side_panel.generation_num < self.model.engine.generation_num:
                self.side_panel.update_ga_data(self.model.engine.generation_num, self.model.engine.best_genome)

            # update statistics time
            cur_t = aux.TimeUtil.current_time_millis()
            cum_t = math.floor((cur_t - self.model.engine.start_total_time) / 1000)
            gen_t = math.floor((cur_t - self.model.engine.start_generation_time) / 1000)
            self.side_panel.update_ga_time(cum_t, gen_t, self.model.engine.generation_sim_time)
            self.side_panel.update_ga_population(len(self.model.engine.robots), self.model.engine.Nagents)
            self.v._window.fill(aux.Color.BLACK)

            self.draw_agents(self.v)

            # draw a black background for the side panel
            self.v.draw_panel_rect()
            self.side_panel.display_ga_info()

            display.flip()
            self.v._t.tick(self.v._fps)
        #
        #
        # if self.v.show_display:
        #     self.evaluate_input()
        #
        # self.draw_aux(self.v)
        # self.v.render()

    def draw_agents(self, v):
        for o in self.model.get_food():
            if o.visible:
                o.draw(v, filled=True if o.amount > 0 else False)
                o.id_box.draw(v, screen_pos=self.space2screen_pos(o.get_position()))

        for g in self.model.engine.robots:
            if g.visible:

                g.draw(v, self)
                g.id_box.draw(v, screen_pos=self.space2screen_pos(g.get_position()))




    def initialize(self):

        v = screen.Viewer.load_from_file(**self.screen_kws)
        if v.show_display:
            from larvaworld.lib.screen.side_panel import SidePanel
            from pygame import display
            # self.get_larvaworld_food()
            self.side_panel = SidePanel(v, self.model.engine.space_dict)
            self.side_panel.update_ga_data(self.model.engine.generation_num, None)
            self.side_panel.update_ga_population(len(self.model.engine.robots), self.model.engine.Nagents)
            self.side_panel.update_ga_time(0, 0, 0)

        # self.display_configuration(v)
        # self.render_aux()
        # self.set_background(*v.display_dims)
        #
        # self.draw_arena(v)

        print('Screen opened')
        return v



class ScreenManager(BaseScreenManager):
    def __init__(self, model, vis_kwargs=None, **kwargs):
        if vis_kwargs is None:
            vis_kwargs = reg.get_null('visualization', mode=None)
        self.vis_kwargs = aux.AttrDict(vis_kwargs)
        self.image_mode = self.vis_kwargs.render.image_mode
        super().__init__(model, mode= self.vis_kwargs.render.mode,
                         show_display= self.vis_kwargs.render.show_display,
                         black_background= self.vis_kwargs.color.black_background, **kwargs)





    def build(self,background_motion=None, traj_color=None,allow_clicks=True, **kwargs):
        self.screen_kws = self.define_screen_kws()


        self.__dict__.update(self.vis_kwargs.draw)
        self.__dict__.update(self.vis_kwargs.color)
        self.__dict__.update(self.vis_kwargs.aux)
        self.dynamic_graphs = []
        self.focus_mode = False
        self.intro_text = self.vis_kwargs.render.intro_text
        self.selected_type = ''


        self.mousebuttondown_pos = None
        self.mousebuttonup_pos = None

        self.selected_agents = []

        self.allow_clicks = allow_clicks

        self.snapshot_interval = int(60 / self.model.dt)

        self.traj_color = traj_color
        if self.trajectory_dt is None:
            self.trajectory_dt = 0.0
        self.background_motion = background_motion

        self.selection_color = np.array([255, 0, 0])

        self.snapshot_counter = 0
        self.odorscape_counter = 0



        self.bg = self.background_motion

        self.pygame_keys = None

        # self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
        #     self.black_background)
        #
        # self.sim_clock = screen.SimulationClock(self.model.dt, color=self.scale_clock_color)
        self.sim_scale = screen.SimulationScale(self.model.space.dims[0], color=self.scale_clock_color)
        self.sim_state = screen.SimulationState(model=self.model, color=self.scale_clock_color)

        self.screen_texts = self.create_screen_texts(color=self.scale_clock_color)
        self.add_screen_texts(list(self.model.odor_layers.keys()), color=self.scale_clock_color)
        self.input_box = screen.InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
                                  center=True, w=120 * 4, h=32 * 4, font=pygame.font.SysFont("comicsansms", 32 * 2))


    def define_screen_kws(self):
        m = self.model
        media_name = self.vis_kwargs.render.media_name
        video_speed = self.vis_kwargs.render.video_speed
        if media_name is None:
            media_name = str(m.id)



        screen_kws = {
            'window_dims': self.window_dims,
            'space_bounds': self.space_bounds,
            'caption': media_name,
            'dt': m.dt,
            'fps': int(video_speed / m.dt),
            'show_display': self.show_display,
            # 'record_video_to':show_display,
        }
        os.makedirs(m.save_to, exist_ok=True)
        if self.mode == 'video':
            screen_kws['record_video_to'] = f'{m.save_to}/{media_name}.mp4'
        if self.mode == 'image':
            screen_kws['record_image_to'] = f'{m.save_to}/{media_name}_{self.image_mode}.png'
        return screen_kws


    def add_screen_texts(self, names, color):
        for name in names:
            text = screen.InputBox(text=name, color_active=color, color_inactive=color)
            self.screen_texts[name] = text

    def step(self, tick=None):
        if self.active :
            self.sim_clock.tick_clock()
            if self.mode == 'video':
                if self.image_mode != 'snapshots' or self.snapshot_tick:
                    self.render(tick)
            elif self.mode == 'image':
                if self.image_mode == 'overlap':
                    self.render(tick)
                elif self.image_mode == 'snapshots' and self.snapshot_tick:
                    self.capture_snapshot(tick)

    def finalize(self, tick=None):
        if self.active:
            if self.image_mode == 'overlap':
                self.v.render()
            elif self.image_mode == 'final':
                self.capture_snapshot(tick)
            if self.v:
                self.v.close()

    def capture_snapshot(self, tick):
        self.render(tick)
        self.model.toggle('snapshot #')
        self.v.render()

    def create_screen_texts(self, color):
        names = [
            'trajectory_dt',
            'trails',
            'focus_mode',
            'draw_centroid',
            'draw_head',
            'draw_midline',
            'draw_contour',
            'draw_sensors',
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
        ]
        return {name: screen.InputBox(text=name, color_active=color, color_inactive=color) for name in names}



    def set_background(self, width, height):
        if self.bg is not None:
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(ROOT_DIR, 'background.png')
            print('Loading background image from', path)
            self.bgimage = pygame.image.load(path)
            self.bgimagerect = self.bgimage.get_rect()
            self.tw = self.bgimage.get_width()
            self.th = self.bgimage.get_height()
            self.th_max = int(height / self.th) + 2
            self.tw_max = int(width / self.tw) + 2
        else:
            self.bgimage = None
            self.bgimagerect = None

    def draw_agents(self, v):
        for o in self.model.get_food():
            if o.visible:
                o.draw(v, filled=True if o.amount > 0 else False)
                o.id_box.draw(v, screen_pos=self.space2screen_pos(o.get_position()))

        for g in self.model.get_flies():
            if g.visible:
                if self.color_behavior:
                    g.update_behavior_dict()
                g.draw(v, self)
                g.id_box.draw(v, screen_pos=self.space2screen_pos(g.get_position()))

        if self.trails:

            self.draw_trajectories()

    def render(self, tick=None):
        if self.bg is not None and tick is not None:
            bg = self.bg[:, tick - 1]
        else:
            bg = [0, 0, 0]
        if self.v is None:
            self.v = self.initialize(bg=bg)
        elif self.v.close_requested():
            self.v.close()
            self.v = None
            self.model.running = False
            return
        if self.image_mode != 'overlap':
            self.draw_arena(self.v, bg)

        self.draw_agents(self.v)

        if self.v.show_display:
            self.evaluate_input()
            self.evaluate_graphs()
        if self.image_mode != 'overlap':

            self.draw_aux(self.v)
            self.v.render()

    def initialize(self, bg):

        v = screen.Viewer(**self.screen_kws)

        self.display_configuration(v)
        self.render_aux()
        self.set_background(*v.display_dims)

        self.draw_arena(v, bg)

        print('Screen opened')
        return v

    def display_configuration(self, v):
        if self.intro_text:
            box = screen.InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
                           text=self.model.configuration_text,
                           color_active=pygame.Color('white'),
                           visible=True,
                           center=True, w=220 * 4, h=200 * 4,
                           font=pygame.font.SysFont("comicsansms", 25))

            for i in range(10):
                box.draw(v)
                v.render()
            box.visible = False



    def draw_aux(self, v):
        v.draw_arena(self.model.space.vertices, self.tank_color, self.screen_color)
        if self.visible_clock:
            self.sim_clock.draw_clock(v)
        if self.visible_scale:
            self.sim_scale.draw_scale(v)
        if self.visible_state:
            self.sim_state.draw_state(v)
        self.draw_screen_texts(v)

    def draw_screen_texts(self, v):
        for text in list(self.screen_texts.values()) + [self.input_box]:
            if text and text.start_time < pygame.time.get_ticks() < text.end_time:
                text.visible = True
                text.draw(v)
            else:
                text.visible = False

    def draw_arena(self, v, bg):
        arena_drawn = False
        for id, layer in self.model.odor_layers.items():
            if layer.visible:
                layer.draw(v)
                arena_drawn = True
                break
        if not arena_drawn and self.model.food_grid is not None:
            self.model.food_grid.draw(v)
            arena_drawn = True

        if not arena_drawn:
            v.draw_polygon(self.model.space.vertices, color=self.tank_color)
            self.draw_background(v, bg)

        if self.model.windscape is not None and self.model.windscape.visible:
            self.model.windscape.draw(v)

        for i, b in enumerate(self.model.borders):
            b.draw(v)

    def render_aux(self):
        self.sim_clock.render_clock(*self.window_dims)
        self.sim_scale.render_scale(*self.window_dims)
        self.sim_state.render_state(*self.window_dims)
        for t in self.screen_texts.values():
            t.render(*self.window_dims)

    def draw_background(self, v, bg):
        if self.bgimage is not None and self.bgimagerect is not None:
            x, y, a = bg
            try:
                min_x = int(np.floor(x))
                min_y = -int(np.floor(y))

                for py in np.arange(min_y - 1, self.th_max + min_y, 1):
                    for px in np.arange(min_x - 1, self.tw_max + min_x, 1):
                        if a != 0.0:
                            # px,py=aux.rotate_point_around_point((px,py),-a)
                            pass
                        p = ((px - x) * (self.tw - 1), (py + y) * (self.th - 1))
                        v._window.blit(self.bgimage, p)
            except:
                pass

    def toggle(self, name, value=None, show=False, minus=False, plus=False, disp=None):
        if disp is None:
            disp = name

        if name == 'snapshot #':
            self.v.snapshot_requested = int(self.model.Nticks * self.model.dt)
            value = self.snapshot_counter
            self.snapshot_counter += 1
        elif name == 'odorscape #':
            reg.graphs.dict['odorscape'](odor_layers = self.model.odor_layers,save_to=self.model.plot_dir, show=show, scale=self.s, idx=self.odorscape_counter)
            value = self.odorscape_counter
            self.odorscape_counter += 1
        elif name == 'trajectory_dt':
            if minus:
                dt = -1
            elif plus:
                dt = +1
            self.trajectory_dt = np.clip(self.trajectory_dt + 5 * dt, a_min=0, a_max=np.inf)
            value = self.trajectory_dt

        if value is None:
            setattr(self, name, not getattr(self, name))
            value = 'ON' if getattr(self, name) else 'OFF'
        self.screen_texts[name].flash_text(f'{disp} {value}')
        # self.screen_texts[name].text = f'{disp} {value}'
        # self.screen_texts[name].end_time = pygame.time.get_ticks() + 2000
        # self.screen_texts[name].start_time = pygame.time.get_ticks() + int(self.dt * 1000)

        if name == 'visible_ids':
            for a in self.model.get_flies() + self.model.get_food():
                a.id_box.visible = not a.id_box.visible
        elif name == 'color_behavior':
            if not self.color_behavior:
                for f in self.model.get_flies():
                    f.set_color(f.default_color)
        elif name == 'random_colors':
            for f in self.model.get_flies():
                color = aux.random_colors(1)[0] if self.random_colors else f.default_color
                f.set_color(color)
        elif name == 'black_background':
            self.update_default_colors()
        elif name == 'larva_collisions':

            self.eliminate_overlap()

    def update_default_colors(self):
        if self.black_background:
            self.tank_color = (0, 0, 0)
            self.screen_color = (50, 50, 50)
            self.scale_clock_color = (255, 255, 255)
        else:
            self.tank_color = (255, 255, 255)
            self.screen_color = (200, 200, 200)
            self.scale_clock_color = (0, 0, 0)
        for i in [self.sim_clock, self.sim_scale, self.sim_state] + list(self.screen_texts.values()):
            i.set_color(self.scale_clock_color)


    def apply_screen_zoom(self, d_zoom):
        self.v.zoom_screen(d_zoom)
        self.sim_scale = SimulationScale(self.model.space.dims[0] * self.v.zoom, color=self.sim_scale.color)
        self.sim_scale.render_scale(*self.window_dims)



    @property
    def snapshot_tick(self):
        return (self.model.Nticks - 1) % self.snapshot_interval == 0


    def add_screen_texts(self, names, color):
        for name in names:
            text = screen.InputBox(text=name, color_active=color, color_inactive=color)
            self.screen_texts[name] = text

    def generate_larva_color(self):
        return aux.random_colors(1)[0] if self.random_colors else self.default_larva_color



    def eliminate_overlap(self):
        pass

    def evaluate_input(self):

        if self.pygame_keys is None:
            self.pygame_keys = reg.controls.load()['pygame_keys']

        d_zoom = 0.01
        ev = pygame.event.get()
        for e in ev:
            if e.type == pygame.QUIT:
                self.v.close()
            if e.type == pygame.KEYDOWN:
                for k, v in self.pygame_keys.items():
                    if e.key == getattr(pygame, v):
                        self.eval_keypress(k)

            if self.allow_clicks:
                if e.type == pygame.MOUSEBUTTONDOWN:
                    self.mousebuttondown_pos = self.v.mouse_position
                elif e.type == pygame.MOUSEBUTTONUP:
                    p = self.v.mouse_position
                    if e.button == 1:
                        if not self.eval_selection(p, ctrl=pygame.key.get_mods() & pygame.KMOD_CTRL):
                            self.model.add_agent(agent_class=self.selected_type, p0=tuple(p),
                                        p1=tuple(self.mousebuttondown_pos))

                    elif e.button == 3:
                        from larvaworld.gui.gui_aux.windows import set_agent_kwargs, object_menu
                        loc = tuple(np.array(self.v.w_loc) + np.array(pygame.mouse.get_pos()))
                        if len(self.selected_agents) > 0:
                            for sel in self.selected_agents:
                                sel = set_agent_kwargs(sel, location=loc)
                        else:
                            self.selected_type = object_menu(self.selected_type, location=loc)
                    elif e.button in [4, 5]:
                        self.apply_screen_zoom(d_zoom=-d_zoom if e.button == 4 else d_zoom)
                        self.toggle(name='zoom', value=self.v.zoom)
        if self.focus_mode and len(self.selected_agents) > 0:
            try:
                sel = self.selected_agents[0]
                self.v.move_center(pos=sel.get_position())
            except:
                pass

    def eval_keypress(self, k):
        from larvaworld.lib.model.agents._larva_sim import LarvaSim
        from larvaworld.lib.model.agents._larva import Larva
        # print(k)
        if k == '▲ trail duration':
            self.toggle('trajectory_dt', plus=True, disp='trail duration')
        elif k == '▼ trail duration':
            self.toggle('trajectory_dt', minus=True, disp='trail duration')
        elif k == 'visible trail':
            self.toggle('trails')
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
                layer.visible = not layer.visible
                self.toggle(layer_id, 'ON' if layer.visible else 'OFF')
            except:
                pass
        elif k == 'snapshot':
            self.toggle('snapshot #')
        elif k == 'windscape':
            try:
                self.model.windscape.visible = not self.model.windscape.visible
                self.toggle('windscape', 'ON' if self.model.windscape.visible else 'OFF')
            except:
                pass
        elif k == 'delete item':
            from larvaworld.gui.gui_aux.windows import delete_objects_window
            if delete_objects_window(self.selected_agents):
                for f in self.selected_agents:
                    self.selected_agents.remove(f)
                    self.delete_agent(f)
        elif k == 'dynamic graph':
            if len(self.selected_agents) > 0:
                sel = self.selected_agents[0]
                if isinstance(sel, Larva):
                    from larvaworld.gui.gui_aux.elements import DynamicGraph
                    self.dynamic_graphs.append(DynamicGraph(agent=sel))
        elif k == 'odor gains':
            if len(self.selected_agents) > 0:
                sel = self.selected_agents[0]
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


    def draw_trajectories(self):
        space_dims = self.model.space.dims * self.s
        agents = self.model.get_flies()
        decay_in_ticks = int(self.trajectory_dt / self.model.dt)

        trajs = [fly.trajectory for fly in agents]
        if self.traj_color is not None:
            traj_cols = [self.traj_color.xs(fly.unique_id, level='AgentID') for fly in agents]
        else:
            traj_cols = [np.array([(0, 0, 0) for t in traj]) for traj, fly in zip(trajs, agents)]

        trajs = [t[-decay_in_ticks:] for t in trajs]
        traj_cols = [t[-decay_in_ticks:] for t in traj_cols]

        for fly, traj, traj_col in zip(agents, trajs, traj_cols):
            # This is the case for simulated larvae where no values are np.nan
            if not np.isnan(traj).any():
                parsed_traj = [traj]
                parsed_traj_col = [traj_col]
            elif np.isnan(traj).all():
                continue
            # This is the case for larva trajectories derived from experiments where some values are np.nan
            else:
                traj_x = np.array([x for x, y in traj])
                ds, de = aux.parse_array_at_nans(traj_x)
                parsed_traj = [traj[s:e] for s, e in zip(ds, de)]
                parsed_traj_col = [traj_col[s:e] for s, e in zip(ds, de)]

            for t, c in zip(parsed_traj, parsed_traj_col):
                # If trajectory has one point, skip

                if len(t) < 2:
                    pass
                else:
                    if self.traj_color is None:
                        self.v.draw_polyline(t, color=fly.default_color, closed=False, width=0.003 * space_dims[0])
                    else:
                        c = [tuple(float(x) for x in s.strip('()').split(',')) for s in c]
                        c = [s if not np.isnan(s).any() else (255, 0, 0) for s in c]
                        self.v.draw_polyline(t, color=c, closed=False, width=0.01 * space_dims[0], dynamic_color=True)


