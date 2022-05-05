import copy
import random
import warnings
import numpy as np
import progressbar
import os
from typing import List, Any
import webcolors
from shapely.geometry import Polygon
from unflatten import unflatten

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from shapely.affinity import affine_transform

from mesa.space import ContinuousSpace

import lib.aux.dictsNlists
import lib.aux.sim_aux
import lib.aux.xy_aux
from lib.conf.base.dtypes import null_dict
from lib.model.agents._larva_sim import LarvaSim
from lib.conf.base import paths
from lib.aux.collecting import NamedRandomActivation
from lib.model.envs._space import FoodGrid, WindScape
import lib.anal.rendering as ren
import lib.aux.colsNstr as fun
from lib.model.envs._maze import Border
from lib.model.agents._agent import LarvaworldAgent
from lib.model.agents._source import Food
from lib.sim.single.input_lib import evaluate_input, evaluate_graphs
from lib.aux.dictsNlists import AttrDict


class LarvaWorld:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        pygame.init()
        W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
        cls.screen_dim_W, cls.screen_dim_H = int(W * 2 / 3/16)*16, int(H * 2 / 3/16)*16
        """Create a new model object_class and instantiate its RNG automatically."""
        cls._seed = kwargs.get("seed", None)
        cls.random = random.Random(cls._seed)
        return object.__new__(cls)

    def __init__(self, env_params, vis_kwargs=None, id='unnamed', dt=0.1, Nsteps=None, save_to='.',
                 background_motion=None, Box2D=False, use_background=False, traj_color=None, allow_clicks=True,
                 experiment=None, progress_bar=None, larva_groups={}, configuration_text=None,larva_collisions=True):
        self.configuration_text = configuration_text
        self.larva_collisions = larva_collisions
        if progress_bar is None:
            progress_bar = progressbar.ProgressBar(Nsteps)
            progress_bar.start()
        self.progress_bar = progress_bar
        self.Box2D = Box2D
        if vis_kwargs is None:
            vis_kwargs = null_dict('visualization', mode=None)
        self.vis_kwargs = AttrDict.from_nested_dicts(vis_kwargs)
        self.__dict__.update(self.vis_kwargs.draw)
        self.__dict__.update(self.vis_kwargs.color)
        self.__dict__.update(self.vis_kwargs.aux)

        self.odor_aura = False
        self.experiment = experiment
        self.dynamic_graphs = []
        self.focus_mode = False
        self.selected_type = ''

        self.borders, self.border_xy, self.border_lines, self.border_bodies = [], [], [], []

        self.mousebuttondown_pos = None
        self.mousebuttonup_pos = None

        self.selected_agents = []
        self.is_running = False
        self.is_paused = False
        self.dt = dt
        self.video_fps = int(self.vis_kwargs.render.video_speed / dt)
        self.allow_clicks = allow_clicks
        self.Nticks = 0
        self.Nsteps = Nsteps
        self.snapshot_interval = int(60 / dt)
        self.id = id

        self._screen = None
        self.save_to = save_to

        os.makedirs(save_to, exist_ok=True)
        if self.vis_kwargs.render.media_name:
            self.media_name = os.path.join(save_to, self.vis_kwargs.render.media_name)
        else:
            self.media_name = os.path.join(save_to, self.id)

        self.traj_color = traj_color
        self.background_motion = background_motion
        self.use_background = use_background
        self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
            self.black_background)

        self.selection_color = np.array([255, 0, 0])
        self.env_pars = AttrDict.from_nested_dicts(env_params)
        self.larva_groups = AttrDict.from_nested_dicts(larva_groups)

        self.snapshot_counter = 0
        self.odorscape_counter = 0
        self.Nodors, self.odor_layers = 0, {}
        self.food_grid = None

        # Add mesa schedule to use datacollector class
        self.create_schedules()
        self.create_arena(**self.env_pars.arena)
        self.space = self.create_space()
        if 'border_list' in self.env_pars.keys() :
            for id, pars in self.env_pars.border_list.items():
                b = Border(model=self, unique_id=id, **pars)
                self.add_border(b)

        self.sim_clock = ren.SimulationClock(self.dt, color=self.scale_clock_color)
        self.sim_scale = ren.SimulationScale(self.arena_dims[0], color=self.scale_clock_color)
        self.sim_state = ren.SimulationState(model=self, color=self.scale_clock_color)

        self.screen_texts = self.create_screen_texts(color=self.scale_clock_color)
        self.input_box = ren.InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
                                      center=True, w=120 * 4, h=32 * 4, font=pygame.font.SysFont("comicsansms", 32 * 2))
        self.configuration_box = ren.InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
                                              color_active=pygame.Color('white'),
                                              center=True, w=220 * 4, h=200 * 4,
                                              font=pygame.font.SysFont("comicsansms", 25))
        self.end_condition_met = False

        # self.env_pars['thermoscape'] = {"plate_temp": 22, "thermo_sources": [[0.5,0.05], [0.05,0.5], [0.5,0.95], [0.95,0.5]], "thermo_source_dTemps" : [8,-8,8,-8]}
        # print(self.env_pars.keys()) # @todo remove the manual coding above when I can work out how to read in thermoscape in env_pars.
        
        # print("Hey")
        # print(self.env_pars.windscape)
        if 'windscape' in self.env_pars.keys() and self.env_pars.windscape is not None and self.env_pars.windscape['wind_speed'] is not None:
            self.windscape = WindScape(model=self, **self.env_pars.windscape)
        else:
            self.windscape = None
        # self.windscape = None

    def toggle(self, name, value=None, show=False, minus=False, plus=False, disp=None):
        if disp is None:
            disp = name

        if name == 'snapshot #':
            import imageio
            record_image_to = f'{self.media_name}_{self.snapshot_counter}.png'
            self._screen._image_writer = imageio.get_writer(record_image_to, mode='i')
            value = self.snapshot_counter
            self.snapshot_counter += 1
        elif name == 'odorscape #':
            self.plot_odorscape(save_to=self.save_to, show=show, scale=self.scaling_factor, idx=self.odorscape_counter)
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
            for a in self.get_flies() + self.get_food():
                a.id_box.visible = not a.id_box.visible
        elif name == 'color_behavior':
            if not self.color_behavior:
                for f in self.get_flies():
                    f.set_color(f.default_color)
        elif name == 'random_colors':
            for f in self.get_flies():
                color = fun.random_colors(1)[0] if self.random_colors else f.default_color
                f.set_color(color)
        elif name == 'black_background':
            self.update_default_colors()
        elif name == 'larva_collisions':

            self.eliminate_overlap()

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

    def create_arena(self, arena_dims, arena_shape):
        self.arena_dims = X, Y = np.array(arena_dims)
        W0, H0 = self.screen_dim_W, self.screen_dim_H
        R0, R = W0 / H0, X / Y
        self.screen_width, self.screen_height = (W0, int(W0 / R/16)*16) if R0 < R else (int(H0 * R/16)*16, H0)
        self.unscaled_space_edges = np.array([(-X / 2, -Y / 2),
                                              (-X / 2, Y / 2),
                                              (X / 2, Y / 2),
                                              (X / 2, -Y / 2)])
        if arena_shape == 'circular':
            # This is a circle_to_polygon shape from the function
            self.unscaled_tank_shape = lib.aux.sim_aux.circle_to_polygon(60, X / 2)
        elif arena_shape == 'rectangular':
            # This is a rectangular shape
            self.unscaled_tank_shape = self.unscaled_space_edges

    def create_space(self):
        s = self.scaling_factor = 1000.0 if self.Box2D else 1.0
        X, Y = self.space_dims = self.arena_dims * s
        self.space_edges = [(x * s, y * s) for (x, y) in self.unscaled_space_edges]
        self.space_edges_for_screen = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
        self.tank_shape = self.unscaled_tank_shape * s
        k = 0.97
        self.tank_polygon = Polygon(self.tank_shape * k)

        if self.Box2D:
            from Box2D import b2World, b2ChainShape, b2EdgeShape
            self._sim_velocity_iterations = 6
            self._sim_position_iterations = 2

            # create the space in Box2D
            space = b2World(gravity=(0, 0), doSleep=True)

            # create a static body for the space borders
            self.tank = space.CreateStaticBody(position=(.0, .0))
            self.tank.CreateFixture(shape=b2ChainShape(vertices=self.tank_shape.tolist()))
            #     create second static body to attach friction
            self.friction_body = space.CreateStaticBody(position=(.0, .0))
            self.friction_body.CreateFixture(shape=b2ChainShape(vertices=self.space_edges))
        else:
            space = ContinuousSpace(x_min=-X / 2, x_max=X / 2, y_min=-Y / 2, y_max=Y / 2, torus=False)
        return space


    def create_schedules(self):
        self.active_larva_schedule = NamedRandomActivation('active_larva_schedule', self)
        self.all_larva_schedule = NamedRandomActivation('all_larva_schedule', self)
        self.active_food_schedule = NamedRandomActivation('active_food_schedule', self)
        self.all_food_schedule = NamedRandomActivation('all_food_schedule', self)

    def delete_agent(self, agent):
        if type(agent) is LarvaSim:
            self.active_larva_schedule.remove(agent)
        elif type(agent) is Food:
            self.active_food_schedule.remove(agent)
        elif type(agent) is Border:
            self.borders.remove(agent)
            agent.delete()

    def close(self):
        self.is_running = False
        if self._screen is not None:
            self._screen.close_requested()

    def get_flies(self) -> List[LarvaworldAgent]:
        return self.active_larva_schedule.agents

    def get_food(self) -> List[LarvaworldAgent]:
        return self.active_food_schedule.agents

    def get_agents(self, agent_class) -> List[LarvaworldAgent]:
        if agent_class == 'Food':
            return self.get_food()
        elif agent_class == 'Larva':
            return self.get_flies()

    def generate_larva_color(self):
        return fun.random_colors(1)[0] if self.random_colors else self.default_larva_color

    def set_background(self, width, height):
        if self.use_background:
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

    def draw_background(self, screen, background_motion):
        if self.bgimage is not None and self.bgimagerect is not None:
            x, y, a = background_motion
            try:
                min_x = int(np.floor(x))
                min_y = -int(np.floor(y))
                if a == 0.0:
                    surface = screen._window
                    for py in np.arange(min_y - 1, self.th_max + min_y, 1):
                        for px in np.arange(min_x - 1, self.tw_max + min_x, 1):
                            p = ((px - x) * (self.tw - 1), (py + y) * (self.th - 1))
                            surface.blit(self.bgimage, p)
            except:
                pass

    def draw_aux(self, screen):
        screen.draw_arena(self.tank_shape, self.tank_color, self.screen_color)
        if self.visible_clock:
            self.sim_clock.draw_clock(screen)
        if self.visible_scale:
            self.sim_scale.draw_scale(screen)
        if self.visible_state:
            self.sim_state.draw_state(screen)
        self.draw_screen_texts(screen)

    def draw_arena(self, screen, background_motion):
        screen.set_bounds(*self.space_edges_for_screen)
        arena_drawn = False
        for id, layer in self.odor_layers.items():
            if layer.visible:
                layer.draw(screen)
                arena_drawn = True
                break
        if not arena_drawn and self.food_grid is not None:
            self.food_grid.draw(screen)
            arena_drawn = True

        if not arena_drawn:
            screen.draw_polygon(self.tank_shape, color=self.tank_color)
            self.draw_background(screen, background_motion)

        if self.windscape is not None and self.windscape.visible:
            self.windscape.draw(screen)

        for i, b in enumerate(self.borders):
            b.draw(screen)

    def render_aux(self, width, height):
        self.sim_clock.render_clock(width, height)
        self.sim_scale.render_scale(width, height)
        self.sim_state.render_state(width, height)
        for t in self.screen_texts.values():
            t.render(width, height)

    def render(self, tick=None, background_motion=None):
        if background_motion is None:
            background_motion = [0, 0, 0]
        if self.background_motion is not None and tick is not None:
            background_motion = self.background_motion[:, tick-1]
        if self._screen is None:
            m = self.vis_kwargs.render.mode
            vid = f'{self.media_name}.mp4' if m == 'video' else None
            im = f'{self.media_name}_{self.snapshot_counter}.png' if m == 'image' else None

            self._screen = ren.Viewer(self.screen_width, self.screen_height, caption=self.id,
                                      fps=self.video_fps, dt=self.dt,
                                      show_display=self.vis_kwargs.render.show_display,
                                      record_video_to=vid, record_image_to=im)
            self.display_configuration(self._screen)
            self.render_aux(self.screen_width, self.screen_height)
            self.set_background(self._screen._window.get_width(), self._screen._window.get_height())

            self.draw_arena(self._screen, background_motion)

            print('Screen opened')
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None
            self.close()
            return
        if self.vis_kwargs['render']['image_mode'] != 'overlap':
            self.draw_arena(self._screen, background_motion)

        for o in self.get_food():
            if o.visible:
                o.draw(self._screen, filled=True if o.amount > 0 else False)
                o.id_box.draw(self._screen)

        for g in self.get_flies():
            if g.visible:
                g.draw(self._screen)
                g.id_box.draw(self._screen)

        if self.trails:
            if self.trajectory_dt is None:
                self.trajectory_dt = 0.0
            ren.draw_trajectories(space_dims=self.space_dims, agents=self.get_flies(), screen=self._screen,
                                  decay_in_ticks=int(self.trajectory_dt / self.dt),
                                  traj_color=self.traj_color)

        evaluate_input(self, self._screen)
        evaluate_graphs(self)
        if self.vis_kwargs['render']['image_mode'] != 'overlap':
            self.draw_aux(self._screen)
            self._screen.render()

    def display_configuration(self, screen):
        if self.configuration_text is not None:
            self.configuration_box.text = self.configuration_text
            self.configuration_box.visible = True
            for i in range(10):
                self.configuration_box.draw(screen)
                screen.render()
            self.configuration_box.visible = False

    def screen2space_pos(self, pos):
        p = (2 * pos[0] / self.screen_width - 1), -(2 * pos[1] / self.screen_height - 1)
        pp = p[0] * self.space_dims[0] / 2, p[1] * self.space_dims[1] / 2
        return pp

    def space2screen_pos(self, pos):
        if pos is None or any(np.isnan(pos)):
            return None
        try:
            return self._screen._transform(pos)
        except:
            p = pos[0] * 2 / self.space_dims[0], pos[1] * 2 / self.space_dims[1]
            pp = ((p[0] + 1) * self.screen_width / 2, (-p[1] + 1) * self.screen_height / 2)
            return pp

    def _place_food(self, food_pars):
        if food_pars is not None:
            if food_pars.food_grid is not None :
                self.food_grid = FoodGrid(**food_pars.food_grid, space_range=self.space_edges_for_screen, model=self)
                # self._create_food_grid(space_range=self.space_edges_for_screen,grid_pars=food_pars.food_grid)
            for gID, gConf in food_pars.source_groups.items():
                ps = lib.aux.xy_aux.generate_xy_distro(**gConf.distribution)
                for i, p in enumerate(ps):
                    self.add_food(id=f'{gID}_{i}', pos=p, group=gID, **gConf)

            for id, f_pars in food_pars.source_units.items():
                self.add_food(id=id, **f_pars)

    def add_food(self, pos, id=None, **food_pars):
        f = Food(unique_id=self.next_id(type='Food') if id is None else id, pos=pos, model=self, **food_pars)
        self.active_food_schedule.add(f)
        self.all_food_schedule.add(f)
        return f

    def add_larva(self, pos, orientation=None, id=None, pars=None, group=None, default_color=None, life_history=None,
                  odor=None):
        print(f'{pos}\n{orientation}\n"id"\n{id}\n{pars}\n{group}\n"dc"\n{default_color}\n{life_history}\n{odor}')
        if group is None and pars is None:
            group, conf = list(self.larva_groups.items())[0]
            print("al0")
            sample_dict = sample_group(conf['sample'], 1, self.sample_ps)
            print("al0.1")
            mod = get_sample_bout_distros(conf['model'], conf['sample'])
            print("al0.15")
            pars = self._generate_larvae(1, sample_dict, mod)
            print("al0.2")
            life_history = conf['life_history']
            print("al0.3")
            odor = conf['odor']
            if default_color is None:
                default_color = conf['default_color']
        print("al1")
        while not lib.aux.sim_aux.inside_polygon([pos], self.tank_polygon):
            pos = tuple(np.array(pos) * 0.999)
        print("al2")
        l = LarvaSim(unique_id=self.next_id(type='Larva') if id is None else id, model=self, pos=pos,
                     orientation=np.random.uniform(0, 2 * np.pi, 1)[0] if orientation is None else orientation,
                     odor=odor, larva_pars=pars, group=group, default_color=default_color, life_history=life_history)
        print("al3")
        self.active_larva_schedule.add(l)
        print("al4")
        self.all_larva_schedule.add(l)
        print("al5")
        return l

    def add_agent(self, agent_class, p0, p1=None):
        try:
            if agent_class == 'Food':
                f = self.add_food(p0)
            elif agent_class == 'Larva':
                f = self.add_larva(p0)
            elif agent_class == 'Border':
                b = Border(model=self, points=[p1, p0])
                self.add_border(b)
        except:
            pass

    def next_id(self, type='Food'):
        if type == 'Food':
            N = self.all_food_schedule.get_agent_count()
            return f'Food_{N}'
        elif type == 'Larva':
            N = self.all_larva_schedule.get_agent_count()
            return f'Larva_{N}'

    def step(self):
        # Overriden by subclasses
        self.Nticks += 1
        # Tick sim_clock
        self.sim_clock.tick_clock()
        if self.windscape is not None:
            self.windscape.update()

    def run(self):

        mode = self.vis_kwargs['render']['mode']
        img_mode = self.vis_kwargs['render']['image_mode']
        self.is_running = True
        warnings.filterwarnings('ignore')
        while self.is_running and self.Nticks < self.Nsteps and not self.end_condition_met:
            # print(self.Nticks)
            if not self.is_paused:
                self.step()
                if self.progress_bar:
                    self.progress_bar.update(self.Nticks)

            if mode == 'video':
                if img_mode != 'snapshots' or self.snapshot_tick:
                    self.render(self.Nticks)
            elif mode == 'image':
                if img_mode == 'overlap':
                    self.render(self.Nticks)
                elif img_mode == 'snapshots' and self.snapshot_tick:
                    self.capture_snapshot()

        if img_mode == 'overlap':
            self._screen.render()
        elif img_mode == 'final':
            self.capture_snapshot()
        if self._screen:
            self._screen.close()

        return self.is_running

    @property
    def snapshot_tick(self):
        return (self.Nticks - 1) % self.snapshot_interval == 0

    def capture_snapshot(self, tick=None, screen=None):
        if tick is None:
            tick = self.Nticks
        self.render(tick)
        self.toggle('snapshot #')
        if screen is None:
            screen = self._screen
        screen.render()

    def move_larvae_to_center(self):
        N = len(self.get_flies())
        orientations = np.random.uniform(low=0.0, high=np.pi * 2, size=N).tolist()
        positions = lib.aux.xy_aux.generate_xy_distro(N=N, mode='uniform', scale=(0.005, 0.015), loc=(0.0, 0.0),
                                                      shape='oval')

        for l, p, o in zip(self.get_flies(), positions, orientations):
            temp = np.array([-np.cos(o), -np.sin(o)])
            head = l.head
            head.set_pose(p, o)
            head.update_vertices(p, o)
            for i, seg in enumerate(l.segs[1:]):
                seg.set_orientation(o)
                prev_p = l.get_global_rear_end_of_seg(seg_index=i)
                new_p = prev_p + temp * l.seg_lengths[i + 1] / 2
                seg.set_position(new_p)
                seg.set_lin_vel(0.0)
                seg.set_ang_vel(0.0)
            l.pos = l.global_midspine_of_body
            self.space.move_agent(l, l.pos)

    def create_borders(self, lines):
        s = self.scaling_factor
        T = [s, 0, 0, s, 0, 0]
        ls = [affine_transform(l, T) for l in lines]
        ps = [l.coords.xy for l in ls]
        xy = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]
        return xy, ls

    def create_border_bodies(self, border_xy):
        from Box2D import b2EdgeShape
        if self.Box2D:
            bs = []
            for xy in border_xy:
                b = self.space.CreateStaticBody(position=(.0, .0))
                b.CreateFixture(shape=b2EdgeShape(vertices=xy.tolist()))
                bs.append(b)
            return bs
        else:
            return []

    # def get_image_path(self):
    #     return f'{self.media_name}_{self.snapshot_counter}.png'
    #     # return None

    def get_agent_list(self, class_name):
        global id
        if class_name == 'Source':
            agents = self.get_food()
        elif class_name in ['LarvaSim', 'LarvaReplay']:
            agents = self.get_flies()
        elif class_name == 'Border':
            agents = self.borders
        pars = list(null_dict(class_name).keys())
        data = {}
        for f in agents:
            dic = {}
            for p in pars:
                if p == 'unique_id':
                    id = f.unique_id
                elif p == 'default_color':
                    try:
                        dic[p] = webcolors.rgb_to_name(tuple([int(x) for x in getattr(f, p)]))
                    except:
                        dic[p] = getattr(f, p)
                else:
                    dic[p] = getattr(f, p)
            data[id] = dic
        return data

    def add_border(self, b):
        self.borders.append(b)
        self.border_xy += b.border_xy
        self.border_lines += b.border_lines
        self.border_bodies += b.border_bodies

    def draw_screen_texts(self, screen):
        for text in list(self.screen_texts.values()) + [self.input_box]:
            if text and text.start_time < pygame.time.get_ticks() < text.end_time:
                text.visible = True
                text.draw(screen)
            else:
                text.visible = False

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
        return {name: ren.InputBox(text=name, color_active=color, color_inactive=color) for name in names}

    def add_screen_texts(self, names, color):
        for name in names:
            text = ren.InputBox(text=name, color_active=color, color_inactive=color)
            self.screen_texts[name] = text

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

    def plot_odorscape(self,scale=1.0,idx=0, **kwargs):
        from lib.anal.plotting import plot_surface
        for id, layer in self.odor_layers.items():
            X, Y = layer.meshgrid
            x = X * 1000 / scale
            y = Y * 1000 / scale
            plot_surface(x=x, y=y, z=layer.get_grid(), vars=[r'x $(mm)$', r'y $(mm)$'], target=r'concentration $(μM)$',
                         title=f'{id} odorscape', save_as=f'{id}_odorscape_{idx}', **kwargs)


    def eliminate_overlap(self):
        pass


def generate_larvae(N, sample_dict, base_model, RefPars=None):

    from lib.aux.dictsNlists import load_dict, flatten_dict
    if RefPars is None:
        RefPars = load_dict(paths.path('ParRef'), use_pickle=False)
    if len(sample_dict) > 0:
        # print(sample_dict)
        all_pars = []
        modF = flatten_dict(base_model)
        for i in range(N):
            lF = copy.deepcopy(modF)
            for p, vs in sample_dict.items():
                p=RefPars[p] if p in RefPars.keys() else p
                lF.update({p: vs[i]})
            dic=AttrDict.from_nested_dicts(unflatten(lF))
            all_pars.append(dic)
    else:
        all_pars = [base_model] * N
    return all_pars


def get_sample_bout_distros(model, sample):
    dic={
        'pause_dist' : ['pause', 'pause_dur'],
        'stridechain_dist' : ['stride', 'run_count'],
        'run_dist' : ['run', 'run_dur'],
         }
    m = AttrDict.from_nested_dicts(copy.deepcopy(model))
    Im=m.brain.intermitter_params
    if Im and sample != {}:

        ds=[ii for ii in ['pause_dist', 'stridechain_dist', 'run_dist'] if (ii in Im.keys()) and (Im[ii] is not None) and ('fit' in Im[ii].keys()) and (Im[ii]['fit'])]
        for d in ds :
            for sample_d in dic[d] :
                if sample_d in sample.bout_distros.keys() and sample.bout_distros[sample_d] is not None :
                    Im[d]=sample.bout_distros[sample_d]
        # for bout, dist in zip(['pause', 'stride'], ['pause_dist', 'stridechain_dist']):
        #     if 'fit' in m.brain.intermitter_params[dist].keys() and m.brain.intermitter_params[dist].fit :
        #         m.brain.intermitter_params[dist] = sample.bout_distros[bout]
    return m


def sample_group(sample, N, sample_ps):
    from lib.stor.larva_dataset import LarvaDataset
    d = LarvaDataset(sample['dir'], load_data=False)
    e = d.read('end')
    ps = [p for p in sample_ps if p in e.columns]
    means = [e[p].mean() for p in ps]

    if len(ps) >= 2:
        base = e[ps].values.T
        cov = np.cov(base)
        vs = np.random.multivariate_normal(means, cov, N).T
    elif len(ps) == 1:
        std = np.std(e[ps].values)
        vs = np.atleast_2d(np.random.normal(means[0], std, N))
    else:
        return {}
    dic = {p: v for p, v in zip(ps, vs)}
    return dic

# if __name__ == '__main__':
#     RefPars = lib.aux.dictsNlists.load_dict(paths.path('ParRef'), use_pickle=False)
#     print(RefPars)
#     sample_ps=list(RefPars.keys())
#
#     from lib.conf.stored.conf import loadConf
#     sample=loadConf('None.200_controls', 'Ref')
#     dic=sample_group(sample, 10, sample_ps)
# print(dic)
