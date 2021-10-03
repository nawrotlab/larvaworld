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

import lib.aux.dictsNlists
import lib.aux.sim_aux
import lib.aux.xy_aux
from lib.aux import colsNstr as fun
from lib.conf.init_dtypes import get_dict
from lib.model.agents._larva_sim import LarvaSim
from lib.stor import paths


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from shapely.affinity import affine_transform
from Box2D import b2World, b2ChainShape, b2EdgeShape
from mesa.space import ContinuousSpace

from lib.aux.collecting import NamedRandomActivation
from lib.envs._space import FoodGrid
import lib.anal.rendering as ren
import lib.aux.colsNstr as fun
from lib.envs._maze import Border
import lib.conf.dtype_dicts as dtypes
from lib.model.agents._agent import LarvaworldAgent
from lib.model.agents._source import Food
from lib.sim.input_lib import evaluate_input, evaluate_graphs


class LarvaWorld:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        pygame.init()
        max_screen_height = pygame.display.Info().current_h
        cls.sim_screen_dim = int(max_screen_height * 2 / 3)
        """Create a new model object_class and instantiate its RNG automatically."""
        cls._seed = kwargs.get("seed", None)
        cls.random = random.Random(cls._seed)
        return object.__new__(cls)

    def __init__(self, env_params, vis_kwargs=None,
                 id='unnamed', dt=0.1, Nsteps=None, save_to='.',
                 background_motion=None, Box2D=False,
                 use_background=False,
                 traj_color=None, allow_clicks=True,
                 experiment=None,
                 progress_bar=None,
                 # space_in_mm=False
                 ):
        # self.space_in_mm = space_in_mm
        self.progress_bar = progress_bar
        self.Box2D = Box2D
        # print(vis_kwargs)
        if vis_kwargs is None:
            vis_kwargs = get_dict('visualization', mode=None)
        self.vis_kwargs = vis_kwargs
        self.__dict__.update(self.vis_kwargs['draw'])
        self.__dict__.update(self.vis_kwargs['color'])
        self.__dict__.update(self.vis_kwargs['aux'])

        self.experiment = experiment
        self.dynamic_graphs = []
        self.focus_mode = False
        self.selected_type = ''

        self.borders, self.border_xy, self.border_lines, self.border_bodies = [], [], [], []

        self.mousebuttondown_time = None
        self.mousebuttonup_time = None
        self.mousebuttondown_pos = None
        self.mousebuttonup_pos = None

        self.input_box = ren.InputBox()
        self.selected_agents = []
        self.is_running = False
        self.is_paused = False
        self.dt = dt
        self.video_fps = int(self.vis_kwargs['render']['video_speed'] / dt)
        # self.video_fps = int(self.video_speed / dt)
        self.allow_clicks = allow_clicks
        # self.touch_sensors = touch_sensors
        self.Nticks = 0
        self.Nsteps = Nsteps
        snapshot_interval_in_sec = 60
        self.snapshot_interval = int(snapshot_interval_in_sec / dt)
        self.id = id

        self._screen = None
        self.save_to = save_to

        os.makedirs(save_to, exist_ok=True)
        if self.vis_kwargs['render']['media_name']:
            self.media_name = os.path.join(save_to, self.vis_kwargs['render']['media_name'])
        else:
            self.media_name = os.path.join(save_to, self.id)

        self.traj_color = traj_color
        self.background_motion = background_motion
        self.use_background = use_background
        self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
            self.black_background)

        self.selection_color = np.array([255, 0, 0])
        self.env_pars = env_params
        # self.larva_pars = larva_pars

        self.snapshot_counter = 0
        self.odorscape_counter = 0
        self.Nodors, self.odor_layers = 0, {}
        self.food_grid = None

        # Add mesa schecule to use datacollector class
        self.create_schedules()
        self.create_arena(**self.env_pars['arena'])
        self.space = self.create_space()
        if 'border_list' in list(self.env_pars.keys()):
            for id, pars in self.env_pars['border_list'].items():
                b = Border(model=self, unique_id=id, **pars)
                self.add_border(b)

        self.sim_clock = ren.SimulationClock(self.dt, color=self.scale_clock_color)
        self.sim_scale = ren.SimulationScale(self.arena_dims[0], color=self.scale_clock_color)
        self.sim_state = ren.SimulationState(model=self, color=self.scale_clock_color)

        self.screen_texts = self.create_screen_texts(color=self.scale_clock_color)

        self.end_condition_met = False

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
            self.plot_odorscape(save_to=self.save_to, show=show)
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
        self.screen_texts[name].text = f'{disp} {value}'
        self.screen_texts[name].end_time = pygame.time.get_ticks() + 2000
        self.screen_texts[name].start_time = pygame.time.get_ticks() + int(self.dt * 1000)

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
        X, Y = arena_dims
        D = self.sim_screen_dim
        self.arena_dims = np.array([X, Y])
        self.screen_width, self.screen_height = (D, int(D * X / Y)) if X <= Y else (int(D * X / Y), D)
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
        s=self.scaling_factor = 1000.0 if self.Box2D else 1.0
        X, Y = self.space_dims = self.arena_dims * s
        self.space_edges = [(x * s, y * s) for (x, y) in self.unscaled_space_edges]
        self.space_edges_for_screen = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
        self.tank_shape = self.unscaled_tank_shape * s
        k = 0.97
        self.tank_polygon = Polygon(self.tank_shape * k)

        if self.Box2D:
            self._sim_velocity_iterations = 6
            self._sim_position_iterations = 2

            # create the space in Box2D
            space = b2World(gravity=(0, 0), doSleep=True)

            # create a static body for the space borders
            self.tank = space.CreateStaticBody(position=(.0, .0))
            tank_shape = b2ChainShape(vertices=self.tank_shape.tolist())
            self.tank.CreateFixture(shape=tank_shape)

            #     create second static body to attach friction
            self.friction_body = space.CreateStaticBody(position=(.0, .0))
            friction_body_shape = b2ChainShape(vertices=self.space_edges)
            self.friction_body.CreateFixture(shape=friction_body_shape)


        else:
            space = ContinuousSpace(x_min=-X / 2, x_max=X / 2, y_min=-Y / 2, y_max=Y / 2, torus=False)
        return space

    def _create_food_grid(self, space_range, grid_pars):
        if grid_pars:
            self.food_grid = FoodGrid(**grid_pars, space_range=space_range)

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
        if self.random_colors:
            color = fun.random_colors(1)[0]
        else:
            color = self.default_larva_color
        return color

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
        self.input_box.draw(screen)

        self.draw_screen_texts(screen)

    def draw_arena(self, screen, background_motion):
        screen.set_bounds(*self.space_edges_for_screen)
        arena_drawn = False
        for id, layer in self.odor_layers.items():
            if layer.visible:
                layer.draw(screen)
                arena_drawn = True
                break
        if not arena_drawn and self.food_grid:
            self.food_grid.draw(screen)
            arena_drawn = True
        if not arena_drawn:
            screen.draw_polygon(self.tank_shape, color=self.tank_color)
            self.draw_background(screen, background_motion)

        for i, b in enumerate(self.borders):
            b.draw(screen)

    def render_aux(self, width, height):
        self.sim_clock.render_clock(width, height)
        self.sim_scale.render_scale(width, height)
        self.sim_state.render_state(width, height)
        for name, text in self.screen_texts.items():
            text.render(width, height)

    def render(self, tick=None, background_motion=None):
        if background_motion is None:
            background_motion = [0, 0, 0]
        if self.background_motion is not None and tick is not None:
            background_motion = self.background_motion[:, tick]
        if self._screen is None:
            m = self.vis_kwargs['render']['mode']
            vid = f'{self.media_name}.mp4' if m == 'video' else None
            im = f'{self.media_name}_{self.snapshot_counter}.png' if m == 'image' else None

            self._screen = ren.GuppiesViewer(self.screen_width, self.screen_height, caption=self.id,
                                             fps=self.video_fps, dt=self.dt,
                                             show_display=self.vis_kwargs['render']['show_display'],
                                             record_video_to=vid, record_image_to=im)
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
            if self.trajectory_dt is None :
                self.trajectory_dt=0.0
            ren.draw_trajectories(space_dims=self.space_dims, agents=self.get_flies(), screen=self._screen,
                                  decay_in_ticks=int(self.trajectory_dt / self.dt),
                                  traj_color=self.traj_color)

        evaluate_input(self, self._screen)
        evaluate_graphs(self)
        # t1 = time.time()
        if self.vis_kwargs['render']['image_mode'] != 'overlap':
            # t2 = time.time()
            self.draw_aux(self._screen)
            # t3 = time.time()
            self._screen.render()

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
        if food_pars is not None :
            pars0 = copy.deepcopy(food_pars)
            if pars0['food_grid'] is not None:
                self._create_food_grid(space_range=self.space_edges_for_screen,
                                       grid_pars=pars0['food_grid'])
            if pars0['source_groups'] is not None:
                for gID, gConf in pars0['source_groups'].items():
                    ps = lib.aux.xy_aux.generate_xy_distro(**gConf['distribution'])
                    for i, p in enumerate(ps) :
                        id =f'{gID}_{i}'
                        self.add_food(id=id, position=p, food_pars=gConf)


            for id, f_pars in pars0['source_units'].items():
                position = f_pars['pos']
                f_pars.pop('pos')
                self.add_food(id=id, position=position, food_pars=f_pars)

    def add_food(self, position, id=None, food_pars={}):
        if id is None:
            id = self.next_id(type='Food')
        f = Food(unique_id=id, pos=position, model=self, **food_pars)
        self.active_food_schedule.add(f)
        self.all_food_schedule.add(f)
        return f

    def add_larva(self, pos, orientation=None, id=None, pars=None, group=None, default_color=None, life=None, odor=None):
        # print(pos, group)
        if group is None and pars is None:
            group, conf = list(self.env_pars['larva_groups'].items())[0]
            sample_dict = sample_group(conf['sample'], 1, self.sample_ps)
            mod = get_sample_bout_distros(conf['model'], conf['sample'])
            pars = self._generate_larvae(1, sample_dict, mod)
            life=conf['life']
            odor=conf['odor']
            if default_color is None:
                default_color = conf['default_color']
        if id is None:
            id = self.next_id(type='Larva')
        if orientation is None:
            orientation = np.random.uniform(0, 2 * np.pi, 1)[0]
        while not lib.aux.sim_aux.inside_polygon([pos], self.tank_polygon)[0] :
            pos=tuple(np.array(pos)*0.999)
        l = LarvaSim(model=self, pos=pos, orientation=orientation, unique_id=id,odor=odor,
                     larva_pars=pars, group=group, default_color=default_color, life=life)
        self.active_larva_schedule.add(l)
        self.all_larva_schedule.add(l)

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
        pass

    def run(self):
        mode = self.vis_kwargs['render']['mode']
        img_mode = self.vis_kwargs['render']['image_mode']
        self.is_running = True
        warnings.filterwarnings('ignore')
        bar = self.progress_bar if self.progress_bar is not None else progressbar.ProgressBar(max_value=self.Nsteps)
        while self.is_running and self.Nticks < self.Nsteps and not self.end_condition_met:
            if not self.is_paused:
                self.step()
                bar.update(self.Nticks)
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

    @ property
    def snapshot_tick(self):
        return (self.Nticks - 1) % self.snapshot_interval == 0

    def capture_snapshot(self, tick=None, screen=None):
        if tick is None :
            tick=self.Nticks
        if screen is None :
            screen=self._screen
        self.render(tick)
        self.toggle('snapshot #')
        screen.render()

    def move_larvae_to_center(self):
        N = len(self.get_flies())
        orientations = np.random.uniform(low=0.0, high=np.pi * 2, size=N).tolist()
        positions = lib.aux.xy_aux.generate_xy_distro(N=N, mode='uniform', scale=(0.005, 0.015), loc=(0.0, 0.0), shape='oval')

        for l, p, o in zip(self.get_flies(), positions, orientations):
            temp = np.array([-np.cos(o), -np.sin(o)])
            head = l.get_head()
            head.set_pose(p, o)
            head.update_vertices(p, o)
            for i, seg in enumerate(l.segs[1:]):
                seg.set_orientation(o)
                prev_p = l.get_global_rear_end_of_seg(seg_index=i)
                new_p = prev_p + temp * l.seg_lengths[i + 1] / 2
                seg.set_position(new_p)
                seg.set_lin_vel(0.0)
                seg.set_ang_vel(0.0)
            l.pos = l.get_global_midspine_of_body()
            self.space.move_agent(l, l.pos)

    def create_borders(self, lines):
        s = self.scaling_factor
        T = [s, 0, 0, s, 0, 0]
        lines = [affine_transform(l, T) for l in lines]
        ps = [p.coords.xy for p in lines]
        border_xy = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]
        return border_xy, lines

    def create_border_bodies(self, border_xy):
        if self.Box2D:
            border_bodies = []
            for xy in border_xy:
                b = self.space.CreateStaticBody(position=(.0, .0))
                border_shape = b2EdgeShape(vertices=xy.tolist())
                b.CreateFixture(shape=border_shape)
                border_bodies.append(b)
            return border_bodies
        else:
            return []

    def get_image_path(self):
        return f'{self.media_name}_{self.snapshot_counter}.png'
        # return None

    def get_agent_list(self, class_name):
        global id
        if class_name == 'Source':
            agents = self.get_food()
        elif class_name in ['LarvaSim', 'LarvaReplay']:
            agents = self.get_flies()
        elif class_name == 'Border':
            agents = self.borders
        pars = list(dtypes.get_dict_dtypes(class_name).keys())
        # pars = list(dtypes.get_dict_dtypes('agent', class_name=class_name).keys())
        # pars = list(agent_dtypes[class_name].keys())
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
        for name, text in self.screen_texts.items():
            if text and text.start_time < pygame.time.get_ticks() < text.end_time:
                text.visible = True
                text.draw(screen)
            else:
                text.visible = False

    def create_screen_texts(self, color):
        texts = {}
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
            'color_behavior',
            'random_colors',
            'black_background',
            'larva_collisions',
            'zoom',
            'snapshot #',
            'odorscape #',
            'is_paused',
        ]

        for name in names:
            text = ren.InputBox(visible=False, text=name,
                                color_active=color, color_inactive=color,
                                screen_pos=None, linewidth=0.01, show_frame=False)
            texts[name] = text
        return texts

    def add_screen_texts(self, names, color):
        for name in names:
            text = ren.InputBox(visible=False, text=name,
                                color_active=color, color_inactive=color,
                                screen_pos=None, linewidth=0.01, show_frame=False)
            self.screen_texts[name] = text

    def update_default_colors(self):
        if self.black_background:
            self.tank_color = (0, 0, 0)
            self.screen_color = (50, 50, 50)
            self.scale_clock_color = (255, 255, 255)
            # self.default_larva_color = np.array([255, 255, 255])
            # for f in self.get_flies():
            #     f.color = fun.invert_color(f.default_color)

        else:
            self.tank_color = (255, 255, 255)
            self.screen_color = (200, 200, 200)
            self.scale_clock_color = (0, 0, 0)
            # self.default_larva_color = np.array([0, 0, 0])
            # for f in self.get_flies():
            #     f.color = f.default_color
        for i in [self.sim_clock, self.sim_scale, self.sim_state] + list(self.screen_texts.values()):
            i.set_color(self.scale_clock_color)

        # for f in self.get_flies():
        #     # fun.invert_color()
        #     color=[abs(c1-c0) for c0,c1 in zip(self.default_larva_color, f.default_color)]
        #     print(f.default_color, color)
        #     f.set_default_color(color)



def generate_larvae(N, sample_dict, base_model, RefPars=None):
    if RefPars is None :
        RefPars = lib.aux.dictsNlists.load_dict(paths.RefParsFile, use_pickle=False)
    if len(sample_dict) > 0:
        all_pars = []
        modF = lib.aux.dictsNlists.flatten_dict(base_model)
        for i in range(N):
            lF = copy.deepcopy(modF)
            for p, vs in sample_dict.items():
                lF.update({RefPars[p]: vs[i]})
            all_pars.append(unflatten(lF))
    else:
        all_pars = [base_model] * N
    return all_pars

def get_sample_bout_distros(model, sample) :
    if model['brain']['intermitter_params'] and sample != {}:
        for bout, dist in zip(['pause', 'stride'], ['pause_dist', 'stridechain_dist']):
            if model['brain']['intermitter_params'][dist]['fit']:
                model['brain']['intermitter_params'][dist] = sample['bout_distros'][bout]
    return model


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
    else :
        return {}
    dic={p:v for p,v in zip(ps,vs)}
    return dic