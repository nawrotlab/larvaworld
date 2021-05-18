import copy
import random
import time
import warnings
import numpy as np
import progressbar
import os
from typing import List, Any
import webcolors
from mesa.datacollection import DataCollector

from lib.conf.conf import loadConfDict

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from shapely.affinity import affine_transform
from unflatten import unflatten
from Box2D import b2World, b2ChainShape, b2EdgeShape
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from lib.aux.collecting import TargetedDataCollector
from lib.envs._space import FoodGrid, DiffusionValueLayer, GaussianValueLayer
import lib.aux.rendering as ren
from lib.aux.sampling import sample_agents, get_ref_bout_distros
import lib.aux.functions as fun
from lib.aux import naming as nam
from lib.envs._maze import Border
import lib.conf.dtype_dicts as dtypes
from lib.model import *
from lib.model._agent import LarvaworldAgent
from lib.sim.input_lib import evaluate_input, evaluate_graphs


class LarvaWorld:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:

        pygame.init()
        max_screen_height = pygame.display.Info().current_h
        cls.sim_screen_dim = int(max_screen_height * 2 / 3)
        """Create a new model object and instantiate its RNG automatically."""
        cls._seed = kwargs.get("seed", None)
        cls.random = random.Random(cls._seed)
        return object.__new__(cls)

    def __init__(self, vis_kwargs, env_params,
                 id='unnamed', dt=0.1, Nsteps=None, save_to='.',
                 background_motion=None, Box2D=False,
                 use_background=False,
                 traj_color=None,
                 touch_sensors=False, allow_clicks=True,
                 experiment=None,
                 progress_bar=None,
                 ):

        self.progress_bar = progress_bar
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
        self.dt = dt
        self.video_fps = int(self.vis_kwargs['render']['video_speed'] / dt)
        # self.video_fps = int(self.video_speed / dt)
        self.allow_clicks = allow_clicks
        self.touch_sensors = touch_sensors
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
        self.create_arena(**self.env_pars['arena_params'])
        self.space = self.create_space(Box2D)
        if 'border_list' in list(self.env_pars.keys()):
            for id, pars in self.env_pars['border_list'].items():
                b = Border(model=self, unique_id=id, **pars)
                self.add_border(b)

        self.sim_clock = ren.SimulationClock(self.dt, color=self.scale_clock_color)
        self.sim_scale = ren.SimulationScale(self.arena_dims[0], self.scaling_factor,
                                             color=self.scale_clock_color)
        self.sim_state = ren.SimulationState(model=self, color=self.scale_clock_color)

        self.screen_texts = self.create_screen_texts(color=self.scale_clock_color)

        self.end_condition_met = False



    def toggle(self, name, value=None, show=False, minus=False, plus=False):

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
        # elif name=='black_background' :
        # elif name=='black_background' :

        if value is None:
            setattr(self, name, not getattr(self, name))
            value = 'ON' if getattr(self, name) else 'OFF'
        self.screen_texts[name].text = f'{name} {value}'
        self.screen_texts[name].end_time = pygame.time.get_ticks() + 2000
        self.screen_texts[name].start_time = pygame.time.get_ticks()+int(self.dt*1000)

        if name == 'visible_ids':
            for a in self.get_flies() + self.get_food():
                a.id_box.visible = not a.id_box.visible
        elif name == 'random_colors':
            for f in self.get_flies():
                f.set_default_color(self.generate_larva_color())
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

    def create_arena(self, arena_xdim, arena_ydim, arena_shape):
        X, Y = arena_xdim, arena_ydim
        self.arena_dims = np.array([X, Y])
        if X <= Y:
            self.screen_width = self.sim_screen_dim
            self.screen_height = int(self.sim_screen_dim * Y / X)
        else:
            self.screen_height = self.sim_screen_dim
            self.screen_width = int(self.sim_screen_dim * X / Y)

        self.unscaled_space_edges_for_screen = np.array([-X / 2, X / 2,
                                                         -Y / 2, Y / 2])
        self.unscaled_space_edges = np.array([(-X / 2, -Y / 2),
                                              (-X / 2, Y / 2),
                                              (X / 2, Y / 2),
                                              (X / 2, -Y / 2)])

        if arena_shape == 'circular':
            tank_radius = X / 2
            # This is a circle_to_polygon shape from the function
            self.unscaled_tank_shape = fun.circle_to_polygon(60, tank_radius)
        elif arena_shape == 'rectangular':
            # This is a rectangular shape
            self.unscaled_tank_shape = self.unscaled_space_edges

    def create_space(self, Box2D):
        if Box2D:
            self.physics_engine = True
            self.scaling_factor = 1000.0
        else:
            self.physics_engine = False
            self.scaling_factor = 1.0
        s = self.scaling_factor
        self.space_dims = self.arena_dims * s
        self.space_edges = [(x * s, y * s) for (x, y) in self.unscaled_space_edges]
        self.space_edges_for_screen = self.unscaled_space_edges_for_screen * s
        self.tank_shape = self.unscaled_tank_shape * s

        if self.physics_engine:
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
            x_min, x_max, y_min, y_max = self.space_edges_for_screen
            space = ContinuousSpace(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                    torus=False)
        return space

    def _create_food_grid(self, space_range, grid_pars):
        if grid_pars:
            self.food_grid = FoodGrid(**grid_pars, space_range=space_range)

    def create_schedules(self):
        self.active_larva_schedule = RandomActivation(self)
        self.all_larva_schedule = RandomActivation(self)
        self.active_food_schedule = RandomActivation(self)
        self.all_food_schedule = RandomActivation(self)

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
        # del self.active_food_schedule
        # del self.active_larva_schedule
        if self._screen is not None:
            self._screen.close_requested()
    # def delete(self):
    #     self.close()
    #     pygame.quit()
    #     del self

    def get_flies(self) -> List[Larva]:
        return self.active_larva_schedule.agents

    def get_food(self) -> List[Food]:
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
        # t1 = time.time()
        screen.draw_arena(self.tank_shape, self.tank_color, self.screen_color)
        # t2 = time.time()
        if self.visible_clock:
            self.sim_clock.draw_clock(screen)
        if self.visible_scale:
            self.sim_scale.draw_scale(screen)
        if self.visible_state:
            self.sim_state.draw_state(screen)
        t3 = time.time()
        self.input_box.draw(screen)

        self.draw_screen_texts(screen)
        t4 = time.time()
        # print()
        # print(np.round([1000*(t4-t3)]))

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

    def render(self, velocity_arrows=False, tick=None, background_motion = [0, 0, 0]):
        # if self.background_motion is None or tick is None:
        #     background_motion = [0, 0, 0]
        # else:
        #     background_motion = self.background_motion[:, tick]
        if self.background_motion and tick :
            background_motion = self.background_motion[:, tick]
        if self._screen is None:
            if self.vis_kwargs['render']['mode'] == 'video':
                self._video_path = f'{self.media_name}.mp4'
            else:
                self._video_path = None
            if self.vis_kwargs['render']['mode'] == 'image':
                self._image_path = f'{self.media_name}_{self.snapshot_counter}.png'
            else:
                self._image_path = None

            self._screen = ren.GuppiesViewer(self.screen_width, self.screen_height, caption=self.id,
                                             fps=self.video_fps, dt=self.dt,
                                             show_display=self.vis_kwargs['render']['show_display'],
                                             # show_display=self.show_display,
                                             record_video_to=self._video_path,
                                             record_image_to=self._image_path)
            self.render_aux(self.screen_width, self.screen_height)
            self.set_background(self._screen._window.get_width(), self._screen._window.get_height())
            self.draw_arena(self._screen, background_motion)
            print('Screen opened')
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None
            self.close()
            return
        # t0=time.time()
        if self.vis_kwargs['render']['image_mode'] != 'overlap':
            self.draw_arena(self._screen, background_motion)

        for o in self.get_food():
            o.draw(self._screen, filled=True if o.amount > 0 else False)
            o.id_box.draw(self._screen)

        for g in self.get_flies():
            g.draw(self._screen)
            g.id_box.draw(self._screen)
            # render velocity arrows
            if velocity_arrows:
                ren.draw_velocity_arrow(self._screen, g)

        if self.trajectories:
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

        # t4 = time.time()
        # print()
        # print(np.round([1000*(t4-t1),1000*(t2-t1), 1000*(t3-t2), 1000*(t4-t3)]))
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

    # def relative2space_pos(self, pos):
    #     x, y = pos
    #     return x * self.space_dims[0] / 2, y * self.space_dims[1] / 2

    def _place_food(self, food_pars):
        pars0 = copy.deepcopy(food_pars)
        if pars0['food_grid'] is not None:
            self._create_food_grid(space_range=self.space_edges_for_screen,
                                   grid_pars=pars0['food_grid'])
        if pars0['source_groups'] is not None:
            distro_pars = ['N', 'mode', 'shape', 'loc', 'scale']
            for group_id, group_pars in pars0['source_groups'].items():
                N, mode, shape, loc, scale = [group_pars[p] for p in distro_pars]
                pars = {p: group_pars[p] for p in group_pars if p not in distro_pars}
                food_positions = fun.generate_xy_distro(mode, shape, N, loc, scale)
                ids = [f'{group_id}_{i}' for i in range(N)]
                for id, p in zip(ids, food_positions):
                    self.add_food(id=id, position=p, food_pars=pars)
        # print(pars0['source_units'])
        # raise
        for id, f_pars in pars0['source_units'].items():
            # print(f_pars)
            position = f_pars['pos']
            f_pars.pop('pos')
            self.add_food(id=id, position=position, food_pars=f_pars)

    def add_food(self, position, id=None, food_pars={}):
        # if food_pars is None:
        #     food_pars = food()
        # food_pars = copy.deepcopy(self.env_pars['food_params'])
        # if 'source_units' in list(food_pars.keys()):
        #     food_pars.pop('source_units')
        if id is None:
            id = self.next_id(type='Food')
        f = Food(unique_id=id, pos=position, model=self, **food_pars)
        self.active_food_schedule.add(f)
        self.all_food_schedule.add(f)
        return f

    def add_larva(self, position, orientation=None, id=None, pars=None, group=None, default_color=None):
        if group is None and pars is None:
            group, distro = list(self.env_pars['larva_params'].items())[0]
            pars = self._generate_larva_pars(1, distro['model'])[0]
            if default_color is None:
                default_color = distro['default_color']
        if id is None:
            id = self.next_id(type='Larva')
        if orientation is None:
            orientation = np.random.uniform(0, 2 * np.pi, 1)[0]
        l = LarvaSim(model=self, pos=position, orientation=orientation, unique_id=id,
                     larva_pars=pars, group=group, default_color=default_color)
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

    def run(self, Nsteps=None):
        mode= self.vis_kwargs['render']['mode']
        img_mode= self.vis_kwargs['render']['image_mode']
        # pygame.init()
        # import pygame_gui
        # manager = pygame_gui.UIManager((800, 600))
        # hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 275), (100, 50)),
        #                                             text='Say Hello',
        #                                             manager=manager)
        # clock = pygame.time.Clock()
        self.is_running = True
        self.sim_paused = False
        if Nsteps is None:
            Nsteps = self.Nsteps
        warnings.filterwarnings('ignore')
        # import time
        if self.progress_bar is None :
            self.progress_bar = progressbar.ProgressBar(max_value=Nsteps)
        bar=self.progress_bar
        while self.is_running and self.Nticks < Nsteps and not self.end_condition_met:
            if not self.sim_paused:
                # t0=time.time()
                self.step()
                bar.update(self.Nticks)
            if mode=='video' :
                if img_mode != 'snapshots':
                    # t1 = time.time()
                    self.render(tick=self.Nticks)
                    # t2 = time.time()
                elif (self.Nticks - 1) % self.snapshot_interval == 0:
                    # print('ss')
                    self.render(tick=self.Nticks)
            elif mode== 'image':
                if img_mode == 'overlap':
                    self.render(tick=self.Nticks)
                elif img_mode == 'snapshots':
                    if (self.Nticks - 1) % self.snapshot_interval == 0:
                        self.render(tick=self.Nticks)
                        self.toggle(name='snapshot #')
                        self._screen.render()

            # print()
            # print((t1-t0)*1000, (t2-t1)*1000)

        if img_mode == 'overlap':
            self._screen.render()
        elif img_mode == 'final':
            self.render(tick=self.Nticks)
            self.toggle(name='snapshot #')
            self._screen.render()
        if self._screen :
            self._screen.close()
        return self.is_running

    def set_end_condition(self):
        if self.experiment == 'capture_the_flag':
            for f in self.get_food():
                if f.unique_id == 'Flag':
                    # print('ss')
                    self.flag = f
                elif f.unique_id == 'Left_base':
                    # print('ssdd')
                    self.l_base = f
                elif f.unique_id == 'Right_base':
                    # print('sssssa')
                    self.r_base = f
            self.l_base_p = self.l_base.get_position()
            self.r_base_p = self.r_base.get_position()
            self.l_dst0 = self.flag.radius * 2 + self.l_base.radius * 2
            self.r_dst0 = self.flag.radius * 2 + self.r_base.radius * 2

        elif self.experiment == 'keep_the_flag':
            for f in self.get_food():
                if f.unique_id == 'Flag':
                    self.flag = f
            self.l_t = 0
            self.r_t = 0

        elif self.experiment == 'catch_me' :
            self.target_group='Left' if random.uniform(0,1)>0.5 else 'Right'
            self.follower_group = 'Right' if self.target_group=='Left' else 'Left'
            for f in self.get_flies() :
                if f.group==self.target_group :
                    f.brain.olfactor.gain = {id : -v for id,v in f.brain.olfactor.gain.items()}
            self.score = {self.target_group : 0.0,
                          self.follower_group : 0.0}

        elif self.experiment == 'odor_pref_train' :
            self.CS_trial_counter=1
            self.UCS_trial_counter=0
            print()
            print(f'Training trial {self.CS_trial_counter} with CS started at {self.sim_clock.minute} : {self.sim_clock.second}')
            for f in self.get_food():
                if f.unique_id == 'CS_source':
                    self.CS_source = f
                elif f.unique_id == 'UCS_source':
                    self.UCS_source = f
            self.CS_source.odor_intensity = 2.0
            self.UCS_source.odor_intensity = 0.0

    def check_end_condition(self):
        if self.experiment == 'capture_the_flag':
            flag_p = self.flag.get_position()
            l_dst = -self.l_dst0 + fun.compute_dst(flag_p, self.l_base_p)
            r_dst = -self.r_dst0 + fun.compute_dst(flag_p, self.r_base_p)
            l_dst = np.round(l_dst * 1000, 2)
            r_dst = np.round(r_dst * 1000, 2)
            if l_dst < 0:
                print('Left group wins')
                self.end_condition_met = True
            elif r_dst < 0:
                print('Right group wins')
                self.end_condition_met = True
            self.sim_state.set_text(f'L:{l_dst} vs R:{r_dst}')

        elif self.experiment == 'keep_the_flag':
            dur = 180
            carrier = self.flag.is_carried_by
            if carrier is None:
                self.l_t = 0
                self.r_t = 0
            elif carrier.group == 'Left':
                self.l_t += self.dt
                self.r_t = 0
                if self.l_t - dur > 0:
                    print('Left group wins')
                    self.end_condition_met = True
            elif carrier.group == 'Right':
                self.r_t += self.dt
                self.l_t = 0
                if self.r_t - dur > 0:
                    print('Right group wins')
                    self.end_condition_met = True
            self.sim_state.set_text(f'L:{np.round(dur - self.l_t, 2)} vs R:{np.round(dur - self.r_t, 2)}')

        elif self.experiment == 'catch_me':
            def set_target_group(group) :
                self.target_group = group
                self.follower_group = 'Right' if self.target_group == 'Left' else 'Left'
                for f in self.get_flies():
                    f.brain.olfactor.gain = {id: -v for id, v in f.brain.olfactor.gain.items()}
            targets={f:f.get_position() for f in self.get_flies() if f.group==self.target_group}
            followers=[f for f in self.get_flies() if f.group==self.follower_group]
            for f in followers :
                if any([f.contained(p) for p in list(targets.values())]):
                    set_target_group(f.group)
                    break
            self.score[self.target_group]+=self.dt
            for group,score in self.score.items() :
                if score>=1200.0 :
                    print(f'{group} group wins')
                    self.end_condition_met = True
            self.sim_state.set_text(f'L:{np.round(self.score["Left"],1)} vs R:{np.round(self.score["Right"],1)}')

        elif self.experiment == 'odor_pref_train':
            if self.sim_clock.timer_opened :
                self.UCS_trial_counter+=1
                print()
                print(f'Starvation trial {self.UCS_trial_counter} with UCS started at {self.sim_clock.minute}:{self.sim_clock.second}')
                self.CS_source.odor_intensity=0.0
                self.UCS_source.odor_intensity=2.0
                self.move_larvae_to_center()
            if self.sim_clock.timer_closed :
                self.CS_trial_counter += 1
                print()
                print(f'Training trial {self.CS_trial_counter} with CS started at {self.sim_clock.minute}:{self.sim_clock.second}')
                self.CS_source.odor_intensity = 2.0
                self.UCS_source.odor_intensity = 0.0
                self.move_larvae_to_center()
            if self.sim_clock.next_on is None and self.sim_clock.next_off is None :
                print()
                print(f'Test trial started at {self.sim_clock.minute}:{self.sim_clock.second}')
                self.CS_source.odor_intensity = 2.0
                self.UCS_source.odor_intensity = 2.0
                self.move_larvae_to_center()
            if self.sim_clock.minute>=35 :
                print(f'Test trial ended at {self.sim_clock.minute}:{self.sim_clock.second}')

    def move_larvae_to_center(self) :
        N=len(self.get_flies())
        orientations = np.random.uniform(low=0.0, high=np.pi*2, size=N).tolist()
        positions = fun.generate_xy_distro(N=N, mode='uniform', scale=(0.005,0.015), loc=(0.0,0.0), shape='oval')

        for l, p, o in zip(self.get_flies(), positions, orientations) :
            temp=np.array([-np.cos(o), -np.sin(o)])
            head = l.get_head()
            head.set_pose(p,o)
            head.update_vertices(p,o)
            for i, seg in enumerate(l.segs[1:]) :
                seg.set_orientation(o)
                prev_p = l.get_global_rear_end_of_seg(seg_index=i)
                new_p = prev_p + temp * l.seg_lengths[i+1] / 2
                seg.set_position(new_p)
                seg.set_lin_vel(0.0)
                seg.set_ang_vel(0.0)
            l.pos = l.get_global_midspine_of_body()
            self.space.move_agent(l, l.pos)



    def create_borders(self, lines):
        s = self.scaling_factor
        X, Y = self.arena_dims
        # if not from_screen:
        #     T = [s, 0, 0, s, -s * X / 2, -s * Y / 2]
        # else:
        #     T = [s, 0, 0, s, 0, 0]
        T = [s, 0, 0, s, 0, 0]
        lines = [affine_transform(l, T) for l in lines]
        ps = [p.coords.xy for p in lines]
        border_xy = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]
        return border_xy, lines

    def create_border_bodies(self, border_xy):
        if self.physics_engine:
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
        pars = list(dtypes.get_dict_dtypes('agent', class_name=class_name).keys())
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
            if text and text.start_time<pygame.time.get_ticks() < text.end_time:
                text.visible = True
                text.draw(screen)
            else:
                text.visible = False

    def create_screen_texts(self, color):
        texts = {}
        names = [
            'trajectory_dt',
            'trajectories',
            'focus_mode',
            'draw_centroid',
            'draw_head',
            'draw_midline',
            'draw_contour',
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
            'sim_paused',
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
        self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
            self.black_background)
        for f in self.get_flies():
            f.set_default_color(self.generate_larva_color())
        for i in [self.sim_clock, self.sim_scale, self.sim_state] + list(self.screen_texts.values()):
            i.set_color(self.scale_clock_color)


class LarvaWorldSim(LarvaWorld):
    def __init__(self, collected_pars=None,
                 id='Unnamed_Simulation', larva_collisions=True, count_bend_errors=False,
                 life_params=None,
                 parameter_dict={}, **kwargs):
        super().__init__(id=id, **kwargs)
        if collected_pars is None:
            collected_pars = {'step': [], 'endpoint': [], 'tables' : {} }

        self.starvation_hours = life_params['starvation_hours']
        if self.starvation_hours is None:
            self.starvation_hours = []
        self.hours_as_larva = life_params['hours_as_larva']
        self.deb_base_f = life_params['deb_base_f']

        self.deb_starvation_hours = [[s0, np.clip(s1, a_min=s0, a_max=self.hours_as_larva)] for [s0, s1] in
                                     self.starvation_hours if s0 < self.hours_as_larva]
        self.sim_starvation_hours = [
            [np.clip(s0 - self.hours_as_larva, a_min=0, a_max=+np.inf), s1 - self.hours_as_larva] for
            [s0, s1] in self.starvation_hours if s1 > self.hours_as_larva]
        if len(self.sim_starvation_hours) > 0:
            on_ticks = [int(s0 * 60 * 60 / self.dt) for [s0, s1] in self.sim_starvation_hours]
            off_ticks = [int(s1 * 60 * 60 / self.dt) for [s0, s1] in self.sim_starvation_hours]
            self.sim_clock.set_timer(on_ticks, off_ticks)
        self.starvation = self.sim_clock.timer_on
        self.count_bend_errors = count_bend_errors

        self.larva_collisions = larva_collisions

        self._place_food(self.env_pars['food_params'])
        self.create_larvae(larva_pars=self.env_pars['larva_params'], parameter_dict=parameter_dict)
        if self.env_pars['odorscape'] is not None:
            self.Nodors, self.odor_layers = self._create_odor_layers(self.env_pars['odorscape'])
        self.add_screen_texts(list(self.odor_layers.keys()), color=self.scale_clock_color)

        self.create_data_collectors(collected_pars)

        if not self.larva_collisions:
            self.eliminate_overlap()

        self.set_end_condition()

    # def prepare_flies(self, timesteps):
    #     for t in range(timesteps):
    #         self.mock_step()
        #     # for g in self.get_flies():
        #     # if np.random.choice([0, 1]) == 0:
        #     #     g.compute_next_action()
        # if Nsec<self.dt :
        #     return
        # for g in self.get_flies():
        #     g.turner.prepare_turner(Nsec)
        # try:
        #     g.crawler.iteration_counter = 0
        #     g.crawler.total_t = 0
        #     g.crawler.t = 0
        # except:
        #     pass
        # try:
        #     g.intermitter.reset()
        # except:
        #     pass
        # try:
        #     g.reset_feeder()
        # except:
        #     pass
        # try:
        #     g.set_ang_activity(0.0)
        #     g.set_lin_activity(0.0)
        # except :
        #     pass
        # raise ValueError

    def prepare_odor_layer(self, timesteps):
        for i in range(timesteps):
            for id, layer in self.odor_layers.items():
                layer.update_values()  # Currently doing something only for the DiffusionValueLayer

    def _create_odor_layers(self, pars):
        sources = self.get_food() + self.get_flies()
        odor_ids = fun.unique_list([s.get_odor_id() for s in sources if s.get_odor_id() is not None])
        Nodors = len(odor_ids)
        odor_colors = fun.N_colors(Nodors, as_rgb=True)
        layers = {}
        odorscape = pars['odorscape']
        for i, (id, color) in enumerate(zip(odor_ids, odor_colors)):
            od_sources = [f for f in sources if f.get_odor_id() == id]
            temp = list(set([s.default_color for s in od_sources]))
            default_color = temp[0] if len(temp) == 1 else color
            kwargs = {
                'unique_id': id,
                'sources': od_sources,
                'default_color': default_color,
                'space_range': self.space_edges_for_screen,
            }
            if odorscape == 'Diffusion':
                layers[id] = DiffusionValueLayer(dt=self.dt, scaling_factor=self.scaling_factor,
                                                 grid_dims=pars['grid_dims'],
                                                 evap_const=pars['evap_const'],
                                                 gaussian_sigma=pars['gaussian_sigma'],
                                                 **kwargs)
            elif odorscape == 'Gaussian':
                layers[id] = GaussianValueLayer(**kwargs)
        return Nodors, layers

    def _generate_larva_pars(self, N, larva_pars, parameter_dict={}):
        if larva_pars['brain']['intermitter_params']:
            for dist in ['pause_dist', 'stridechain_dist']:
                if larva_pars['brain']['intermitter_params'][dist] == 'fit':
                    larva_pars['brain']['intermitter_params'][dist] = get_ref_bout_distros(dist)
        flat_larva_pars = fun.flatten_dict(larva_pars)
        sample_pars = [p for p in flat_larva_pars if flat_larva_pars[p] == 'sample']
        if len(sample_pars) >= 1:
            pars, samples = sample_agents(pars=sample_pars, N=N)

            all_larva_pars = []
            for i in range(N):
                l = copy.deepcopy(larva_pars)
                flat_l = fun.flatten_dict(l)
                for p, s in zip(pars, samples):
                    flat_l.update({p: s[i]})
                all_larva_pars.append(unflatten(flat_l))
        else :
            all_larva_pars=[larva_pars]*N

        for k, vs in parameter_dict.items():
            for l, v in zip(all_larva_pars, vs):
                l[k].update(v)
        return all_larva_pars

    def create_larvae(self, larva_pars, parameter_dict={}):
        for group_id, group_pars in larva_pars.items():
            N=group_pars['N']
            a1, a2 = np.deg2rad(group_pars['orientation_range'])
            orientations = np.random.uniform(low=a1, high=a2, size=N).tolist()
            positions = fun.generate_xy_distro(N=N, **{k:group_pars[k] for k in ['mode', 'shape', 'loc', 'scale']})
            all_pars = self._generate_larva_pars(N, group_pars['model'], parameter_dict=parameter_dict)

            for i, (p, o, pars) in enumerate(zip(positions, orientations, all_pars)):
                self.add_larva(position=p, orientation=o, id=f'{group_id}_{i}', pars=pars, group=group_id,
                               default_color=group_pars['default_color'])
                # print(pars['brain']['olfactor_params'])
                # print(i, pars['brain']['olfactor_params']['odor_dict']['CS']['mean'])
            # raise
            # self._place_larvae(positions, orientations, ids, all_pars, group=group_id)

    def step(self):

        # Tick sim_clock
        self.sim_clock.tick_clock()
        self.Nticks += 1

        if len(self.sim_starvation_hours) > 0:
            self.starvation = self.sim_clock.timer_on
            if self.sim_clock.timer_opened:
                # print('Starvation period starts!')
                if self.food_grid is not None:
                    self.food_grid.empty_grid()
            if self.sim_clock.timer_closed:
                # print('Starvation period ended!')
                if self.food_grid is not None:
                    self.food_grid.reset()

        if not self.larva_collisions:
            self.larva_bodies = self.get_larva_bodies()
        # Update value_layers
        for id, layer in self.odor_layers.items():
            layer.update_values()  # Currently doing something only for the DiffusionValueLayer

        for l in self.get_flies():
            l.compute_next_action()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
        if self.physics_engine:
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            self.update_trajectories(self.get_flies())
        self.larva_step_col.collect(self)
        # if self.table_collector is not None and self.Nticks%(int(3*60/self.dt))==0 :
        #     for name, table in self.table_collector.tables.items() :
        #         cols=list(table.keys())
        #         for l in self.get_flies() :
        #             self.table_collector.add_table_row(table_name=name, row={col : getattr(l, col) for col in cols})

        self.check_end_condition()

    def update_trajectories(self, flies):
        for fly in flies:
            fly.update_trajectory()

    def space_to_mm(self, array):
        return array * 1000 / self.scaling_factor

    def plot_odorscape(self, save_to=None, show=False):
        from lib.anal.plotting import plot_surface
        for id, layer in self.odor_layers.items():
            title = f'{id} odorscape'
            X, Y = layer.meshgrid
            V = layer.get_grid()
            x = self.space_to_mm(X)
            y = self.space_to_mm(Y)
            plot_surface(x=x, y=y, z=V,
                         labels=[r'x $(mm)$', r'y $(mm)$', r'concentration $(μM)$'], title=title,
                         save_to=save_to, save_as=f'{id}_odorscape_{self.odorscape_counter}', show=show)

    def get_larva_bodies(self, scale=1.0):
        larva_bodies = {}
        for l in self.get_flies():
            larva_bodies[l.unique_id] = l.get_polygon(scale=scale)
        return larva_bodies

    def larva_bodies_except(self, id):
        return {k: v for k, v in self.larva_bodies.items() if k != id}

    def detect_collisions(self, id):
        body = self.larva_bodies[id]
        ids = []
        for id0, body0 in self.larva_bodies_except(id).items():
            if body.intersects(body0):
                ids.append(id0)
        return ids

    def collisions_exist(self, scale=1.0):
        self.larva_bodies = self.get_larva_bodies(scale=scale)
        for l in self.get_flies():
            ids = self.detect_collisions(l.unique_id)
            if len(ids) > 0:
                return True
        return False

    def create_data_collectors(self, collected_pars):
        self.larva_step_col = TargetedDataCollector(schedule_id='active_larva_schedule', mode='step',
                                                    pars=collected_pars['step'])

        self.larva_end_col = TargetedDataCollector(schedule_id='active_larva_schedule', mode='endpoint',
                                                   pars=collected_pars['endpoint'])

        self.food_end_col = TargetedDataCollector(schedule_id='all_food_schedule', mode='endpoint',
                                                  pars=['initial_amount', 'final_amount'])

        self.table_collector = DataCollector(tables=collected_pars['tables']) if len(collected_pars['tables'])>0 else None

    def eliminate_overlap(self):
        scale = 3.0
        while self.collisions_exist(scale=scale):
            self.larva_bodies = self.get_larva_bodies(scale=scale)
            for l in self.get_flies():
                dx, dy = np.random.randn(2) * l.sim_length / 10
                overlap = True
                while overlap:
                    ids = self.detect_collisions(l.unique_id)
                    if len(ids) > 0:
                        l.move_body(dx, dy)
                        self.larva_bodies[l.unique_id] = l.get_polygon(scale=scale)
                    else:
                        # overlap=False
                        break


class LarvaWorldReplay(LarvaWorld):
    def __init__(self, step_data, endpoint_data, dataset=None, pos_xy_pars=[],
                 id='Unnamed_Replay', draw_Nsegs=None, experiment='replay', **kwargs):

        super().__init__(id=id, experiment=experiment, **kwargs)

        self.dataset = dataset
        self.pos_pars = pos_xy_pars
        self.draw_Nsegs = draw_Nsegs

        self.step_data = step_data
        self.endpoint_data = endpoint_data
        self.agent_ids = self.step_data.index.unique('AgentID').values
        self.num_agents = len(self.agent_ids)

        self.available_pars = [p for p in self.step_data.columns.values if
                               p not in fun.flatten_list(self.dataset.contour_xy) + fun.flatten_list(
                                   self.dataset.points_xy)]

        # self.starting_tick = self.step_data.index.unique('Step')[0]
        try:
            self.lengths = self.endpoint_data['length'].values
        except:
            self.lengths = np.ones(self.num_agents) * 5

        self.pars = self.step_data.columns.values
        self.mid_pars = [p for p in fun.flatten_list(dataset.points_xy) if p in self.pars]
        self.Npoints = int(len(self.mid_pars) / 2)

        self.con_pars = [p for p in fun.flatten_list(dataset.contour_xy) if p in self.pars]
        self.Ncontour = int(len(self.con_pars) / 2)

        self.cen_pars = [p for p in dataset.cent_xy if p in self.pars]

        Nsegs = self.draw_Nsegs
        if Nsegs is not None:
            if Nsegs == self.Npoints - 1:
                self.or_pars = [p for p in nam.orient(dataset.segs) if p in self.pars]
                self.Nors = len(self.or_pars)
                self.angle_pars = []
                self.Nangles = 0
                if self.Nors != Nsegs:
                    raise ValueError(
                        f'Orientation values are not present for all body segments : {self.Nors} of {Nsegs}')
            elif Nsegs == 2:

                self.or_pars = [p for p in ['front_orientation'] if p in self.pars]
                self.Nors = len(self.or_pars)
                self.angle_pars = [p for p in ['bend'] if p in self.pars]
                self.Nangles = len(self.angle_pars)
                if self.Nors != 1 or self.Nangles != 1:
                    raise ValueError(
                        f'{self.Nors} orientation and {Nsegs} angle values are present and 1,1 are needed.')
            else:
                raise ValueError(f'Defined number of segments {Nsegs} must be either 2 or {self.Npoints - 1}')
        else:
            self.Nors, self.Nangles = 0, 0
            self.angle_pars, self.or_pars = [], []

        self.create_flies()

        if 'food_params' in self.env_pars.keys():
            self._place_food(self.env_pars['place_params']['initial_num_food'],
                             self.env_pars['place_params']['initial_food_positions'],
                             food_pars=self.env_pars['food_params'])

    def create_flies(self):
        for i, id in enumerate(self.agent_ids):
            data = self.step_data.xs(id, level='AgentID', drop_level=True)
            f = LarvaReplay(model=self, unique_id=id, length=self.lengths[i], data=data)
            self.active_larva_schedule.add(f)
            self.space.place_agent(f, (0, 0))

    def step(self):
        self.Nticks += 1
        # Tick sim_clock
        self.sim_clock.tick_clock()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
