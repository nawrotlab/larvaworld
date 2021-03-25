import copy
import random
import time
import warnings
import numpy as np
import progressbar
import os
from typing import List, Any, Optional

import lib.conf.sim_modes
from lib.model.envs._maze import Border
from lib.aux import naming as nam
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from mesa.datacollection import DataCollector
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, LineString
from unflatten import unflatten
from Box2D import b2World, b2ChainShape, b2Vec2, b2EdgeShape
from mesa.space import ContinuousSpace
from mesa import Model, Agent
from mesa.time import RandomActivation

from lib.aux.collecting import TargetedDataCollector
from lib.model.envs._space import GaussianValueLayer, DiffusionValueLayer, ValueGrid
from lib.model.agents._larva import LarvaSim, LarvaReplay
from lib.model.agents._agent import Food
from lib.anal.plotting import plot_surface
from lib.aux.rendering import SimulationState, InputBox
from lib.aux import rendering
import lib.aux.functions as fun
from lib.aux.rendering import SimulationClock, SimulationScale, draw_velocity_arrow, draw_trajectories
from lib.aux.sampling import sample_agents, get_ref_bout_distros
import lib.sim.gui_lib as gui
from lib.conf.sim_modes import agent_pars


class LarvaWorld:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:

        pygame.init()
        max_screen_height = pygame.display.Info().current_h
        cls.sim_screen_dim = int(max_screen_height * 2 / 3)
        """Create a new model object and instantiate its RNG automatically."""
        cls._seed = kwargs.get("seed", None)
        cls.random = random.Random(cls._seed)
        return object.__new__(cls)

    def __init__(self, env_params, fly_params=None, id='unnamed', dt=0.1, Nsteps=None, save_to='.',
                 background_motion=None,
                 use_background=False, black_background=False, mode='video', image_mode='final', media_name=None,
                 trajectories=True, trail_decay_in_sec=0.0, trajectory_colors=None, visible_state=True,
                 random_larva_colors=False, color_behavior=False, draw_head=False, draw_contour=True,
                 draw_centroid=False, draw_midline=True, show_display=True, video_speed=None,
                 snapshot_interval_in_sec=60 * 60 * 10, touch_sensors=False, allow_clicks=True, visible_clock=True,
                 *args: [], **kwargs: {'seed': 1}):
        # super().__init__(*args, **kwargs)
        self.visible_ids = False
        self.visible_state = visible_state
        self.visible_clock = visible_clock
        self.selected_type = 'Food'

        self.borders, self.border_xy, self.border_lines, self.border_bodies = [], [], [], []

        self.mousebuttondown_time = None
        self.mousebuttonup_time = None
        self.mousebuttondown_pos = None
        self.mousebuttonup_pos = None

        self.input_box = InputBox()
        self.selected_agents = []
        self.is_running = False
        self.dt = dt
        if video_speed is None:
            self.video_fps = int(1 / dt)
        else:
            self.video_fps = int(video_speed / dt)
        self.allow_clicks = allow_clicks
        self.touch_sensors = touch_sensors
        self.show_display = show_display

        self.Nsteps = Nsteps
        self.snapshot_interval = int(snapshot_interval_in_sec / dt)
        self.id = id

        self._screen = None
        self.mode = mode
        self.image_mode = image_mode

        os.makedirs(save_to, exist_ok=True)
        if media_name:
            self.media_name = os.path.join(save_to, media_name)
        else:
            self.media_name = os.path.join(save_to, self.id)

        self.trajectories = trajectories
        self.trajectory_colors = trajectory_colors
        self.trail_decay_in_ticks = int(trail_decay_in_sec / self.dt)

        self.random_larva_colors = random_larva_colors
        self.color_behavior = color_behavior

        self.draw_head = draw_head
        self.draw_contour = draw_contour
        self.draw_centroid = draw_centroid
        self.draw_midline = draw_midline

        # if background_motion is None:
        #     self.background_motion = np.zeros((3, self.Nsteps))
        # else:
        #     self.background_motion = background_motion
        self.background_motion = background_motion
        self.use_background = use_background
        self.black_background = black_background
        if self.black_background:
            self.tank_color = (0, 0, 0)
            self.screen_color = (50, 50, 50)
            self.scale_clock_color = (255, 255, 255)
            self.default_larva_color = np.array([255, 255, 255])
        else:
            self.tank_color = (255, 255, 255)
            self.screen_color = (200, 200, 200)
            self.scale_clock_color = (0, 0, 0)
            self.default_larva_color = np.array([0, 0, 0])
        self.selection_color = np.array([255, 0, 0])
        self.env_pars = env_params
        self.larva_pars = fly_params

        self.snapshot_counter = 0
        self.food_grid = None

        # Add mesa schecule to use datacollector class

        self.create_schedules()
        self.create_arena(**self.env_pars['arena_params'])
        self.space = self.create_space(**self.env_pars['space_params'])
        if 'border_list' in self.env_pars.keys():
            for border_pars in self.env_pars['border_list'] :
                b = Border(model=self, **border_pars)
                self.add_border(b)

        self.sim_clock = SimulationClock(self.dt, color=self.scale_clock_color)
        self.sim_scale = SimulationScale(self.arena_dims[0], self.scaling_factor,
                                         color=self.scale_clock_color)
        self.sim_state = SimulationState(model=self, color=self.scale_clock_color)

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

    def create_space(self, physics_engine, scaling_factor):
        self.physics_engine = physics_engine

        if scaling_factor is None:
            scaling_factor = 1.0
        self.scaling_factor = scaling_factor
        self.space_dims = self.arena_dims * self.scaling_factor
        self.space_edges = [(x * scaling_factor, y * scaling_factor) for (x, y) in self.unscaled_space_edges]
        self.space_edges_for_screen = self.unscaled_space_edges_for_screen * scaling_factor
        self.tank_shape = self.unscaled_tank_shape * scaling_factor

        # print(self.space_edges)
        # print(type(self.space_edges))
        # print(len(self.space_edges))

        if physics_engine:
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
            # space = LimitedSpace(x_max=self.space_x_range[1], y_max=self.space_y_range[1],
            #                      torus=False, x_min=self.space_x_range[0], y_min=self.space_y_range[0])
        return space

    def create_schedules(self):
        self.active_larva_schedule = RandomActivation(self)
        self.all_larva_schedule = RandomActivation(self)
        self.active_food_schedule = RandomActivation(self)
        self.all_food_schedule = RandomActivation(self)

    def destroy(self):
        self.is_running = False
        del self.active_food_schedule
        del self.active_larva_schedule
        if self._screen is not None:
            self._screen.close()
            self._screen = None
        pygame.quit()

    def delete(self, agent):
        if type(agent) is LarvaSim:
            self.active_larva_schedule.remove(agent)
        elif type(agent) is Food:
            self.active_food_schedule.remove(agent)
        elif type(agent) is Border:
            self.borders.remove(agent)
            agent.delete()

    def close(self):
        self.destroy()

    def get_flies(self) -> List[Agent]:
        return self.active_larva_schedule.agents

    def get_food(self) -> List[Agent]:
        # print(self.active_food_schedule.agents)
        return self.active_food_schedule.agents

    def seed(self, seed=None):
        if seed is not None:
            self.__seed = seed
        return [self.__seed]

    def reset(self):
        self.destroy()
        self.create_schedules()
        self.populate_space(self.env_pars)

        if self.physics_engine:
            # step to resolve
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            self.space.ClearForces()

    def get_fly_positions(self):
        return np.array([g.get_position() for g in self.get_flies()])

    def generate_larva_color(self):
        if self.random_larva_colors:
            color = fun.random_colors(1)[0]
        else:
            color = self.default_larva_color
        return color

    def set_background(self):
        if self.use_background:
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(ROOT_DIR, 'background.png')
            print('Loading background image from', path)
            self.bgimage = pygame.image.load(path)
            self.bgimagerect = self.bgimage.get_rect()
            self.tw = self.bgimage.get_width()
            self.th = self.bgimage.get_height()
            self.th_max = int(self._screen._window.get_height() / self.th) + 2
            self.tw_max = int(self._screen._window.get_width() / self.tw) + 2
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
        if self.visible_clock:
            self.sim_clock.draw_clock(screen)
        self.sim_scale.draw_scale(screen)
        if self.visible_state:
            self.sim_state.draw_state(screen)
        self.input_box.draw(screen)

    def draw_arena(self, screen):
        screen.set_bounds(*self.space_edges_for_screen)
        screen.draw_polygon(self.space_edges, color=self.screen_color)
        screen.draw_polygon(self.tank_shape, color=self.tank_color)
        for i, b in enumerate(self.borders):
            b.draw(screen)
        # for i,b in enumerate(self.border_xy):

    def render_aux(self):
        if self.visible_clock:
            self.sim_clock.render_clock(self.screen_width, self.screen_height)
        if self.visible_state:
            self.sim_scale.render_scale(self.screen_width, self.screen_height)
        self.sim_state.render_state(self.screen_width, self.screen_height)

    def render(self, velocity_arrows=False, tick=None):

        if self.background_motion is None or tick is None:
            background_motion = [0, 0, 0]
        else:
            background_motion = self.background_motion[:, tick]
        if self._screen is None:
            # caption = self.spec.id if self.spec else ""
            if self.mode == 'video':
                self._video_path = f'{self.media_name}.mp4'
            else:
                self._video_path = None
            if self.mode == 'image':
                self._image_path = f'{self.media_name}_{self.snapshot_counter}.png'
            else:
                self._image_path = None

            self._screen = rendering.GuppiesViewer(self.screen_width, self.screen_height, caption=self.id,
                                                   fps=self.video_fps, dt=self.dt, show_display=self.show_display,
                                                   record_video_to=self._video_path,
                                                   record_image_to=self._image_path)
            self.render_aux()
            self.set_background()
            self.draw_arena(self._screen)
            self.draw_background(self._screen, background_motion)
            print('Screen opened')
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None
            self.is_running = False
            return None

        if self.image_mode != 'overlap':
            self.draw_arena(self._screen)
            self.draw_background(self._screen, background_motion)

        if self.food_grid:
            self.food_grid.draw(self._screen)
        for o in self.get_food():
            o.draw(self._screen)
            o.id_box.draw(self._screen)

        for g in self.get_flies():
            g.draw(self._screen)
            g.id_box.draw(self._screen)
            # render velocity arrows
            if velocity_arrows:
                draw_velocity_arrow(self._screen, g)

        if self.trajectories:
            draw_trajectories(space_dims=self.space_dims, agents=self.get_flies(), screen=self._screen,
                              decay_in_ticks=self.trail_decay_in_ticks, trajectory_colors=self.trajectory_colors)

        self.evaluate_input()
        if self.image_mode != 'overlap':
            self.draw_aux(self._screen)
            self._screen.render()
            # return image

    def screen2space_pos(self, pos):
        p = (2 * pos[0] / self.screen_width - 1), -(2 * pos[1] / self.screen_height - 1)
        pp = p[0] * self.space_dims[0] / 2, p[1] * self.space_dims[1] / 2
        return pp

    def space2screen_pos(self, pos):
        if pos is None or any(np.isnan(pos)):
            return None
        try :
            return self._screen._transform(pos)
        except :
            p = pos[0] * 2 / self.space_dims[0], pos[1] * 2 / self.space_dims[1]
            pp = ((p[0] + 1) * self.screen_width / 2, (-p[1] + 1) * self.screen_height / 2)
            return pp

    def _place_food(self, N=0, positions=None, food_pars={}):
        pars = copy.deepcopy(food_pars)
        if len(pars['food_list']) == 0:
            if N == 0:
                return
            food_positions = self._generate_food_positions(N, positions)
            for i, p in enumerate(food_positions):
                self.add_food(position=p, food_pars=pars)
        else:
            for f in pars['food_list']:
                id = f['unique_id']
                position = f['pos']
                f.pop('unique_id')
                f.pop('pos')
                self.add_food(id=id, position=position, food_pars=f)

    def _generate_food_positions(self, N, positions):
        raw_food_positions = []
        if positions['mode'] == 'defined':
            raw_food_positions = positions['loc']
        elif positions['mode'] == 'uniform':
            for i in range(N):
                th = np.random.uniform(0, 2 * np.pi, 1)
                r = float(np.sqrt(np.random.uniform(0, 1, 1)))
                x = r * np.cos(th)
                y = r * np.sin(th)
                pos = (float(x), float(y))
                raw_food_positions.append(pos)
        elif positions['mode'] == 'normal':
            raw_food_positions = np.random.normal(loc=positions[1],
                                                  scale=positions[2],
                                                  size=(N, 2))
        # Scale positions to the tank dimensions
        food_positions = [(x * self.space_dims[0] / 2, y * self.space_dims[1] / 2) for (x, y) in
                          raw_food_positions]
        return food_positions

    def add_food(self, position, id=None, food_pars=None):
        if food_pars is None:
            food_pars = copy.deepcopy(self.env_pars['food_params'])
        if 'food_list' in list(food_pars.keys()):
            food_pars.pop('food_list')
        if id is None:
            id = self.next_id(type='Food')
        f = Food(unique_id=id, position=position, model=self, **food_pars)
        self.active_food_schedule.add(f)
        self.all_food_schedule.add(f)
        return f

    def add_larva(self, position, orientation=None, id=None, pars=None):
        if pars is None:
            ids, all_pars = self._generate_larva_pars(1, copy.deepcopy(self.larva_pars))
            pars = all_pars[0]
        if id is None:
            id = self.next_id(type='Larva')
        if orientation is None:
            orientation = np.random.uniform(0, 2 * np.pi, 1)[0]
        l = LarvaSim(model=self, pos=position, orientation=orientation, unique_id=id, fly_params=pars)
        self.active_larva_schedule.add(l)
        self.all_larva_schedule.add(l)
        return l

    def next_id(self, type='Food'):
        if type == 'Food':
            N = self.all_food_schedule.get_agent_count()
            return f'Food_{N}'
        elif type == 'Larva':
            N = self.all_larva_schedule.get_agent_count()
            return f'Larva_{N}'

    def run(self, Nsteps=None):
        # pygame.init()
        self.is_running = True
        if Nsteps is None:
            Nsteps = self.Nsteps
        warnings.filterwarnings('ignore')
        with progressbar.ProgressBar(max_value=Nsteps) as bar:
            if self.mode == 'video':
                for i in range(Nsteps):
                    if not self.is_running:
                        self.close()
                        return False
                    self.step()
                    # TODO Figure this out for multiple agents. Now only the first is used
                    self.render(tick=i)
                    bar.update(i)
                    # print(self.is_running)

            elif self.mode == 'image':
                if self.image_mode == 'snapshots':
                    for i in range(Nsteps):
                        self.step()
                        if (self.active_larva_schedule.time - 1) % self.snapshot_interval == 0:
                            self.snapshot_counter += 1
                            self.render()
                            self._screen.close()
                            self._screen = None
                        bar.update(i)
                elif self.image_mode == 'overlap':
                    for i in range(Nsteps):
                        self.step()
                        self.render()
                        bar.update(i)
                    self._screen.render()
                    self._screen.close()

                elif self.image_mode == 'final':
                    if isinstance(self, LarvaWorldSim):
                        for i in range(Nsteps):
                            self.step()
                            bar.update(i)
                    elif isinstance(self, LarvaWorldReplay):
                        self.active_larva_schedule.steps = Nsteps - 1
                        self.step()
                    self.render()
            else:
                if isinstance(self, LarvaWorldSim):
                    for i in range(Nsteps):
                        self.step()
                        bar.update(i)
                elif isinstance(self, LarvaWorldReplay):
                    raise ValueError('When running a replay, set mode to video or image')
        return True

    def create_borders(self, lines, from_screen=False):
        s = self.scaling_factor
        X, Y = self.arena_dims
        if not from_screen:
            T = [s, 0, 0, s, -s * X / 2, -s * Y / 2]
        else:
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

    def evaluate_input(self):
        d_zoom=0.05
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                self._screen.close_requested()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.toggle_ids()
                elif event.key == pygame.K_t:
                    self.toggle_clock()
                elif event.key == pygame.K_s:
                    self.toggle_state()
                elif event.key == pygame.K_b:
                    self.toggle_behavior()
                elif event.key == pygame.K_m:
                    self.toggle_midline()
                elif event.key == pygame.K_c:
                    self.toggle_contour()
                elif event.key == pygame.K_h:
                    self.toggle_head()
                elif event.key == pygame.K_e:
                    self.toggle_centroid()
                # elif event.key == pygame.K_MINUS:
                #     print(self._screen._fps)
                #     self._screen._fps-=10
                #     print(self._screen._fps)
                # elif event.key == pygame.K_PLUS:
                #     print(self._screen._fps)
                #     self._screen._fps+=10
                #     print(self._screen._fps)
                elif event.key == pygame.K_LEFT:
                    self._screen.move_center(-0.05,0)
                elif event.key == pygame.K_RIGHT:
                    self._screen.move_center(+0.05,0)
                elif event.key == pygame.K_UP:
                    self._screen.move_center(0,+0.05)
                elif event.key == pygame.K_DOWN:
                    self._screen.move_center(0,-0.05)

                elif event.key == pygame.K_DELETE:
                    if gui.delete_objects_window(self.selected_agents):
                        for f in self.selected_agents:
                            self.selected_agents.remove(f)
                            self.delete(f)
            if self.allow_clicks:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mousebuttondown_pos = self._screen.get_mouse_position()
                    # self.mousebuttondown_time = time.time()
                elif event.type == pygame.MOUSEBUTTONUP:
                    # self.mousebuttonup_time = time.time()
                    # dt = self.mousebuttonup_time - self.mousebuttondown_time
                    p = self._screen.get_mouse_position()
                    if event.button == 1:
                        res = self.eval_selection(p)
                        # self.mousebuttondown_time = time.time()
                        if not res:
                            if self.selected_type == 'Food':
                                f = self.add_food(p)
                            elif self.selected_type == 'Larva':
                                f = self.add_larva(p)
                            elif self.selected_type == 'Border':
                                b = Border(model=self, points=[tuple(self.mousebuttondown_pos), tuple(p)], from_screen=True)
                                self.add_border(b)
                    elif event.button == 3:
                        if len(self.selected_agents) > 0:
                            sel = self.selected_agents[0]
                            sel = gui.set_agent_kwargs(sel)
                        else:
                            self.selected_type = gui.object_menu(self.selected_type)
                    elif event.button == 4:
                        self._screen.zoom_screen(d_zoom=-d_zoom)
                    elif event.button == 5:
                        self._screen.zoom_screen(d_zoom=+d_zoom)

                self.input_box.get_input(event)


    def get_place_params(self):
        return {**self.get_food_placement(), **self.get_larva_placement()}

    def get_image_path(self):
        return f'{self.media_name}_{self.snapshot_counter}.png'
        # return None

    def get_food_placement(self):
        Nfood = len(self.get_food())
        if Nfood > 0:
            food_loc = np.array([f.pos for f in self.get_food()]) * 2 / self.arena_dims
            food_pos = {'mode': 'defined', 'loc': food_loc}
        else:
            food_pos = None
        food_placement = {'initial_num_food': Nfood,
                          'initial_food_positions': food_pos}
        return food_placement

    def get_larva_placement(self):
        Nlarvae = len(self.get_flies())
        if Nlarvae > 0:
            larva_loc = np.array([f.current_pos for f in self.get_flies()]) * 2 / self.arena_dims
            larva_or = np.array([f.front_orientation_in_deg for f in self.get_flies()])
            larva_pos = {'mode': 'defined', 'loc': larva_loc, 'orientation': larva_or}
        else:
            larva_pos = None
        larva_placement = {'initial_num_flies': Nlarvae,
                           'initial_fly_positions': larva_pos}
        return larva_placement

    def toggle_ids(self):
        self.visible_ids = not self.visible_ids
        for a in self.get_flies() + self.get_food():
            a.id_box.visible = self.visible_ids

    def toggle_clock(self):
        self.visible_clock = not self.visible_clock

    def toggle_state(self):
        self.visible_state = not self.visible_state

    def toggle_behavior(self):
        self.color_behavior = not self.color_behavior

    def toggle_midline(self):
        self.draw_midline = not self.draw_midline

    def toggle_contour(self):
        self.draw_contour = not self.draw_contour

    def toggle_head(self):
        self.draw_head = not self.draw_head

    def toggle_centroid(self):
        self.draw_centroid = not self.draw_centroid

    def eval_selection(self, p):
        res = False
        for f in self.get_food() + self.get_flies() + self.borders:
            if f.contained(p):
                res = True
                if not f.selected:
                    f.selected = True
                    self.selected_agents.append(f)
            else:
                if f.selected:
                    res = True
                    f.selected = False
                    self.selected_agents.remove(f)
        return res

    def get_agent_list(self, class_name):
        if class_name == 'Food':
            agents = self.get_food()
        elif class_name in ['LarvaSim', 'LarvaReplay']:
            agents = self.get_flies()
        elif class_name == 'Border':
            agents = self.borders
        pars = agent_pars[class_name]
        data = []
        for f in agents:
            dic = {}
            for p in pars:
                dic[p] = getattr(f, p)
            data.append(dic)
        return data

    def add_border(self, b):
        self.borders.append(b)
        self.border_xy += b.border_xy
        self.border_lines += b.border_lines
        self.border_bodies += b.border_bodies


class LarvaWorldSim(LarvaWorld):
    def __init__(self, collected_pars=None,
                 id='Unnamed_Simulation', allow_collisions=True, count_bend_errors=False,
                 starvation_hours=[], hours_as_larva=0, deb_base_f=1, parameter_dict={},**kwargs):
        super().__init__(id=id, **kwargs)
        if collected_pars is None:
            collected_pars = {'step': [], 'endpoint': []}
        self.starvation_hours = starvation_hours
        self.hours_as_larva = hours_as_larva
        self.deb_base_f = deb_base_f
        self.deb_starvation_hours = [[s0, np.clip(s1, a_min=s0, a_max=hours_as_larva)] for [s0, s1] in
                                     self.starvation_hours if s0 < hours_as_larva]
        self.sim_starvation_hours = [[np.clip(s0 - hours_as_larva, a_min=0, a_max=+np.inf), s1 - hours_as_larva] for
                                     [s0, s1] in self.starvation_hours if s1 > hours_as_larva]
        if len(self.sim_starvation_hours) > 0:
            on_ticks = [int(s0 * 60 * 60 / self.dt) for [s0, s1] in self.sim_starvation_hours]
            off_ticks = [int(s1 * 60 * 60 / self.dt) for [s0, s1] in self.sim_starvation_hours]
            self.sim_clock.set_timer(on_ticks, off_ticks)
        self.starvation = self.sim_clock.timer_on
        self.count_bend_errors = count_bend_errors

        self.allow_collisions = allow_collisions

        self.populate_space(env_pars=self.env_pars, larva_pars=self.larva_pars, parameter_dict=parameter_dict)
        self.create_data_collectors(collected_pars)

    def populate_space(self, env_pars, larva_pars, parameter_dict={}):
        food_pars = env_pars['food_params']
        if food_pars:
            self._place_food(self.env_pars['place_params']['initial_num_food'],
                             self.env_pars['place_params']['initial_food_positions'],
                             food_pars=food_pars)
            if 'grid_pars' in list(food_pars.keys()):
                self._create_food_grid(space_range=self.space_edges_for_screen,
                                       food_pars=food_pars['grid_pars'])
        # odor_params = environment_params['odor_params']
        self.Nodors, self.odor_layers = self._create_odor_layers(odor_pars=env_pars['odor_params'])

        self.create_larvae(N=env_pars['place_params']['initial_num_flies'],
                           pos_conf=env_pars['place_params']['initial_fly_positions'],
                           larva_pars=larva_pars, parameter_dict=parameter_dict)

    def prepare_flies(self, timesteps):
        for t in range(timesteps):
            self.mock_step()
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
        if self.odor_layers:
            for i in range(timesteps):
                self.odor_layers.update_values()  # Currently doing something only for the DiffusionValueLayer

    def _create_food_grid(self, space_range, food_pars):
        if food_pars and 'grid_resolution' in food_pars:
            self.food_grid = ValueGrid(**food_pars, space_range=space_range,
                                       distribution='uniform')

    def _create_odor_layers(self, odor_pars):
        if odor_pars:
            # landscape = self.odor_params['odor_landscape']
            # odor_ids = self.food_params['odor_id_list']
            Nodors = len(odor_pars['odor_id_list'])
            layers = dict.fromkeys(odor_pars['odor_id_list'])
            odor_colors = fun.random_colors(Nodors)

            if odor_pars['odor_carriers'] == 'food':
                sources = self.get_food()
            elif odor_pars['odor_carriers'] == 'flies':
                sources = self.get_flies()
            else:
                raise ('Currently only food or larvae can be odor carriers')
            self.allocate_odors(sources,
                                odor_pars['odor_id_list'],
                                # odor_pars['odor_intensity_list'],
                                # odor_pars['odor_spread_list'],
                                odor_pars['odor_source_allocation']
                                )
            for i, (odor_id, odor_color) in enumerate(zip(odor_pars['odor_id_list'], odor_colors)):
                if odor_pars['odor_landscape'] == 'Diffusion':
                    layers[odor_id] = DiffusionValueLayer(world=self.space, unique_id=odor_id,
                                                          sources=[f for f in sources if
                                                                   f.get_odor_id() == odor_id],
                                                          world_range=[self.world_x_range,
                                                                       self.world_y_range],
                                                          grid_resolution=odor_pars[
                                                              'odor_layer_grid_resolution'],
                                                          evap_const=odor_pars['odor_evaporation_rate'],
                                                          diff_const=odor_pars['odor_diffusion_rate'], color=odor_color)
                elif odor_pars['odor_landscape'] == 'Gaussian':
                    layers[odor_id] = GaussianValueLayer(world=self.space, unique_id=odor_id,
                                                         sources=[f for f in sources if
                                                                  f.get_odor_id() == odor_id], color=odor_color)
                for f in layers[odor_id].sources:
                    f.set_default_color(layers[odor_id].color)
            return Nodors, layers
        else:
            return 0, {}

    def _generate_larva_poses(self, N, positions):
        mode = positions['mode']
        raw_larva_positions = None
        if mode == 'identical':
            raw_larva_positions = np.zeros((N, 2)) + positions['loc']
            larva_orientations = np.zeros(N) + positions['orientation']
        elif mode == 'normal':
            raw_larva_positions = np.random.normal(loc=positions['loc'], scale=positions['scale'], size=(N, 2))
            larva_orientations = np.random.rand(N) * 2 * np.pi - np.pi
        elif mode == 'facing_right':
            raw_larva_positions = np.random.normal(loc=positions['loc'], scale=positions['scale'], size=(N, 2))
            larva_orientations = np.random.rand(N) * 2 * np.pi / 6 - np.pi / 6
        elif mode == 'spiral':
            raw_larva_positions = [(0.0, 0.8)] * 8 + [(0.6, 0)] * 8 + [(0.0, -0.4)] * 8 + [(-0.2, 0.0)] * 8
            larva_orientations = [i * np.pi / 4 for i in range(8)] * 4
        elif mode == 'uniform':
            raw_larva_positions = np.random.uniform(low=-1, high=1, size=(N, 2))
            larva_orientations = np.random.rand(N) * 2 * np.pi - np.pi
        elif mode == 'uniform_circ':
            raw_larva_positions = []
            for i in range(N):
                theta = np.random.uniform(0, 2 * np.pi, 1)
                r = float(np.sqrt(np.random.uniform(0, 1, 1)))
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pos = (float(x), float(y))
                raw_larva_positions.append(pos)
                larva_orientations = np.random.rand(N) * 2 * np.pi - np.pi
        elif mode == 'defined':
            raw_larva_positions = positions['loc']
            larva_orientations = positions['orientation']
        elif mode == 'scal':
            larva_positions = positions['loc']
            larva_orientations = positions['orientation']

        # Scale positions to the tank dimensions
        if raw_larva_positions is not None:
            larva_positions = [(x * self.space_dims[0] / 2, y * self.space_dims[1] / 2) for (x, y) in
                               raw_larva_positions]
        else:
            pass
        return larva_positions, larva_orientations

    def _generate_larva_pars(self, N, larva_pars,  parameter_dict={}):
        if not isinstance(larva_pars, list):
            larva_confs = [larva_pars]
        else:
            larva_confs = larva_pars
        ls = larva_confs[:]
        Ns = [int(N / len(ls)) for i in range(len(ls) - 1)]
        Ns.append(N - sum(Ns))
        all_larva_pars = []
        larva_ids = []
        for l, n in zip(ls, Ns):
            if 'id_prefix' in l:
                id_prefix = l['id_prefix']
                # FIXME This should be deleted but then the input larva_pars is changed which causes problem in sequential simulations.
                #  Even shallow copying the list as ls does not save it!!!
                #  I had to add **kwargs to the LarvaBody class in order to just ignore this argument
                # del l['id_prefix']
            else:
                id_prefix = 'Larva'
            for i in range(n):
                larva_ids.append(f'{id_prefix}_{i}')
            for dist in ['pause_dist', 'stridechain_dist']:
                if l['neural_params']['intermitter_params'][dist] == 'fit':
                    l['neural_params']['intermitter_params'][dist] = get_ref_bout_distros(dist)
            type_larva_pars = [l] * n
            flat_larva_pars = fun.flatten_dict(l)
            sample_pars = [p for p in flat_larva_pars if flat_larva_pars[p] == 'sample']
            if len(sample_pars) >= 1:
                pars, samples = sample_agents(pars=sample_pars, N=n)
                for i, c in enumerate(type_larva_pars):
                    flat_c = fun.flatten_dict(c)
                    for p, s in zip(pars, samples):
                        flat_c.update({p: s[i]})
                    type_larva_pars[i] = unflatten(flat_c)
            all_larva_pars.append(type_larva_pars)
        all_larva_pars=fun.flatten_list(all_larva_pars)
        for k,vs in parameter_dict.items() :
            # if len(all_larva_pars)!=len(vs) :
            #     raise ValueError (f'Parameter {k} has {len(vs)} values but number of larvae is {len(all_larva_pars)}')
            for larva_pars,v in zip(all_larva_pars, vs) :
                # print(v)
                # print(v)
                larva_pars[k].update(v)
                # larva_pars['neural_params'][k]=v
                # print(list(larva_pars.keys()))
                # print(list(larva_pars['neural_params'].keys()))
                # print(list(larva_pars['neural_params']['olfactor_params'].keys()))
                # print(larva_pars['neural_params']['olfactor_params']['olfactor_gain_mean'])
        return larva_ids, all_larva_pars

    def _place_larvae(self, positions, orientations, ids, all_pars):
        for i, (p, o, id, pars) in enumerate(zip(positions, orientations, ids, all_pars)):
            self.add_larva(position=p, orientation=o, id=id, pars=pars)

    def create_larvae(self, N, pos_conf, larva_pars, parameter_dict={}):
        positions, orientations = self._generate_larva_poses(N, pos_conf)
        ids, all_pars = self._generate_larva_pars(N, larva_pars, parameter_dict=parameter_dict)
        self._place_larvae(positions, orientations, ids, all_pars)

    def allocate_odors(self, agents, odor_id_list,allocation_mode='iterative'):
        ids = self.compute_odor_parameters(len(agents), odor_id_list,
                                           # odor_intensity_list,
                                           # odor_spread_list,
                                           allocation_mode)
        for a, id in zip(agents, ids):
            a.set_odor_id(id)

    def compute_odor_parameters(self, N, odor_id_list,allocation_mode='iterative'):
        N_o = len(odor_id_list)
        # N_i = len(odor_intensity_list)
        # N_s = len(odor_spread_list)
        if allocation_mode == 'iterative':
            ids = [odor_id_list[i % N_o] for i in range(N)]
            # intensities = [odor_intensity_list[i % N_i] for i in range(N)]
            # spreads = [odor_spread_list[i % N_s] for i in range(N)]
        return ids

    def step(self):
        # Tick sim_clock
        self.sim_clock.tick_clock()

        if len(self.sim_starvation_hours) > 0:
            self.starvation = self.sim_clock.timer_on
            if self.sim_clock.timer_opened:
                # print(self.sim_clock.hour, self.sim_clock.minute)
                if self.food_grid is not None:
                    self.food_grid.empty_grid()
            if self.sim_clock.timer_closed:
                # print(self.sim_clock.hour, self.sim_clock.minute)
                # try:
                #     for l in self.get_flies():
                #         l.brain.intermitter.explore2exploit_bias = l.brain.intermitter.base_explore2exploit_bias
                # except:
                #     pass
                if self.food_grid is not None:
                    self.food_grid.reset()

        # print(self.sim_clock.dmsecond)
        # Update value_layers

        if self.odor_layers:
            for layer_id in self.odor_layers:
                self.odor_layers[layer_id].update_values()  # Currently doing something only for the DiffusionValueLayer

        if not self.allow_collisions:
            self.larva_bodies = self.get_larva_bodies()
        for l in self.get_flies():
            l.compute_next_action()

        self.active_larva_schedule.step()
        self.active_food_schedule.step()

        # step space
        if self.physics_engine:
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            self.update_trajectories(self.get_flies())
        self.larva_step_collector.collect(self)

        # self.table_collector.add_table_row(table_name='Torque', )

    def mock_step(self):
        if self.odor_layers:
            for layer_id in self.odor_layers:
                self.odor_layers[layer_id].update_values()  # Currently doing something only for the DiffusionValueLayer
        for i, g in enumerate(self.get_flies()):
            if np.random.choice([0, 1]) == 0:
                # p,o=g.get_midpoint_position()
                # FIXME now preparing only turner
                # g.compute_next_action()
                # g.step()
                try:
                    g.turner.step()
                except:
                    pass

    # update trajectories
    def update_trajectories(self, flies):
        for fly in flies:
            fly.update_trajectory()

    def space_to_mm(self, array):
        return array * 1000 / self.scaling_factor

    def plot_odorscape(self, title=False, save_to=None):

        radx = self.space_dims[0] / 2
        rady = self.space_dims[1] / 2
        delta = np.min([radx, rady]) / 50
        x = np.arange(-radx, radx, delta)
        y = np.arange(-rady, rady, delta)
        X, Y = np.meshgrid(x, y)

        @np.vectorize
        def func(a, b):
            v = layer.get_value((a, b))
            return v

        for layer_id in self.odor_layers:
            layer = self.odor_layers[layer_id]
            V = func(X, Y)
            num_sources = layer.get_num_sources()
            name = f'{layer_id} odorscape'
            plot_surface(x=self.space_to_mm(X), y=self.space_to_mm(Y), z=V,
                         labels=[r'x $(mm)$', r'y $(mm)$', r'concentration $(μM)$'], title=title,
                         save_to=save_to, save_as=f'{layer_id}_odorscape')
        # plt.figure()
        # CS = plt.contour(X, Y, V)
        # plt.clabel(CS, inline=1, fontsize=10)
        # plt.title(f'Odorant concentration landscape from {num_sources} sources ')
        # plt.show()

    def get_larva_bodies(self):
        larva_bodies = fun.flatten_list([[Polygon(v[0]) for v in l.seg_vertices] for l in self.get_flies()])
        return larva_bodies

    def create_data_collectors(self, collected_pars):
        self.larva_step_collector = TargetedDataCollector(schedule_id='active_larva_schedule', mode='step',
                                                          pars=collected_pars['step'])

        self.larva_endpoint_collector = TargetedDataCollector(schedule_id='active_larva_schedule', mode='endpoint',
                                                              pars=collected_pars['endpoint'])

        self.food_endpoint_collector = TargetedDataCollector(schedule_id='all_food_schedule', mode='endpoint',
                                                             pars=['initial_amount', 'final_amount'])

        # self.table_collector = DataCollector(tables={"Torque": ["unique_id", "torque"]})


class LarvaWorldReplay(LarvaWorld):
    def __init__(self, step_data, endpoint_data, dataset=None, pos_xy_pars=[],
                 id='Unnamed_Replay', draw_Nsegs=None, **kwargs):

        super().__init__(id=id, **kwargs)

        self.dataset = dataset
        self.pos_pars = pos_xy_pars
        self.draw_Nsegs = draw_Nsegs

        self.step_data = step_data
        self.endpoint_data = endpoint_data
        self.agent_ids = self.step_data.index.unique('AgentID').values
        self.num_agents = len(self.agent_ids)

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
                    raise ValueError(f'Orientation values are not present for all body segments : {self.Nors} of {Nsegs}')
            elif Nsegs == 2:
                self.or_pars = [p for p in ['front_orientation'] if p in self.pars]
                self.Nors = len(self.or_pars)
                self.angle_pars = [p for p in ['bend'] if p in self.pars]
                self.Nangles = len(self.angle_pars)
                if self.Nors != 1 or self.Nangles != 1:
                    raise ValueError(f'{self.Nors} orientation and {Nsegs} angle values are present and 1,1 are needed.')
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
            f = LarvaReplay(model=self, unique_id=id,length=self.lengths[i],data=data)
            self.active_larva_schedule.add(f)
            self.space.place_agent(f, (0, 0))

    def step(self):
        # Tick sim_clock
        self.sim_clock.tick_clock()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
