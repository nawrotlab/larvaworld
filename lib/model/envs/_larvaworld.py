import warnings

import numpy as np
import progressbar
import pygame
import os
from Box2D import b2World, b2ChainShape
from mesa.space import ContinuousSpace
from unflatten import unflatten

from lib.aux.collecting import TargetedDataCollector
from lib.model.envs._space import GaussianValueLayer, DiffusionValueLayer, ValueGrid
from lib.model.larva._larva import LarvaSim
from lib.model.envs._food import Food
from lib.anal.plotting import plot_surface
from lib.aux.rendering import SimulationState
from lib.aux import rendering
from gym.envs.registration import EnvSpec
from mesa import Model
from mesa.time import RandomActivation

from lib.model.larva._larva import LarvaReplay
import lib.aux.functions as fun
from lib.aux.rendering import SimulationClock, SimulationScale, draw_velocity_arrow, draw_trajectories
from lib.aux.sampling import sample_agents, get_ref_bout_distros

pygame.init()
max_screen_height = pygame.display.Info().current_h
sim_screen_dim = int(max_screen_height * 2 / 3)

class LarvaWorld(Model):
    def __init__(self, id, dt,
                 env_params, Nsteps, save_to,
                 background_motion=None, use_background=False, black_background=False,
                 mode='video', image_mode='final', media_name=None,
                 trajectories=True, trail_decay_in_sec=None, trajectory_colors=None,
                 show_state=True, random_larva_colors=False, color_behavior=False,
                 draw_head=False, draw_contour=True, draw_centroid=False, draw_midline=True,
                 show_display=True, video_fps=None, snapshot_interval_in_sec=20):

        self.dt = dt
        if video_fps is None :
            self.video_fps=int(1/dt)
        else :
            self.video_fps = int(video_fps/dt)

        self.show_display=show_display
        self.sim_screen_dim = sim_screen_dim
        self.Nsteps = Nsteps
        self.snapshot_interval = int(snapshot_interval_in_sec / self.dt)

        self.id = id
        self.spec = EnvSpec(id=f'{id}-v0')

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

        if trail_decay_in_sec is None:
            self.trail_decay_in_ticks = None
        else:
            self.trail_decay_in_ticks = int(trail_decay_in_sec / self.dt)

        self.show_state = show_state
        self.random_larva_colors = random_larva_colors
        self.color_behavior = color_behavior

        self.draw_head = draw_head
        self.draw_contour = draw_contour
        self.draw_centroid = draw_centroid
        self.draw_midline = draw_midline

        if background_motion is None:
            self.background_motion = np.zeros((3, self.Nsteps))
        else:
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

        self.env_pars = env_params

        self.snapshot_counter = 0
        self.food_grid = None

        # Add mesa schecule to use datacollector class

        self.create_schedules()
        self.create_arena(**self.env_pars['arena_params'])
        self.space = self.create_space(**self.env_pars['space_params'])
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
        del self.active_food_schedule
        del self.active_larva_schedule
        if self._screen is not None:
            self._screen.close()
            self._screen = None

    def delete(self, agent):
        if type(agent) is LarvaSim:
            self.active_larva_schedule.remove(agent)
        elif type(agent) is Food:
            self.active_food_schedule.remove(agent)

    def close(self):
        self.destroy()

    def get_flies(self):
        return self.active_larva_schedule.agents

    def get_food(self):
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
        self.sim_clock.draw_clock(screen)
        self.sim_scale.draw_scale(screen)
        if self.show_state:
            self.sim_state.draw_state(screen)

    def draw_arena(self, screen):
        screen.set_bounds(*self.space_edges_for_screen)
        screen.draw_polygon(self.space_edges, color=self.screen_color)
        screen.draw_polygon(self.tank_shape, color=self.tank_color)

    def render_aux(self):
        self.sim_clock.render_clock(self.screen_width, self.screen_height)
        self.sim_scale.render_scale(self.screen_width, self.screen_height)
        self.sim_state.render_state(self.screen_width, self.screen_height)

    def render(self, velocity_arrows=False, background_motion=[0, 0, 0]):

        if self._screen is None:
            caption = self.spec.id if self.spec else ""
            if self.mode == 'video':
                _video_path=f'{self.media_name}.mp4'
            else:
                _video_path = None
            if self.mode == 'image':
                _image_path = f'{self.media_name}_{self.snapshot_counter}.png'
            else:
                _image_path = None

            self._screen = rendering.GuppiesViewer(self.screen_width, self.screen_height, caption=caption,
                                                   fps=self.video_fps, dt=self.dt, display=self.show_display,
                                                   record_video_to=_video_path,
                                                   record_image_to=_image_path)
            self.render_aux()
            self.set_background()
            self.draw_arena(self._screen)
            self.draw_background(self._screen, background_motion)
            print('Screen opened')
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None


        if self.image_mode != 'overlap':
            self.draw_arena(self._screen)
            self.draw_background(self._screen, background_motion)

        if self.food_grid:
            self.food_grid.draw(self._screen)
        # render food
        for o in self.get_food():
            o.draw(self._screen)

        for g in self.get_flies():
            g.draw(self._screen)
            # render velocity arrows
            if velocity_arrows:
                draw_velocity_arrow(self._screen, g)



        if self.trajectories:
            draw_trajectories(space_dims=self.space_dims, agents=self.get_flies(), screen=self._screen,
                              decay_in_ticks=self.trail_decay_in_ticks, trajectory_colors=self.trajectory_colors)
        if self.image_mode != 'overlap':
            self.draw_aux(self._screen)
            self._screen.render()



    def _place_food(self, num_food, positions, food_params):
        # num_food = self.place_params['initial_num_ood']
        if num_food == 0:
            return
        # positions = self.place_params['initial_food_positions']
        # Food positions in the range [-1,1] for x and y
        unscaled_food_positions = []
        if positions['mode'] == 'defined':
            unscaled_food_positions = positions['loc']
        elif positions['mode'] == 'uniform':
            for i in range(num_food):
                theta = np.random.uniform(0, 2 * np.pi, 1)
                r = float(np.sqrt(np.random.uniform(0, 1, 1)))
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pos = (float(x), float(y))
                unscaled_food_positions.append(pos)
        elif positions['mode'] == 'normal':
            # base_pos = tuple([self.tank_radius * x for x in initial_food_positions[1]])
            # food_positions = np.random.normal(loc=base_pos, scale=initial_food_positions[2] * self.tank_radius,
            #                                   size=(initial_num_food, 2))
            unscaled_food_positions = np.random.normal(loc=positions[1],
                                                       scale=positions[2],
                                                       size=(num_food, 2))
        # Scale positions to the tank dimensions
        food_positions = [(x * self.space_dims[0] / 2, y * self.space_dims[1] / 2) for (x, y) in
                          unscaled_food_positions]
        food_ids = [f'Food_{i}' for i in range(num_food)]
        for i, p in enumerate(food_positions):
            f = Food(unique_id=food_ids[i], model=self, position=p,
                     amount=food_params['amount'],
                     shape_radius=food_params['shape_radius'])
            self.active_food_schedule.add(f)
            self.all_food_schedule.add(f)

    def run(self, Nsteps=None):
        if Nsteps is None:
            Nsteps = self.Nsteps
        warnings.filterwarnings('ignore')
        with progressbar.ProgressBar(max_value=Nsteps) as bar:
            if self.mode == 'video':
                for i in range(Nsteps):
                    self.step()
                    # TODO Figure this out for multiple agents. Now only the first is used
                    self.render(background_motion=self.background_motion[:, i])
                    bar.update(i)
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


class LarvaWorldSim(LarvaWorld):
    def __init__(self, fly_params,
                 collected_pars={'step': [], 'endpoint': []},
                 id='Unnamed_Simulation',
                 **kwargs):

        super().__init__(id=id,
                         **kwargs)

        self.fly_params = fly_params
        for dist in ['pause_dist', 'stridechain_dist'] :
            if self.fly_params['neural_params']['intermitter_params'][dist]=='fit' :
                self.fly_params['neural_params']['intermitter_params'][dist] = get_ref_bout_distros(dist)

        self.odor_layers = {}
        self.Nodors=0

        self.populate_space(self.env_pars)

        self.larva_step_collector = TargetedDataCollector(target_schedule='active_larva_schedule', mode='step',
                                                          pars=collected_pars['step'])
        self.larva_endpoint_collector = TargetedDataCollector(target_schedule='active_larva_schedule', mode='endpoint',
                                                              pars=collected_pars['endpoint'])

        self.food_endpoint_collector = TargetedDataCollector(target_schedule='all_food_schedule', mode='endpoint',
                                                             pars=['initial_amount', 'final_amount'])

    def populate_space(self, environment_params):
        food_params = environment_params['food_params']
        if food_params:
            self._place_food(environment_params['place_params']['initial_num_food'],
                             environment_params['place_params']['initial_food_positions'],
                             food_params=food_params)
            self._create_food_grid(space_range=self.space_edges_for_screen,
                                   food_params=food_params)
        odor_params = environment_params['odor_params']
        if odor_params:
            self._create_odor_layers(odor_params=odor_params)
        self._place_flies(environment_params['place_params']['initial_num_flies'],
                          environment_params['place_params']['initial_fly_positions'])

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

    def _create_food_grid(self, space_range, food_params):
        if food_params and 'grid_resolution' in food_params:
            self.food_grid = ValueGrid(**food_params, space_range=space_range,
                                       distribution='uniform')

    def _create_odor_layers(self, odor_params):
        if odor_params:
            # landscape = self.odor_params['odor_landscape']
            # odor_ids = self.food_params['odor_id_list']
            self.Nodors = len(odor_params['odor_id_list'])
            self.odor_layers = dict.fromkeys(odor_params['odor_id_list'])
            if odor_params['odor_carriers'] == 'food':
                sources = self.get_food()
            elif odor_params['odor_carriers'] == 'flies':
                sources = self.get_flies()
            else:
                raise ('Currently only food or flies can be odor carriers')
            self.allocate_odor_parameters(sources, odor_params['odor_id_list'],
                                          odor_params['odor_intensity_list'],
                                          odor_params['odor_spread_list'],
                                          odor_params['odor_source_allocation'])
            for i, odor_id in enumerate(odor_params['odor_id_list']):
                if odor_params['odor_landscape'] == 'Diffusion':
                    self.odor_layers[odor_id] = DiffusionValueLayer(world=self.space, unique_id=odor_id,
                                                                    sources=[f for f in sources if
                                                                             f.get_odor_id() == odor_id],
                                                                    world_range=[self.world_x_range,
                                                                                 self.world_y_range],
                                                                    grid_resolution=odor_params[
                                                                        'odor_layer_grid_resolution'],
                                                                    evap_const=odor_params['odor_evaporation_rate'],
                                                                    diff_const=odor_params['odor_diffusion_rate'])
                elif odor_params['odor_landscape'] == 'Gaussian':
                    self.odor_layers[odor_id] = GaussianValueLayer(world=self.space, unique_id=odor_id,
                                                                   sources=[f for f in sources if
                                                                            f.get_odor_id() == odor_id])

    def _place_flies(self, num_flies, positions):
        mode = positions['mode']
        unscaled_fly_positions = None
        if mode == 'identical':
            unscaled_fly_positions = np.zeros((num_flies, 2)) + positions['loc']
            fly_orientations = np.zeros(num_flies) + positions['orientation']
        elif mode == 'normal':
            unscaled_fly_positions = np.random.normal(loc=positions['loc'], scale=positions['scale'],
                                                      size=(num_flies, 2))
            fly_orientations = np.random.rand(num_flies) * 2 * np.pi - np.pi
        elif mode == 'facing_right':
            unscaled_fly_positions = np.random.normal(loc=positions['loc'], scale=positions['scale'],
                                                      size=(num_flies, 2))
            fly_orientations = np.random.rand(num_flies) * 2 * np.pi / 6 - np.pi / 6
        elif mode == 'spiral':
            unscaled_fly_positions = [(0.0, 0.8)] * 8 + [(0.6, 0)] * 8 + [(0.0, -0.4)] * 8 + [(-0.2, 0.0)] * 8
            fly_orientations = [i * np.pi / 4 for i in range(8)] * 4
        elif mode == 'uniform':
            unscaled_fly_positions = np.random.uniform(low=-1, high=1, size=(num_flies, 2))
            fly_orientations = np.random.rand(num_flies) * 2 * np.pi - np.pi
        elif mode == 'uniform_circ':
            unscaled_fly_positions = []
            for i in range(num_flies):
                theta = np.random.uniform(0, 2 * np.pi, 1)
                r = float(np.sqrt(np.random.uniform(0, 1, 1)))
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pos = (float(x), float(y))
                unscaled_fly_positions.append(pos)
                fly_orientations = np.random.rand(num_flies) * 2 * np.pi - np.pi
        elif mode == 'defined':
            unscaled_fly_positions = positions['loc']
            fly_orientations = positions['orientation']
        elif mode == 'scal':
            fly_positions = positions['loc']
            fly_orientations = positions['orientation']

        # Scale positions to the tank dimensions
        if unscaled_fly_positions is not None:
            fly_positions = [(x * self.space_dims[0] / 2, y * self.space_dims[1] / 2) for (x, y) in
                             unscaled_fly_positions]
        else:
            pass

        fly_ids = [f'Larva_{i}' for i in range(num_flies)]
        each_fly_params = [self.fly_params for i in range(num_flies)]

        flat_fly_params = fun.flatten_dict(self.fly_params)
        sample_pars = [p for p in flat_fly_params if flat_fly_params[p] == 'sample']
        if len(sample_pars) >= 1:
            pars, samples = sample_agents(pars=sample_pars, num_agents=num_flies)
            # print(f'Sampling parameters {parameters} from sample file')

            for i, config in enumerate(each_fly_params):
                flat_config = fun.flatten_dict(config)
                for p, s in zip(pars, samples):
                    flat_config.update({p: s[i]})
                config = unflatten(flat_config)
                each_fly_params[i] = config
        for i, (p, o, single_fly_params) in enumerate(zip(fly_positions, fly_orientations, each_fly_params)):
            f = LarvaSim(model=self, pos=p, orientation=o, unique_id=fly_ids[i], fly_params=single_fly_params)
            self.active_larva_schedule.add(f)
            # print(self.active_larva_schedule.agents)
            self.all_larva_schedule.add(f)

    def allocate_odor_parameters(self, agents, odor_id_list, odor_intensity_list, odor_spread_list,
                                 allocation_mode='iterative'):
        ids, intensities, spreads = self.compute_odor_parameters(len(agents), odor_id_list,
                                                                 odor_intensity_list,
                                                                 odor_spread_list,
                                                                 allocation_mode)
        for agent, id, intensity, spread in zip(agents, ids, intensities, spreads):
            agent.set_odor_id(id)
            agent.set_scaled_odor_intensity(intensity)
            agent.set_scaled_odor_spread(spread)
            agent.set_odor_dist()

    def compute_odor_parameters(self, num_agents, odor_id_list, odor_intensity_list, odor_spread_list,
                                allocation_mode='iterative'):
        agent_odor_ids = []
        agent_odor_intensities = []
        agent_odor_spreads = []
        num_odors = len(odor_id_list)
        num_intensities = len(odor_intensity_list)
        num_spreads = len(odor_spread_list)
        if allocation_mode == 'iterative':
            for i in range(num_agents):
                id_index = i % num_odors
                intensity_index = i % num_intensities
                spread_index = i % num_spreads
                agent_odor_ids.append(odor_id_list[id_index])
                agent_odor_intensities.append(odor_intensity_list[intensity_index])
                agent_odor_spreads.append(odor_spread_list[spread_index])
        return agent_odor_ids, agent_odor_intensities, agent_odor_spreads

    def step(self):
        # Tick sim_clock
        self.sim_clock.tick_clock()
        # print(self.sim_clock.dmsecond)
        # Update value_layers

        if self.odor_layers:
            for layer_id in self.odor_layers:
                self.odor_layers[layer_id].update_values()  # Currently doing something only for the DiffusionValueLayer

        # s0 = time.time()
        for fly in self.get_flies():
            fly.compute_next_action()

        # s1 = time.time()
        self.active_larva_schedule.step()
        # s2 = time.time()

        self.active_food_schedule.step()

        # step space
        if self.physics_engine:
            self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)
            self.update_trajectories(self.get_flies())
        self.larva_step_collector.collect(self)
        # s3 = time.time()
        # if self.sim_clock.second < 0.001:
        #     print(self.sim_clock.hour, self.sim_clock.minute)
        #     print(np.round(s3 - s0, 5))
            # print(np.round(s3 - s0, 5), np.round(s1 - s0, 5), np.round(s2 - s1, 5), np.round(s3 - s2, 5))

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
            # g.step()
            # if self.physics_engine:
            #     self.space.Step(self.dt, self._sim_velocity_iterations, self._sim_position_iterations)

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
            plot_surface(x=self.space_to_mm(X), y=self.space_to_mm(Y), z=V, name=name, title=title,
                         save_to=save_to, save_as=f'{layer_id}_odorscape')
        # plt.figure()
        # CS = plt.contour(X, Y, V)
        # plt.clabel(CS, inline=1, fontsize=10)
        # plt.title(f'Odorant concentration landscape from {num_sources} sources ')
        # plt.show()


class LarvaWorldReplay(LarvaWorld):
    def __init__(self, step_data, endpoint_data, dataset=None,
                 segment_ratio=None,
                 pos_xy_pars=[],
                 id='Unnamed_Replay',
                 draw_Nsegs=None,
                 **kwargs):

        super().__init__(
            id=id,
            **kwargs)

        self.dataset = dataset
        self.pos_xy_pars = pos_xy_pars
        self.draw_Nsegs = draw_Nsegs

        self.step_data = step_data
        self.endpoint_data = endpoint_data
        self.agent_ids = self.step_data.index.unique('AgentID').values
        self.num_agents = len(self.agent_ids)

        self.starting_tick = self.step_data.index.unique('Step')[0]
        try:
            self.lengths = self.endpoint_data['length'].values
        except:
            self.lengths = np.ones(self.num_agents) * 5

        self.create_flies()

        if 'food_params' in self.env_pars.keys():
            self._place_food(self.env_pars['place_params']['initial_num_food'],
                             self.env_pars['place_params']['initial_food_positions'],
                             food_params=self.env_pars['food_params'])

    def create_flies(self):
        for i, agent_id in enumerate(self.agent_ids):
            data = self.step_data.xs(agent_id, level='AgentID', drop_level=True)
            f = LarvaReplay(model=self, unique_id=agent_id, schedule=self.active_larva_schedule,
                            length=self.lengths[i],
                            data=data)
            self.active_larva_schedule.add(f)
            self.space.place_agent(f, (0, 0))

    def step(self):
        # Tick sim_clock
        self.sim_clock.tick_clock()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()
