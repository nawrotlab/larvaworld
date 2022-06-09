import copy
import os
import sys
import random
import multiprocessing
import math
import threading
import time
from typing import Tuple

import pandas as pd
import progressbar
import numpy as np
import pygame
from unflatten import unflatten

from lib.aux.sim_aux import get_source_xy
from lib.aux.xy_aux import comp_rate
from lib.conf.base import paths
import lib.aux.dictsNlists as dNl
from lib.conf.base.pars import getPar, ParDict
from lib.conf.stored.conf import copyConf, kConfDict, loadRef, saveConf, loadConf
from lib.ga.robot.larva_robot import LarvaRobot, LarvaOffline
from lib.ga.scene.scene import Scene

from lib.ga.util.color import Color
from lib.ga.util.side_panel import SidePanel
from lib.ga.util.time_util import TimeUtil
from lib.model.envs._base_larvaworld import BaseLarvaWorld
# from lib.model.envs._larvaworld_sim import LarvaWorldSim
from lib.process.angular import comp_angular
from lib.process.aux import cycle_curve_dict
from lib.process.spatial import scale_to_length


class GAselector:
    def __init__(self, model, Ngenerations=None, Nagents=30, Nelits=3, Pmutation=0.3, Cmutation=0.1,
                 selection_ratio=0.3, verbose=0, bestConfID = None):

        self.bestConfID = bestConfID
        self.model = model
        self.Ngenerations = Ngenerations
        self.Nagents = Nagents
        self.Nelits = Nelits
        self.Pmutation = Pmutation
        self.Cmutation = Cmutation
        self.selection_ratio = selection_ratio
        self.verbose = 1
        self.sorted_genomes = None
        self.genomes = []
        self.genome_dict = None
        self.genome_dicts = []
        self.best_genome = None
        self.best_fitness = None
        self.generation_num = 1
        self.num_cpu = multiprocessing.cpu_count()
        self.start_total_time = TimeUtil.current_time_millis()
        self.start_generation_time = self.start_total_time
        self.generation_step_num = 0
        self.generation_sim_time = 0

    def printd(self, min_debug_level, *args):
        if self.verbose >= min_debug_level:
            msg = ''

            for arg in args:
                msg += str(arg) + ' '

            print(msg)

    def create_new_generation(self):
        genomes_selected = self.ga_selection()  # parents of the new generation
        self.printd(1, '\ngenomes selected:', genomes_selected)
        self.generation_num += 1
        self.genomes = self.ga_crossover_mutation(genomes_selected)

        self.generation_step_num = 0
        self.generation_sim_time = 0
        self.start_generation_time = TimeUtil.current_time_millis()
        self.printd(1, '\nGeneration', self.generation_num, 'started')

    def sort_genomes(self):
        # sort genomes by fitness
        self.sorted_genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)

        best_new_genome = self.sorted_genomes[0]
        if self.best_genome is None or best_new_genome.fitness > self.best_genome.fitness:
            self.best_genome = best_new_genome
            self.best_fitness = self.best_genome.fitness
            self.printd(1, 'New best:', self.best_genome.to_string())
            if self.bestConfID is not None:
                saveConf(self.best_genome.mConf, 'Model', self.bestConfID, verbose=self.verbose)


    def ga_selection(self):
        num_genomes_to_select = round(self.Nagents * self.selection_ratio)
        if num_genomes_to_select < 2:
            raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                             'Please increase population (' + str(self.Nagents) + ') or selection ratio (' +
                             str(self.selection_ratio) + ')')

        genomes_selected = []

        # elitism: keep the best genomes in the new generation
        for i in range(self.Nelits):
            elite_genome = self.sorted_genomes.pop(0)
            genomes_selected.append(elite_genome)
            num_genomes_to_select -= 1

        while num_genomes_to_select > 0:
            genome_selected = self.roulette_select(self.sorted_genomes)
            genomes_selected.append(genome_selected)
            self.sorted_genomes.remove(genome_selected)
            num_genomes_to_select -= 1

        return genomes_selected

    def roulette_select(self, genomes):
        fitness_sum = 0

        for genome in genomes:
            fitness_sum += genome.fitness

        value = random.uniform(0, fitness_sum)
        # print(fitness_sum, value)

        for i in range(len(genomes)):
            value -= genomes[i].fitness

            if value < 0:
                return genomes[i]

        return genomes[-1]

    def ga_crossover_mutation(self, parents):
        num_genomes_to_create = self.Nagents
        new_genomes = []

        # elitism: keep the best genomes in the new generation
        for i in range(self.Nelits):
            new_genomes.append(parents[i])
            num_genomes_to_create -= 1

        while num_genomes_to_create > 0:
            parent_a, parent_b = self.choose_parents(parents)
            new_genome = parent_a.crossover(parent_b, self.generation_num)
            new_genome.mutation(Pmut=self.Pmutation, Cmut=self.Cmutation)
            new_genomes.append(new_genome)
            num_genomes_to_create -= 1

        return new_genomes

    def choose_parents(self, parents):
        pos_a = random.randrange(len(parents))
        parent_a = parents[pos_a]
        parents.remove(parent_a)  # avoid choosing the same parent two times
        pos_b = random.randrange(len(parents))
        parent_b = parents[pos_b]
        parents.insert(pos_a, parent_a)  # reinsert the first parent in the list
        return parent_a, parent_b


def initConf(init_mode, space_dict, mConf0):
    if init_mode == 'random':
        kws = {}
        for k, vs in space_dict.items():
            if vs['dtype'] == bool:
                kws[k] = random.choice([True, False])
            elif vs['dtype'] == str:
                kws[k] = random.choice(vs['choices'])
            elif vs['dtype'] == Tuple[float]:
                vv0 = random.uniform(vs['min'], vs['max'])
                vv1 = random.uniform(vv0, vs['max'])
                kws[k] = (vv0, vv1)
            elif vs['dtype'] == int:
                kws[k] = random.randint(vs['min'], vs['max'])
            else:
                kws[k] = random.uniform(vs['min'], vs['max'])
        # initConf = dNl.update_nestdict(mConf0, kws)
        # return initConf
    elif init_mode == 'default':
        kws = {k: vs['initial_value'] for k, vs in space_dict.items()}
        # initConf = dNl.update_nestdict(mConf0, kws)
        # return initConf
    elif init_mode == 'model':
        kws = {k: mConf0[k] for k, vs in space_dict.items()}
    return kws


class GAbuilder(GAselector):
    def __init__(self, scene, side_panel=None, space_dict=None, robot_class=LarvaRobot, base_model='Sakagiannis2022',
                 multicore=True, fitness_func=None, fitness_target_kws={}, fitness_target_refID=None,
                 exclude_func=None, plot_func=None, bestConfID=None, init_mode='random', progress_bar=True, **kwargs):
        super().__init__(bestConfID = bestConfID,**kwargs)
        # self.robot_ids = [i for i in range(Nagents))]
        # print(fitness_target_kws)
        # raise

        self.is_running = True
        if progress_bar and self.Ngenerations is not None:
            self.progress_bar = progressbar.ProgressBar(self.Ngenerations)
            self.progress_bar.start()
        else:
            self.progress_bar = None

        self.fit_dict = self.arrange_fitness(fitness_func, fitness_target_refID, fitness_target_kws)

        self.exclude_func = exclude_func
        self.multicore = multicore
        self.scene = scene
        self.robot_class = robot_class
        self.space_dict = space_dict
        self.mConf0 = loadConf(base_model, 'Model')

        gConfs = [initConf(init_mode, space_dict, self.mConf0) for i in range(self.Nagents)]
        self.genomes = [Genome(gConf=gConf,mConf=dNl.update_nestdict(self.mConf0, gConf), space_dict=space_dict, generation_num=self.generation_num) for gConf in gConfs]


        self.robots = self.build_generation()
        self.step_df = self.init_step_df()

        # print(self.model.Nsteps, self.Nagents)
        # raise
        self.printd(1, 'Generation', self.generation_num, 'started')
        self.printd(1, 'multicore:', self.multicore, 'num_cpu:', self.num_cpu)


    def init_step_df(self):


        c = dNl.AttrDict.from_nested_dicts(
            {'id': self.model.id, 'group_id': 'GA_robots', 'dt': self.model.dt, 'fr': 1 / self.model.dt,
             'agent_ids': np.arange(self.Nagents), 'duration': self.model.Nsteps * self.model.dt,
             'Npoints': 3, 'Ncontour': 0, 'point': '', 'N': self.Nagents, 'Nticks': self.model.Nsteps,
             'mID': self.bestConfID,
             'color': 'blue', 'env_params': self.model.env_pars})

        self.my_index = pd.MultiIndex.from_product([np.arange(c.Nticks), c.agent_ids],
                                                   names=['Step', 'AgentID'])
        self.df_columns = getPar(['b', 'fov', 'rov', 'v', 'x', 'y'])
        self.df_Ncols=len(self.df_columns)
        step_df = np.ones([c.Nticks, c.N, self.df_Ncols]) * np.nan

        e = pd.DataFrame(index=c.agent_ids)
        e['cum_dur'] = c.duration
        e['num_ticks'] = c.Nticks
        e['length'] = [robot.real_length for robot in self.robots]

        self.dataset=dNl.AttrDict.from_nested_dicts({'step_data' : None, 'endpoint_data' : e, 'config' : c})

        return step_df

    def finalize_step_df(self):
        t0 = time.time()
        e, c = self.dataset.endpoint_data, self.dataset.config
        self.step_df[:, :, :3] = np.rad2deg(self.step_df[:, :, :3])
        self.step_df = self.step_df.reshape(c.Nticks* c.N, self.df_Ncols)
        s = pd.DataFrame(self.step_df, index=self.my_index, columns=self.df_columns)
        s = s.astype(float)

        cycle_ks,eval_ks =None,None
        scale_to_length(s, e, c, pars=None, keys=['v'])
        self.dataset.step_data = s
        dic0=self.fit_dict.robot_dict

        ks=[]
        if 'eval' in dic0.keys():
            eval_ks=self.fit_dict.target_kws.eval_shorts
            ks+=eval_ks
        if 'cycle_curves' in dic0.keys():
            cycle_ks = list(self.fit_dict.target_kws.pooled_cycle_curves.keys())
            ks +=cycle_ks
        ks=dNl.unique_list(ks)
        for k in ks:
            ParDict.compute(k, self.dataset)

        for i, g in self.genome_dict.genomes.items():
            if g.fitness is None :
                # t0 = time.time()
                ss = self.dataset.step_data.xs(i, level='AgentID')
                gdict=dNl.AttrDict.from_nested_dicts({k:[] for k in dic0.keys()})
                gdict['step']=ss
                if cycle_ks:
                    gdict['cycle_curves'] = cycle_curve_dict(s=ss, dt=self.model.dt, shs=cycle_ks)
                if eval_ks:
                    gdict['eval']={sh: ss[getPar(sh)].dropna().values for sh in eval_ks}
                    # t1 = time.time()
                    # print('--1--', t1 - t0)
                    # t00 = time.time()
                g.fitness, g.fitness_dict = self.get_fitness(gdict)
                # print(g.fitness)
                # g.fitness
                # t11 = time.time()
                # print('--2--', t11 - t00)


    def build_generation(self):
        self.genome_dict = dNl.AttrDict.from_nested_dicts(
            {'generation': self.generation_num, 'genomes': {i: g for i, g in enumerate(self.genomes)}})
        self.genome_dicts.append(self.genome_dict)

        robots = []
        for i, g in self.genome_dict.genomes.items():
            robot = self.robot_class(unique_id=i, model=self.model, larva_pars=g.mConf)
            robot.genome = g
            robots.append(robot)
            self.scene.put(robot)
        return robots

    def step(self):
        start_time = TimeUtil.current_time_millis()

        if self.multicore:
            threads = []
            num_robots = len(self.robots)
            num_robots_per_cpu = math.floor(num_robots / self.num_cpu)

            self.printd(2, 'num_robots_per_cpu:', num_robots_per_cpu)

            for i in range(self.num_cpu - 1):
                start_pos = i * num_robots_per_cpu
                end_pos = (i + 1) * num_robots_per_cpu
                self.printd(2, 'core:', i + 1, 'positions:', start_pos, ':', end_pos)
                robot_list = self.robots[start_pos:end_pos]

                thread = GA_thread(robot_list)
                thread.start()
                self.printd(2, 'thread', i + 1, 'started')
                threads.append(thread)

            # last sublist of robots
            start_pos = (self.num_cpu - 1) * num_robots_per_cpu
            self.printd(2, 'last core, start_pos', start_pos)
            robot_list = self.robots[start_pos:]

            thread = GA_thread(robot_list)
            thread.start()
            self.printd(2, 'last thread started')
            threads.append(thread)

            for t in threads:
                t.join()

            if self.verbose >= 2:
                end_time = TimeUtil.current_time_millis()
                partial_duration = end_time - start_time
                print('Step partial duration', partial_duration)

            for robot in self.robots[:]:
                self.check(robot)
        else:
            for robot in self.robots[:]:
                robot.sense_and_act()
                self.check(robot)
        self.generation_sim_time += self.model.dt
        self.generation_step_num += 1
        if self.generation_step_num == self.model.Nsteps:

            self.finalize_step_df()
            for robot in self.robots[:]:
                self.destroy_robot(robot)

        # check population extinction
        if not self.robots:

            self.sort_genomes()

            if self.Ngenerations is None or self.generation_num < self.Ngenerations:

                self.create_new_generation()
                if self.progress_bar:
                    self.progress_bar.update(self.generation_num)

                self.robots = self.build_generation()
                self.step_df = self.init_step_df()

            else:
                self.finalize()



    def destroy_robot(self, robot, excluded=False):
        if excluded :
            robot.genome.fitness = -np.inf
        self.scene.remove(robot)
        self.robots.remove(robot)

    def get_fitness2(self, robot):
        if self.fit_dict.func is not None:
            return self.fit_dict.func(robot, **self.fit_dict.target_kws)
        else:
            return None

    def get_fitness(self, gdict):
        return self.fit_dict.func(gdict, **self.fit_dict.target_kws)


    def finalize(self):
        self.is_running = False
        if self.progress_bar:
            self.progress_bar.finish()
        self.printd(0, 'Best genome:', self.best_genome.to_string())
        self.printd(0, 'Best fittness:', self.best_genome.fitness)

    def check(self, robot):
        if not self.model.offline:
            if robot.x < 0 or robot.x > self.scene.width or robot.y < 0 or robot.y > self.scene.height:
                # if robot.x < 0 or robot.x > self.scene.width or robot.y < 0 or robot.y > self.scene.height:
                self.destroy_robot(robot)

            # destroy robot if it collides an obstacle
            if robot.collision_with_object:
                self.destroy_robot(robot)

        if self.exclude_func is not None:
            if self.exclude_func(robot):
                self.destroy_robot(robot, excluded=True)

    def arrange_fitness(self, fitness_func, fitness_target_refID, fitness_target_kws):
        robot_dict=dNl.AttrDict.from_nested_dicts({})
        if fitness_target_refID is not None:
            d = loadRef(fitness_target_refID)
            if 'eval_shorts' in fitness_target_kws.keys():
                shs = fitness_target_kws['eval_shorts']
                eval_pars, eval_lims, eval_labels = getPar(shs, to_return=['d', 'lim', 'lab'])
                fitness_target_kws['eval'] = {sh: d.get_par(p, key='distro').dropna().values for p, sh in
                                              zip(eval_pars, shs)}
                robot_dict['eval']= {sh: [] for p, sh in zip(eval_pars, shs)}
                fitness_target_kws['eval_labels'] = eval_labels
            if 'pooled_cycle_curves' in fitness_target_kws.keys():
                curves = d.config.pooled_cycle_curves
                shorts = fitness_target_kws['pooled_cycle_curves']
                dic={}
                for sh in shorts :
                    dic[sh] = 'abs' if sh == 'sv' else 'norm'

                fitness_target_kws['cycle_curve_keys']=dic
                fitness_target_kws['pooled_cycle_curves'] = {sh: curves[sh] for sh in shorts}
                robot_dict['cycle_curves']= {sh: [] for sh in shorts}

            fitness_target = d
        else:
            fitness_target = None
        if 'source_xy' in fitness_target_kws.keys():
            fitness_target_kws['source_xy'] = self.model.source_xy
        return dNl.AttrDict.from_nested_dicts({'func': fitness_func, 'target_refID': fitness_target_refID,
                                               'target_kws': fitness_target_kws, 'target': fitness_target, 'robot_dict' : robot_dict})


class BaseGAlauncher(BaseLarvaWorld):
    SCENE_MAX_SPEED = 3000

    SCENE_MIN_SPEED = 1
    SCENE_SPEED_CHANGE_COEFF = 1.5

    SIDE_PANEL_WIDTH = 600

    # SCREEN_MARGIN = 12

    def __init__(self, sim_params, scene='no_boxes', scene_speed=0, env_params=None, experiment='exploration',
                 caption=None, save_to=None, offline=False, **kwargs):
        self.offline = offline
        id = sim_params.sim_ID
        self.sim_params = sim_params
        dt = sim_params.timestep
        Nsteps = int(sim_params.duration * 60 / dt)
        if save_to is None:
            save_to = paths.path("SIM")
        self.save_to = save_to
        self.dir_path = f'{save_to}/{sim_params.path}/{id}'
        self.plot_dir = f'{self.dir_path}/plots'
        os.makedirs(self.plot_dir, exist_ok=True)
        if not self.offline:
            super().__init__(id=id, dt=dt, Box2D=sim_params.Box2D, env_params=env_params,
                             save_to=f'{self.dir_path}/visuals',
                             Nsteps=Nsteps, experiment=experiment, **kwargs)
            self.arena_width, self.arena_height = self.env_pars.arena.arena_dims
        else:
            self.env_pars = env_params
            self.scaling_factor = 1
            X, Y = self.arena_width, self.arena_height = self.env_pars.arena.arena_dims
            self.space_edges_for_screen = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
            self.experiment = experiment
            self.dt = dt
            self.Nticks = 0
            self.Nsteps = Nsteps
            self.id = id
            self.save_to = save_to
            self.Box2D = False
        self.source_xy = get_source_xy(self.env_pars['food_params'])
        # print(self.source_xy)
        # raise
        if caption is None:
            caption = f'GA {experiment} : {self.id}'
        self.caption = caption
        self.scene_file = f'{paths.path("GA_SCENE")}/{scene}.txt'
        self.scene_speed = scene_speed
        self.obstacles = []

    def build_box(self, x, y, size, color):
        from lib.ga.scene.box import Box

        box = Box(x, y, size, color)
        self.obstacles.append(box)
        return box

    def build_wall(self, point1, point2, color):
        from lib.ga.scene.wall import Wall
        wall = Wall(point1, point2, color)
        self.obstacles.append(wall)
        return wall

    def get_larvaworld_food(self):

        for label,pos in self.source_xy.items():
            x, y = self.screen_pos(pos)
            # size = ff.radius * self.scaling_factor
            # col = ff.default_color
            box = self.build_box(x, y, 4, 'green')
            box.label = label
            self.scene.put(box)

    def screen_pos(self, real_pos):
        return np.array(real_pos) * self.scaling_factor + np.array([self.scene.width / 2, self.scene.height / 2])

    def init_scene(self):
        self.scene = Scene.load_from_file(self.scene_file, self.scene_speed, self.SIDE_PANEL_WIDTH)

        self.scene.set_bounds(*self.space_edges_for_screen)

    def increase_scene_speed(self):
        if self.scene.speed < self.SCENE_MAX_SPEED:
            self.scene.speed *= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)

    def decrease_scene_speed(self):
        if self.scene.speed > self.SCENE_MIN_SPEED:
            self.scene.speed /= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)


class GAlauncher(BaseGAlauncher):
    def __init__(self, ga_build_kws, ga_select_kws, show_screen=True, **kwargs):
        super().__init__(**kwargs)
        if self.offline:
            ga_build_kws.robot_class = LarvaOffline
            show_screen = False
        self.ga_build_kws = ga_build_kws
        self.ga_select_kws = ga_select_kws
        self.show_screen = show_screen
        if show_screen:
            pygame.init()
            pygame.display.set_caption(self.caption)
            self.clock = pygame.time.Clock()
        self.initialize(**ga_build_kws, **ga_select_kws)

    def run(self):
        while True and self.engine.is_running:
            from pygame import KEYDOWN, K_ESCAPE, K_r, K_MINUS, K_PLUS, K_s, K_e

            t0 = TimeUtil.current_time_millis()
            self.engine.step()
            t1 = TimeUtil.current_time_millis()
            self.printd(2, 'Step duration: ', t1 - t0)
            if self.show_screen:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                        sys.exit()
                    elif e.type == KEYDOWN and e.key == K_r:
                        self.initialize(**self.ga_select_kws, **self.ga_build_kws)
                    elif e.type == KEYDOWN and (e.key == K_PLUS or e.key == 93 or e.key == 270):
                        self.increase_scene_speed()
                    elif e.type == KEYDOWN and (e.key == K_MINUS or e.key == 47 or e.key == 269):
                        self.decrease_scene_speed()
                    elif e.type == KEYDOWN and e.key == K_s:
                        pass
                        # self.engine.save_genomes()
                    # elif e.type == KEYDOWN and e.key == K_e:
                    #     self.engine.evaluation_mode = 'preparing'

                if self.side_panel.generation_num < self.engine.generation_num:
                    self.side_panel.update_ga_data(self.engine.generation_num, self.engine.best_genome)

                # update statistics time
                cur_t = TimeUtil.current_time_millis()
                cum_t = math.floor((cur_t - self.engine.start_total_time) / 1000)
                gen_t = math.floor((cur_t - self.engine.start_generation_time) / 1000)
                self.side_panel.update_ga_time(cum_t, gen_t, self.engine.generation_sim_time)
                self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
                self.screen.fill(Color.BLACK)

                for obj in self.scene.objects:
                    obj.draw(self.scene)

                # draw a black background for the side panel
                side_panel_bg_rect = pygame.Rect(self.scene.width, 0, self.SIDE_PANEL_WIDTH, self.scene.height)
                pygame.draw.rect(self.screen, Color.BLACK, side_panel_bg_rect)

                self.display_info()

                pygame.display.flip()
                self.clock.tick(int(round(self.scene.speed)))
        return self.engine.best_genome

    def printd(self, min_debug_level, *args):
        if self.engine.verbose >= min_debug_level:
            msg = ''

            for arg in args:
                msg += str(arg) + ' '

            print(msg)

    def display_info(self):
        self.side_panel.display_ga_info()

    def initialize(self, **kwargs):
        self.init_scene()

        self.engine = GAbuilder(scene=self.scene, model=self, **kwargs)
        if self.show_screen:
            if not self.offline:
                self.get_larvaworld_food()
            self.scene.screen = pygame.display.set_mode((self.scene.width + self.scene.panel_width, self.scene.height))
            self.screen = self.scene.screen
            self.side_panel = SidePanel(self.scene)
            self.side_panel.update_ga_data(self.engine.generation_num, None)
            self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
            self.side_panel.update_ga_time(0, 0, 0)


class Genome:

    def __init__(self, mConf,gConf, space_dict, generation_num=None, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.gConf = gConf
        self.mConf = mConf
        self.generation_num = generation_num
        self.fitness = None
        self.fitness_dict = None
        self.space_dict = space_dict



    def crossover(self, other_parent, generation_num):
        gConf1=self.gConf
        gConf2=other_parent.gConf

        gConf = {k: gConf1[k] if random.random() < 0.5 else gConf2[k] for k in self.space_dict.keys()}
        mConf_new=dNl.update_nestdict(self.mConf, gConf)
        # apply uniform crossover to generate a new genome
        return Genome(mConf=mConf_new,gConf=gConf,  generation_num=generation_num, space_dict=self.space_dict)

    def mutation(self, **kwargs):
        gConf=self.gConf
        # mConf_f=dNl.flatten_dict(self.mConf)
        for k, vs in self.space_dict.items():
            v = self.gConf[k]
            if vs['dtype'] == bool:
                vv = self.mutate_with_probability(v, choices=[True, False], **kwargs)
            elif vs['dtype'] == str:
                vv = self.mutate_with_probability(v, choices=vs['choices'], **kwargs)
            else:
                r0, r1 = vs['min'], vs['max']
                range = r1 - r0
                if vs['dtype'] == Tuple[float]:
                    v0, v1 = v
                    vv0 = self.mutate_with_probability(v0, range=range, **kwargs)
                    vv1 = self.mutate_with_probability(v1, range=range, **kwargs)
                    vv = (vv0, vv1)
                else:
                    vv = self.mutate_with_probability(v, range=range, **kwargs)
            self.gConf[k]=vv
            # dNl.flatten_dict(self.mConf)[k]=vv
            # setattr(self.mConf, k, vv)

        self.check_parameter_bounds()
        self.mConf = dNl.update_nestdict(self.mConf, self.gConf)

    def mutate_with_probability(self, v, Pmut, Cmut, choices=None, range=None):
        if random.random() < Pmut:
            if choices is None:
                if v is None:
                    return v
                else:
                    if range is None:
                        return random.gauss(v, Cmut * v)
                    else:
                        return random.gauss(v, Cmut * range)
            else:
                return random.choice(choices)
        else:
            return v

    def check_parameter_bounds(self):
        # mConf_f = dNl.flatten_dict(self.mConf)
        for k, vs in self.space_dict.items():
            if vs['dtype'] in [bool, str]:
                continue

            else:
                r0, r1 = vs['min'], vs['max']
                v = self.gConf[k]
                if v is None:
                    self.gConf[k] = v
                    continue
                else:

                    if vs['dtype'] == Tuple[float]:
                        vv0, vv1 = v
                        if vv0 < r0:
                            vv0 = r0
                        if vv1 > r1:
                            vv1 = r1
                        if vv0 > vv1:
                            vv0 = vv1
                        self.gConf[k] = (vv0, vv1)
                        # setattr(self.mConf, k, (vv0, vv1))
                        continue
                    if vs['dtype'] == int:
                        v = int(v)
                    if v < r0:
                        self.gConf[k] = r0
                        # setattr(self.mConf, k, r0)
                    elif v > r1:
                        self.gConf[k] = r1
                        # setattr(self.mConf, k, r1)
                    else:
                        self.gConf[k] = v
                        # setattr(self.mConf, k, v)
        self.mConf = dNl.update_nestdict(self.mConf, self.gConf)

    def __repr__(self):
        fitness = None if self.fitness is None else round(self.fitness, 2)
        return self.__class__.__name__ + '(fitness:' + repr(fitness) + ' generation_num:' + repr(
            self.generation_num) + ')'

    def get(self, rounded=False):
        dic = {}
        for k, vs in self.space_dict.items():
            v = self.gConf[k]
            if v is not None and rounded:
                if vs['dtype'] == float:
                    v = round(v, 2)
                elif vs['dtype'] == Tuple[float]:
                    v = (round(v[0], 2), round(v[1], 2))
            dic[k] = v
        return dic

    def to_string(self):
        fitness = None if self.fitness is None else round(self.fitness, 2)
        kwstrings = [f' {vs["name"]}:' + repr(self.get(rounded=True)[k]) for k, vs in self.space_dict.items()]
        kwstr = ''
        for ii in kwstrings:
            kwstr = kwstr + ii

        return '(fitness:' + repr(fitness) + kwstr + ')'


class GA_thread(threading.Thread):
    def __init__(self, robots):
        threading.Thread.__init__(self)
        self.robots = robots

    def run(self):
        for robot in self.robots:
            robot.sense_and_act()


if __name__ == '__main__':
    # print(null_dict.__class__)
    print(kConfDict('Ga'))
