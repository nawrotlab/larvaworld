import math
import multiprocessing
import os
import random
import sys
import random
import multiprocessing
import math
from typing import Tuple

from scipy.stats import ks_2samp
import numpy as np
import pygame
from pygame import KEYDOWN, K_ESCAPE, K_r, K_MINUS, K_PLUS, K_s, K_e
from unflatten import unflatten

import lib.aux.dictsNlists as dNl
from lib.aux.xy_aux import eudi5x
from lib.conf.base import paths
from lib.conf.base.dtypes import null_dict, ga_dict
from lib.conf.base.par import ParDict, getPar
from lib.conf.stored.conf import copyConf, kConfDict, loadRef, expandConf, saveConf, next_idx
from lib.ga.robot.larva_robot import LarvaRobot
from lib.ga.scene.box import Box
from lib.ga.scene.scene import Scene
from lib.ga.scene.wall import Wall
from lib.ga.util.color import Color
from lib.ga.util.side_panel import SidePanel
from lib.ga.util.thread_ga_robot import ThreadGaRobot
from lib.ga.util.time_util import TimeUtil
from lib.model.envs._larvaworld_sim import LarvaWorldSim


class GA_selector:
    def __init__(self, model, Nagents=30, Nelits=3, Pmutation=0.3, Cmutation=0.1,
                 selection_ratio=0.3, max_Nticks=1000, max_dur=None, verbose=0):
        self.model = model
        self.Nagents = Nagents
        self.Nelits = Nelits
        self.Pmutation = Pmutation
        self.Cmutation = Cmutation
        self.selection_ratio = selection_ratio
        if max_dur is not None:
            max_Nticks = int(max_dur * 60 / self.model.dt)
        self.max_Nticks = max_Nticks
        self.verbose = verbose

        self.stored_genomes = None
        self.genomes = []
        self.genomes_last_generation = []
        self.best_genome = None
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
        self.genomes_last_generation = self.genomes
        genomes_selected = self.ga_selection()  # parents of the new generation
        self.printd(1, '\ngenomes selected:', genomes_selected)
        self.generation_num += 1
        new_genomes = self.ga_crossover_mutation(genomes_selected)
        self.genomes = new_genomes


        self.generation_step_num = 0
        self.generation_sim_time = 0
        # reset generation time
        self.start_generation_time = TimeUtil.current_time_millis()

        print('\nGeneration', self.generation_num, 'started')

    def ga_selection(self):
        # sort genomes by fitness
        sorted_genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)
        best_new_genome = sorted_genomes[0]
        if self.best_genome is None or best_new_genome.fitness > self.best_genome.fitness:
            self.best_genome = best_new_genome
            print('New best:', self.best_genome.to_string())

        num_genomes_to_select = round(self.Nagents * self.selection_ratio)

        if num_genomes_to_select < 2:
            raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                             'Please increase population (' + str(self.Nagents) + ') or selection ratio (' +
                             str(self.selection_ratio) + ')')

        genomes_selected = []

        # elitism: keep the best genomes in the new generation
        for i in range(self.Nelits):
            elite_genome = sorted_genomes.pop(0)
            genomes_selected.append(elite_genome)
            num_genomes_to_select -= 1
            print("Elite:", elite_genome.to_string())

        while num_genomes_to_select > 0:
            genome_selected = self.roulette_select(sorted_genomes)
            genomes_selected.append(genome_selected)
            sorted_genomes.remove(genome_selected)
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
            new_genome.mutation(Pmut=self.Pmutation, Cmut= self.Cmutation)
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


class GA_builder(GA_selector):
    def __init__(self, scene, side_panel, space_dict=None, robot_class=LarvaRobot, base_model='Sakagiannis2022',
                 multicore=True, fitness_func=None,fitness_target_kws={},fitness_target_refID=None, exclude_func = None, plot_func=None, bestConfID=None, init_mode='random', **kwargs):
        super().__init__(**kwargs)

        self.bestConfID = bestConfID
        self.evaluation_mode = None
        self.fitness_func = fitness_func

        self.fitness_target_refID = fitness_target_refID
        if fitness_target_refID is not None :
            d = loadRef(fitness_target_refID)
            d.load(contour=False)
            if 'eval_shorts' in fitness_target_kws.keys():
                shs=fitness_target_kws['eval_shorts']
                # s, e, c = d.step_data, d.endpoint_data, d.config
                dic = ParDict(mode='load').dict
                fitness_target_kws['eval'] = {sh: d.step_data[dic[sh]['d']].dropna().values for sh in shs}
                eval_lims, eval_labels = getPar(shs, to_return=['lim', 'lab'])
                fitness_target_kws['eval_labels'] = eval_labels
                # return eval
            if 'target_fov_curve' in fitness_target_kws.keys():
                fitness_target_kws['target_fov_curve'] = np.array(d.config.pooled_cycle_curves.fov)
            self.fitness_target=d
        else :
            self.fitness_target = None
        self.fitness_target_kws = fitness_target_kws
        self.exclude_func = exclude_func
        self.plot_func = plot_func
        self.multicore = multicore
        self.scene = scene
        self.side_panel = side_panel
        self.robot_class = robot_class
        self.space_dict = space_dict
        if type(base_model) == str and base_model in kConfDict('Model'):
            self.larva_pars = copyConf(base_model, 'Model')
        elif isinstance(base_model, dict):
            self.larva_pars = base_model

        g_kws={
            'space_dict' : self.space_dict,
            'generation_num' : self.generation_num,
        }
        for i in range(self.Nagents):
            if init_mode == 'default':
                g = BaseGenome.default(**g_kws)
            elif init_mode == 'random':
                g = BaseGenome.random(**g_kws)
            elif init_mode == 'base_model':
                kws = {k : dNl.flatten_dict(self.larva_pars)[k] for k in self.space_dict.keys()}
                g = BaseGenome(**kws, **g_kws)
            self.genomes.append(g)

        self.robots = self.build_generation()
        self.side_panel.update_ga_data(self.generation_num, None, None)
        self.side_panel.update_ga_population(len(self.robots))
        self.side_panel.update_ga_time(0, 0, 0)

        print('\nGeneration', self.generation_num, 'started')

        self.printd(1, 'multicore:', self.multicore, 'num_cpu:', self.num_cpu)

    def build_generation(self):
        if self.bestConfID is not None and self.best_genome is not None:
            temp = dNl.flatten_dict(self.larva_pars)
            for k, vs in self.space_dict.items():
                temp[k] = self.best_genome.get(rounded=True)[k]
            best = dNl.AttrDict.from_nested_dicts(unflatten(temp))
            saveConf(best, 'Model', self.bestConfID)

        self.last_robots = []
        if self.evaluation_mode == 'preparing' and self.best_genome is not None:
            self.stored_genomes = self.genomes
            self.genomes = [self.best_genome] * len(self.genomes)
            self.evaluation_mode = 'plotting'
        if self.evaluation_mode == 'plotting':
            self.genomes = self.stored_genomes
            self.stored_genomes = None
        robots = []
        for i, genome in enumerate(self.genomes):
            robot = genome.build_robot(larva_pars=self.larva_pars, robot_class=self.robot_class, unique_id=i,
                                       model=self.model)
            robot.genome = genome
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

                thread = ThreadGaRobot(robot_list)
                thread.start()
                self.printd(2, 'thread', i + 1, 'started')
                threads.append(thread)

            # last sublist of robots
            start_pos = (self.num_cpu - 1) * num_robots_per_cpu
            self.printd(2, 'last core, start_pos', start_pos)
            robot_list = self.robots[start_pos:]

            thread = ThreadGaRobot(robot_list)
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
                # ensure robot doesn't accidentaly go outside of the scene
                if robot.x < 0 or robot.x > self.scene.width or robot.y < 0 or robot.y > self.scene.height:
                    self.destroy_robot(robot)

                # destroy robot if it collides an obstacle
                if robot.collision_with_object:
                    self.destroy_robot(robot)

                if self.exclude_func is not None :
                    if self.exclude_func(robot) :
                        self.destroy_robot(robot, excluded=True)
        else:
            # multicore = False
            for robot in self.robots[:]:
                robot.sense_and_act()

                # ensure robot doesn't accidentaly go outside of the scene
                if robot.x < 0 or robot.x > self.scene.width or robot.y < 0 or robot.y > self.scene.height:
                    self.destroy_robot(robot)

                # destroy robot if it collides an obstacle
                if robot.collision_with_object:
                    self.destroy_robot(robot)

                if self.exclude_func is not None :
                    if self.exclude_func(robot) :
                        self.destroy_robot(robot, excluded=True)

        # create new obstacles for long lasting generations
        if self.generation_step_num == self.max_Nticks:
            print('Time limit reached (' + str(self.max_Nticks) +
                  ' steps), destroying all remaining robots')

            for robot in self.robots[:]:
                self.destroy_robot(robot)

        # check population extinction
        if not self.robots:
            if self.evaluation_mode == 'plotting':
                if self.plot_func is not None:
                    self.plot_func(self.last_robots, generation_num=self.generation_num, save_to=self.model.plot_dir, **self.fitness_target_kws)
                self.evaluation_mode = None  # raise
            print('Generation', self.generation_num, 'terminated')
            self.create_new_generation()

            self.robots = self.build_generation()
            self.side_panel.update_ga_data(self.generation_num, self.best_genome, self.best_genome.fitness)

        # update statistics time
        cur_t = TimeUtil.current_time_millis()
        cum_t = math.floor((cur_t - self.start_total_time) / 1000)
        gen_t = math.floor((cur_t - self.start_generation_time) / 1000)
        self.generation_sim_time += self.model.dt
        self.side_panel.update_ga_time(cum_t, gen_t, self.generation_sim_time)
        self.side_panel.update_ga_population(len(self.robots))
        self.generation_step_num += 1

    def destroy_robot(self, robot, excluded=False):
        if excluded :
            fitness=-np.inf
        else :
            self.last_robots.append(robot)
            fitness = self.get_fitness(robot)
        robot.genome.fitness = fitness

        self.scene.remove(robot)
        self.robots.remove(robot)
        self.printd(1, 'Destroyed robot with fitness value', fitness)

    def get_fitness(self, robot):
        if self.fitness_func is not None:
            return self.fitness_func(robot, **self.fitness_target_kws)
        else:
            return None


class Base_runner(LarvaWorldSim):
    SCENE_MAX_SPEED = 3000

    SCENE_MIN_SPEED = 1
    SCENE_SPEED_CHANGE_COEFF = 1.5

    SIDE_PANEL_WIDTH = 600

    SCREEN_MARGIN = 12

    def __init__(self, scene_file, scene_speed=0,id=None, env_params=None, experiment='exploration',dt=0.1, caption='Template',save_to=None, **kwargs):
        if save_to is None:
            save_to = paths.path("GA")
        if env_params is None:
            env_params = null_dict('env_conf')
        if id is None :
            id=f'{experiment}_{next_idx(experiment, type="ga")}'
        super().__init__(env_params=env_params, dt=dt,save_to=save_to,experiment=experiment,id=id, **kwargs)
        self.dir_path = f'{self.save_to}/{self.experiment}/{self.id}'
        self.plot_dir = f'{self.dir_path}/plots'
        os.makedirs(self.plot_dir, exist_ok=True)

        self.arena_width, self.arena_height = env_params.arena.arena_dims

        self.caption = caption
        self.scene_file = scene_file
        self.scene_speed = scene_speed
        self.obstacles = []

    def build_box(self, x, y, size, color):
        box = Box(x, y, size, color)
        self.obstacles.append(box)
        return box

    def build_wall(self, point1, point2, color):
        wall = Wall(point1, point2, color)
        self.obstacles.append(wall)
        return wall

    def get_larvaworld_food(self):
        for ff in self.get_food():
            x, y = self.screen_pos(ff.pos)
            size = ff.radius * self.scaling_factor
            col = ff.default_color
            box = self.build_box(x, y, size, col)
            box.label = ff.unique_id
            self.scene.put(box)

    # @ property
    def real_pos(self, screen_pos):
        return (np.array(screen_pos) - np.array([self.scene.width / 2, self.scene.height / 2])) / self.scaling_factor

    # @property
    def screen_pos(self, real_pos):
        return np.array(real_pos) * self.scaling_factor + np.array([self.scene.width / 2, self.scene.height / 2])

    def init_scene(self):
        self.scene = Scene.load_from_file(self.scene_file, self.scene_speed, self.SIDE_PANEL_WIDTH)
        self.scaling_factor = self.scene.width / self.arena_width
        self.get_larvaworld_food()

        self.screen = self.scene.screen
        self.side_panel = SidePanel(self.scene)

    def increase_scene_speed(self):
        if self.scene.speed < self.SCENE_MAX_SPEED:
            self.scene.speed *= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)

    def decrease_scene_speed(self):
        if self.scene.speed > self.SCENE_MIN_SPEED:
            self.scene.speed /= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)


class GA_runner(Base_runner):
    def __init__(self, ga_kws={},show_screen=True, **kwargs):
        super().__init__(**kwargs)
        pygame.init()
        pygame.display.set_caption(self.caption)
        clock = pygame.time.Clock()
        self.initialize(**ga_kws)
        while True:
            if show_screen:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                        sys.exit()
                    elif e.type == KEYDOWN and e.key == K_r:
                        self.initialize(**ga_kws)
                    elif e.type == KEYDOWN and (e.key == K_PLUS or e.key == 93 or e.key == 270):
                        self.increase_scene_speed()
                    elif e.type == KEYDOWN and (e.key == K_MINUS or e.key == 47 or e.key == 269):
                        self.decrease_scene_speed()
                    elif e.type == KEYDOWN and e.key == K_s:
                        self.engine.save_genomes()
                    elif e.type == KEYDOWN and e.key == K_e:
                        self.engine.evaluation_mode = 'preparing'

            t0 = TimeUtil.current_time_millis()
            self.engine.step()
            t1 = TimeUtil.current_time_millis()
            self.printd(2, 'Step duration: ', t1 - t0)

            if show_screen:
                self.screen.fill(Color.BLACK)

                for obj in self.scene.objects:
                    obj.draw(self.screen)

                    if issubclass(type(obj), self.engine.robot_class) and obj.unique_id is not None:
                        obj.draw_label(self.screen)

                # draw a black background for the side panel
                side_panel_bg_rect = pygame.Rect(self.scene.width, 0, self.SIDE_PANEL_WIDTH, self.scene.height)
                pygame.draw.rect(self.screen, Color.BLACK, side_panel_bg_rect)

                self.display_info()

                pygame.display.flip()
            clock.tick(int(round(self.scene.speed)))

    def printd(self, min_debug_level, *args):
        if self.engine.verbose >= min_debug_level:
            msg = ''

            for arg in args:
                msg += str(arg) + ' '

            print(msg)

    def display_info(self):
        self.side_panel.display_ga_info(self.engine.space_dict)

    def initialize(self, **kwargs):
        self.init_scene()
        self.engine = GA_builder(scene=self.scene, side_panel=self.side_panel, model=self, **kwargs)


class BaseGenome:

    def __init__(self, space_dict, generation_num=None, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.generation_num = generation_num
        self.fitness = None
        self.fitness_dict = None
        self.space_dict = space_dict

    @staticmethod
    def random(generation_num, space_dict):
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
        return BaseGenome(**kws, generation_num=generation_num, space_dict=space_dict)

    @staticmethod
    def default(generation_num, space_dict):
        kws = {k : vs['initial_value'] for k, vs in space_dict.items()}
        return BaseGenome(**kws, generation_num=generation_num, space_dict=space_dict)

    def crossover(self, other_parent, generation_num):
        kws = {k: getattr(self, k) if random.random() < 0.5 else getattr(other_parent, k) for k in
               self.space_dict.keys()}
        # apply uniform crossover to generate a new genome
        return BaseGenome(**kws, generation_num=generation_num, space_dict=self.space_dict)

    def mutation(self, **kwargs):
        for k, vs in self.space_dict.items():
            v = getattr(self, k)
            if vs['dtype'] == bool:
                vv = self.mutate_with_probability(v, choices=[True, False], **kwargs)
            elif vs['dtype'] == str:
                vv = self.mutate_with_probability(v, choices=vs['choices'], **kwargs)
            elif vs['dtype'] == Tuple[float]:
                v0, v1 = v
                vv0 = self.mutate_with_probability(v0, **kwargs)
                vv1 = self.mutate_with_probability(v1, **kwargs)
                vv = (vv0, vv1)
            else:
                vv = self.mutate_with_probability(v, **kwargs)
            setattr(self, k, vv)
        self.check_parameter_bounds()

    def mutate_with_probability(self, v, Pmut, Cmut, choices=None):
        if random.random() < Pmut:
            if choices is None:
                if v is None :
                    return v
                else:
                    return random.gauss(v, Cmut * v)
            else:
                return random.choice(choices)
        else:
            return v

    def check_parameter_bounds(self):
        for k, vs in self.space_dict.items():
            if vs['dtype'] in [bool, str]:
                continue

            else:
                r0, r1 = vs['min'], vs['max']
                v = getattr(self, k)
                if v is None:
                    setattr(self, k, v)
                    continue
                else :

                    if vs['dtype'] == Tuple[float]:
                        vv0, vv1 = v
                        if vv0 < r0:
                            vv0 = r0
                        if vv1 > r1:
                            vv1 = r1
                        if vv0 > vv1:
                            vv0 = vv1

                        setattr(self, k, (vv0, vv1))
                        continue
                    if vs['dtype'] == int:
                        v=int(v)
                    # else:
                    if v < r0:
                        setattr(self, k, r0)
                    elif v > r1:
                        setattr(self, k, r1)
                    else :
                        setattr(self, k, v)

    def __repr__(self):
        fitness = None if self.fitness is None else round(self.fitness, 2)
        return self.__class__.__name__ + '(fitness:' + repr(fitness) + ' generation_num:' + repr(
            self.generation_num) + ')'

    def get(self, rounded=False):
        dic={}
        for k, vs in self.space_dict.items():
            v = getattr(self, k)
            if v is not None and rounded:
                if vs['dtype'] == float:
                    v = round(v, 2)
                elif vs['dtype'] == Tuple[float]:
                    v = (round(v[0], 2), round(v[1], 2))
            dic[k]=v
        return dic


    def to_string(self):
        fitness = None if self.fitness is None else round(self.fitness, 2)
        kwstrings = [f' {vs["name"]}:' + repr(self.get(rounded=True)[k]) for k, vs in self.space_dict.items()]
        # for k, vs in self.space_dict.items():
        #     kwstrings.append(f' {vs["name"]}:' + repr(self.get(rounded=True)[k]))
        kwstr = ''
        for ii in kwstrings:
            kwstr = kwstr + ii

        return self.__class__.__name__ + '(fitness:' + repr(fitness) + ' generation_num:' + repr(
            self.generation_num) + kwstr + ')'

    def get_saved_genome_repr(self):
        kwstr = ''
        for k in self.space_dict.keys():
            kwstr = kwstr + str(getattr(self, k)) + ' '
        return kwstr + str(self.generation_num) + ' ' + str(self.fitness)

    def build_robot(self, larva_pars, robot_class, unique_id, model):

        larva_pars_f = dNl.flatten_dict(larva_pars)

        for k in self.space_dict.keys():
            larva_pars_f[k] = getattr(self, k)
        larva2_pars = dNl.AttrDict.from_nested_dicts(unflatten(larva_pars_f))
        robot = robot_class(unique_id=unique_id, model=model, larva_pars=larva2_pars)
        return robot


if __name__ == '__main__':
    print(null_dict('GAselection'))
