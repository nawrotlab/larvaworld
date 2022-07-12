import copy
import random
import multiprocessing
import math
import threading
import time
from typing import Tuple

import pandas as pd
import progressbar
import numpy as np

import lib.aux.dictsNlists as dNl
# from lib.ga.util.genome import Genome


from lib.registry.pars import preg

from lib.ga.robot.larva_robot import LarvaRobot

from lib.ga.util.time_util import TimeUtil


class GAselector:
    def __init__(self, model, Ngenerations=None, Nagents=30, Nelits=3, Pmutation=0.3, Cmutation=0.1,
                 selection_ratio=0.3, verbose=0, bestConfID=None):
        self.M = preg.larva_conf_dict
        self.bestConfID = bestConfID
        self.model = model
        self.Ngenerations = Ngenerations
        self.Nagents = Nagents
        self.Nelits = Nelits
        self.Pmutation = Pmutation
        self.Cmutation = Cmutation
        self.selection_ratio = selection_ratio
        self.verbose = verbose
        self.sorted_genomes = None
        self.gConfs = []
        self.genome_df = None
        self.all_genomes_dic = []
        self.genome_dict = {}
        self.genome_dicts = {}
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

    def create_new_generation(self, space_dict):
        self.genome_dict={}
        self.gConfs =None
        self.generation_num += 1
        gConfs_selected = self.ga_selection()  # parents of the new generation
        self.printd(1, '\ngenomes selected:', gConfs_selected)

        self.gConfs = self.ga_crossover_mutation(gConfs_selected, space_dict)

        self.generation_step_num = 0
        self.generation_sim_time = 0
        self.start_generation_time = TimeUtil.current_time_millis()
        self.printd(1, '\nGeneration', self.generation_num, 'started')

    def sort_genomes(self):
        sorted_idx = sorted(list(self.genome_dict.keys()), key=lambda i: self.genome_dict[i].fitness, reverse=True)
        self.sorted_genomes = [self.genome_dict[i] for i in sorted_idx]
        if self.best_genome is None or self.sorted_genomes[0].fitness > self.best_genome.fitness:
            self.best_genome = self.sorted_genomes[0]
            self.best_fitness = self.best_genome.fitness

            if self.bestConfID is not None:
                self.M.saveConf(conf=self.best_genome.mConf, mID=self.bestConfID, verbose=self.verbose)
        end_generation_time = TimeUtil.current_time_millis()
        total_generation_time=end_generation_time-self.start_generation_time

        if self.verbose >= 2:
            print(f'Generation {self.generation_num} duration',total_generation_time)


    def ga_selection(self):
        num_gConfs_to_select = round(self.Nagents * self.selection_ratio)
        if num_gConfs_to_select < 2:
            raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                             'Please increase population (' + str(self.Nagents) + ') or selection ratio (' +
                             str(self.selection_ratio) + ')')

        gConfs_selected = []

        # elitism: keep the best genomes in the new generation
        for i in range(self.Nelits):
            g = self.sorted_genomes.pop(0)
            gConfs_selected.append(g.gConf)
            num_gConfs_to_select -= 1

        while num_gConfs_to_select > 0:
            g = self.roulette_select(self.sorted_genomes)
            gConfs_selected.append(g.gConf)
            self.sorted_genomes.remove(g)
            num_gConfs_to_select -= 1

        return gConfs_selected

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

    def ga_crossover_mutation(self, gConfs, space_dict):
        num_gConfs_to_create = self.Nagents
        new_gConfs = []

        # elitism: keep the best genomes in the new generation
        for i in range(self.Nelits):

            new_gConfs.append(gConfs[i])
            num_gConfs_to_create -= 1

        while num_gConfs_to_create > 0:
            gConf_a, gConf_b = self.choose_parents(gConfs)
            gConf = self.crossover(gConf_a, gConf_b)
            space_dict=self.M.update_mdict(space_dict,gConf)
            self.M.mutate(space_dict, Pmut=self.Pmutation, Cmut=self.Cmutation)
            gConf=self.M.conf(space_dict)
            new_gConfs.append(gConf)
            num_gConfs_to_create -= 1

        return new_gConfs

    def choose_parents(self, gConfs):
        pos_a = random.randrange(len(gConfs))
        gConf_a = gConfs[pos_a]
        gConfs.remove(gConf_a)  # avoid choosing the same parent two times
        pos_b = random.randrange(len(gConfs))
        gConf_b = gConfs[pos_b]
        gConfs.insert(pos_a, gConf_a)  # reinsert the first parent in the list
        return gConf_a, gConf_b

    def new_genome(self, gConf, mConf0):
        mConf = dNl.update_nestdict(mConf0, gConf)
        return dNl.NestDict({'fitness': None, 'fitness_dict': None, 'gConf': gConf, 'mConf': mConf})

    def crossover(self, gConf_a, gConf_b):
        gConf={}
        for k in gConf_a.keys():
            if np.random.uniform(0, 1, 1) >= 0.5:
                gConf[k]=gConf_a[k]
            else :
                gConf[k] = gConf_b[k]
        return gConf

        pass


# def initConf(init_mode, space_dict, mConf0):
#     if init_mode == 'random':
#         kws = {}
#         for k, vs in space_dict.items():
#             if vs['dtype'] == bool:
#                 kws[k] = random.choice([True, False])
#             elif vs['dtype'] == str:
#                 kws[k] = random.choice(vs['choices'])
#             elif vs['dtype'] == Tuple[float]:
#                 vv0 = random.uniform(vs['min'], vs['max'])
#                 vv1 = random.uniform(vv0, vs['max'])
#                 kws[k] = (vv0, vv1)
#             elif vs['dtype'] == int:
#                 kws[k] = random.randint(vs['min'], vs['max'])
#             else:
#                 kws[k] = random.uniform(vs['min'], vs['max'])
#         # initConf = dNl.update_nestdict(mConf0, kws)
#         # return initConf
#     elif init_mode == 'default':
#         kws = {k: vs['initial_value'] for k, vs in space_dict.items()}
#         # initConf = dNl.update_nestdict(mConf0, kws)
#         # return initConf
#     elif init_mode == 'model':
#         # print(mConf0)
#         kws = {k: dNl.flatten_dict(mConf0)[k] for k, vs in space_dict.items()}
#     return kws


class GAbuilder(GAselector):
    def __init__(self, scene, side_panel=None, space_mkeys=[], robot_class=LarvaRobot, base_model='explorer',
                 multicore=True, fitness_func=None, fitness_target_kws=None, fitness_target_refID=None,fit_dict =None,
                 exclude_func=None, plot_func=None, bestConfID=None, init_mode='random', progress_bar=True, **kwargs):
        super().__init__(bestConfID=bestConfID, **kwargs)


        self.is_running = True
        if progress_bar and self.Ngenerations is not None:
            self.progress_bar = progressbar.ProgressBar(self.Ngenerations)
            self.progress_bar.start()
        else:
            self.progress_bar = None
        if fit_dict is None :
            if fitness_target_kws is None:
                fitness_target_kws = {}
            from lib.ga.util.functions import arrange_fitness
            fit_dict = arrange_fitness(fitness_func, fitness_target_refID, fitness_target_kws,dt=self.model.dt,source_xy=self.model.source_xy)
        self.fit_dict =fit_dict
        self.exclude_func = exclude_func
        self.multicore = multicore
        self.scene = scene
        self.robot_class = robot_class

        self.mConf0 = self.M.loadConf(base_model)
        self.space_dict = self.M.space_dict(mkeys=space_mkeys, mConf0=self.mConf0)
        self.dataset0=self.init_dataset()



        if init_mode=='default' :
            gConf=self.M.conf(self.space_dict)
            self.gConfs=[gConf]*self.Nagents
        elif init_mode=='model':
            mF=dNl.flatten_dict(self.mConf0)
            gConf={k:mF[k] for k,p in self.space_dict.items()}
            self.gConfs = [gConf] * self.Nagents
        elif init_mode == 'random':
            self.gConfs=[]
            for i in range(self.Nagents):
                self.M.randomize(self.space_dict)
                gConf = self.M.conf(self.space_dict)
                self.gConfs.append(gConf)
        # print(self.space_dict.keys())
        # print(self.gConfs)
        self.robots = self.build_generation()
        self.step_df = self.init_step_df()

        self.printd(1, 'Generation', self.generation_num, 'started')
        self.printd(1, 'multicore:', self.multicore, 'num_cpu:', self.num_cpu)



    def init_dataset(self):
        c = dNl.NestDict(
            {'id': self.model.id, 'group_id': 'GA_robots', 'dt': self.model.dt, 'fr': 1 / self.model.dt,
             'agent_ids': np.arange(self.Nagents), 'duration': self.model.Nsteps * self.model.dt,
             'Npoints': 3, 'Ncontour': 0, 'point': '', 'N': self.Nagents, 'Nticks': self.model.Nsteps,
             'mID': self.bestConfID,
             'color': 'blue', 'env_params': self.model.env_pars})

        self.my_index = pd.MultiIndex.from_product([np.arange(c.Nticks), c.agent_ids],
                                                   names=['Step', 'AgentID'])
        self.df_columns = preg.getPar(['b', 'fov', 'rov', 'v', 'x', 'y'])
        self.df_Ncols = len(self.df_columns)

        e = pd.DataFrame(index=c.agent_ids)
        e['cum_dur'] = c.duration
        e['num_ticks'] = c.Nticks

        return dNl.NestDict({'step_data': None, 'endpoint_data': e, 'config': c})

    def init_step_df(self):
        self.dataset = dNl.copyDict(self.dataset0)

        step_df = np.ones([self.dataset.config.Nticks, self.dataset.config.N, self.df_Ncols]) * np.nan
        self.dataset.endpoint_data['length'] = [robot.real_length for robot in self.robots]
        # print(self.dataset.endpoint_data['length'])
        return step_df

    def finalize_step_df(self):
        # t0 = time.time()

        e, c = self.dataset.endpoint_data, self.dataset.config
        self.step_df[:, :, :3] = np.rad2deg(self.step_df[:, :, :3])
        self.step_df = self.step_df.reshape(c.Nticks * c.N, self.df_Ncols)
        s = pd.DataFrame(self.step_df, index=self.my_index, columns=self.df_columns)
        s = s.astype(float)


        from lib.process.spatial import scale_to_length

        scale_to_length(s, e, c, pars=None, keys=['v'])
        self.dataset.step_data = s
        # FK=self.fit_dict.keys
        for k in self.fit_dict.keys:
            preg.compute(k, self.dataset)

        for i, g in self.genome_dict.items():
            if g.fitness is None:
                # ss = self.dataset.step_data.xs(i, level='AgentID')
                gdict =self.fit_dict.robot_func(ss=self.dataset.step_data.xs(i, level='AgentID'))
                g.fitness, g.fitness_dict = self.get_fitness(gdict)
                # print(i,g.fitness, g.fitness_dict )

    def build_generation(self):
        robots = []
        for i, gConf in enumerate(self.gConfs):
            g=self.new_genome(gConf, self.mConf0)
            self.genome_dict[i] = g



            robot = self.robot_class(unique_id=i, model=self.model, larva_pars=g.mConf)
            robot.genome = g
            robots.append(robot)
            self.scene.put(robot)

        if self.multicore:
            self.threads=self.build_threads(robots)

        return robots

    def step(self):

        if self.multicore:

            for thr in self.threads:
                thr.step()



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
            # print(self.generation_num, self.best_fitness)

            if self.model.sim_params.store_data:
                self.all_genomes_dic += [
                {'generation': self.generation_num, **{p.name : g.gConf[k] for k,p in self.space_dict.items()},
                 'fitness': g.fitness, **dNl.flatten_dict(g.fitness_dict)}
                for g in self.sorted_genomes if g.fitness_dict is not None]

            if self.Ngenerations is None or self.generation_num < self.Ngenerations:

                self.create_new_generation(self.space_dict)
                if self.progress_bar:
                    self.progress_bar.update(self.generation_num)

                self.robots = self.build_generation()
                self.step_df = self.init_step_df()

            else:
                self.finalize()

    def destroy_robot(self, robot, excluded=False):
        if excluded:
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
        self.printd(0, 'Best fittness:', self.best_genome.fitness)
        if self.model.sim_params.store_data :
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.model.dir_path)



    def check(self, robot):
        # print(self.model.sim_params.store_data)
        # raise
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

    def store_genomes(self, dic, save_to):
        self.genome_df = pd.DataFrame.from_records(dic)
        self.genome_df = self.genome_df.round(3)
        self.genome_df.sort_values(by='fitness', ascending=False, inplace=True)
        preg.graph_dict.dict['mpl'](data=self.genome_df, font_size=18, save_to=save_to,
                                    name=self.bestConfID, verbose=self.verbose)
        self.genome_df.to_csv(f'{save_to}/{self.bestConfID}.csv')
        # print(f'GA dataframe saved at {filepath}')

    def build_threads(self, robots):
        # if self.multicore:

        threads = []
        num_robots = len(robots)
        num_robots_per_cpu = math.floor(num_robots / self.num_cpu)

        self.printd(2, 'num_robots_per_cpu:', num_robots_per_cpu)

        for i in range(self.num_cpu - 1):
            start_pos = i * num_robots_per_cpu
            end_pos = (i + 1) * num_robots_per_cpu
            self.printd(2, 'core:', i + 1, 'positions:', start_pos, ':', end_pos)
            robot_list = robots[start_pos:end_pos]

            thread = GA_thread(robot_list)
            thread.start()
            self.printd(2, 'thread', i + 1, 'started')
            threads.append(thread)

        # last sublist of robots
        start_pos = (self.num_cpu - 1) * num_robots_per_cpu
        self.printd(2, 'last core, start_pos', start_pos)
        robot_list = robots[start_pos:]

        thread = GA_thread(robot_list)
        thread.start()
        self.printd(2, 'last thread started')
        threads.append(thread)

        for t in threads:
            t.join()
        return threads


class GA_thread(threading.Thread):
    def __init__(self, robots):
        threading.Thread.__init__(self)
        self.robots = robots

    def step(self):
        for robot in self.robots:
            robot.sense_and_act()


if __name__ == '__main__':
    # print(null_dict.__class__)
    pass
