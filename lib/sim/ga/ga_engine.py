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

from lib.aux import dictsNlists as dNl, naming as nam, colsNstr as cNs
from lib.aux.colsNstr import get_class_by_name
# from lib.ga.util.genome import Genome
from lib.sim.ga.functions import GA_optimization, get_robot_class

from lib.registry.pars import preg
from lib.registry import reg
from lib.model.robot.larva_robot import LarvaRobot

from lib.aux.time_util import TimeUtil


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
                self.M.saveConf(conf=self.best_genome.mConf, mID=self.bestConfID)
        end_generation_time = TimeUtil.current_time_millis()
        total_generation_time=end_generation_time-self.start_generation_time
        reg.vprint(f'Generation {self.generation_num} best_fitness : {self.best_fitness}',2)
        # reg.vprint(f'Generation {self.generation_num} duration : {total_generation_time}',self.verbose)




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
            gConf0 = self.crossover(gConf_a, gConf_b)
            space_dict=self.M.update_mdict(space_dict,gConf0)
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
        return dNl.NestDict({'fitness': None, 'fitness_dict': {}, 'gConf': gConf, 'mConf': mConf})

    def crossover(self, gConf_a, gConf_b):
        gConf={}
        for k in gConf_a.keys():
            if np.random.uniform(0, 1, 1) >= 0.5:
                gConf[k]=gConf_a[k]
            else :
                gConf[k] = gConf_b[k]
        return gConf

        pass


class GAbuilder(GAselector):
    def __init__(self, viewer, side_panel=None, space_mkeys=[], robot_class=None, base_model='explorer',
                 multicore=True, fitness_func=None, fitness_target_kws=None, fitness_target_refID=None,fit_dict =None,
                 exclude_func=None, plot_func=None,exclusion_mode=False, bestConfID=None, init_mode='random', progress_bar=True, **kwargs):
        super().__init__(bestConfID=bestConfID, **kwargs)

        self.is_running = True
        if progress_bar and self.Ngenerations is not None:
            self.progress_bar = progressbar.ProgressBar(self.Ngenerations)
            self.progress_bar.start()
        else:
            self.progress_bar = None
        self.exclude_func = exclude_func
        self.multicore = multicore
        self.viewer = viewer
        self.robot_class = get_robot_class(robot_class, self.model.offline)
        self.mConf0 = self.M.loadConf(base_model)
        self.space_dict = self.M.space_dict(mkeys=space_mkeys, mConf0=self.mConf0)
        self.excluded_ids = []
        self.Nagents_min = round(self.Nagents * self.selection_ratio)
        self.exclusion_mode = exclusion_mode

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

        self.robots = self.build_generation()





        if self.exclusion_mode :
            self.fit_dict =None
            self.dataset =None
            self.step_df =None


        else :
            if fit_dict is None :
                if fitness_target_refID is not None:
                    fit_dict=GA_optimization(fitness_target_refID, fitness_target_kws)
                else :
                    from lib.sim.ga.functions import arrange_fitness
                    fit_dict = arrange_fitness(fitness_func,source_xy=self.model.source_xy)
            self.fit_dict =fit_dict
            arg=self.fit_dict.func_arg
            if arg=='s':

                self.dataset0=self.init_dataset()
                self.step_df = self.init_step_df()
            elif arg=='robot':

                self.dataset = None
                self.step_df = None

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
        return step_df

    def finalize_step_df(self):
        e, c = self.dataset.endpoint_data, self.dataset.config
        self.step_df[:, :, :3] = np.rad2deg(self.step_df[:, :, :3])
        self.step_df = self.step_df.reshape(c.Nticks * c.N, self.df_Ncols)
        s = pd.DataFrame(self.step_df, index=self.my_index, columns=self.df_columns)
        s = s.astype(float)

        s.drop(self.excluded_ids, level='AgentID', axis=0, inplace=True)
        e=e.drop(index=self.excluded_ids)
        for id in self.excluded_ids:
            self.genome_dict.pop(id, None)
        self.dataset.config.agent_ids = s.index.unique('AgentID').values
        self.dataset.config.N=len(self.dataset.config.agent_ids)

        # s[nam.scal('velocity')] = pd.concat([g / e['length'].loc[id] for id, g in s['velocity'].groupby('AgentID')])
        # s[preg.getPar('sv')] = (s[preg.getPar('v')].values.T / ls).T


        from lib.process.spatial import scale_to_length

        scale_to_length(s, e, c, pars=None, keys=['v'])
        self.dataset.step_data = s
        if 'keys' in self.fit_dict.keys():
            for k in self.fit_dict.keys:
                preg.par_dict.compute(k, self.dataset)
        fit_dicts=self.fit_dict.func(s=self.dataset.step_data)

        valid_gs={}
        for i, g in self.genome_dict.items():
            g.fitness_dict = {k: dic[i] for k, dic in fit_dicts.items()}
            mus = {k: -np.mean(list(dic.values())) for k, dic in g.fitness_dict.items()}
            if len(mus) == 1:
                g.fitness = list(mus.values())[0]
            else:
                coef_dict = {'KS': 10, 'RSS': 1}
                g.fitness = np.sum([coef_dict[k] * mean for k, mean in mus.items()])
            if not np.isnan(g.fitness) :

                valid_gs[i]=g
        self.genome_dict = valid_gs






    def eval(self, gd):
        for i, g in gd.items():
            if g.fitness is None:
                gdict =self.fit_dict.robot_func(ss=g['step'])
                g.fitness, g.fitness_dict = self.get_fitness(gdict)


    def build_generation(self):
        robots = []
        for i, gConf in enumerate(self.gConfs):
            g=self.new_genome(gConf, self.mConf0)
            self.genome_dict[i] = g



            robot = self.robot_class(unique_id=i, model=self.model, larva_pars=g.mConf)
            robot.genome = g
            robots.append(robot)
            self.viewer.put(robot)

        if self.multicore:
            self.threads=self.build_threads(robots)

        return robots

    def step(self):
        if self.step_df is not None:
            for robot in self.robots[:]:
                self.step_df[self.generation_step_num, robot.unique_id, :] = robot.collect

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


        if self.generation_step_num == self.model.Nsteps or len(self.robots)<=self.Nagents_min:
            self.end_generation()
            if self.Ngenerations is None or self.generation_num < self.Ngenerations:
                self.excluded_ids = []
                self.create_new_generation(self.space_dict)
                if self.progress_bar:
                    self.progress_bar.update(self.generation_num)
                self.robots = self.build_generation()
                if not self.exclusion_mode and self.dataset is not None:
                    self.step_df = self.init_step_df()
            else:
                self.finalize()






    def end_generation(self):
        if self.step_df is not None:
            self.finalize_step_df()
        else :
            self.eval_robots()



        for robot in self.robots[:]:
            self.destroy_robot(robot)

        # check population extinction
        # if not self.robots:



        self.sort_genomes()

        if self.model.sim_params.store_data:
            self.all_genomes_dic += [
            {'generation': self.generation_num, **{p.name : g.gConf[k] for k,p in self.space_dict.items()},
             'fitness': g.fitness, **dNl.flatten_dict(g.fitness_dict)}
            for g in self.sorted_genomes if g.fitness_dict is not None]






    def destroy_robot(self, robot, excluded=False):
        if excluded:

            self.excluded_ids.append(robot.unique_id)

            robot.genome.fitness = -np.inf
        if self.exclusion_mode:
            robot.genome.fitness =robot.Nticks
        self.viewer.remove(robot)
        self.robots.remove(robot)


    def get_fitness2(self, robot):
        if self.fit_dict.func is not None:
            return self.fit_dict.func(robot, **self.fit_dict.target_kws)
        else:
            return None

    def get_fitness2(self, gdict):
        return self.fit_dict.func(gdict, **self.fit_dict.target_kws)

    def finalize(self):
        self.is_running = False
        if self.progress_bar:
            self.progress_bar.finish()
        self.printd(0, 'Best fittness:', self.best_genome.fitness)
        if self.model.sim_params.store_data :
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.model.data_dir)



    def check(self, robot):
        if not self.model.offline:
            if robot.x < 0 or robot.x > self.viewer.width or robot.y < 0 or robot.y > self.viewer.height:
                # if robot.x < 0 or robot.x > self.viewer.width or robot.y < 0 or robot.y > self.viewer.height:
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
                                    name=self.bestConfID)
        self.genome_df.to_csv(f'{save_to}/{self.bestConfID}.csv')

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

    def eval_robots(self):
        for robot in self.robots :
            i=robot.unique_id
            self.genome_dict[i].fitness=self.fit_dict.func(robot)


class GA_thread(threading.Thread):
    def __init__(self, robots):
        threading.Thread.__init__(self)
        self.robots = robots

    def step(self):
        for robot in self.robots:
            robot.sense_and_act()

