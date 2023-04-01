import random
import multiprocessing
import math
import sys
import threading
import warnings

import agentpy
import pandas as pd
import progressbar
import numpy as np

from larvaworld.lib import reg, aux, util
from larvaworld.lib.screen import Viewer, GA_ScreenManager
from larvaworld.lib.sim.base_run import BaseRun


class GAlauncher(BaseRun):
    def __init__(self, **kwargs):
        super().__init__(runtype = 'Ga', **kwargs)

    def setup(self):
        self.collections=['pose']


        self.odor_ids = aux.get_all_odors({}, self.p.env_params.food_params)
        self.build_env(self.p.env_params)

        self.screen_manager=GA_ScreenManager(model=self,show_display=self.show_display,
                                           panel_width=600,caption = f'GA {self.p.experiment} : {self.id}',
                                           space_bounds=aux.get_arena_bounds(self.space.dims, self.scaling_factor))
        self.initialize(**self.p.ga_build_kws, **self.p.ga_select_kws)


    def simulate(self):
        self.running = True
        self.setup(**self._setup_kwargs)
        while self.running:
            self.t+=1
            self.engine.sim_step()
            self.screen_manager.render(self.t)
        return self.engine.best_genome

    def initialize(self, **kwargs):
        self.engine = GAengine(model=self, **kwargs)

class GAselector:
    def __init__(self, model, Ngenerations=None, Nagents=30, Nelits=3, Pmutation=0.3, Cmutation=0.1,
                 selection_ratio=0.3, verbose=0, bestConfID=None, **kwargs):
        super().__init__(**kwargs)
        self.bestConfID = bestConfID
        self.model = model
        self.Ngenerations = Ngenerations
        self.Nagents = Nagents
        self.Nelits = Nelits

        self.selection_ratio = selection_ratio

        self.Nagents_min = round(self.Nagents * self.selection_ratio)
        if self.Nagents_min < 2:
            raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                             'Please increase population (' + str(self.Nagents) + ') or selection ratio (' +
                             str(self.selection_ratio) + ')')
        self.Pmutation = Pmutation
        self.Cmutation = Cmutation
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
        self.start_total_time = aux.TimeUtil.current_time_millis()
        self.start_generation_time = self.start_total_time
        self.generation_step_num = 0
        self.generation_sim_time = 0

    def create_new_generation(self, space_dict):
        self.genome_dict={}
        # self.gConfs =None
        self.generation_num += 1
        gConfs_selected = self.ga_selection()  # parents of the new generation
        reg.vprint(f'genomes selected: {gConfs_selected}', 1)

        self.gConfs = self.ga_crossover_mutation(gConfs_selected, space_dict)

        self.generation_step_num = 0
        self.generation_sim_time = 0
        self.start_generation_time = aux.TimeUtil.current_time_millis()
        reg.vprint(f'Generation {self.generation_num} started', 1)

    def sort_genomes(self):
        sorted_idx = sorted(list(self.genome_dict.keys()), key=lambda i: self.genome_dict[i].fitness, reverse=True)
        self.sorted_genomes = [self.genome_dict[i] for i in sorted_idx]

        if self.best_genome is None or self.sorted_genomes[0].fitness > self.best_genome.fitness:
            self.best_genome = self.sorted_genomes[0]
            self.best_fitness = self.best_genome.fitness

            if self.bestConfID is not None:
                reg.saveConf(conf=self.best_genome.mConf, conftype='Model', id=self.bestConfID)
        reg.vprint(f'Generation {self.generation_num} best_fitness : {self.best_fitness}',2)




    def ga_selection(self):
        gConfs_selected = []

        # elitism: keep the best genomes in the new generation
        for i in range(self.Nelits):
            g = self.sorted_genomes.pop(0)
            gConfs_selected.append(g.gConf)

        while len(gConfs_selected) < self.Nagents_min:
            g = self.roulette_select(self.sorted_genomes)
            gConfs_selected.append(g.gConf)
            self.sorted_genomes.remove(g)
        return gConfs_selected

    def roulette_select(self, genomes):
        fitness_sum = 0
        for g in genomes:
            fitness_sum += g.fitness
        v = random.uniform(0, fitness_sum)
        for i in range(len(genomes)):
            v -= genomes[i].fitness
            if v < 0:
                return genomes[i]
        return genomes[-1]

    def ga_crossover_mutation(self, gConfs, space_dict):
        new_gConfs = []

        # elitism: keep the best genomes in the new generation
        for i in range(self.Nelits):
            new_gConfs.append(gConfs[i])

        while len(new_gConfs) < self.Nagents:
            gConf_a, gConf_b = self.choose_parents(gConfs)
            gConf0 = self.crossover(gConf_a, gConf_b)
            space_dict=reg.model.update_mdict(space_dict,gConf0)
            reg.model.mutate(space_dict, Pmut=self.Pmutation, Cmut=self.Cmutation)
            gConf=reg.model.conf(space_dict)
            new_gConfs.append(gConf)
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
        mConf = mConf0.update_nestdict(gConf)
        return aux.AttrDict({'fitness': None, 'fitness_dict': {}, 'gConf': gConf, 'mConf': mConf})

    def crossover(self, gConf_a, gConf_b):
        gConf={}
        for k in gConf_a.keys():
            if np.random.uniform(0, 1, 1) >= 0.5:
                gConf[k]=gConf_a[k]
            else :
                gConf[k] = gConf_b[k]
        return gConf

class GAengine(GAselector):
    def __init__(self, space_mkeys=[], robot_class=None, base_model='explorer',
                 multicore=True, fitness_func=None, fitness_target_kws=None, fitness_target_refID=None,fit_dict =None,
                 exclude_func=None, exclusion_mode=False, bestConfID=None, init_mode='random', progress_bar=True, **kwargs):
        super().__init__(bestConfID=bestConfID, **kwargs)
        if progress_bar and self.Ngenerations is not None:
            self.progress_bar = progressbar.ProgressBar(self.Ngenerations)
            self.progress_bar.start()
        else:
            self.progress_bar = None
        self.exclude_func = exclude_func
        self.multicore = multicore
        self.robot_class = get_robot_class(robot_class, self.model.offline)
        self.mConf0 = reg.loadConf(id=base_model, conftype='Model')
        self.space_dict = reg.model.space_dict(mkeys=space_mkeys, mConf0=self.mConf0)
        self.excluded_ids = []
        self.exclusion_mode = exclusion_mode

        self.gConfs=self.create_first_generation(init_mode, self.Nagents, self.space_dict, self.mConf0)

        self.build_generation()

        if self.exclusion_mode :
            self.fit_dict =None
        else :
            if fit_dict is None :
                if fitness_target_refID is not None:
                    fit_dict=util.GA_optimization(fitness_target_refID, fitness_target_kws)
                else :
                    fit_dict = arrange_fitness(fitness_func,source_xy=self.model.source_xy)
            self.fit_dict =fit_dict

        reg.vprint(f'Generation {self.generation_num} started', 1)
        reg.vprint(f'multicore: {self.multicore} num_cpu: {self.num_cpu}', 1)


    def create_first_generation(self, mode, N, space_dict, baseConf):
        if mode=='default' :
            gConf=reg.model.conf(space_dict)
            gConfs=[gConf]*N
        elif mode=='model':
            gConf={k:baseConf.flatten()[k] for k,p in space_dict.items()}
            gConfs = [gConf] * N
        elif mode == 'random':
            gConfs=[]
            for i in range(N):
                reg.model.randomize(space_dict)
                gConf = reg.model.conf(space_dict)
                gConfs.append(gConf)
        return gConfs



    def build_generation(self):
        self.genome_dict = {i : self.new_genome(gConf, self.mConf0) for i, gConf in enumerate(self.gConfs)}

        confs= [{'larva_pars' : g.mConf, 'unique_id' : id, 'genome' : g} for id, g in self.genome_dict.items()]
        self.model.place_agents(confs, self.robot_class)

        # robots = []
        # for i, gConf in enumerate(self.gConfs):
        #     g=self.new_genome(gConf, self.mConf0)
        #     self.genome_dict[i] = g
        #
        #     robot = self.robot_class(unique_id=i, model=self.model, larva_pars=g.mConf)
        #     robot.genome = g
        #     robots.append(robot)
        # self.model.space.add_agents(robots, positions=[a.pos for a in robots])
        # self.robots = agentpy.AgentList(model=self.model, objs=robots)
        self.collectors = reg.get_reporters(collections=self.model.collections, agents=self.model.agents)
        if self.multicore:
            self.threads=self.build_threads(self.model.agents)


    def sim_step(self):
        self.step()
        self.update()
        self.generation_sim_time += self.model.dt
        self.generation_step_num += 1
        if self.generation_step_num == self.model.Nsteps or len(self.model.agents) <= self.Nagents_min:
            self.end_generation()
            if self.Ngenerations is None or self.generation_num < self.Ngenerations:
                self.excluded_ids = []
                self.create_new_generation(self.space_dict)
                if self.progress_bar:
                    self.progress_bar.update(self.generation_num)
                self.build_generation()
            else:
                self.finalize()

    def step(self):
        if self.multicore:
            for thr in self.threads:
                thr.step()
            for robot in self.model.agents:
                self.check(robot)
        else:
            for robot in self.model.agents:
                robot.step()
                self.check(robot)

    def update(self):
        self.model.agents.nest_record(self.collectors['step'])


    def end_generation(self):
        self.model.agents.nest_record(self.collectors['end'])
        self.model.create_output()
        self.eval_robots2(self.model.output.variables)
        for a in self.model.agents[:]:
            self.destroy_robot(a)
        self.sort_genomes()

        if self.model.store_data:
            self.all_genomes_dic += [
            {'generation': self.generation_num, **{p.name : g.gConf[k] for k,p in self.space_dict.items()},
             'fitness': g.fitness, **g.fitness_dict.flatten()}
            for g in self.sorted_genomes if g.fitness_dict is not None]
        self.model._logs = {}
        self.model.t = 0

    def destroy_robot(self, robot, excluded=False):
        if excluded:
            self.excluded_ids.append(robot.unique_id)
            robot.genome.fitness = -np.inf
        if self.exclusion_mode:
            robot.genome.fitness =robot.cum_dur
        self.model.agents.remove(robot)
        self.model.space.remove_agents([robot])

    def finalize(self):
        self.model.running = False
        if self.progress_bar:
            self.progress_bar.finish()
        reg.vprint(f'Best fittness: {self.best_genome.fitness}', 2)
        if self.model.store_data :
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.model.data_dir)



    def check(self, robot):
        if not self.model.offline:
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
        reg.graphs.dict['mpl'](data=self.genome_df, font_size=18, save_to=save_to,
                                    name=self.bestConfID)
        self.genome_df.to_csv(f'{save_to}/{self.bestConfID}.csv')

    def build_threads(self, robots):
        threads = []
        num_robots = len(robots)
        num_robots_per_cpu = math.floor(num_robots / self.num_cpu)

        reg.vprint(f'num_robots_per_cpu: {num_robots_per_cpu}', 2)

        for i in range(self.num_cpu - 1):
            start_pos = i * num_robots_per_cpu
            end_pos = (i + 1) * num_robots_per_cpu
            reg.vprint(f'core: {i + 1} positions: {start_pos} : {end_pos}', 1)
            robot_list = robots[start_pos:end_pos]

            thread = GA_thread(robot_list)
            thread.start()
            reg.vprint(f'thread {i + 1} started', 1)
            threads.append(thread)

        # last sublist of robots
        start_pos = (self.num_cpu - 1) * num_robots_per_cpu
        reg.vprint(f'last core, start_pos {start_pos}', 1)
        robot_list = robots[start_pos:]

        thread = GA_thread(robot_list)
        thread.start()
        reg.vprint(f'last thread started', 1)
        threads.append(thread)

        for t in threads:
            t.join()
        return threads

    def convert_output_to_dataset(self,df, id):
        from larvaworld.lib.process.dataset import LarvaDataset
        df.index.set_names(['AgentID', 'Step'], inplace=True)
        df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
        df.sort_index(level=['Step', 'AgentID'], inplace=True)

        end = df[list(self.collectors['end'].keys())].xs(df.index.get_level_values('Step').max(), level='Step')
        step = df[list(self.collectors['step'].keys())]
        d = LarvaDataset(dir=None, id=id,
                         load_data=False, env_params=self.model.p.env_params,
                         source_xy=self.model.source_xy,
                         fr=1 / self.model.dt)
        d.set_data(step=step, end=end, food=None)
        return d


    def eval_robots2(self, variables):
        for gID,df in variables.items():
            d=self.convert_output_to_dataset(df.copy(), id=f'{self.model.id}_generation:{self.generation_num}')
            d._enrich(proc_keys=['angular', 'spatial'])

            s, e, c = d.step_data, d.endpoint_data, d.config

            fit_dicts=self.fit_dict.func(s=s)

            valid_gs={}
            for i, g in self.genome_dict.items():
                g.fitness_dict = aux.AttrDict({k: dic[i] for k, dic in fit_dicts.items()})
                mus = aux.AttrDict({k: -np.mean(list(dic.values())) for k, dic in g.fitness_dict.items()})
                if len(mus) == 1:
                    g.fitness = list(mus.values())[0]
                else:
                    coef_dict = {'KS': 10, 'RSS': 1}
                    g.fitness = np.sum([coef_dict[k] * mean for k, mean in mus.items()])
                if not np.isnan(g.fitness) :
                    valid_gs[i]=g
            self.genome_dict = valid_gs


class GA_thread(threading.Thread):
    def __init__(self, robots):
        threading.Thread.__init__(self)
        self.robots = robots

    def step(self):
        for robot in self.robots:
            robot.step()

def get_robot_class(robot_class=None, offline=False):
    if offline:
        robot_class = 'LarvaOffline'
    if robot_class is None:
        robot_class = 'LarvaRobot'

    if type(robot_class) == str:
        if robot_class == 'LarvaRobot':
            class_name = f'larvaworld.lib.model.agents.larva_robot.LarvaRobot'
        elif robot_class == 'ObstacleLarvaRobot':
            class_name = f'larvaworld.lib.model.agents.larva_robot.ObstacleLarvaRobot'
        elif robot_class == 'LarvaOffline':
            class_name = f'larvaworld.lib.model.agents.larva_offline.LarvaOffline'
        else :
            raise
        return aux.get_class_by_name(class_name)
    elif type(robot_class) == type:
        return robot_class

def arrange_fitness(fitness_func, **kwargs):
    def func(robot):
        return fitness_func(robot, **kwargs)
    return aux.AttrDict({'func': func, 'func_arg': 'robot'})





def optimize_mID(mID0, mID1=None, fit_dict=None, refID=None, space_mkeys=['turner', 'interference'], init='model',
               exclusion_mode=False,experiment='exploration',
                 id=None, dt=1 / 16, dur=0.5, save_to=None,  Nagents=30, Nelits=6, Ngenerations=20,
                 **kwargs):

    warnings.filterwarnings('ignore')
    if mID1 is None:
        mID1 = mID0



    kws = {
        # 'sim_params': reg.get_null('sim_params', duration=dur,dt=dt),
        # 'show_display': show_display,
        # 'offline': offline,
        # 'store_data': store_data,
        'experiment': experiment,
        'env_params': 'arena_200mm',
        'ga_select_kws': reg.get_null('ga_select_kws', Nagents=Nagents, Nelits=Nelits, Ngenerations=Ngenerations, selection_ratio=0.1),
        'ga_build_kws': reg.get_null('ga_build_kws', init_mode=init, space_mkeys=space_mkeys, base_model=mID0,exclusion_mode=exclusion_mode,
                                      bestConfID=mID1, fitness_target_refID=refID)
    }

    conf = reg.get_null('Ga', **kws)
    conf.env_params = reg.expandConf(id=conf.env_params, conftype='Env')

    conf.ga_build_kws.fit_dict = fit_dict

    GA = GAlauncher(parameters=conf, save_to=save_to, id=id, duration=dur,dt=dt)
    best_genome = GA.simulate()
    entry = {mID1: best_genome.mConf}
    return entry
