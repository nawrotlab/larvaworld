import random
import multiprocessing
import math
import threading
import warnings
import param
import pandas as pd
import progressbar
import numpy as np

from larvaworld.lib import reg, aux, util
from larvaworld.lib.screen import GA_ScreenManager
from larvaworld.lib.sim.base_run import BaseRun




class GAselector(param.Parameterized):
    Ngenerations = param.Integer(default=None, allow_None=True,
                                 doc='Number of generations to run for the genetic algorithm engine')
    Nagents = param.Integer(default=30, doc='Number of agents per generation')
    Nelits = param.Integer(default=3, doc='Number of best agents to include in the next generation')

    selection_ratio = param.Magnitude(default=0.3, doc='Fraction of agent population to include in the next generation')
    Pmutation = param.Magnitude(default=0.3, doc='Probability of mutation for each agent in the next generation')
    Cmutation = param.Number(default=0.1, doc='Fraction of allowed parameter range to mutate within')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Nagents_min = round(self.Nagents * self.selection_ratio)
        if self.Nagents_min < 2:
            raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                             'Please increase population (' + str(self.Nagents) + ') or selection ratio (' +
                             str(self.selection_ratio) + ')')

    def crossover(self, g1, g2):
        g = {}
        for k in g1.keys():
            if np.random.uniform(0, 1, 1) >= 0.5:
                g[k] = g1[k]
            else:
                g[k] = g2[k]
        return g


class GAconf_generator(GAselector):
    init_mode = param.Selector(default='random', objects=['random', 'model', 'default'],
                               doc='Mode of initial generation')
    base_model = param.Selector(default='explorer', objects=reg.storedConf('Model'), doc='ID of the model to optimize')
    bestConfID = param.String(default=None, doc='ID for the optimized model')
    space_mkeys = param.List(default=[], doc='Keys of the modules where the optimization parameters are')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mConf0 = reg.loadConf(id=self.base_model, conftype='Model')
        self.space_dict = reg.model.space_dict(mkeys=self.space_mkeys, mConf0=self.mConf0)
        self.space_columns = [p.name for k, p in self.space_dict.items()]
        self.gConf0 = reg.model.conf(self.space_dict)



    def create_first_generation(self):
        mode=self.init_mode
        N=self.Nagents
        d=self.space_dict
        if mode == 'default':
            gConf = reg.model.conf(d)
            gConfs = [gConf] * N
        elif mode == 'model':
            gConf = {k: self.mConf0.flatten()[k] for k, p in d.items()}
            gConfs = [gConf] * N
        elif mode == 'random':
            gConfs = []
            for i in range(N):
                reg.model.randomize(d)
                gConf = reg.model.conf(d)
                gConfs.append(gConf)
        else :
            raise ValueError('Not implemented')
        return gConfs

    def new_genome(self, gConf, mConf0):
        mConf = mConf0.update_nestdict(gConf)
        return aux.AttrDict({'fitness': None, 'fitness_dict': {}, 'gConf': gConf, 'mConf': mConf})

    def create_new_generation(self, sorted_gs):
        if len(sorted_gs) < self.Nagents_min:
            raise ValueError(
                f'The number of genomes ({len(sorted_gs)}) is lower than the minimum number required to breed a new generation ({self.Nagents_min})')
        gs0 = [sorted_gs[i].gConf for i in range(self.Nagents_min)]

        # elitism: keep the best genomes in the new generation
        gs = [gs0[i] for i in range(self.Nelits)]

        for i in range(self.Nagents - self.Nelits):
            g1, g2 = random.sample(gs0, 2)
            g0 = self.crossover(g1, g2)
            space_dict = reg.model.update_mdict(self.space_dict, g0)
            for d, p in space_dict.items():
                p.mutate(Pmut=self.Pmutation, Cmut=self.Cmutation)

            g = reg.model.generate_configuration(space_dict)
            gs.append(g)
        return gs

    def create_generation(self, sorted_gs=None):
        if sorted_gs is None :
            self.gConfs = self.create_first_generation()
        else :
            self.gConfs = self.create_new_generation(sorted_gs)
        self.genome_dict = {i: self.new_genome(gConf, self.mConf0) for i, gConf in enumerate(self.gConfs)}

class GAevaluator(GAconf_generator):
    robot_class = param.String(default=None, doc='The agent class', allow_None=True)
    multicore = param.Boolean(default=True, doc='Whether to use parallel processing')
    exclusion_mode = param.Boolean(default=False, doc='Whether to apply exclusion mode')
    offline = param.Boolean(default=False, doc='Whether to simulate offline')
    exclude_func = param.Callable(default=None, doc='The function that evaluates exclusion', allow_None=True)
    fitness_func = param.Callable(default=None, doc='The function that evaluates fitness', allow_None=True)
    fitness_target_refID = param.Selector(default=None, allow_None=True, objects=reg.storedConf('Ref'), doc='ID of the reference dataset')
    fitness_target_kws = param.Parameter(default=None, doc='The target metrics to optimize against')
    fit_dict = param.Dict(default=None, doc='The complete dictionary of the fitness evaluation process')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def __init__(self, ga_select_kws, robot_class=None, multicore=True, exclude_func=None, exclusion_mode=False,
    #              space_mkeys=[], base_model='explorer', bestConfID=None, init_mode='random', offline=False, **kwargs):
    #     super().__init__(ga_select_kws=ga_select_kws, space_mkeys=space_mkeys, base_model=base_model,
    #                      bestConfID=bestConfID, init_mode=init_mode)
    #     self.exclusion_mode = exclusion_mode
    #     self.exclude_func = exclude_func
    #     self.multicore = multicore
        self.robot_class_func = get_robot_class(self.robot_class, self.offline)

        self.best_genome = None
        self.best_fitness = None

        self.all_genomes_dic = []

        self.num_cpu = multiprocessing.cpu_count()

        if self.exclusion_mode:
            self.fit_dict = None
        elif self.fit_dict is None:
            if self.fitness_target_refID is not None:
                self.fit_dict = util.GA_optimization(self.fitness_target_refID, self.fitness_target_kws)
            else:
                self.fit_dict = arrange_fitness(self.fitness_func, source_xy=self.source_xy)







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
        else:
            raise
        return aux.get_class_by_name(class_name)
    elif type(robot_class) == type:
        return robot_class


def arrange_fitness(fitness_func, **kwargs):
    def func(robot):
        return fitness_func(robot, **kwargs)

    return aux.AttrDict({'func': func, 'func_arg': 'robot'})


class GAlauncher(BaseRun, GAevaluator):
    def __init__(self, **kwargs):
        BaseRun.__init__(self,runtype='Ga', **kwargs)
        GAevaluator.__init__(self, **self.p.ga_build_kws, **self.p.ga_select_kws, offline=self.offline)
    def setup(self):
        # GAevaluator.__init__(self, **self.p.ga_build_kws, ga_select_kws=self.p.ga_select_kws, offline=self.offline)
        # self.selector = GAevaluator(**self.p.ga_build_kws, ga_select_kws=self.p.ga_select_kws, offline=self.offline)
        self.generation_num = 0
        self.start_total_time = aux.TimeUtil.current_time_millis()
        reg.vprint(f'--- Genetic Algorithm  "{self.id}" initialized!--- ', 2)
        temp = self.Ngenerations if self.Ngenerations is not None else 'unlimited'
        reg.vprint(f'Launching {temp} generations of {self.duration} seconds, with {self.Nagents} agents each!', 2)
        # reg.vprint('', 2)
        if self.Ngenerations is not None:
            self.progress_bar = progressbar.ProgressBar(self.Ngenerations)
            self.progress_bar.start()


        else:
            self.progress_bar = None
        self.collections = ['pose']

        self.odor_ids = aux.get_all_odors({}, self.p.env_params.food_params)
        self.build_env(self.p.env_params)

        self.screen_manager = GA_ScreenManager(model=self, show_display=self.show_display,
                                               panel_width=600, caption=f'GA {self.p.experiment} : {self.id}',
                                               space_bounds=aux.get_arena_bounds(self.space.dims, self.scaling_factor))
        # self.initialize(**self.p.ga_build_kws)
        self.build_generation()



    def simulate(self):

        self.running = True
        self.setup(**self._setup_kwargs)
        while self.running:
            self.t += 1
            self.sim_step()
            self.screen_manager.render(self.t)
        return self.best_genome

    def build_generation(self, sorted_genomes=None):

        self.create_generation(sorted_genomes)
        # self.genome_dict = {i : self.new_genome(gConf, self.mConf0) for i, gConf in enumerate(gConfs)}

        confs = [{'larva_pars': g.mConf, 'unique_id': id, 'genome': g} for id, g in self.genome_dict.items()]
        self.place_agents(confs, self.robot_class_func)

        self.collectors = reg.get_reporters(collections=self.collections, agents=self.agents)
        self.step_output_keys=list(self.collectors['step'].keys())
        self.end_output_keys=list(self.collectors['end'].keys())
        if self.multicore:
            self.threads = self.build_threads(self.agents)
        else:
            self.threads = None
        self.generation_num += 1
        self.generation_step_num = 0
        self.generation_sim_time = 0
        self.start_generation_time = aux.TimeUtil.current_time_millis()
        reg.vprint(f'Generation {self.generation_num} started', 1)
        if self.progress_bar:
            self.progress_bar.update(self.generation_num)

    def convert_output_to_dataset(self, df, id):
        from larvaworld.lib.process.dataset import LarvaDataset
        df.index.set_names(['AgentID', 'Step'], inplace=True)
        df = df.reorder_levels(order=['Step', 'AgentID'], axis=0)
        df.sort_index(level=['Step', 'AgentID'], inplace=True)

        end = df[self.end_output_keys].xs(df.index.get_level_values('Step').max(), level='Step')
        # step = df[self.step_output_keys]
        d = LarvaDataset(dir=None, id=id,
                         load_data=False, env_params=self.p.env_params,
                         source_xy=self.source_xy,
                         fr=1 / self.dt)
        d.set_data(step=df[self.step_output_keys], end=end, food=None)
        return d

    def eval_robots(self):
        reg.vprint(f'Evaluating generation {self.generation_num}', 1)
        data = df_from_log(self._logs)
        # data=self.output.variables
        if self.fit_dict.func_arg!='s' :
            raise ValueError ('Evaluation function must take step data as argument')
        func=self.fit_dict.func
        for gID, df in data.items():
            d = self.convert_output_to_dataset(df.copy(), id=f'{self.id}_generation:{self.generation_num}')
            d._enrich(proc_keys=['angular', 'spatial'])

            # s, e, c = d.step_data, d.endpoint_data, d.config

            fit_dicts = func(s=d.step_data)

            valid_gs = {}
            for i, g in self.genome_dict.items():
                g.fitness_dict = aux.AttrDict({k: dic[i] for k, dic in fit_dicts.items()})
                mus = aux.AttrDict({k: -np.mean(list(dic.values())) for k, dic in g.fitness_dict.items()})
                if len(mus) == 1:
                    g.fitness = list(mus.values())[0]
                else:
                    coef_dict = {'KS': 10, 'RSS': 1}
                    g.fitness = np.sum([coef_dict[k] * mean for k, mean in mus.items()])
                if not np.isnan(g.fitness):
                    valid_gs[i] = g
            sorted_gs = [valid_gs[i] for i in
                         sorted(list(valid_gs.keys()), key=lambda i: valid_gs[i].fitness, reverse=True)]
            self.store(sorted_gs)
            reg.vprint(f'Generation {self.generation_num} evaluated', 1)
            return sorted_gs

    def store(self, sorted_gs):
        if self.best_genome is None or sorted_gs[0].fitness > self.best_genome.fitness:
            self.best_genome = sorted_gs[0]
            self.best_fitness = self.best_genome.fitness

            if self.bestConfID is not None:
                reg.saveConf(conf=self.best_genome.mConf, conftype='Model', id=self.bestConfID)
        reg.vprint(f'Generation {self.generation_num} best_fitness : {self.best_fitness}', 1)
        self.all_genomes_dic += [
            {'generation': self.generation_num, **{p.name: g.gConf[k] for k, p in self.space_dict.items()},
             'fitness': g.fitness, **g.fitness_dict.flatten()}
            for g in sorted_gs if g.fitness_dict is not None]

    @property
    def generation_completed(self):
        return self.generation_step_num >= self.Nsteps or len(self.agents) <= self.Nagents_min

    @property
    def max_generation_completed(self):
        return self.Ngenerations is not None and self.generation_num >= self.Ngenerations

    def sim_step(self):
        self.step()
        self.update()
        self.generation_sim_time += self.dt
        self.generation_step_num += 1
        if self.generation_completed:
            self.agents.nest_record(self.collectors['end'])

            # self.create_output()
            sorted_genomes = self.eval_robots()
            self.delete_agents()
            self._logs = {}
            self.t = 0
            if not self.max_generation_completed:

                self.build_generation(sorted_genomes)

            else:
                self.finalize()

    def step(self):
        if self.threads:
            for thr in self.threads:
                thr.step()

        else:
            self.agents.step()

    def update(self):
        if self.exclude_func is not None:
            for robot in self.agents:
                if self.exclude_func(robot):
                    robot.genome.fitness = -np.inf
                    self.delete_agent(robot)
        self.agents.nest_record(self.collectors['step'])

    def finalize(self):
        self.running = False
        if self.progress_bar:
            self.progress_bar.finish()
        reg.vprint(f'Best fittness: {self.best_genome.fitness}', 2)
        if self.store_data:
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.data_dir)

    def store_genomes(self, dic, save_to):
        self.genome_df = pd.DataFrame.from_records(dic)
        self.genome_df = self.genome_df.round(3)
        self.genome_df.sort_values(by='fitness', ascending=False, inplace=True)
        reg.graphs.dict['mpl'](data=self.genome_df, font_size=18, save_to=save_to,
                               name=self.bestConfID)
        self.genome_df.to_csv(f'{save_to}/{self.bestConfID}.csv')

        self.corr_df=self.genome_df[['fitness']+self.space_columns].corr()
        self.diff_df, row_colors=reg.model.diff_df(mIDs=[self.base_model, self.bestConfID], ms=[self.mConf0,self.best_genome.mConf])

    def build_threads(self, robots):
        N = self.num_cpu
        threads = []
        num_robots = len(robots)
        num_robots_per_cpu = math.floor(num_robots / N)

        reg.vprint(f'num_robots_per_cpu: {num_robots_per_cpu}', 2)

        for i in range(N - 1):
            start_pos = i * num_robots_per_cpu
            end_pos = (i + 1) * num_robots_per_cpu
            reg.vprint(f'core: {i + 1} positions: {start_pos} : {end_pos}', 1)
            robot_list = robots[start_pos:end_pos]

            thread = GA_thread(robot_list)
            thread.start()
            reg.vprint(f'thread {i + 1} started', 1)
            threads.append(thread)

        # last sublist of robots
        start_pos = (N - 1) * num_robots_per_cpu
        reg.vprint(f'last core, start_pos {start_pos}', 1)
        robot_list = robots[start_pos:]

        thread = GA_thread(robot_list)
        thread.start()
        reg.vprint(f'last thread started', 1)
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
            robot.step()


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

def df_from_log(log_dict):
    ddf={}
    obj_types = {}
    for obj_type, log_subdict in log_dict.items():

        if obj_type not in obj_types.keys():
            obj_types[obj_type] = {}

        for obj_id, log in log_subdict.items():

            # Add object id/key to object log
            log['obj_id'] = [obj_id] * len(log['t'])

            # Add object log to aggregate log
            for k, v in log.items():
                if k not in obj_types[obj_type]:
                    obj_types[obj_type][k] = []
                obj_types[obj_type][k].extend(v)

    # Transform logs into dataframes
    for obj_type, log in obj_types.items():
        index_keys = ['obj_id', 't']
        df = pd.DataFrame(log)
        df = df.set_index(index_keys)
        ddf[obj_type] = df
        return ddf