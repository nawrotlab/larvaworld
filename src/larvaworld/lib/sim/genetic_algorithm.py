import random
import multiprocessing
import math
import threading
import warnings
import param
import pandas as pd
import progressbar
import numpy as np

import larvaworld
from larvaworld.lib import reg, aux, util
from larvaworld.lib.param import NestedConf, ClassAttr, class_generator, SimOps, OptionalSelector
from larvaworld.lib.process.evaluation import Evaluation
from larvaworld.lib.screen import GA_ScreenManager
from larvaworld.lib.sim.base_run import BaseRun




def dst2source_evaluation(robot, source_xy):
    traj = np.array(robot.trajectory)
    dst = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    cum_dst = np.sum(dst)
    l=[]
    for label, pos in source_xy.items():
        l.append(aux.eudi5x(traj, pos))
    fitness= - np.mean(np.min(np.vstack(l),axis=0))/ cum_dst
    return fitness

def cum_dst(robot, **kwargs):
    return robot.cum_dst / robot.real_length


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 20:
        return True
    else:
        return False


fitness_funcs = aux.AttrDict({
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})



exclusion_funcs = aux.AttrDict({
    'bend_errors': bend_error_exclusion
})




class GAevaluation(Evaluation):
    exclusion_mode = param.Boolean(default=False,label='exclusion mode', doc='Whether to apply exclusion mode')
    exclude_func_name = OptionalSelector(default=None,objects=list(exclusion_funcs.keys()),
                                       label='name of exclusion function',doc='The function that evaluates exclusion', allow_None=True)
    fitness_func_name = OptionalSelector(default=None,objects=list(fitness_funcs.keys()),
                                       label='name of fitness function',doc='The function that evaluates fitness', allow_None=True)

    fit_kws = param.Dict(default={}, label='fitness metrics to evaluate', doc='The target metrics to optimize against')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.exclude_func = exclusion_funcs[self.exclude_func_name] if type(self.exclude_func_name)==str else None

        if self.exclusion_mode:
            self.fit_func = None
        elif self.fitness_func_name and self.fitness_func_name in fitness_funcs.keys() :

            def func(robot):
                return fitness_funcs[self.fitness_func_name](robot, **self.fit_kws)

            self.fit_func_arg = 'robot'
            self.fit_func = func
        elif self.target:
            self.fit_func_arg = 's'
            self.fit_func = self.fit_func_solo
        else:
            raise


class GAselector(NestedConf):
    Ngenerations = param.Integer(default=None, allow_None=True, label='# generations',
                                 doc='Number of generations to run for the genetic algorithm engine')
    Nagents = param.Integer(default=30, label='# agents per generation', doc='Number of agents per generation')
    Nelits = param.Integer(default=3, label='# best agents for next generation',
                           doc='Number of best agents to include in the next generation')

    selection_ratio = param.Magnitude(default=0.3, label='selection ratio',
                                      doc='Fraction of agent population to include in the next generation')
    Pmutation = param.Magnitude(default=0.3, label='mutation probability',
                                doc='Probability of mutation for each agent in the next generation')
    Cmutation = param.Number(default=0.1, label='mutation coeficient',
                             doc='Fraction of allowed parameter range to mutate within')
    bestConfID = param.String(default='best_model', label='model ID for optimized model',
                              doc='ID for the optimized model')

    init_mode = param.Selector(default='random', objects=['random', 'model', 'default'],
                               label='mode of initial generation',doc='Mode of initial generation')
    base_model = reg.conf.Model.confID_selector('loco_default')

    space_mkeys = param.ListSelector(default=[], objects=reg.model.mkeys,
                                     label='keys of modules to include in space search',doc='Keys of the modules where the optimization parameters are')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.Nagents_min = round(self.Nagents * self.selection_ratio)
        if self.Nagents_min < 2:
            raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                             'Please increase population (' + str(self.Nagents) + ') or selection ratio (' +
                             str(self.selection_ratio) + ')')


        self.mConf0 = reg.conf.Model.getID(self.base_model)
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
        mConf.life_history = {'age': 0.0, 'epochs': {}}
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

    def crossover(self, g1, g2):
        g = {}
        for k in g1.keys():
            if np.random.uniform(0, 1, 1) >= 0.5:
                g[k] = g1[k]
            else:
                g[k] = g2[k]
        return g











# class GAlauncher2(BaseRun):
#     def __init__(self, parameters, **kwargs):
#         super().__init__(runtype='Ga', parameters=parameters, **kwargs)


class GAlauncher(BaseRun,GAselector,GAevaluation):


    def __init__(self,dataset=None, **kwargs):
        '''
        Simulation mode 'Ga' launches a genetic algorith optimization simulation of a specified agent model.

        Args:
            **kwargs: Arguments passed to the setup method

        '''

        BaseRun.__init__(self, runtype='Ga',dataset=dataset, **kwargs)
        # raise
        GAevaluation.__init__(self,dataset=dataset,**self.p.ga_eval_kws)
        # GAselector.__init__(self,**self.p.ga_select_kws,dataset=dataset,**self.p.ga_eval_kws)
        GAselector.__init__(self,**self.p.ga_select_kws)



    # def initialize_superclasses(self, parameters,dataset=None):
    #     # print('DD')
    #     GAevaluation.__init__(self, dataset=dataset, **parameters.ga_eval_kws)
    #     # print('DDD')
    #     GAselector.__init__(self, **parameters.ga_select_kws,dataset=dataset, **parameters.ga_eval_kws)
    #     # print('DDDD')


    def setup(self):
        self.best_genome = None
        self.best_fitness = None
        self.sorted_genomes = None
        self.all_genomes_dic = []
        self.num_cpu = multiprocessing.cpu_count()
        self.generation_num = 0
        self.start_total_time = aux.TimeUtil.current_time_millis()



        reg.vprint(f'--- Genetic Algorithm  "{self.id}" initialized!--- ', 2)
        temp = self.Ngenerations if self.Ngenerations is not None else 'unlimited'
        reg.vprint(f'Launching {temp} generations of {self.duration} minutes, with {self.Nagents} agents each!', 2)
        if self.Ngenerations is not None:
            self.progress_bar = progressbar.ProgressBar(self.Ngenerations)
            self.progress_bar.start()
        else:
            self.progress_bar = None
        self.p.collections = ['pose']
        # self.odor_ids = self.get_all_odors()
        self.build_env(self.p.env_params)
        self.screen_manager = GA_ScreenManager(model=self)
        self.build_generation()



    def simulate(self):
        self.sim_setup()
        while self.running:
            self.sim_step()
        return self.best_genome

    def build_generation(self, sorted_genomes=None):
        self.create_generation(sorted_genomes)
        confs = [{'larva_pars': g.mConf, 'unique_id': str(id), 'genome': g} for id, g in self.genome_dict.items()]
        self.place_agents(confs)
        self.set_collectors(self.p.collections)
        if self.multicore:
            self.threads = self.build_threads(self.agents)
        else:
            self.threads = None
        self.generation_num += 1
        self.generation_step_num = 0
        self.start_generation_time = aux.TimeUtil.current_time_millis()
        reg.vprint(f'Generation {self.generation_num} started', 1)
        if self.progress_bar:
            self.progress_bar.update(self.generation_num)



    def eval_robots(self, Ngen, genome_dict):
        reg.vprint(f'Evaluating generation {Ngen}', 1)
        assert self.fit_func_arg =='s'

        self.data_collection = larvaworld.lib.LarvaDatasetCollection.from_agentpy_output(self.output)
        for d in self.data_collection.datasets:
            d.enrich(proc_keys=['angular', 'spatial'], is_last=False)
            valid_gs = {}
            for i, g in genome_dict.items():
            # for id in d.config.agent_ids:
                ss = d.step_data.xs(str(i), level='AgentID')
                g.fitness_dict = self.fit_func(ss)


            # fit_dicts = self.fit_func(s=d.step_data)

            # for i, g in genome_dict.items():
            #     g.fitness_dict = aux.AttrDict({k: dic[str(i)] for k, dic in fit_dicts.items()})
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
            self.store(sorted_gs, Ngen)
            reg.vprint(f'Generation {Ngen} evaluated', 1)
            return sorted_gs

    def store(self, sorted_gs, Ngen):
        if self.best_genome is None or sorted_gs[0].fitness > self.best_genome.fitness:
            self.best_genome = sorted_gs[0]
            self.best_fitness = self.best_genome.fitness

            if self.bestConfID is not None:
                reg.conf.Model.setID(self.bestConfID, self.best_genome.mConf)
        reg.vprint(f'Generation {Ngen} best_fitness : {self.best_fitness}', 1)
        self.all_genomes_dic += [
            {'generation': Ngen, **{p.name: g.gConf[k] for k, p in self.space_dict.items()},
             'fitness': g.fitness, **g.fitness_dict.flatten()}
            for g in sorted_gs if g.fitness_dict is not None]

    @property
    def generation_completed(self):
        return self.generation_step_num >= self.Nsteps or len(self.agents) <= self.Nagents_min

    @property
    def max_generation_completed(self):
        return self.Ngenerations is not None and self.generation_num >= self.Ngenerations

    def sim_step(self):
        self.t += 1
        self.step()
        self.update()
        self.generation_step_num += 1
        if self.generation_completed:
            self.end()
            if not self.max_generation_completed:
                self.build_generation(self.sorted_genomes)
            else:
                self.finalize()

    def step(self):
        if self.threads:
            for thr in self.threads:
                thr.step()
        else:
            self.agents.step()
        if self.exclude_func is not None:
            for robot in self.agents:
                if self.exclude_func(robot):
                    robot.genome.fitness = -np.inf
                    # self.delete_agent(robot)
        self.screen_manager.render()

    def end(self):
        self.agents.nest_record(self.collectors['end'])
        self.create_output()
        self.sorted_genomes = self.eval_robots(Ngen=self.generation_num, genome_dict=self.genome_dict)
        self.delete_agents()
        self._logs = {}
        self.t = 0


    def update(self):

        self.agents.nest_record(self.collectors['step'])

    def finalize(self):
        self.running = False
        self.screen_manager.finalize()
        if self.progress_bar:
            self.progress_bar.finish()
        reg.vprint(f'Best fittness: {self.best_genome.fitness}', 1)
        if self.store_data:
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.data_dir)

    def store_genomes(self, dic, save_to):
        df = pd.DataFrame.from_records(dic)
        df = df.round(3)
        df.sort_values(by='fitness', ascending=False, inplace=True)
        reg.graphs.dict['mpl'](data=df, font_size=18, save_to=save_to,
                               name=self.bestConfID)
        df.to_csv(f'{save_to}/{self.bestConfID}.csv')

        self.corr_df=df[['fitness']+self.space_columns].corr()
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


def optimize_mID(mID0, mID1=None, refID=None, space_mkeys=['turner', 'interference'], init='model',
               exclusion_mode=False,experiment='exploration',
                 id=None, dt=1 / 16, dur=0.5, dir=None,  Nagents=30, Nelits=6, Ngenerations=20,
                 **kwargs):

    warnings.filterwarnings('ignore')
    if mID1 is None:
        mID1 = mID0

    gaconf = reg.gen.Ga(ga_select_kws=reg.gen.GAselector(Nagents=Nagents, Nelits=Nelits, Ngenerations=Ngenerations,
                                      selection_ratio=0.1,init_mode=init, space_mkeys=space_mkeys,
                                      base_model=mID0,bestConfID=mID1),
                      ga_eval_kws=reg.gen.GAevaluation(refID=refID,**kwargs),
                      env_params='arena_200mm',
                      experiment=experiment).nestedConf


    gaconf.env_params = reg.conf.Env.getID(gaconf.env_params)

    # conf.ga_build_kws.

    GA = GAlauncher(parameters=gaconf, dir=dir, id=id, duration=dur,dt=dt)
    best_genome = GA.simulate()
    entry = {mID1: best_genome.mConf}
    return entry


reg.gen.GAselector=class_generator(GAselector)
reg.gen.GAevaluation=class_generator(GAevaluation)

class GAconf(SimOps):
    env_params = reg.conf.Env.confID_selector()
    experiment = reg.conf.Ga.confID_selector()
    ga_eval_kws = ClassAttr(reg.gen.GAevaluation, doc='The GA evaluation configuration')
    ga_select_kws = ClassAttr(reg.gen.GAselector, doc='The GA selection configuration')

    scene = param.String('no_boxes', doc='The name of the scene to load')

# reg.gen.Ga=class_generator(GAlauncher)
reg.gen.Ga=class_generator(GAconf)