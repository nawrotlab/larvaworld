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
from lib.ga.util.genome import Genome

from lib.registry.pars import preg

from lib.ga.robot.larva_robot import LarvaRobot

from lib.ga.util.time_util import TimeUtil


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
        self.verbose = verbose
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
                preg.saveConf(conf=self.best_genome.mConf, conftype='Model', id=self.bestConfID, verbose=self.verbose)


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
        # print(mConf0)
        kws = {k: dNl.flatten_dict(mConf0)[k] for k, vs in space_dict.items()}
    return kws


class GAbuilder(GAselector):
    def __init__(self, scene, side_panel=None, space_dict=None, robot_class=LarvaRobot, base_model='explorer',
                 multicore=True, fitness_func=None, fitness_target_kws=None, fitness_target_refID=None,
                 exclude_func=None, plot_func=None, bestConfID=None, init_mode='random', progress_bar=True, **kwargs):
        super().__init__(bestConfID = bestConfID,**kwargs)

        if fitness_target_kws is None:
            fitness_target_kws = {}
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
        self.mConf0 = preg.loadConf(id=base_model, conftype='Model')


        gConfs = [initConf(init_mode, space_dict, self.mConf0) for i in range(self.Nagents)]



        self.genomes = [Genome(gConf=gConf,mConf=dNl.update_nestdict(self.mConf0, gConf), space_dict=space_dict, generation_num=self.generation_num) for gConf in gConfs]


        self.robots = self.build_generation()
        self.step_df = self.init_step_df()

        # print(self.model.Nsteps, self.Nagents)
        # raise
        self.printd(1, 'Generation', self.generation_num, 'started')
        self.printd(1, 'multicore:', self.multicore, 'num_cpu:', self.num_cpu)


    def init_step_df(self):


        c = dNl.NestDict(
            {'id': self.model.id, 'group_id': 'GA_robots', 'dt': self.model.dt, 'fr': 1 / self.model.dt,
             'agent_ids': np.arange(self.Nagents), 'duration': self.model.Nsteps * self.model.dt,
             'Npoints': 3, 'Ncontour': 0, 'point': '', 'N': self.Nagents, 'Nticks': self.model.Nsteps,
             'mID': self.bestConfID,
             'color': 'blue', 'env_params': self.model.env_pars})

        self.my_index = pd.MultiIndex.from_product([np.arange(c.Nticks), c.agent_ids],
                                                   names=['Step', 'AgentID'])
        self.df_columns = preg.getPar(['b', 'fov', 'rov', 'v', 'x', 'y'])
        self.df_Ncols=len(self.df_columns)
        step_df = np.ones([c.Nticks, c.N, self.df_Ncols]) * np.nan

        e = pd.DataFrame(index=c.agent_ids)
        e['cum_dur'] = c.duration
        e['num_ticks'] = c.Nticks
        e['length'] = [robot.real_length for robot in self.robots]

        self.dataset=dNl.NestDict({'step_data' : None, 'endpoint_data' : e, 'config' : c})

        return step_df

    def finalize_step_df(self):
        t0 = time.time()
        e, c = self.dataset.endpoint_data, self.dataset.config
        self.step_df[:, :, :3] = np.rad2deg(self.step_df[:, :, :3])
        self.step_df = self.step_df.reshape(c.Nticks* c.N, self.df_Ncols)
        s = pd.DataFrame(self.step_df, index=self.my_index, columns=self.df_columns)
        s = s.astype(float)

        cycle_ks,eval_ks =None,None
        from lib.process.spatial import scale_to_length

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
            preg.compute(k, self.dataset)

        for i, g in self.genome_dict.genomes.items():
            if g.fitness is None :
                # t0 = time.time()
                ss = self.dataset.step_data.xs(i, level='AgentID')
                gdict=dNl.NestDict({k:[] for k in dic0.keys()})
                gdict['step']=ss
                if cycle_ks:
                    from lib.process.aux import cycle_curve_dict

                    gdict['cycle_curves'] = cycle_curve_dict(s=ss, dt=self.model.dt, shs=cycle_ks)
                if eval_ks:
                    gdict['eval']={sh: ss[preg.getPar(sh)].dropna().values for sh in eval_ks}
                    # t1 = time.time()
                    # print('--1--', t1 - t0)
                    # t00 = time.time()
                g.fitness, g.fitness_dict = self.get_fitness(gdict)
                # print(g.fitness)
                # g.fitness
                # t11 = time.time()
                # print('--2--', t11 - t00)


    def build_generation(self):
        self.genome_dict = dNl.NestDict(
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
        robot_dict=dNl.NestDict()
        if fitness_target_refID is not None:
            d = preg.loadRef(fitness_target_refID)
            if 'eval_shorts' in fitness_target_kws.keys():
                shs = fitness_target_kws['eval_shorts']
                eval_pars, eval_lims, eval_labels = preg.getPar(shs, to_return=['d', 'lim', 'lab'])
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
        return dNl.NestDict({'func': fitness_func, 'target_refID': fitness_target_refID,
                                               'target_kws': fitness_target_kws, 'target': fitness_target, 'robot_dict' : robot_dict})





class GA_thread(threading.Thread):
    def __init__(self, robots):
        threading.Thread.__init__(self)
        self.robots = robots

    def run(self):
        for robot in self.robots:
            robot.sense_and_act()


if __name__ == '__main__':
    # print(null_dict.__class__)
    pass
