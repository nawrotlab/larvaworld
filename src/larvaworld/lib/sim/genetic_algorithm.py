import random
import multiprocessing
import math
import sys
import threading
import warnings

import pandas as pd
import progressbar
import numpy as np

from larvaworld.lib import reg, aux, util
from larvaworld.lib.screen import Viewer
from larvaworld.lib.sim.run_template import BaseRun


class GAlauncher(BaseRun):
    def __init__(self, **kwargs):
        super().__init__(runtype = 'Ga', **kwargs)

    def setup(self):

        self.odor_ids = aux.get_all_odors({}, self.p.env_params.food_params)
        self.build_env(self.p.env_params)
        self.initialize(**self.p.ga_build_kws, **self.p.ga_select_kws)


    def simulate(self):
        self.setup(**self._setup_kwargs)
        while True and self.engine.is_running:
            self.engine.step()
            if self.viewer.show_display:
                from pygame import KEYDOWN, K_ESCAPE, K_r, K_MINUS, K_PLUS, K_s, QUIT, event, Rect, draw, display
                for e in event.get():
                    if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                        sys.exit()
                    elif e.type == KEYDOWN and e.key == K_r:
                        self.initialize(**self.p.ga_select_kws, **self.p.ga_build_kws)
                    elif e.type == KEYDOWN and (e.key == K_PLUS or e.key == 93 or e.key == 270):
                        self.viewer.increase_fps()
                    elif e.type == KEYDOWN and (e.key == K_MINUS or e.key == 47 or e.key == 269):
                        self.viewer.decrease_fps()
                    elif e.type == KEYDOWN and e.key == K_s:
                        pass
                        # self.engine.save_genomes()
                    # elif e.type == KEYDOWN and e.key == K_e:
                    #     self.engine.evaluation_mode = 'preparing'

                if self.side_panel.generation_num < self.engine.generation_num:
                    self.side_panel.update_ga_data(self.engine.generation_num, self.engine.best_genome)

                # update statistics time
                cur_t = aux.TimeUtil.current_time_millis()
                cum_t = math.floor((cur_t - self.engine.start_total_time) / 1000)
                gen_t = math.floor((cur_t - self.engine.start_generation_time) / 1000)
                self.side_panel.update_ga_time(cum_t, gen_t, self.engine.generation_sim_time)
                self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
                self.viewer._window.fill(aux.Color.BLACK)

                for obj in self.viewer.objects:
                    obj.draw(self.viewer)

                # draw a black background for the side panel
                self.viewer.draw_panel_rect()
                self.side_panel.display_ga_info()

                display.flip()
                self.viewer._t.tick(self.viewer._fps)
        return self.engine.best_genome

    def initialize(self, **kwargs):
        self.viewer = Viewer.load_from_file(f'{reg.ROOT_DIR}/lib/sim/ga_scenes/{self.p.scene}.txt',
                                            show_display=self.p.show_screen and not self.p.offline,
                                           panel_width=600,caption = f'GA {self.p.experiment} : {self.id}',
                                           space_bounds=aux.get_arena_bounds(self.space.dims, self.scaling_factor))

        self.engine = GAengine(model=self, **kwargs)
        if self.viewer.show_display:
            from larvaworld.lib.screen.side_panel import SidePanel

            from pygame import display
            self.get_larvaworld_food()
            self.side_panel = SidePanel(self.viewer, self.engine.space_dict)
            self.side_panel.update_ga_data(self.engine.generation_num, None)
            self.side_panel.update_ga_population(len(self.engine.robots), self.engine.Nagents)
            self.side_panel.update_ga_time(0, 0, 0)



    def get_larvaworld_food(self):
        for label,ff in self.p.env_params.food_params.source_units.items():
            x, y = self.screen_pos(ff.pos)
            size = ff.radius * self.scaling_factor
            col = ff.default_color
            box = self.build_box(x, y, size, col)
            box.label = label
            self.viewer.put(box)

    def screen_pos(self, real_pos):
        return np.array(real_pos) * self.scaling_factor + np.array([self.viewer.width / 2, self.viewer.height / 2])



class GAselector:
    def __init__(self, model, Ngenerations=None, Nagents=30, Nelits=3, Pmutation=0.3, Cmutation=0.1,
                 selection_ratio=0.3, verbose=0, bestConfID=None):
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
        self.is_running = True
        if progress_bar and self.Ngenerations is not None:
            self.progress_bar = progressbar.ProgressBar(self.Ngenerations)
            self.progress_bar.start()
        else:
            self.progress_bar = None
        self.exclude_func = exclude_func
        self.multicore = multicore
        self.robot_class = get_robot_class(robot_class, self.model.p.offline)
        self.mConf0 = reg.loadConf(id=base_model, conftype='Model')
        self.space_dict = reg.model.space_dict(mkeys=space_mkeys, mConf0=self.mConf0)
        self.excluded_ids = []
        self.exclusion_mode = exclusion_mode

        self.gConfs=self.create_first_generation(init_mode, self.Nagents, self.space_dict, self.mConf0)

        self.robots = self.build_generation()

        if self.exclusion_mode :
            self.fit_dict =None
            self.dataset =None
            self.step_df =None
        else :
            if fit_dict is None :
                if fitness_target_refID is not None:
                    fit_dict=util.GA_optimization(fitness_target_refID, fitness_target_kws)
                else :
                    fit_dict = arrange_fitness(fitness_func,source_xy=self.model.source_xy)
            self.fit_dict =fit_dict
            arg=self.fit_dict.func_arg
            if arg=='s':
                self.dataset0=self.init_dataset()
                self.step_df = self.init_step_df()
            elif arg=='robot':
                self.dataset = None
                self.step_df = None

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

    def init_dataset(self):
        c = aux.AttrDict(
            {'id': self.model.id, 'group_id': 'GA_robots', 'dt': self.model.dt, 'fr': 1 / self.model.dt,
             'agent_ids': np.arange(self.Nagents), 'duration': self.model.Nsteps * self.model.dt,
             'Npoints': 3, 'Ncontour': 0, 'point': '', 'N': self.Nagents, 'Nticks': self.model.Nsteps,
             'mID': self.bestConfID,
             'color': 'blue', 'env_params': self.model.p.env_params})

        self.my_index = pd.MultiIndex.from_product([np.arange(c.Nticks), c.agent_ids],
                                                   names=['Step', 'AgentID'])
        self.df_columns = reg.getPar(['b', 'fov', 'rov', 'v', 'x', 'y'])
        self.df_Ncols = len(self.df_columns)

        e = pd.DataFrame(index=c.agent_ids)
        e['cum_dur'] = c.duration
        e['num_ticks'] = c.Nticks

        return aux.AttrDict({'step_data': None, 'endpoint_data': e, 'config': c})

    def init_step_df(self):
        self.dataset = self.dataset0.get_copy()

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

        from larvaworld.lib.process.spatial import scale_to_length
        scale_to_length(s, e, c, pars=None, keys=['v'])
        self.dataset.step_data = s
        try :
            for k in self.fit_dict.keys:
                reg.par.compute(k, self.dataset)
        except:
            pass
        fit_dicts=self.fit_dict.func(s=self.dataset.step_data)

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


    def build_generation(self):
        robots = []
        for i, gConf in enumerate(self.gConfs):
            g=self.new_genome(gConf, self.mConf0)
            self.genome_dict[i] = g

            robot = self.robot_class(unique_id=i, model=self.model, larva_pars=g.mConf)
            robot.genome = g
            robots.append(robot)
            self.model.viewer.put(robot)

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

        self.sort_genomes()

        if self.model.store_data:
            self.all_genomes_dic += [
            {'generation': self.generation_num, **{p.name : g.gConf[k] for k,p in self.space_dict.items()},
             'fitness': g.fitness, **g.fitness_dict.flatten()}
            for g in self.sorted_genomes if g.fitness_dict is not None]

    def destroy_robot(self, robot, excluded=False):
        if excluded:
            self.excluded_ids.append(robot.unique_id)
            robot.genome.fitness = -np.inf
        if self.exclusion_mode:
            robot.genome.fitness =robot.Nticks
        self.model.viewer.remove(robot)
        self.robots.remove(robot)

    def finalize(self):
        self.is_running = False
        if self.progress_bar:
            self.progress_bar.finish()
        reg.vprint(f'Best fittness: {self.best_genome.fitness}', 2)
        if self.model.store_data :
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.model.data_dir)



    def check(self, robot):
        if not self.model.p.offline:
            if robot.xx < 0 or robot.xx > self.model.viewer.width or robot.yy < 0 or robot.yy > self.model.viewer.height:
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

def get_robot_class(robot_class=None, offline=False):
    if offline:
        robot_class = 'LarvaOffline'
    if robot_class is None:
        robot_class = 'LarvaRobot'

    if type(robot_class) == str:
        if robot_class == 'LarvaRobot':
            class_name = f'lib.model.agents.larva_robot.LarvaRobot'
        elif robot_class == 'ObstacleLarvaRobot':
            class_name = f'lib.model.agents.larva_robot.ObstacleLarvaRobot'
        elif robot_class == 'LarvaOffline':
            class_name = f'lib.model.agents.larva_offline.LarvaOffline'
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
                 offline=False,show_screen=False,exclusion_mode=False,experiment='exploration',
                 id=None, dt=1 / 16, dur=0.5, save_to=None, store_data=False, Nagents=30, Nelits=6, Ngenerations=20,
                 **kwargs):

    warnings.filterwarnings('ignore')
    if mID1 is None:
        mID1 = mID0



    kws = {
        'sim_params': reg.get_null('sim_params', duration=dur,timestep=dt),
        'show_screen': show_screen,
        'offline': offline,
        'store_data': store_data,
        'experiment': experiment,
        'env_params': 'arena_200mm',
        'ga_select_kws': reg.get_null('ga_select_kws', Nagents=Nagents, Nelits=Nelits, Ngenerations=Ngenerations, selection_ratio=0.1),
        'ga_build_kws': reg.get_null('ga_build_kws', init_mode=init, space_mkeys=space_mkeys, base_model=mID0,exclusion_mode=exclusion_mode,
                                      bestConfID=mID1, fitness_target_refID=refID)
    }

    conf = reg.get_null('Ga', **kws)
    conf.env_params = reg.expandConf(id=conf.env_params, conftype='Env')

    conf.ga_build_kws.fit_dict = fit_dict

    GA = GAlauncher(parameters=conf, save_to=save_to, id=id)
    best_genome = GA.simulate()
    entry = {mID1: best_genome.mConf}
    return entry
