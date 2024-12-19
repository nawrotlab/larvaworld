import math
import multiprocessing
import random
import threading
from typing import List
import numpy as np
import pandas as pd
import param
import progressbar

from ... import vprint
from ..process.dataset import LarvaDatasetCollection, LarvaDataset
from .. import reg, util
from ..model import SpaceDict
from ..param import ClassAttr, OptionalSelector, SimOps, class_generator
from ..plot.table import diff_df
from ..process.evaluation import Evaluation
from ..util import AttrDict
from .base_run import BaseRun

__all__ = [
    "GAevaluation",
    "GAselector",
    "GAlauncher",
    "optimize_mID",
]


def dst2source_evaluation(robot, source_xy):
    traj = np.array(robot.trajectory)
    dst = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    cum_dst = np.sum(dst)
    l = []
    for label, pos in source_xy.items():
        l.append(util.eudi5x(traj, pos))
    fitness = -np.mean(np.min(np.vstack(l), axis=0)) / cum_dst
    return fitness


def cum_dst(robot, **kwargs):
    return robot.cum_dst / robot.length


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 20:
        return True
    else:
        return False


fitness_funcs = AttrDict(
    {
        "dst2source": dst2source_evaluation,
        "cum_dst": cum_dst,
    }
)

exclusion_funcs = AttrDict({"bend_errors": bend_error_exclusion})


class GAevaluation(Evaluation):
    """Class that implements the genetic algorithm evaluation process."""

    exclusion_mode = param.Boolean(
        default=False, label="exclusion mode", doc="Whether to apply exclusion mode"
    )
    exclude_func_name = OptionalSelector(
        default=None,
        objects=list(exclusion_funcs.keys()),
        label="name of exclusion function",
        doc="The function that evaluates exclusion",
        allow_None=True,
    )
    fitness_func_name = OptionalSelector(
        default=None,
        objects=list(fitness_funcs.keys()),
        label="name of fitness function",
        doc="The function that evaluates fitness",
        allow_None=True,
    )

    fit_kws = param.Dict(
        default={},
        label="fitness metrics to evaluate",
        doc="The target metrics to optimize against",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.exclude_func = (
            exclusion_funcs[self.exclude_func_name]
            if type(self.exclude_func_name) == str
            else None
        )

        if self.exclusion_mode:
            self.fit_func = None
        elif self.fitness_func_name and self.fitness_func_name in fitness_funcs:

            def func(robot):
                return fitness_funcs[self.fitness_func_name](robot, **self.fit_kws)

            self.fit_func_arg = "robot"
            self.fit_func = func
        elif self.target:
            self.fit_func_arg = "s"
            self.fit_func = self.fit_func_solo
        else:
            raise


class GAselector(SpaceDict):
    """Class that implements the genetic algorithm selection process."""

    Ngenerations = param.Integer(
        default=None,
        allow_None=True,
        label="# generations",
        doc="Number of generations to run for the genetic algorithm engine",
    )
    Nagents = param.Integer(
        default=20,
        label="# agents per generation",
        doc="Number of agents per generation",
    )
    Nelits = param.Integer(
        default=3,
        label="# best agents for next generation",
        doc="Number of best agents to include in the next generation",
    )

    selection_ratio = param.Magnitude(
        default=0.3,
        label="selection ratio",
        doc="Fraction of agent population to include in the next generation",
    )
    bestConfID = param.String(
        default=None,
        label="model ID for optimized model",
        doc="ID for the optimized model",
    )

    def __init__(self, **kwargs):
        """
        Initialize the GeneticAlgorithm class.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to the superclass initializer.

        Raises
        ------
        ValueError
            If the number of parents selected to breed a new generation is less than 2.

        Notes
        -----
        - If `bestConfID` is None, it generates a unique configuration ID for the best model.
        - Calculates the minimum number of agents required for selection based on the selection ratio.
        - Ensures that at least 2 agents are selected for breeding a new generation.
        """
        super().__init__(**kwargs)
        if self.bestConfID is None:
            for i in range(1000):
                id = f"{self.base_model}_fit{i}"
                if id not in reg.conf.Model.confIDs:
                    self.bestConfID = id
                    break

        self.Nagents_min = round(self.Nagents * self.selection_ratio)
        if self.Nagents_min < 2:
            raise ValueError(
                "The number of parents selected to breed a new generation is < 2. "
                + "Please increase population ("
                + str(self.Nagents)
                + ") or selection ratio ("
                + str(self.selection_ratio)
                + ")"
            )

    def new_genome(self, gConf, mConf0):
        """
        Create a new genome with the given genetic and mutation configurations.

        Args:
            gConf (dict): Genetic configuration dictionary.
            mConf0 (AttrDict): Initial mutation configuration as an AttrDict.

        Returns:
            AttrDict: A dictionary-like object containing the new genome's fitness,
                  fitness dictionary, genetic configuration, and updated mutation configuration.
        """
        mConf = mConf0.update_nestdict(gConf)
        mConf.life_history = {"age": 0.0, "epochs": {}}
        return AttrDict(
            {"fitness": None, "fitness_dict": {}, "gConf": gConf, "mConf": mConf}
        )

    def create_new_generation(self, sorted_gs):
        """
        Create a new generation of genomes using elitism and crossover.

        Args:
            sorted_gs (list): A list of sorted genomes from the previous generation.

        Returns:
            list: A new list of genomes for the next generation.

        Raises:
            ValueError: If the number of genomes is lower than the minimum required to breed a new generation.

        Notes:
            - The best genomes are preserved through elitism.
            - New genomes are created by randomly selecting and combining attributes from two parent genomes.
            - The new genomes are then mutated before being added to the new generation.
        """
        if len(sorted_gs) < self.Nagents_min:
            raise ValueError(
                f"The number of genomes ({len(sorted_gs)}) is lower than the minimum number required to breed a new generation ({self.Nagents_min})"
            )
        gs0 = [sorted_gs[i].gConf for i in range(self.Nagents_min)]

        # elitism: keep the best genomes in the new generation
        gs = [gs0[i] for i in range(self.Nelits)]

        for i in range(self.Nagents - self.Nelits):
            g1, g2 = random.sample(gs0, 2)
            g0 = AttrDict(
                {
                    k: g1[k] if np.random.uniform(0, 1, 1) >= 0.5 else g2[k]
                    for k in self.space_ks
                }
            )
            g0 = self.mutate(g0)
            gs.append(g0)
        return gs

    def create_generation(self, sorted_gs=None):
        """
        Creates a new generation of genomes.

        If `sorted_gs` is provided, it creates a new generation based on the sorted genomes.
        Otherwise, it creates the first generation with a specified number of agents.

        Args:
            sorted_gs (list, optional): A list of sorted genomes to base the new generation on. Defaults to None.

        Returns:
            dict: A dictionary where keys are indices and values are new genomes generated from the configuration.
        """
        if sorted_gs is None:
            self.gConfs = self.create_first_generation(self.Nagents)
        else:
            self.gConfs = self.create_new_generation(sorted_gs)
        return {
            i: self.new_genome(gConf, self.mConf0)
            for i, gConf in enumerate(self.gConfs)
        }


class GAlauncher(BaseRun):
    def __init__(self, dataset=None, evaluator=None, **kwargs):
        """
        Simulation mode 'Ga' launches a genetic algorith optimization simulation of a specified agent model.
        """
        super().__init__(runtype="Ga", **kwargs)
        if evaluator is None:
            evaluator = GAevaluation(dataset=dataset, **self.p.ga_eval_kws)
        self.evaluator = evaluator
        self.selector = GAselector(**self.p.ga_select_kws)

    def setup(self):
        """
        Initializes the genetic algorithm setup.

        This method sets up the initial parameters and configurations for the genetic algorithm,
        including initializing genome-related attributes, setting up the progress bar, and building
        the environment and the first generation.

        Attributes:
            genome_dict (dict): Dictionary to store genome information.
            best_genome (Any): Stores the best genome found.
            best_fitness (Any): Stores the fitness value of the best genome.
            sorted_genomes (Any): Stores sorted genomes.
            all_genomes_dic (list): List to store all genomes.
            generation_num (int): Counter for the number of generations.
            progress_bar (progressbar.ProgressBar or None): Progress bar for tracking generations.
            p.collections (list): List of parameter collections to be recorded.

        Prints:
            Initialization message and the number of generations to be launched.
        """
        self.genome_dict = None
        self.best_genome = None
        self.best_fitness = None
        self.sorted_genomes = None
        self.all_genomes_dic = []
        self.generation_num = 0

        Ngens = self.selector.Ngenerations

        vprint(f'--- Genetic Algorithm  "{self.id}" initialized!--- ', 1)
        if Ngens is not None:
            self.progress_bar = progressbar.ProgressBar(Ngens)
            self.progress_bar.start()
            temp = Ngens
        else:
            self.progress_bar = None
            temp = "unlimited"
        vprint(
            f"Launching {temp} generations of {self.duration} minutes, with {self.selector.Nagents} agents each!",
            1,
        )
        self.p.collections = ["pose", "brain"]
        self.build_env(self.p.env_params)

        self.build_generation()

    def simulate(self):
        """
        Simulates the genetic algorithm process.

        This method sets up the simulation, runs it in a loop until the
        simulation is no longer running, and then returns the best genome
        found during the simulation.

        Returns:
            Genome: The best genome found during the simulation.
        """
        self.sim_setup()
        while self.running:
            self.sim_step()
        return self.best_genome

    def build_generation(self, sorted_genomes=None):
        """
        Builds a new generation of genomes for the genetic algorithm.

        Args:
            sorted_genomes (dict, optional): A dictionary of genomes sorted by fitness. Defaults to None.

        This method performs the following steps:
        1. Records the current time as the pre-start generation time.
        2. Increments the generation number.
        3. Creates a new generation of genomes using the selector.
        4. Prepares configuration dictionaries for each genome, including parameters, unique ID, genome object, and a random bright color.
        5. Places agents in the simulation based on the configurations.
        6. Sets up data collectors.
        7. Initializes threads for agents if multicore processing is enabled.
        8. Updates the progress bar if it is enabled.
        9. Records the current time as the start generation time.
        10. Calculates and prints the duration taken to load the generation.
        """
        self.prestart_generation_time = util.TimeUtil.current_time_sec()
        self.generation_num += 1
        self.genome_dict = self.selector.create_generation(sorted_genomes)
        confs = [
            {
                "larva_pars": g.mConf,
                "unique_id": str(id),
                "genome": g,
                "color": util.Color.random_bright(),
            }
            for id, g in self.genome_dict.items()
        ]
        self.place_agents(confs)
        self.set_collectors(self.p.collections)
        if self.multicore:
            self.threads = self.build_threads(self.agents)
        else:
            self.threads = None

        # self.generation_step_num = 0

        if self.progress_bar:
            self.progress_bar.update(self.generation_num)
        # self.gen_progressbar.start()
        self.start_generation_time = util.TimeUtil.current_time_sec()
        gen_load_dur = np.round(
            self.start_generation_time - self.prestart_generation_time
        )
        vprint(f"Generation {self.generation_num} started in {gen_load_dur} sec", 1)

    def eval_robots(self, ds: List[LarvaDataset], Ngen: int, genome_dict: dict) -> list:
        """
        Evaluate the fitness of robots for a given generation.

        Parameters:
        ds (list): A list of LarvaDataset objects to be enriched and evaluated.
        Ngen (int): The current generation number.
        genome_dict (dict): A dictionary where keys are genome identifiers and values are genome objects.

        Returns:
        list: A sorted list of valid genome objects based on their fitness, in descending order.
        """
        vprint(f"Evaluating generation {Ngen}", 1)
        assert self.evaluator.fit_func_arg == "s"
        for d in ds:
            d.enrich(
                proc_keys=["angular", "spatial"],
                dsp_starts=[],
                dsp_stops=[],
                tor_durs=[],
                is_last=False,
            )
            valid_gs = {}
            for i, g in genome_dict.items():
                ss = d.step_data.xs(str(i), level="AgentID")
                g.fitness_dict = self.evaluator.fit_func(ss)
                mus = AttrDict(
                    {
                        k: -np.mean(list(dic.values()))
                        for k, dic in g.fitness_dict.items()
                    }
                )
                if len(mus) == 1:
                    g.fitness = list(mus.values())[0]
                else:
                    coef_dict = {"KS": 10, "RSS": 1}
                    g.fitness = np.sum([coef_dict[k] * mean for k, mean in mus.items()])
                if not np.isnan(g.fitness):
                    valid_gs[i] = g
            sorted_gs = [
                valid_gs[i]
                for i in sorted(
                    list(valid_gs.keys()),
                    key=lambda i: valid_gs[i].fitness,
                    reverse=True,
                )
            ]
            self.store(sorted_gs, Ngen)
            # reg.vprint(f'Generation {Ngen} evaluated', 1)
            return sorted_gs

    def store(self, sorted_gs, Ngen):
        """
        Stores the best genome from the current generation and updates the best fitness value.
        Optionally, stores data for all genomes in the current generation.

        Args:
            sorted_gs (list): A list of genomes sorted by their fitness in descending order.
            Ngen (int): The current generation number.

        Returns:
            None
        """
        if len(sorted_gs) > 0:
            g0 = sorted_gs[0]
            if self.best_genome is None or g0.fitness > self.best_genome.fitness:
                self.best_genome = g0
                self.best_fitness = self.best_genome.fitness
                reg.conf.Model.setID(self.selector.bestConfID, self.best_genome.mConf)
        vprint(f"Generation {Ngen} best_fitness : {self.best_fitness}", 1)
        if self.store_data:
            self.all_genomes_dic += [
                {
                    "generation": Ngen,
                    **{p.name: g.gConf[k] for k, p in self.selector.space_objs.items()},
                    "fitness": g.fitness,
                    **g.fitness_dict.flatten(),
                }
                for g in sorted_gs
                if g.fitness_dict is not None
            ]

    @property
    def generation_completed(self):
        """
        Check if the current generation is completed.

        A generation is considered completed if the number of steps taken (self.t)
        is greater than or equal to the maximum number of steps allowed (self.Nsteps),
        or if the number of agents is less than or equal to the minimum number of agents
        required by the selector (self.selector.Nagents_min).

        Returns:
            bool: True if the generation is completed, False otherwise.
        """
        return self.t >= self.Nsteps or len(self.agents) <= self.selector.Nagents_min

    @property
    def max_generation_completed(self):
        """
        Check if the maximum number of generations has been completed.

        Returns:
            bool: True if the current generation number is greater than or equal to the
                  maximum number of generations specified by the selector, otherwise False.
        """
        return (
            self.selector.Ngenerations is not None
            and self.generation_num >= self.selector.Ngenerations
        )

    def sim_step(self):
        """
        Advances the simulation by one step.

        This method increments the simulation time, performs a simulation step,
        updates the screen manager, and updates the simulation state. If the
        current generation is completed, it either builds a new generation or
        finalizes the simulation if the maximum number of generations is reached.
        """
        self.t += 1
        self.step()
        self.screen_manager.step()
        self.update()
        # self.generation_step_num += 1
        if self.generation_completed:
            self.end()
            if not self.max_generation_completed:
                self.build_generation(self.sorted_genomes)
            else:
                self.finalize()

    def step(self):
        """
        Executes a single step of the genetic algorithm simulation.

        If threads are available, each thread will perform a step.
        Otherwise, the agents will perform a step.

        Additionally, if an exclusion function is defined in the evaluator,
        it will be applied to each agent. If an agent is excluded by the function,
        its genome's fitness will be set to negative infinity.
        """
        if self.threads:
            for thr in self.threads:
                thr.step()
        else:
            self.agents.step()
        if self.evaluator.exclude_func is not None:
            for robot in self.agents:
                if self.evaluator.exclude_func(robot):
                    robot.genome.fitness = -np.inf

    def end(self):
        """
        Finalizes the current generation by performing the following steps:
        1. Records the end time of the generation and calculates its duration.
        2. Logs the completion of the generation.
        3. Records the agents' final states.
        4. Creates output data for the generation.
        5. Collects and processes the data into a LarvaDatasetCollection.
        6. Evaluates the robots based on the collected data and sorts their genomes.
        7. Deletes the agents for the current generation.
        8. Resets internal logs and time counters.
        9. Records the end time of the evaluation and calculates its duration.
        10. Logs the completion of the evaluation.

        This method is typically called at the end of each generation in a genetic algorithm simulation.
        """
        self.end_generation_time = util.TimeUtil.current_time_sec()
        gen_dur = self.end_generation_time - self.start_generation_time
        vprint(f"Generation {self.generation_num} completed in {gen_dur} sec", 1)
        self.agents.nest_record(self.collectors["end"])
        self.create_output()
        self.data_collection = LarvaDatasetCollection.from_agentpy_output(self.output)
        self.sorted_genomes = self.eval_robots(
            ds=self.data_collection.datasets,
            Ngen=self.generation_num,
            genome_dict=self.genome_dict,
        )
        self.delete_agents()
        self._logs = {}
        self.t = 0
        # self.gen_progressbar.finish()
        self.end_generation_eval_time = util.TimeUtil.current_time_sec()
        gen_eval_dur = self.end_generation_eval_time - self.prestart_generation_time
        vprint(f"Generation {self.generation_num} evaluated in {gen_eval_dur} sec", 1)

    def update(self):
        """
        Updates the state of the genetic algorithm simulation.

        This method performs the following actions:
        1. Records the current state of the agents using the step collector.
        2. (Commented out) Updates the progress bar for the genetic algorithm generation.

        Note: The progress bar update is currently commented out.
        """
        self.agents.nest_record(self.collectors["step"])
        # self.gen_progressbar.update(self.t)

    def finalize(self):
        """
        Finalizes the genetic algorithm simulation.

        This method performs the following actions:
        - Sets the running flag to False, indicating the simulation has ended.
        - Calls the finalize method on the screen manager to clean up any resources.
        - If a progress bar is being used, it finishes the progress bar.
        - Prints the best fitness value found during the simulation.
        - If data storage is enabled, stores all genomes to the specified directory.

        Returns:
            None
        """
        self.running = False
        self.screen_manager.finalize()
        if self.progress_bar:
            self.progress_bar.finish()
        vprint(f"Best fittness: {self.best_genome.fitness}", 1)
        if self.store_data:
            self.store_genomes(dic=self.all_genomes_dic, save_to=self.data_dir)

    def store_genomes(self, dic: dict, save_to: str) -> None:
        """
        Stores the genomes in a DataFrame, sorts them by fitness, and saves the results to a CSV file.
        Additionally, generates a correlation DataFrame and a difference DataFrame if possible.

        Parameters:
        dic (dict): Dictionary containing genome data.
        save_to (str): Path to the directory where the results will be saved.

        Returns:
        None
        """
        df = pd.DataFrame.from_records(dic)
        df = df.round(3)
        df.sort_values(by="fitness", ascending=False, inplace=True)
        reg.graphs.dict["mpl"](
            data=df, font_size=18, save_to=save_to, name=self.selector.bestConfID
        )
        df.to_csv(f"{save_to}/{self.selector.bestConfID}.csv")
        try:
            cols = [p.name for k, p in self.selector.space_objs.items()]
            self.corr_df = df[["fitness"] + cols].corr()
        except:
            pass
        try:
            self.diff_df, row_colors = diff_df(
                mIDs=[self.selector.base_model, self.selector.bestConfID],
                ms=[self.selector.mConf0, self.best_genome.mConf],
            )
        except:
            pass

    def build_threads(self, robots: list) -> list:
        """
        Distributes the given robots among multiple threads for parallel processing.

        This method divides the list of robots into sublists, each assigned to a separate thread.
        The number of threads created is equal to the number of CPU cores available. Each thread
        processes a subset of the robots list.

        Args:
            robots (list): A list of robot instances to be processed.

        Returns:
            list: A list of threads that were created and executed.
        """
        N = multiprocessing.cpu_count()
        threads = []
        N_per_cpu = math.floor(len(robots) / N)
        vprint(f"num_robots_per_cpu: {N_per_cpu}", 0)

        for i in range(N - 1):
            p0 = i * N_per_cpu
            p1 = (i + 1) * N_per_cpu
            vprint(f"core: {i + 1} positions: {p0} : {p1}", 0)
            thread = GA_thread(robots[p0:p1])
            thread.start()
            vprint(f"thread {i + 1} started", 0)
            threads.append(thread)

        # last sublist of robots
        p0 = (N - 1) * N_per_cpu
        vprint(f"last core, start_pos {p0}", 0)
        thread = GA_thread(robots[p0:])
        thread.start()
        vprint("last thread started", 0)
        threads.append(thread)

        for t in threads:
            t.join()
        return threads


class GA_thread(threading.Thread):
    """
    A thread class for running a genetic algorithm on a list of robots.

    Attributes:
        robots (list): A list of robot instances that the genetic algorithm will operate on.

    Methods:
        step():
            Executes the step method for each robot in the robots list.
    """

    def __init__(self, robots: list):
        threading.Thread.__init__(self)
        self.robots = robots

    def step(self):
        for robot in self.robots:
            robot.step()


def optimize_mID(
    mID0,
    ks,
    evaluator,
    mID1=None,
    experiment="exploration",
    Nagents=10,
    Nelits=2,
    Ngenerations=3,
    duration=0.5,
    **kwargs,
):
    """
    Optimize a model configuration (stored under a unique ID) using a genetic algorithm.

    Parameters:
    mID0 (str): The initial model ID to start the optimization from.
    ks (list): List of keys defining the search space for the genetic algorithm.
    evaluator (callable): A function to evaluate the fitness of each genome.
    mID1 (str, optional): The model ID to store the best configuration. Defaults to mID0.
    experiment (str, optional): The type of experiment to run. Defaults to "exploration".
    Nagents (int, optional): Number of agents in the population. Defaults to 10.
    Nelits (int, optional): Number of elite genomes to carry over to the next generation. Defaults to 2.
    Ngenerations (int, optional): Number of generations to run the genetic algorithm. Defaults to 3.
    duration (float, optional): Duration of each simulation in the genetic algorithm. Defaults to 0.5.
    **kwargs: Additional keyword arguments to pass to the GAlauncher.

    Returns:
    dict: A dictionary with the model ID as the key and the best genome configuration as the value.
    """
    if mID1 is None:
        mID1 = mID0

    p = AttrDict(
        {
            "ga_select_kws": {
                "Nagents": Nagents,
                "Nelits": Nelits,
                "Ngenerations": Ngenerations,
                "init_mode": "model",
                "space_mkeys": ks,
                "base_model": mID0,
                "bestConfID": mID1,
            },
            "env_params": reg.conf.Env.getID("arena_200mm"),
            "experiment": experiment,
        }
    )
    GA = GAlauncher(parameters=p, evaluator=evaluator, duration=duration, **kwargs)
    best_genome = GA.simulate()
    return {mID1: best_genome.mConf}


reg.gen.GAselector = class_generator(GAselector)
reg.gen.GAevaluation = class_generator(GAevaluation)


class GAconf(SimOps):
    """
    Configuration class for the genetic algorithm (GA) simulation.
    """

    env_params = reg.conf.Env.confID_selector()
    experiment = reg.conf.Ga.confID_selector()
    ga_eval_kws = ClassAttr(reg.gen.GAevaluation, doc="The GA evaluation configuration")
    ga_select_kws = ClassAttr(reg.gen.GAselector, doc="The GA selection configuration")

    scene = param.String("no_boxes", doc="The name of the scene to load")


# reg.gen.Ga=class_generator(GAlauncher)
reg.gen.Ga = class_generator(GAconf)
