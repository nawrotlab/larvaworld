import sys

import numpy as np
import pandas as pd
import random


from datetime import datetime
from agentpy.version import __version__
from agentpy.datadict import DataDict
from agentpy.sample import Range, Values

from agentpy.tools import make_list, InfoStr

from larvaworld.lib.model import Object
from larvaworld.lib import reg, aux
from larvaworld.lib.param import SimOps


class BasicABModel(Object):
    '''
        Basic Class for the Agent-based model
        Extends the agentpy Model class

    '''

    def __init__(self, id='ABModel', parameters=None, _run_id=None, **kwargs):

        # Prepare parameters
        self.p = aux.AttrDict()
        if parameters:
            for k, v in parameters.items():
                if isinstance(v, (Range, Values)):
                    v = v.vdef
                self.p[k] = v

        # Iniate model as model object with id 0
        self._id_counter = -1
        super().__init__(model=self, id=id)

        # Simulation attributes
        self.t = 0
        self.running = False
        self._run_id = _run_id

        # Random number generators
        # Can be re-initiated with seed by Model.run()
        self.random = random.Random()
        self.nprandom = np.random.default_rng()

        # Recording
        self._logs = {}
        self.reporters = {}
        self.output = DataDict()
        self.output.info = {
            'model_type': self.type,
            'time_stamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'agentpy_version': __version__,
            'python_version': sys.version[:5],
            'experiment': False,
            'completed': False
        }

        # Private variables
        self._steps = None
        self._partly_run = False
        self._setup_kwargs = kwargs
        self._set_var_ignore()

    def __repr__(self):
        return self.type

    # Class Methods --------------------------------------------------------- #

    @classmethod
    def as_function(cls, **kwargs):
        """ Converts the model into a function that can be used with the
        `ema_workbench <https://emaworkbench.readthedocs.io/>`_ library.

        Arguments:
            **kwargs: Additional keyword arguments that will passed
                to the model in addition to the parameters.

        Returns:
            function:
                The model as a function that takes
                parameter values as keyword arguments and
                returns a dictionary of reporters.
        """

        superkwargs = kwargs

        def agentpy_model_as_function(**kwargs):
            model = cls(kwargs, **superkwargs)
            model.run(display=False)
            return model.reporters

        agentpy_model_as_function.__doc__ = f"""
        Performs a simulation of the model '{cls.__name__}'.

        Arguments:
            **kwargs: Keyword arguments with parameter values.

        Returns:
            dict: Reporters of the model.
        """

        return agentpy_model_as_function

    # Properties ------------------------------------------------------------ #

    @property
    def info(self):
        rep = f"Agent-based model {{"
        items = list(self.__dict__.items())
        for k, v in items:
            if k[0] != '_':
                v = v._short_repr() if '_short_repr' in dir(v) else v
                rep += f"\n'{k}': {v}"
        rep += '\n}'
        return InfoStr(rep)

    # Handling object ids --------------------------------------------------- #

    def _new_id(self):
        """ Returns a new unique object id (int). """
        self._id_counter += 1
        return self._id_counter

    # Recording ------------------------------------------------------------- #

    def report(self, rep_keys, value=None):
        """ Reports a new simulation result.
        Reporters are meant to be 'summary statistics' or 'evaluation measures'
        of the simulation as a whole, and only one value can be stored per run.
        In comparison, variables that are recorded with :func:`Model.record`
        can be recorded multiple times for each time-step and object.

        Arguments:
            rep_keys (str or list of str):
                Name(s) of the reporter(s) to be documented.
            value (int or float, optional): Value to be reported.
                The same value will be used for all `rep_keys`.
                If none is given, the values of object attributes
                with the same name as each rep_key will be used.

        Examples:

            Store a reporter `x` with a value `42`::

                model.report('x', 42)

            Define a custom model that stores a reporter `sum_id`
            with the sum of all agent ids at the end of the simulation::

                class MyModel(ap.Model):
                    def setup(self):
                        agents = ap.AgentList(self, self.p.agents)
                    def end(self):
                        self.report('sum_id', sum(self.agents.id))

            Running an experiment over different numbers of agents for this
            model yields the following datadict of reporters::

                >>> sample = ap.sample({'agents': (1, 3)}, 3)
                >>> exp = ap.Experiment(MyModel, sample)
                >>> results = exp.run()
                >>> results.reporters
                        sum_id
                run_id
                0            1
                1            3
                2            6
        """
        for rep_key in make_list(rep_keys):
            if value is not None:
                self.reporters[rep_key] = value
            else:
                self.reporters[rep_key] = getattr(self, rep_key)

    # Placeholder methods for custom simulation methods --------------------- #

    def setup(self):
        """ Defines the model's actions before the first simulation step.
        Can be overwritten to initiate agents and environments."""
        pass

    def step(self):
        """ Defines the model's actions
        during each simulation step (excluding `t==0`).
        Can be overwritten to define the models' main dynamics."""
        pass

    def update(self):
        """ Defines the model's actions
        after each simulation step (including `t==0`).
        Can be overwritten for the recording of dynamic variables. """
        pass

    def end(self):
        """ Defines the model's actions after the last simulation step.
        Can be overwritten for final calculations and reporting."""
        pass

    # Simulation routines (in line with ipysimulate) ------------------------ #

    def set_parameters(self, parameters):
        """ Adds and/or updates the parameters of the model. """
        self.p.update(parameters)

    def sim_setup(self, steps=None, seed=None):
        """ Prepares time-step 0 of the simulation.
        Initiates (additional) steps and the two random number generators,
        and then calls :func:`Model.setup` and :func:`Model.update`. """

        # Prepare random number generators if initial run
        if self._partly_run is False:
            if seed is None:
                if 'seed' in self.p:
                    seed = self.p['seed']  # Take seed from parameters
                else:
                    seed = random.getrandbits(128)
            if not ('report_seed' in self.p and not self.p['report_seed']):
                self.report('seed', seed)
            self.random = random.Random(seed)
            npseed = self.random.getrandbits(128)
            self.nprandom = np.random.default_rng(seed=npseed)

        # Prepare simulation steps
        if steps is None:
            self._steps = self.p['steps'] if 'steps' in self.p else np.nan
        else:
            self._steps = self.t + steps

        # Initiate simulation
        self.running = True
        self._partly_run = True

        # Execute setup and first update
        self.setup(**self._setup_kwargs)
        self.update()

        # Stop simulation if t too high
        if self.t >= self._steps:
            self.running = False

    def sim_step(self):
        """ Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`."""
        self.t += 1
        self.step()
        self.update()
        if self.t >= self._steps:
            self.running = False

    def sim_reset(self):
        """ Reset model to initial conditions. """
        # TODO Remove attributes
        self.record = super().record
        self.__init__(parameters=self.p,
                      _run_id=self._run_id,
                      **self._setup_kwargs)

    # Main simulation method for direct use --------------------------------- #

    def stop(self):
        """ Stops :meth:`Model.run` during an active simulation. """
        self.running = False

    def run(self, steps=None, seed=None, display=True):
        """ Executes the simulation of the model.
        Can also be used to continue a partly-run simulation
        for a given number of additional steps.

        It starts by calling :func:`Model.run_setup` and then calls
        :func:`Model.run_step` until the method :func:`Model.stop` is called
        or `steps` is reached. After that, :func:`Model.end` and
        :func:`Model.create_output` are called. The simulation results can
        be found in :attr:`Model.output`.

        Arguments:
            steps (int, optional):
                Number of (additional) steps for the simulation to run.
                If passed, the parameter 'Model.p.steps' will be ignored.
                The simulation can still be stopped with :func:'Model.stop'.
            seed (int, optional):
                Seed to initialize the model's random number generators.
                If none is given, the parameter 'Model.p.seed' is used.
                If there is no such parameter, a random seed will be used.
                For a partly-run simulation, this argument will be ignored.
            display (bool, optional):
                Whether to display simulation progress (default True).

        Returns:
            DataDict: Recorded variables and reporters.

        """

        dt0 = datetime.now()
        self.sim_setup(steps, seed)
        while self.running:
            self.sim_step()
            if display:
                print(f"\rCompleted: {self.t} steps", end='')
        self.end()
        self.create_output()

        self.output.info['completed'] = True
        self.output.info['created_objects'] = self._id_counter
        self.output.info['completed_steps'] = self.t
        self.output.info['run_time'] = ct = str(datetime.now() - dt0)

        if display:
            print(f"\nRun time: {ct}\nSimulation finished")

        return self.output

    # Data management ------------------------------------------------------- #

    def create_output(self):
        """ Generates a :class:`DataDict` with dataframes of all recorded
        variables and reporters, which will be stored in :obj:`Model.output`.
        """

        def output_from_obj_list(self, log_dict, columns):
            # Aggregate logs per object type
            # Log dict structure: {obj_type: obj_id: log}
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
                if obj_type == self.type:
                    del log['obj_id']
                    index_keys = ['t']
                else:
                    index_keys = ['obj_id', 't']
                df = pd.DataFrame(log)
                for k, v in columns.items():
                    df[k] = v  # Set additional index columns
                df = df.set_index(list(columns.keys()) + index_keys)
                self.output['variables'][obj_type] = df

        # 1 - Document parameters
        if self.p:
            self.output['parameters'] = DataDict()
            self.output['parameters']['constants'] = self.p.copy()

        # 2 - Define additional index columns
        columns = {}
        if self._run_id is not None:
            if self._run_id[0] is not None:
                columns['sample_id'] = self._run_id[0]
            if len(self._run_id) > 1 and self._run_id[1] is not None:
                columns['iteration'] = self._run_id[1]

        # 3 - Create variable output
        if self._logs:
            self.output['variables'] = DataDict()
            output_from_obj_list(self, self._logs, columns)

        # 4 - Create reporters output
        if self.reporters:
            d = {k: [v] for k, v in self.reporters.items()}
            for key, value in columns.items():
                d[key] = value
            df = pd.DataFrame(d)
            if columns:
                df = df.set_index(list(columns.keys()))
            self.output['reporters'] = df



class ABModel(BasicABModel,reg.SimConfigurationParams):

    def __init__(self, **kwargs):
        '''
        Basic simulation class that extends the agentpy.Model class and creates a larvaworld agent-based model (ABM).
        Further extended by classes supporting the various simulation modes in larvaworld.
        Specifies the simulation mode, type of experiment and simulation duration and timestep.
        Specifies paths for saving simulated data and results.

        Args:
            runtype: The simulation mode as defined by a subclass
            parameters: Dictionary of configuration parameters to be passed to the ABM model
            store_data: Whether to store simulation data. Defaults to True
            save_to: Path to store data. If not specified, it is automatically set to the runtype-specific subdirectory under the platform's ROOT/DATA directory
            id: Unique ID of the simulation. If not specified it is automatically set according to the simulation mode and experiment type.
            experiment: The experiment simulated
            offline: Whether to perform the simulation without launching a spatial arena. Defaults to False
            Box2D: Whether to implement the Box2D physics engine. Defaults to False
            larva_collisions: Whether to allow overlap between larva bodies. Defaults to True
            dt: The simulation timestep in seconds. Defaults to 0.1
            duration: The simulation duration in seconds. Defaults to None for unlimited duration. Computed from Nsteps if specified.
            Nsteps: The number of simulation timesteps. Defaults to None for unlimited timesteps. Computed from duration if specified.
            **kwargs: Arguments passed to the setup method
        '''

        reg.SimConfigurationParams.__init__(self, **kwargs)
        # self.initialize_superclasses(self.parameters)
        self.parameters.steps = self.Nsteps
        self.parameters.agentpy_output_kws = {'exp_name': self.experiment, 'exp_id': self.id,
                                     'path': f'{self.data_dir}/agentpy_output'}
        BasicABModel.__init__(self, parameters=self.parameters, id=self.id)


    # def initialize_superclasses(self, parameters,**kwargs):
    #     pass
