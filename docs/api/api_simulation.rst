Simulation API
==============

Simulation engine classes for running experiments, batch runs, evaluations, genetic algorithms, and replay.

----

ExpRun
------

**Purpose**: Run a single behavioral experiment

**Location**: ``larvaworld.lib.sim.single_run.ExpRun``

**Key Methods**:

.. code-block:: python

   from larvaworld.lib.sim import ExpRun

   run = ExpRun(experiment="dish", N=10, duration=5.0)
   run.simulate()
   dataset = run.datasets[0]

**Parameters**:

- ``experiment`` (str): Experiment ID (e.g., "dish", "chemotaxis")
- ``N`` (int): Number of larvae
- ``duration`` (float): Simulation duration (minutes)
- ``env_params`` (dict): Environment configuration
- ``larva_groups`` (list): Larva group configurations
- ``screen_kws`` (dict): Visualization options

**Related**: :doc:`../working_with_larvaworld/single_experiments`

----

BatchRun
--------

**Purpose**: Run multiple simulations with parameter sweeps

**Location**: ``larvaworld.lib.sim.batch_run.BatchRun``

**Key Methods**:

.. code-block:: python

   from larvaworld.lib import reg
   from larvaworld.lib.sim import BatchRun

   batch_conf = reg.conf.Batch.getID("PItest_off")
   batch = BatchRun(experiment="PItest_off", **batch_conf)
   par_df, figs = batch.simulate(n_jobs=4)

**Related**: :doc:`../working_with_larvaworld/batch_runs_advanced`

----

EvalRun
-------

**Purpose**: Compare multiple models against reference data

**Location**: ``larvaworld.lib.sim.model_evaluation.EvalRun``

**Key Methods**:

.. code-block:: python

   from larvaworld.lib.sim import EvalRun

   eval_run = EvalRun(
       refID='exploration.30controls',
       modelIDs=['explorer', 'navigator'],
       duration=5.0
   )
   eval_run.simulate()
   eval_run.plot_results()

**Related**: :doc:`../working_with_larvaworld/model_evaluation`

----

GAlauncher
----------

**Purpose**: Genetic algorithm optimization

**Location**: ``larvaworld.lib.sim.genetic_algorithm.GAlauncher``

**Helper Function**:

.. code-block:: python

   from larvaworld.lib.sim.genetic_algorithm import GAevaluation, optimize_mID

   evaluator = GAevaluation(refID="exploration.30controls")
   results = optimize_mID(
       mID0="explorer",
       ks=["crawler", "turner"],  # Module names to optimize
       evaluator=evaluator,
       Ngenerations=50
   )

**Related**: :doc:`../working_with_larvaworld/ga_optimization_advanced`

----

ReplayRun
---------

**Purpose**: Replay existing datasets without simulation

**Location**: ``larvaworld.lib.sim.dataset_replay.ReplayRun``

**Key Methods**:

.. code-block:: python

   from larvaworld.lib.sim import ReplayRun

   replay = ReplayRun(
       refID='exploration.30controls',
       screen_kws={'vis_mode': 'screen'}
   )
   replay.run()

**Related**: :doc:`../working_with_larvaworld/replay`

----

BaseRun
-------

**Purpose**: Base class for all simulation modes

**Location**: ``larvaworld.lib.sim.base_run.BaseRun``

**Inherited by**: ``ExpRun``, ``BatchRun``, ``EvalRun``, ``GAlauncher``, ``ReplayRun``

**Common Methods**:

- ``simulate()``: Execute simulation
- ``store()``: Save to HDF5
- ``plot()``: Generate plots
- ``analyze()``: Compute metrics

----

See Also
--------

- :doc:`../concepts/simulation_modes` - Mode comparison
- :doc:`../concepts/module_interaction` - Runtime interactions
- AutoAPI: :doc:`autoapi/larvaworld/lib/sim/index`
