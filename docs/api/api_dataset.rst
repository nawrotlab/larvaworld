Dataset API
===========

Data structures for managing, processing, and analyzing larva trajectories.

----

LarvaDataset
------------

**Purpose**: Store and process trajectory data for a group of larvae

**Location**: ``larvaworld.lib.process.dataset.LarvaDataset``

**Key Attributes**:

.. code-block:: python

   dataset = run.datasets[0]

   dataset.e           # Endpoint data (DataFrame)
   dataset.s           # Step-wise data (DataFrame)
   dataset.c           # Configuration (AttrDict)
   dataset.agent_ids   # List of larva IDs

**Key Methods**:

Preprocessing
~~~~~~~~~~~~~

.. code-block:: python

   dataset.preprocess(
       drop_collisions=True,
       interpolate_nans=True,
       filter_f=3.0,
       rescale_by=0.001,
       transposition="center"
   )

Processing
~~~~~~~~~~

.. code-block:: python

   dataset.process(
       proc_keys=["angular", "spatial"],
       dsp_starts=[0],
       dsp_stops=[60],
       tor_durs=[5, 10]
   )

Annotation
~~~~~~~~~~

.. code-block:: python

   dataset.annotate(
       anot_keys=[
           "bout_detection",
           "bout_distribution",
           "interference"
       ]
   )

Plotting
~~~~~~~~

.. code-block:: python

   from larvaworld.lib.process import LarvaDatasetCollection

   collection = LarvaDatasetCollection(datasets=[dataset])
   # Use graph IDs such as "trajectories" or "distros"
   collection.plot(ids=["trajectories"])

   # For more plotting functions, see :doc:`../visualization/plotting_api`.

**Related**: :doc:`../data_pipeline/data_processing`

----

LarvaDatasetCollection
----------------------

**Purpose**: Manage multiple datasets

**Location**: ``larvaworld.lib.process.dataset.LarvaDatasetCollection``

**Usage**:

.. code-block:: python

   from larvaworld.lib.process.dataset import LarvaDatasetCollection

   collection = LarvaDatasetCollection(datasets=[ds1, ds2, ds3])
   collection.plot_comparison()

----

Helper Functions
----------------

comp_PI
~~~~~~~

**Purpose**: Compute Preference Index

**Location**: ``larvaworld.lib.util``

**Usage**:

.. code-block:: python

   from larvaworld.lib import util

   xs = dataset.e["x"].values
   arena_xdim = dataset.c.env_params.arena.dims[0]
   PI = util.comp_PI(arena_xdim=arena_xdim, xs=xs)

----

eval_fast
~~~~~~~~~

**Purpose**: Fast model evaluation (KS tests)

**Location**: ``larvaworld.lib.process.evaluation.eval_fast``

**Usage**:

.. code-block:: python

   from larvaworld.lib.process.evaluation import eval_fast

   ks_results = eval_fast(
       datasets=[model_dataset],
       refDataset=ref_dataset,
       metric_definition="angular"
   )

----

See Also
--------

- :doc:`../data_pipeline/data_processing` - Processing pipeline
- :doc:`../working_with_larvaworld/model_evaluation` - Evaluation workflows
- AutoAPI: :doc:`autoapi/larvaworld/lib/process/index`
