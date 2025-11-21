Generators API
==============

Configuration generators and data import utilities.

----

LabFormat
---------

**Purpose**: Import lab-specific experimental datasets

**Location**: ``larvaworld.lib.reg.generators.LabFormat``

**Usage**:

.. code-block:: python

   from larvaworld.lib import reg

   lab = reg.gen.LabFormat(labID="Schleyer")
   lab.import_dataset(
       parent_dir="exploration",
       merged=True,
       max_Nagents=30,
       min_duration_in_sec=60,
       id="my_experiment",
       refID="my_experiment",
       save_dataset=True,
   )

**Supported Labs**:

- ``"Schleyer"`` - 16 Hz, 12-point midline, 22-point contour
- ``"Jovanic"`` - 11.27 Hz, 11-point midline, convex hull
- ``"Berni"`` - 2 Hz, centroid only
- ``"Arguello"`` - 10 Hz, 5-point midline

**Related**: :doc:`../data_pipeline/lab_formats_import`

----

Configuration Classes
---------------------

EnvConf
~~~~~~~

**Purpose**: Environment configuration

**Location**: ``larvaworld.lib.reg.conf``

**Usage**:

.. code-block:: python

   from larvaworld.lib import reg

   env_conf = reg.conf.Env.getID("arena_200mm")

----

ModelConf
~~~~~~~~~

**Purpose**: Larva model configuration

**Usage**:

.. code-block:: python

   model_conf = reg.conf.Model.getID("explorer")

----

ExpConf
~~~~~~~

**Purpose**: Experiment configuration

**Usage**:

.. code-block:: python

   exp_conf = reg.conf.Exp.getID("chemotaxis")

----

BatchConf
~~~~~~~~~

**Purpose**: Batch run configuration

**Usage**:

.. code-block:: python

   batch_conf = reg.conf.Batch.getID("PItest_off")

----

GaConf
~~~~~~

**Purpose**: Genetic algorithm configuration

**Usage**:

.. code-block:: python

   ga_conf = reg.conf.Ga.getID("my_ga_config")

----

RefConf
~~~~~~~

**Purpose**: Reference dataset configuration

**Usage**:

.. code-block:: python

   ref_dataset = reg.loadRef(id="exploration.30controls", load=True)

----

Registry Access
---------------

The ``reg`` module provides unified access to all configurations:

.. code-block:: python

   from larvaworld.lib import reg

   # List available IDs
   print(reg.conf.Exp.confIDs)      # All experiments
   print(reg.conf.Model.confIDs)    # All models
   print(reg.conf.Env.confIDs)      # All environments
   print(reg.conf.Ref.confIDs)      # All references

   # Get configuration
   conf = reg.conf.Exp.getID("chemotaxis")

   # Load reference dataset
   dataset = reg.loadRef(id="exploration.30controls", load=True)

----

See Also
--------

- :doc:`../concepts/experiment_configuration_pipeline` - Configuration system
- :doc:`../data_pipeline/lab_formats_import` - Data import
- AutoAPI: :doc:`autoapi/larvaworld/lib/reg/index`
