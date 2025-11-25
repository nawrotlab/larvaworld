API Reference
=============

Complete API documentation for all Larvaworld modules and classes.

.. note::
   Detailed per-class/per-function API documentation is auto-generated from docstrings using Sphinx AutoAPI. Use the navigation below to explore specific modules.

----

Core Modules
------------

Simulation
~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   api_simulation

**Classes**: ``ExpRun``, ``BatchRun``, ``EvalRun``, ``GAlauncher``, ``ReplayRun``, ``BaseRun``

**Purpose**: Simulation engine and execution modes

----

Dataset
~~~~~~~

.. toctree::
   :maxdepth: 2

   api_dataset

**Classes**: ``LarvaDataset``, ``LarvaDatasetCollection``

**Purpose**: Data management, processing, and analysis

----

Generators
~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   api_generators

**Classes**: ``LabFormat``, ``EnvConf``, ``SimConfiguration``, ``EnrichConf``

**Purpose**: Configuration generators and data import

----

Dashboards
~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   api_dashboards

**Classes**: Dashboard applications (Panel-based)

**Purpose**: Interactive web applications

----

CLI
~~~

.. toctree::
   :maxdepth: 2

   api_cli

**Module**: ``larvaworld.cli``

**Purpose**: Command-line interface

----

AutoAPI Documentation
---------------------

For exhaustive API documentation, see the auto-generated reference:

.. toctree::
   :maxdepth: 3

   autoapi/index

----

Related Documentation
---------------------

- :doc:`../usage` - Basic usage guide
- :doc:`../concepts/architecture_overview` - Platform architecture
- Workflow guides: :doc:`../working_with_larvaworld/single_experiments`, :doc:`../working_with_larvaworld/model_evaluation`, :doc:`../working_with_larvaworld/replay`, :doc:`../working_with_larvaworld/ga_optimization_advanced`
- Tutorials: :doc:`../tutorials/configuration`, :doc:`../tutorials/simulation`, :doc:`../tutorials/data`, :doc:`../tutorials/development`
