#########
Tutorials
#########

A guided course through Larvaworld, from running your first virtual experiment to importing your
own tracking data and writing your own behavioral modules.

Every lesson is a Jupyter notebook. You can read it on this site, or run it yourself and change the
numbers — which is the point. The lessons build on each other, but each one states what it assumes,
so you can also enter the course wherever your work starts.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`beaker;1.5em;sd-mr-1` Coming from the lab?
      :class-card: sd-border-1

      Start with :doc:`2_experimental_data/index`. You will import a published tracking dataset,
      have Larvaworld compute the standard kinematic metrics for you, and compare groups — without
      simulating anything.

   .. grid-item-card:: :octicon:`cpu;1.5em;sd-mr-1` Coming from modeling?
      :class-card: sd-border-1

      Start with :doc:`1_getting_started/index`, then go to
      :doc:`3_models_and_environments/index`. You will run a preconfigured experiment, then take
      apart the configuration that produced it.

.. dropdown:: Before you start
   :icon: checklist

   **Install Larvaworld** — see :doc:`/installation`. The tutorials use no optional dependencies
   unless a lesson says otherwise.

   **Know where your data lives.** Everything Larvaworld reads and writes — the configuration
   registry, imported datasets, simulation output — lives under a single data directory, printed by

   .. code-block:: python

      import larvaworld
      print(larvaworld.DATA_DIR)

   **Pick how you want to run the notebooks:**

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Way
        - What to do
      * - **From the Portal**
        - Run ``larvaworld-portal``, open a workflow card and press its notebook button. The
          notebook is copied into your workspace, so the shipped copy stays clean.
      * - **In Jupyter**
        - ``jupyter lab`` from a clone of the repository, then open ``docs/tutorials/``.
      * - **Read-only**
        - Right here. Every page shows the code and the output it produced.

   **A note on the switches.** Notebooks that can start a long simulation, open a window or write
   video guard those cells behind ``RUN_*`` switches that are off by default. The pages on this site
   show the output of everything *except* those cells. Flip a switch in your own copy when you want
   the heavy version.

The course
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` 1 · Getting started
      :link: 1_getting_started/index
      :link-type: doc

      The idea behind the platform, the command line, your first simulation, and the Python API
      that everything else is built on.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~75 min` :bdg-success:`no data needed`

   .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` 2 · Experimental data
      :link: 2_experimental_data/index
      :link-type: doc

      Import tracking data from any supported lab format, let Larvaworld annotate it, and replay it
      as a simulation. Four complete worked examples on published datasets.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~2 h` :bdg-warning:`downloads data`

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1` 3 · Models & environments
      :link: 3_models_and_environments/index
      :link-type: doc

      The configuration registry, arenas and food sources, and the odor, temperature and wind
      landscapes a virtual larva senses.

      +++
      :bdg-primary:`intermediate` :bdg-secondary:`~60 min` :bdg-success:`no data needed`

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` 4 · Optimization & evaluation
      :link: 4_optimization_and_evaluation/index
      :link-type: doc

      Measure how well a model reproduces real behavior, then let a genetic algorithm close the
      gap — including one full worked optimization.

      +++
      :bdg-primary:`intermediate` :bdg-secondary:`~90 min` :bdg-info:`uses a reference dataset`

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` 5 · Extending Larvaworld
      :link: 5_extending_larvaworld/index
      :link-type: doc

      Write your own behavioral module and plug it into the brain, or drive a module from an
      external simulator over a socket.

      +++
      :bdg-primary:`advanced` :bdg-secondary:`~60 min` :bdg-danger:`writes code`

Suggested paths
===============

.. tab-set::

   .. tab-item:: Experimentalist

      You have tracking data and want metrics, figures and group comparisons.

      #. :doc:`2_experimental_data/import_datasets` — what a lab format is and how an import works
      #. :doc:`2_experimental_data/import_free_exploration_dataset` — a complete worked example
      #. :doc:`2_experimental_data/replay` — see your recordings move again
      #. :doc:`4_optimization_and_evaluation/model_evaluation` — compare your data against a model

   .. tab-item:: Modeler

      You want to build, run and fit virtual larvae.

      #. :doc:`1_getting_started/single_simulation` — run a preconfigured experiment
      #. :doc:`1_getting_started/python_api_basics` — configure your own
      #. :doc:`3_models_and_environments/configuration_registry` — where configurations come from
      #. :doc:`4_optimization_and_evaluation/ga_turner_noise_optimization` — fit a module to data

   .. tab-item:: Developer

      You want to extend the platform itself.

      #. :doc:`1_getting_started/python_api_basics` — the objects you will be working with
      #. :doc:`3_models_and_environments/configuration_registry` — the registry contract
      #. :doc:`5_extending_larvaworld/custom_brain_modules` — write and register a module
      #. :doc:`5_extending_larvaworld/remote_model_interface` — talk to an external simulator

Reference material
==================

The tutorials stay practical on purpose. When you want the reasoning behind a design, or the full
list of options for something, these are the pages to read:

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Concepts
      :link: /concepts/theory_overview
      :link-type: doc
      :class-card: sd-border-1

      The model, the environment and the analysis pipeline, explained in full.

   .. grid-item-card:: Working with Larvaworld
      :link: /working_with_larvaworld/single_experiments
      :link-type: doc
      :class-card: sd-border-1

      Task-oriented reference for experiments, evaluation, replay, batch runs and GA.

   .. grid-item-card:: Data pipeline
      :link: /data_pipeline/lab_formats_import
      :link-type: doc
      :class-card: sd-border-1

      Supported lab formats, the processing steps and the reference datasets.

   .. grid-item-card:: Web applications
      :link: /visualization/web_applications
      :link-type: doc
      :class-card: sd-border-1

      The browser Portal: what each app does and how to launch it.

.. toctree::
   :hidden:
   :maxdepth: 2

   1_getting_started/index
   2_experimental_data/index
   3_models_and_environments/index
   4_optimization_and_evaluation/index
   5_extending_larvaworld/index
