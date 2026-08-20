.. rst-class:: lw-centered

###################
Getting started
###################

Four lessons that take you from *what is this platform for* to *I can configure and run my own
virtual experiment from Python*. Nothing here needs experimental data, an internet connection or a
long-running simulation — every lesson runs in under a minute of compute.

.. rubric:: Prerequisites

Larvaworld installed (:doc:`/installation`) and a Python environment you can run notebooks in.
No prior knowledge of the package is assumed.

The lessons
===========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`book;1.5em;sd-mr-1` Theoretical background
      :link: theoretical_background
      :link-type: doc

      What it means to model a whole behaving organism, and the five components — energetics,
      homeostatic interface, nervous system, sensorimotor interface, environment — that Larvaworld
      is built out of.

      +++
      :bdg-secondary:`reading` :bdg-secondary:`~15 min`

   .. grid-item-card:: :octicon:`terminal;1.5em;sd-mr-1` The command line interface
      :link: command_line_interface
      :link-type: doc

      Run any of the five simulation modes without writing Python, and find out which arguments
      each mode accepts.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~15 min`

   .. grid-item-card:: :octicon:`play;1.5em;sd-mr-1` Your first simulation
      :link: single_simulation
      :link-type: doc

      Launch a preconfigured experiment from Python, watch it run headless, and look at the
      datasets it produced.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~20 min`

   .. grid-item-card:: :octicon:`code;1.5em;sd-mr-1` The Python API
      :link: python_api_basics
      :link-type: doc

      Load an experiment configuration, change an odor source, swap the model a larva group uses,
      and launch the modified experiment.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~25 min`

.. dropdown:: What you will be able to do afterwards
   :icon: light-bulb

   - Explain what a Larvaworld *experiment*, *larva group* and *model* are, and how they relate.
   - Run any simulation mode from a terminal, and read its ``--help`` output without guessing.
   - Launch an experiment from Python, headless or with a window, and control its duration and
     population size.
   - Find a stored configuration in the registry by ID, inspect it, and change one of its values.
   - Say where a simulation wrote its output.

.. dropdown:: Related reference pages
   :icon: link

   - :doc:`/concepts/theory_overview` — the full conceptual tour of the platform
   - :doc:`/concepts/simulation_modes` — what each of the five modes is for
   - :doc:`/concepts/experiment_types` — the catalog of preconfigured experiments
   - :doc:`/usage` — the same ground as a compact reference page
   - :doc:`/working_with_larvaworld/single_experiments` — worked case studies

----

**Next:** :doc:`../2_simulated_experiments/index` — see the platform reproduce published
behavioral assays.

.. toctree::
   :hidden:
   :maxdepth: 1

   theoretical_background
   command_line_interface
   single_simulation
   python_api_basics
