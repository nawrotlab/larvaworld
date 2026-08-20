###############################
3 · Models & environments
###############################

Everything a simulation needs — the arena, the food, the odor plume, the wind, the larvae
themselves — is a *configuration*: a nested, typed, named object kept in Larvaworld's registry.
This section is about reading those configurations, changing them, and storing your own.

.. rubric:: Prerequisites

:doc:`../1_getting_started/python_api_basics`. You should be comfortable loading a configuration by
ID and printing it.

The lessons
===========

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`archive;1.5em;sd-mr-1` The configuration registry
      :link: configuration_registry
      :link-type: doc

      What a *conftype* is, how the registry stores one, and the full lifecycle of a
      configuration: look up, copy, modify, save under a new ID, delete.

      +++
      :bdg-primary:`intermediate` :bdg-secondary:`~20 min`

   .. grid-item-card:: :octicon:`container;1.5em;sd-mr-1` Building an environment
      :link: environment_configuration
      :link-type: doc

      Arena geometry and dimensions, food sources and grids, borders, and the sensory layers that
      sit on top of them.

      +++
      :bdg-primary:`intermediate` :bdg-secondary:`~20 min`

   .. grid-item-card:: :octicon:`broadcast;1.5em;sd-mr-1` Sensory landscapes
      :link: sensory_landscapes
      :link-type: doc

      The three modalities a virtual larva can sense — odor, temperature, wind — and how to render
      each one so you can see what the agent is in.

      +++
      :bdg-primary:`intermediate` :bdg-secondary:`~20 min`

.. dropdown:: What you will be able to do afterwards
   :icon: light-bulb

   - Name the conftypes Larvaworld manages and list the stored IDs of any of them.
   - Copy a stored configuration, change a field, and save it under your own ID — or delete it.
   - Assemble an arena with sources, borders and a food grid from scratch.
   - Choose between a Gaussian and a diffusion odorscape, and say what changes.
   - Render a static or animated view of an environment without running a full experiment.

.. dropdown:: Related reference pages
   :icon: link

   - :doc:`/agents_environments/arenas_and_substrates` — arenas, substrates and food
   - :doc:`/agents_environments/larva_agent_architecture` — what a larva agent is made of
   - :doc:`/agents_environments/brain_module_architecture` — the modules inside the brain
   - :doc:`/concepts/experiment_configuration_pipeline` — how a configuration becomes a run
   - :doc:`/visualization/web_applications` — the Portal's Environment Builder does this in a browser

----

**Next:** :doc:`../4_optimization_and_evaluation/index` — find out how good your model actually is.

.. toctree::
   :hidden:
   :maxdepth: 1

   configuration_registry
   environment_configuration
   sensory_landscapes
