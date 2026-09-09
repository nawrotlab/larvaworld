.. rst-class:: lw-centered

#############################
Optimization & evaluation
#############################

A model is only interesting if you can say how well it reproduces the animal. This section is about
measuring that distance and then closing it: first evaluating a set of candidate models against a
reference dataset, then letting a genetic algorithm search a model's parameter space for you.

.. rubric:: Prerequisites

:doc:`../1_getting_started/single_simulation` and a reference dataset. The lessons use
``reg.default_refID``, which ships with the package, so nothing has to be downloaded.

The lessons
===========

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`checklist;1.5em;sd-mr-1` Model evaluation
      :link: model_evaluation
      :link-type: doc

      Run several candidate models against the same reference dataset and compare them on
      kinematic metrics — the measurement step that everything else builds on.

      +++
      :bdg-primary:`intermediate` :bdg-secondary:`~25 min`

   .. grid-item-card:: :octicon:`git-branch;1.5em;sd-mr-1` Genetic algorithm optimization
      :link: genetic_algorithm_optimization
      :link-type: doc

      How the GA is put together: the parameter space, the selection rules that build each
      generation, and the fitness that ranks a genome.

      +++
      :bdg-primary:`intermediate` :bdg-secondary:`~25 min`

   .. grid-item-card:: :octicon:`flame;1.5em;sd-mr-1` Worked example: turner noise
      :link: ga_turner_noise_optimization
      :link-type: doc

      One optimization from start to finish — two parameters of the turner module, fitted against
      real turning behavior, with before/after videos and a model diff.

      +++
      :bdg-success:`worked example` :bdg-secondary:`~40 min`

.. dropdown:: What you will be able to do afterwards
   :icon: light-bulb

   - Point an evaluation at a reference dataset by ID and read the resulting error tables.
   - Choose evaluation metrics deliberately instead of accepting the defaults.
   - Build an optimization space from whole modules, or narrow it to individual parameters.
   - Set generation count, population size and elitism, and know what each one costs you.
   - Save the best genome as a new model configuration and re-run any experiment with it.
   - Compare the optimized model against both the original model and the real animals.

.. dropdown:: Related reference pages
   :icon: link

   - :doc:`/working_with_larvaworld/model_evaluation` — evaluation as a reference page
   - :doc:`/working_with_larvaworld/ga_optimization_advanced` — the algorithm in detail
   - :doc:`/working_with_larvaworld/batch_runs_advanced` — parameter sweeps, the other search mode
   - :doc:`/data_pipeline/reference_datasets` — what the shipped reference datasets contain

----

**Next:** :doc:`../6_extending_larvaworld/index` — when fitting parameters is not enough and you
need new behavior.

.. toctree::
   :hidden:
   :maxdepth: 1

   model_evaluation
   genetic_algorithm_optimization
   ga_turner_noise_optimization
