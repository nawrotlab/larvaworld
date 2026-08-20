##########################
2 · Experimental data
##########################

Larvaworld reads tracking data from several labs' formats and converts it into one standardized
dataset structure — the *same* structure a simulation produces. Everything downstream (metrics,
figures, replay, model evaluation) therefore works identically on recorded and simulated animals.

This section covers the import machinery once, then four complete worked examples on published
datasets. Each example downloads its own data and needs nothing but the cells being run in order.

.. rubric:: Prerequisites

:doc:`../1_getting_started/index`, or at least enough familiarity to know that Larvaworld keeps its
configurations in a registry. Disk space and patience for the downloads noted on each card.

Start here
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` Importing experimental data
      :link: import_datasets
      :link-type: doc

      The machinery: what a **lab format** is, where raw and processed data live, and how one
      ``import_dataset`` call turns a folder of tracker output into an analysable dataset.

      +++
      :bdg-primary:`start here` :bdg-secondary:`~30 min`

   .. grid-item-card:: :octicon:`repo-template;1.5em;sd-mr-1` Import a public dataset — template
      :link: import_public_dataset_template
      :link-type: doc

      A blank worked example to copy for your own data: fetch, identify the tracker, import,
      analyse, visualize.

      +++
      :bdg-info:`template` :bdg-secondary:`~30 min`

Worked examples
===============

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`pulse;1.5em;sd-mr-1` Free exploration
      :link: import_free_exploration_dataset
      :link-type: doc

      Naive third-instar larvae crawling in an empty dish (Schleyer lab, G-Node). The baseline
      case: what a larva does when nothing is done to it.

      +++
      :bdg-warning:`935 MB download`

   .. grid-item-card:: :octicon:`video;1.5em;sd-mr-1` DeepLabCut exports
      :link: import_DeepLabCut_dataset
      :link-type: doc

      Pose-estimation output rather than a purpose-built tracker: how a DLC CSV/HDF5 export becomes
      a Larvaworld dataset.

      +++
      :bdg-warning:`downloads data`

   .. grid-item-card:: :octicon:`beaker;1.5em;sd-mr-1` Feeding state
      :link: import_feeding_state_locomotion_dataset
      :link-type: doc

      Fed, sucrose-fed and starved groups compared: a real group contrast, imported and analysed
      end to end.

      +++
      :bdg-warning:`downloads data`

And then
========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`play;1.5em;sd-mr-1` Replaying experiments
      :link: replay
      :link-type: doc

      Drive the simulator with recorded tracks instead of a model: dispersal views, single-animal
      close-ups, body reconstruction from the midline, video export.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~25 min`

.. dropdown:: What you will be able to do afterwards
   :icon: light-bulb

   - Recognize which lab format matches your tracker's output, and what to do when none does.
   - Import one recording or a whole batch of dishes, filtered by track duration and animal count.
   - Have Larvaworld compute angular and spatial metrics and annotate strides, turns and pauses.
   - Register an imported dataset under a reference ID and reload it in a later session without
     touching the raw files again.
   - Produce the standard comparison figures — trajectories, endpoint boxplots, dispersal — for
     several groups at once.
   - Replay a dataset and export the video.

.. dropdown:: Related reference pages
   :icon: link

   - :doc:`/data_pipeline/lab_formats_import` — every supported format and its fields
   - :doc:`/data_pipeline/data_processing` — what preprocessing and annotation actually compute
   - :doc:`/data_pipeline/reference_datasets` — the datasets shipped with the package
   - :doc:`/working_with_larvaworld/replay` — replay as a reference page
   - :doc:`/visualization/plotting_api` — the plotting catalog

----

**Next:** :doc:`../3_models_and_environments/index` — build the virtual side of the comparison.

.. toctree::
   :hidden:
   :maxdepth: 1

   import_datasets
   import_public_dataset_template
   import_free_exploration_dataset
   import_DeepLabCut_dataset
   import_feeding_state_locomotion_dataset
   replay
