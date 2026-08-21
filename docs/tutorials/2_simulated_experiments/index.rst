.. rst-class:: lw-centered

#########################
Simulated experiments
#########################

Five behavioral assays, each reproduced by running one **stored experiment** and changing nothing
but how many larvae take part.

These lessons are the shortest path from *what does this platform do?* to an answer. They assume no
knowledge of how a configuration is built — every experiment used here already exists in the
registry, and the notebooks read it rather than write it. The experiments and the model behind them
are those of `Sakagiannis et al. (2025) <https://doi.org/10.7554/eLife.104262>`_.

.. rubric:: Prerequisites

:doc:`../1_getting_started/index`, or at least
:doc:`../1_getting_started/single_simulation`. No data has to be downloaded and no configuration
has to be understood.

The experiments
===============

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`beaker;1.5em;sd-mr-1` Free exploration
      :link: free_exploration
      :link-type: doc

      Twenty larvae in an empty 10 cm dish. The baseline behavior every other assay is built on,
      and the one the locomotory model was calibrated against.

      +++
      :bdg-primary:`start here` :bdg-secondary:`~10 min`

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Dispersal
      :link: dispersion
      :link-type: doc

      The same animals released from a single point in a larger arena, and their spread over time
      compared directly against a recording of real larvae.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~12 min` :bdg-info:`compares with real data`

   .. grid-item-card:: :octicon:`location;1.5em;sd-mr-1` Chemotaxis
      :link: chemotaxis
      :link-type: doc

      An odor source and a larva that can smell it: climbing a gradient from a distance, and
      exploring locally once the source has been found.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~15 min`

   .. grid-item-card:: :octicon:`git-compare;1.5em;sd-mr-1` Odor preference
      :link: odor_preference
      :link-type: doc

      Two odors on opposite sides of a dish and one population between them, summarized by the
      single number the standard group assay reports.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~10 min`

   .. grid-item-card:: :octicon:`container;1.5em;sd-mr-1` Feeding
      :link: feeding
      :link-type: doc

      A floor made entirely of food: how much the larvae eat, how they split their time between
      crawling and feeding, and the substrate visibly disappearing beneath them.

      +++
      :bdg-primary:`beginner` :bdg-secondary:`~15 min` :bdg-info:`writes a video`

.. dropdown:: What you will be able to do afterwards
   :icon: light-bulb

   - Run any stored experiment by name, with your own population size.
   - Read what a stored experiment contains - arena, sources, larva group, model, timing - without
     having to build one.
   - Find the datasets a run produced and say what is in them.
   - Produce the standard figures for an assay by calling plots by name.
   - Compare a simulated population against a recorded one on the same axes.
   - Compute and interpret a preference index.
   - Read a time budget of crawling, pausing and feeding, and render a depleting substrate.

.. dropdown:: Related reference pages
   :icon: link

   - :doc:`/concepts/experiment_types` - the full catalog of preconfigured experiments
   - :doc:`/working_with_larvaworld/single_experiments` - the same runs as a reference page
   - :doc:`/concepts/theory_overview` - the model behind the behavior
   - :doc:`/visualization/plotting_api` - the plotting catalog these notebooks draw from

----

**Next:** :doc:`../3_experimental_data/index` — do the same with your own recordings.

.. toctree::
   :hidden:
   :maxdepth: 1

   free_exploration
   dispersion
   chemotaxis
   odor_preference
   feeding
