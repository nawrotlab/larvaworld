.. rst-class:: lw-centered

########################
Extending Larvaworld
########################

Larvaworld's brain is deliberately modular: each behavioral module is a small Python class that maps
an input state to an output state, and each one can be swapped for your own. This section shows both
ways of doing that — writing the module in Python, and delegating it to an external simulator over a
socket.

.. rubric:: Prerequisites

:doc:`../1_getting_started/python_api_basics` and :doc:`../4_models_and_environments/configuration_registry`.
The second lesson additionally needs ``brian2`` if you want to run it rather than read it.

The lessons
===========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Custom brain modules
      :link: custom_brain_modules
      :link-type: doc

      Subclass an existing module, implement ``update()``, register it as a new mode, and point a
      model configuration at it — a custom olfactor and a custom thermosensor, end to end.

      +++
      :bdg-primary:`advanced` :bdg-secondary:`~30 min`

   .. grid-item-card:: :octicon:`plug;1.5em;sd-mr-1` Remote model interface
      :link: remote_model_interface
      :link-type: doc

      Run a module in another process — here a brian2 neuron model — and exchange state with the
      simulation every timestep over Larvaworld's IPC layer.

      +++
      :bdg-primary:`advanced` :bdg-secondary:`~30 min` :bdg-warning:`needs brian2`

.. dropdown:: What you will be able to do afterwards
   :icon: light-bulb

   - Say what a brain module receives, what it must produce, and when it is called.
   - Register a new implementation as a *mode* of an existing module type.
   - Activate your implementation from a model configuration and confirm it is being used.
   - Split a module across processes and keep its state consistent across timesteps.
   - Recognize when an external simulator needs a warm-up phase and state snapshotting.

.. dropdown:: Related reference pages
   :icon: link

   - :doc:`/agents_environments/brain_module_architecture` — the module hierarchy
   - :doc:`/agents_environments/larva_agent_architecture` — where modules sit inside an agent
   - :doc:`/concepts/module_interaction` — how state flows between modules
   - :doc:`/contributing` — getting your extension into the package

----

**Back to** :doc:`../index`.

.. toctree::
   :hidden:
   :maxdepth: 1

   custom_brain_modules
   remote_model_interface
