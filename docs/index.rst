####################################
Larvaworld Documentation
####################################

.. image:: https://img.shields.io/pypi/v/larvaworld.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/larvaworld/
   :alt: PyPI

.. image:: https://img.shields.io/badge/python-3.10%20%7C%203.11-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/github/license/nawrotlab/larvaworld
   :target: https://github.com/nawrotlab/larvaworld/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/docs%20by-sphinx-blue.svg
   :target: https://www.sphinx-doc.org/
   :alt: Documentation Built by Sphinx

----

Welcome to Larvaworld's documentation
=====================================

**Larvaworld** is an open-source Python package and virtual laboratory for the study of *Drosophila melanogaster* larval behavior. It combines:

- agent-based simulations of virtual larvae,
- modular behavioral and neural control,
- energetics and life-history models, and
- standardized pipelines for analysis of both simulated and experimental locomotion data.

If you work with larval behavior – experimentally, computationally, or both – Larvaworld aims to give you a single coherent environment for simulating, analyzing and comparing behavioral datasets.

Project links
-------------

- **Source code (GitHub)**: https://github.com/nawrotlab/larvaworld
- **PyPI package**: https://pypi.org/project/larvaworld/
- **Larvaworld paper (preprint)**: *Larvaworld: A behavioral simulation and analysis platform for Drosophila larva*
  bioRxiv 2025.06.15.659765; https://doi.org/10.1101/2025.06.15.659765

For the scientific background, model assumptions and validation results, please refer to the paper above and to the :doc:`concepts/theory_overview` section of this documentation.

----

.. toctree::
   :maxdepth: 1
   :hidden:

   concepts/theory_overview

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   installation
   concepts/dependencies
   usage

.. toctree::
   :maxdepth: 1
   :caption: Concepts & architecture
   :hidden:

   concepts/architecture_overview
   concepts/simulation_modes
   concepts/experiment_configuration_pipeline
   concepts/module_interaction

.. toctree::
   :maxdepth: 1
   :caption: Working with Larvaworld
   :hidden:

   concepts/experiment_types
   working_with_larvaworld/single_experiments
   working_with_larvaworld/model_evaluation
   working_with_larvaworld/replay
   working_with_larvaworld/ga_optimization_advanced

.. toctree::
   :maxdepth: 1
   :caption: Agents & environments
   :hidden:

   agents_environments/larva_agent_architecture
   agents_environments/brain_module_architecture
   agents_environments/arenas_and_substrates

.. toctree::
   :maxdepth: 1
   :caption: Data pipeline
   :hidden:

   data_pipeline/lab_formats_import
   data_pipeline/data_processing
   data_pipeline/reference_datasets

.. toctree::
   :maxdepth: 1
   :caption: Visualization & interfaces
   :hidden:

   visualization/keyboard_controls
   visualization/web_applications
   visualization/visualization_snapshots
   visualization/plotting_api

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   "Configuration"
   tutorials/CONFTYPES.ipynb
   tutorials/environment_configuration.ipynb
   tutorials/sensorscapes.ipynb

   "Simulation"
   tutorials/cli.ipynb
   tutorials/single_simulation.ipynb
   tutorials/model_evaluation.ipynb
   tutorials/genetic_algorithm_optimization.ipynb

   "Data"
   tutorials/import_datasets.ipynb
   tutorials/replay.ipynb

   "Development"
   tutorials/library_interface.ipynb
   tutorials/custom_module.ipynb
   tutorials/remote_model_interface.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:
   
   concepts/code_structure
   contributing
   API Reference <autoapi/larvaworld/index>



.. include:: CITATION.rst

----

About the Authors
=================

Larvaworld was originally developed by **Panagiotis Sakagiannis** at the Computational Neuroscience lab of the University of Cologne (PI: Martin Paul Nawrot) and is currently maintained and further developed by Panagiotis Sakagiannis and Alexandros Marantis.

For inquiries, contact: `p.sakagiannis@uni-koeln.de <mailto:p.sakagiannis@uni-koeln.de>`_

The project is open source, and many features were made possible by contributors volunteering their time at the `Computational Neuroscience lab <https://computational-systems-neuroscience.de/>`_. See the `Contributors Page <https://github.com/nawrotlab/larvaworld/graphs/contributors>`_ to learn more.
