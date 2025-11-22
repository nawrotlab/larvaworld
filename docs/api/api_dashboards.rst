Dashboards API
==============

Interactive web applications built with Panel (Holoviz stack).

----

Launching Dashboards
--------------------

**CLI**:

.. code-block:: bash

   larvaworld-app

**Access**: ``http://localhost:5006``

----

Available Dashboards
--------------------

Experiment Viewer
~~~~~~~~~~~~~~~~~

**Purpose**: Interactive exploration of simulation results

**File**: ``larvaworld.dashboards.experiment_viewer``

**Features**:

- Load saved experiments
- Plot trajectories, metrics, distributions
- Filter by time window, agent ID
- Export plots

----

Track Viewer
~~~~~~~~~~~~

**Purpose**: Detailed trajectory inspection

**File**: ``larvaworld.dashboards.track_viewer``

**Features**:

- 2D trajectory plots
- Velocity/acceleration profiles
- Zoom and pan
- Multi-agent comparison

----

Model Inspector
~~~~~~~~~~~~~~~

**Purpose**: Explore model parameters

**File**: ``larvaworld.dashboards.model_inspector``

**Features**:

- Browse available models
- View parameter values
- Compare configurations

----

Module Inspector
~~~~~~~~~~~~~~~~

**Purpose**: Inspect behavioral modules

**File**: ``larvaworld.dashboards.module_inspector``

**Features**:

- Crawler, Turner, Feeder modules
- Real-time parameter adjustment

----

Neural Oscillator Inspector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Visualize neural oscillators

**File**: ``larvaworld.dashboards.neural_oscillator_inspector``

**Features**:

- Phase plots
- Frequency analysis
- Coupling visualization

----

Development Status
------------------

.. warning::
   Dashboards are functional but under active development. API may change in future releases.

----

See Also
--------

- :doc:`../visualization/web_applications` - Dashboard usage guide
- AutoAPI: :doc:`autoapi/larvaworld/dashboards/index`
