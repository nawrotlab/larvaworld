CLI API
=======

Command-line interface for running simulations.

----

Entry Point
-----------

**Command**: ``larvaworld``

**Location**: ``larvaworld.cli.main.main()``

**Usage**:

.. code-block:: bash

   larvaworld --help

----

Simulation Modes
----------------

Exp (Experiment)
~~~~~~~~~~~~~~~~

**Purpose**: Run a single experiment

.. code-block:: bash

   larvaworld Exp dish -N 10 -duration 5.0

**Options**:

- ``-N, --Nagents``: Number of larvae
- ``--duration``: Simulation duration (minutes)
- ``--dt``: Timestep (seconds)
- ``--Box2D``: Enable Box2D physics
- ``--show_display``: Show visualization window

----

Batch
~~~~~

**Purpose**: Run batch simulations

.. code-block:: bash

   larvaworld Batch PItest_off -Nsims 10

----

Eval (Evaluation)
~~~~~~~~~~~~~~~~~

**Purpose**: Model evaluation

.. code-block:: bash

   larvaworld Eval -refID exploration.30controls -mIDs explorer navigator

----

Ga (Genetic Algorithm)
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Optimization

.. code-block:: bash

   larvaworld Ga exploration -Ngenerations 50

----

Replay
~~~~~~

**Purpose**: Dataset replay

.. code-block:: bash

   larvaworld Replay -refID exploration.30controls -video_name replay.mp4

----

Common Options
--------------

**Output**:

- ``-d, --dir``: Output directory
- ``--id``: Simulation ID

**Visualization**:

- ``--show_display``: Real-time display
- ``--video_name``: Export video

**Physics**:

- ``--Box2D``: Enable Box2D multisegment body
- ``--dt``: Timestep (default: 0.1 s)

----

Argument Parser
---------------

**Class**: ``SimModeParser``

**Location**: ``larvaworld.cli.argparser.SimModeParser``

**Purpose**: Parse command-line arguments for simulation modes

----

See Also
--------

- :doc:`../usage` - Basic usage guide
- :doc:`../tutorials/cli` - CLI tutorial
- AutoAPI: :doc:`autoapi/larvaworld/cli/index`
