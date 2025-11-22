# Figure 3: CLI Simulation Modes

## 📊 Image

![Simulation Modes](fig3_simulation_modes.png)

## Description

**CLI simulation modes. The simulation modes available in Larvaworld along with the respective argument to launch them via the command-line interface.**

This figure shows the different simulation modes that can be launched from the command line using the `larvaworld` command.

### Simulation Modes:

1. **Exp** (`-exp`): Single experiment simulation
2. **Batch** (`-batch`): Multiple simulations with parameter sweeps
3. **Ga** (`-ga`): Genetic algorithm optimization
4. **Eval** (`-eval`): Model evaluation against reference data
5. **Replay** (`-rep`): Replay stored or imported datasets

Each mode has:

- **Shortcut**: CLI argument (e.g., `-exp`)
- **Purpose**: What it's used for
- **Configuration**: Parameter requirements
- **Output**: What it produces

---

## Purpose

This figure serves to:

- ✅ **Guide users** on available simulation modes
- ✅ **Show CLI usage** for each mode
- ✅ **Explain differences** between modes
- ✅ **Quick reference** for developers

Developer note: This should be included as requested.

---

## Usage in ReadTheDocs

**Placement**: Getting Started / CLI Usage section

```rst
Simulation Modes
~~~~~~~~~~~~~~~~

Larvaworld provides five main simulation modes, each serving a distinct purpose.

.. figure:: _static/images/fig3_simulation_modes.png
   :alt: CLI Simulation Modes
   :align: center
   :width: 90%

   **Figure 3**: Available simulation modes in Larvaworld. Each mode can be
   launched via the command-line interface using the respective shortcut.

Command-Line Usage
^^^^^^^^^^^^^^^^^^

All simulation modes are accessed through the ``larvaworld`` command:

.. code-block:: bash

   # Single experiment
   larvaworld -exp <experiment_id>

   # Batch run
   larvaworld -batch <batch_id>

   # Genetic algorithm
   larvaworld -ga <ga_id>

   # Model evaluation
   larvaworld -eval <eval_id>

   # Dataset replay
   larvaworld -rep <dataset_id>

Mode Descriptions
^^^^^^^^^^^^^^^^^

**Exp (Experiment)**
   Run a single simulation with specified parameters. Fastest mode for testing
   and exploration.

**Batch**
   Execute multiple simulations with parameter sweeps. Useful for systematic
   exploration of parameter space.

**Ga (Genetic Algorithm)**
   Optimize model parameters using evolutionary algorithms to match target data.

**Eval (Evaluation)**
   Compare locomotory models against reference datasets using statistical tests.

**Replay**
   Visualize stored simulations or imported real experimental data.

For detailed information on each mode, see :ref:`simulation-modes-detailed`.
```

---

## Related Content

- **Table 2**: Simulation modes (detailed descriptions)
- **Diagram 11**: Simulation Types Comparison (from mermaid_diagrams)
- **CLI Tutorial**: `docs/tutorials/cli.ipynb`

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **File**: `/images/sim_modes.png`
- **Caption** (LaTeX line 445): "CLI simulation modes. The simulation modes available in Larvaworld along with the respective argument to launch them via the command-line interface."
- **Label**: `fig:sim_modes`
- **Related text**: Lines 436-439 in main.tex
