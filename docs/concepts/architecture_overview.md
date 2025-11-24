# Architecture Overview

Larvaworld follows a **layered architecture** pattern that separates concerns and enables flexible extension. This design allows users to access the platform through multiple interfaces while maintaining a unified core engine for simulation and analysis.

---

## Five-Layer Architecture

Larvaworld is organized into **five interconnected layers**, each with distinct responsibilities:

```{mermaid}
graph TB
    UI[User Interfaces<br/>CLI • Web • GUI]
    Sim[Simulation Engine<br/>Exp • Batch • GA • Eval]
    Model[Models<br/>Agents • Envs • Modules]
    Data[Data Layer<br/>Import • Storage • Analysis]
    Viz[Visualization<br/>Plots • Video • Dashboards]

    UI --> Sim
    Sim --> Model
    Model --> Data
    Data --> Viz
    Viz --> UI

    style UI fill:#2196f3,stroke:#1976d2,stroke-width:2px,color:#fff
    style Sim fill:#4caf50,stroke:#388e3c,stroke-width:2px,color:#000
    style Model fill:#9c27b0,stroke:#7b1fa2,stroke-width:2px,color:#fff
    style Data fill:#f44336,stroke:#d32f2f,stroke-width:2px,color:#fff
    style Viz fill:#ff9800,stroke:#f57c00,stroke-width:2px,color:#000
```

---

## Layer 1: User Interfaces

### Command-Line Interface (CLI)

**Location**: `/src/larvaworld/cli/`

The CLI provides direct access to all simulation modes through an `argparse`-based interface:

```bash
larvaworld Exp dish -N 5 -duration 3.0
larvaworld Eval -refID exploration.30controls -mIDs explorer navigator
larvaworld Batch PItest_off -Nsims 10
```

**Key Components**:

- `main.py`: Entry point with `main()` function
- `argparser.py`: `SimModeParser` class for argument parsing
- **Modes**: Exp, Batch, Ga, Eval, Replay

**Use case**: Quick simulation launching

---

### Web Dashboards

**Location**: `/src/larvaworld/dashboards/`

Interactive web applications built with **Panel** (Holoviz stack) and **Bokeh**:

| Dashboard                 | Purpose                               |
| ------------------------- | ------------------------------------- |
| **Experiment Viewer**     | View experiment results interactively |
| **Track Viewer**          | Inspect larva trajectories            |
| **Larva-model Inspector** | Explore larva models                  |
| **Module Inspector**      | Inspect behavioral modules            |

Launch with:

```bash
larvaworld-app
```

Access via: `http://localhost:5006`

**Use case**: Interactive exploration, parameter tuning, visualization

---

### GUI (Deprecated)

**Location**: `/src/larvaworld/gui/`

:::{warning}
The desktop GUI built with **PySimpleGUI** is **deprecated** but still present for backward compatibility. It is not included in the accessible entry points and may not be fully supported in current releases. **Use CLI or Web apps instead**.
:::

---

## Layer 2: Simulation Engine

**Location**: `/src/larvaworld/lib/sim/`

The simulation engine provides **five specialized modes**, each optimized for different workflows:

| Mode       | Class        | File                   | Purpose                        |
| ---------- | ------------ | ---------------------- | ------------------------------ |
| **Exp**    | `ExpRun`     | `single_run.py`        | Single experiment run          |
| **Batch**  | `BatchRun`   | `batch_run.py`         | Multiple parallel experiments  |
| **Ga**     | `GAlauncher` | `genetic_algorithm.py` | Genetic algorithm optimization |
| **Eval**   | `EvalRun`    | `model_evaluation.py`  | Model evaluation vs. real data |
| **Replay** | `ReplayRun`  | `dataset_replay.py`    | Replay recorded trajectories   |

**Common Base**:

- **BaseRun**: Base class for all simulation modes
- **ABModel**: Agentpy-based agent-based model

**Key Methods**:

- `simulate()`: Run the simulation loop
- `store()`: Save results to HDF5
- `analyze()`: Perform data analysis
- `plot()`: Generate analysis plots

For detailed information on simulation modes, see {doc}`simulation_modes`.

---

## Layer 3: Models

**Location**: `/src/larvaworld/lib/model/`

This layer contains the **agent models** (larvae) and **environment models** (arenas).

### Agent Models

**Key Classes**:

- **LarvaSim**: Complete larva agent (inherits from `LarvaMotile` and `BaseController`)
- **LarvaMotile**: Core agent step-by-step behavior (inherits from `LarvaSegmented`)
- **LarvaSegmented**: Morphology (shape, segments, sensors)
- **Brain**: Sensory integration and locomotor control
- **Locomotor**: Unified crawling, turning, and feeding

**Modules** (in `modules/`):

- **Sensors**: Olfactor, Touch, Windsensor, Thermo, Feeder
- **Brain**: DefaultBrain, NengoBrain, Brian2Brain
- **Locomotor**: Crawler, Turner, Feeder, Interference
- **Energetics**: DEB (Dynamic Energy Budget)

For detailed architecture, see {doc}`../agents_environments/larva_agent_architecture` and {doc}`../agents_environments/brain_module_architecture`.

### Environment Models

**Key Classes**:

- **Arena**: Arena with food/odor sources, borders
- **Source**: Single/distributed food/odor sources
- **Odorscape**: Gaussian or diffusion-generated gradients

**Configuration**:

- Configurable via `EnvConf` registry entries
- Pre-configured arenas (e.g., `'dish'`, `'arena_200mm'`)

For details, see {doc}`../agents_environments/arenas_and_substrates`.

---

## Layer 4: Data Layer

**Location**: `/src/larvaworld/lib/process/` and `/src/larvaworld/lib/reg/`

This layer handles **data import, storage, processing, and analysis**.

### Dataset Management

**LarvaDataset** (`dataset.py`):

- **Time-series data**: Pose, velocity, acceleration, orientation
- **Endpoint data**: Summary statistics per larva
- **Bout data**: Annotated behavioral events (strides, turns, pauses)
- **Storage**: HDF5 format for efficient I/O

**Methods**:

- `preprocess()`: Filtering, scaling, alignment
- `process()`: Compute metrics (angular, spatial, tortuosity)
- `annotate()`: Detect bouts (strides, turns, pauses)

For detailed workflows, see {doc}`../data_pipeline/data_processing`.

### Data Import

**LabFormat** (`reg/generators.py`):

- Import experimental datasets from diverse tracking systems
- **Supported lab-specific formats**: Schleyer, Jovanic, Berni, Arguello

For details, see {doc}`../data_pipeline/lab_formats_import`.

### Registry

**Configuration Registry** (`/src/larvaworld/lib/reg/`):

- **stored_confs/**: Preconfigured experiments, models, environments
- **reg.conf**: Access configurations (`Env`, `Model`, `Exp`, `Batch`, `Ga`)
- **reg.gen**: Generators for configurations and datasets

For details, see {doc}`experiment_configuration_pipeline`.

---

## Layer 5: Visualization

**Location**: `/src/larvaworld/lib/plot/` and `/src/larvaworld/lib/screen/`

This layer provides **plotting**, **real-time rendering**, and **video export**.

### Plotting

**Modules** (in `plot/`):

- `traj.py`: Trajectory plots
- `time.py`: Time-series plots
- `hist.py`: Histograms and distributions
- `bearing.py`: Bearing plots (odor navigation)
- `stridecycle.py`: Stride cycle analysis
- `deb.py`: DEB energetics plots

For details, see {doc}`../visualization/plotting_api`.

### Real-Time Visualization

**Screen** (`screen/`):

- Real-time 2D rendering using Pygame
- Configurable drawing (midline, contour, trails)
- Interactive controls (zoom, pause, snapshot)
- Video export (MP4, AVI)

For keyboard controls, see {doc}`../visualization/keyboard_controls`.

---

## Mapping to Codebase

| Layer                 | Primary Folders               | Key Classes/Files                                          |
| --------------------- | ----------------------------- | ---------------------------------------------------------- |
| **User Interfaces**   | `/cli`, `/dashboards`, `/gui` | `cli/main.py`, `dashboards/main.py`                        |
| **Simulation Engine** | `/lib/sim`                    | `ExpRun`, `BatchRun`, `EvalRun`, `GAlauncher`, `ReplayRun` |
| **Models**            | `/lib/model`                  | `LarvaSim`, `Brain`, `Locomotor`, `Env`                    |
| **Data Layer**        | `/lib/process`, `/lib/reg`    | `LarvaDataset`, `LabFormat`, `reg.conf`                    |
| **Visualization**     | `/lib/plot`, `/lib/screen`    | `plot/traj.py`, `screen/drawing.py`                        |

---

## Data Flow

### Experiment Execution Flow

```
1. User Input (CLI/Web/Python)
   ↓
2. Configuration Loading (reg.conf)
   ↓
3. Simulation Setup (ExpRun/BatchRun/EvalRun)
   ↓
4. Agent Creation (LarvaSim × N)
   ↓
5. Simulation Loop (BaseRun.simulate())
   ↓
6. Data Collection (LarvaDataset)
   ↓
7. Storage (HDF5)
   ↓
8. Analysis & Plotting (dataset.process(), LarvaDatasetCollection.plot())
   ↓
9. Visualization (Screen / Video / Dashboards)
```

For detailed runtime interactions, see {doc}`module_interaction`.

---

## Related Documentation

- {doc}`simulation_modes` - Detailed simulation mode comparison
- {doc}`code_structure` - Code metrics and folder organization
- {doc}`dependencies` - Third-party dependencies
- {doc}`module_interaction` - Runtime module interactions
- {doc}`experiment_types` - Pre-configured experiments
