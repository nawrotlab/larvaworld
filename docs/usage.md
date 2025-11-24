(usage)=

# Basic Usage

Assuming you've followed the {ref}`installation steps <installation>`, you're now ready to run your first simulation!

Larvaworld can be used in **two main ways**:

1. **Command-Line Interface (CLI)** – Quick experiments via terminal
2. **Python API** – Programmatic control for custom workflows

---

## Command-Line Interface (CLI)

Larvaworld provides an `argparse`-based CLI for running simulations directly from the terminal.

### Check Installation

Verify that the CLI is working:

```bash
larvaworld --version
```

### Get Help

View available commands and options:

```bash
larvaworld --help
```

### Run a Simple Experiment

Run a **dish exploration** experiment with 5 larvae for 3 minutes:

```bash
larvaworld Exp dish -N 5 -duration 3.0 -vis_mode video
```

**What happens:**

- 5 larvae are placed in a circular arena (Petri dish)
- The simulation runs for 3 minutes (simulated time)
- A visualization window opens showing the larvae exploring (requires `-vis_mode video`)
- Data is saved to the default output directory

### Common CLI Options

| Option            | Description                      | Example           |
| ----------------- | -------------------------------- | ----------------- |
| `-N`, `--Nagents` | Number of larvae                 | `-N 20`           |
| `--duration`      | Simulation duration (minutes)    | `--duration 10.0` |
| `--dt`            | Simulation timestep (seconds)    | `--dt 0.05`       |
| `-vis_mode`       | Visualization mode (`video` for real-time display, `image` for snapshots) | `-vis_mode video` |

### Save Output to a Custom Directory

```bash
larvaworld Exp dish -N 5 -duration 3.0 -dir /path/to/output
```

For more CLI details, see the full {doc}`CLI tutorial <tutorials/cli>`.

---

## Python API

For more control, use Larvaworld programmatically in Python scripts or Jupyter notebooks.

### Basic Experiment (Dish Exploration)

```python
from larvaworld.lib.sim import ExpRun

# Create and run a dish exploration experiment
run = ExpRun(
    experiment="dish",
    N=5,
    duration=3.0,
    screen_kws={"vis_mode": "video"}
)

# Run the simulation
run.simulate()

# Access the first dataset
dataset = run.datasets[0]
print(dataset.e)       # Endpoint (summary) data
print(dataset.s.head())  # Step-wise time-series sample
```

### Run Without Visualization (Faster)

```python
run = ExpRun(
    experiment="chemotaxis",
    N=20,
    duration=10.0,
    screen_kws={}  # No visualization (headless)
)
run.simulate()
```

### Custom Configuration

```python
from larvaworld.lib.sim.single_run import ExpRun
from larvaworld.lib import reg

# Load a stored environment configuration
env_conf = reg.conf.Env.getID("arena_200mm")

# Load a stored model configuration
model_conf = reg.conf.Model.getID("explorer")

# Create a custom experiment
run = ExpRun(
    experiment="dish",
    env_params=env_conf,
    larva_groups={
        "explorer": {
            "model": model_conf,
            "distribution": {"N": 10}
        }
    },
    duration=5.0
)
run.simulate()
```

### Save and Load Results

```python
# Save dataset to HDF5
run.store()

# Load a saved dataset later via the reference registry
from larvaworld.lib import reg

dataset = reg.loadRef(id="exploration.30controls", load=True)
print(dataset.e)   # Endpoint data
print(dataset.s)   # Step-wise data
```

---

## Simulation Modes

Larvaworld supports **five simulation modes**, each accessed via a different run class:

| Mode       | Python Class   | CLI Command                   | Use Case                       |
| ---------- | -------------- | ----------------------------- | ------------------------------ |
| **Exp**    | `ExpRun`       | `larvaworld Exp <experiment>` | Single behavioral experiment   |
| **Batch**  | `BatchRun`     | `larvaworld Batch <batch_id>` | Parameter sweeps (advanced)    |
| **Eval**   | `EvalRun`      | `larvaworld Eval <refID>`     | Model evaluation vs. data      |
| **Replay** | `ReplayRun`    | `larvaworld Replay <dataset>` | Visualize existing datasets    |
| **Ga**     | `optimize_mID` | `larvaworld Ga <config>`      | Genetic algorithm optimization |

For a complete guide to simulation modes, see {doc}`concepts/simulation_modes`.

---

## Pre-configured Experiments

Larvaworld includes **20+ pre-configured experiments** spanning:

- **Exploration** (close-view, dish, dispersion)
- **Chemotaxis** (navigation, local search)
- **Odor Preference** (train & test, on/off food)
- **Foraging** (patchy food, uniform food)
- **Growth** (rearing, rovers vs. sitters)
- **Imitation** (realistic bodies, dataset imitation)
- **Games** (maze, capture the flag)

See {doc}`concepts/experiment_types` for the full list.

---
