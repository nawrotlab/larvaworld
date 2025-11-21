# Single Experiments

This page covers how to run **single behavioral experiments** in Larvaworld using the `Exp` simulation mode. We'll explore two case studies: **Dish Exploration** and **Olfactory Learning**.

---

## Overview

The `Exp` mode runs **one simulation** with fixed parameters, making it ideal for:

- ✅ Quick testing of configurations
- ✅ Video generation for presentations
- ✅ Single-condition analysis
- ✅ Model demonstrations

For more details on simulation modes, see {doc}`../concepts/simulation_modes`.

---

## Case Study 1: Dish Exploration

### Purpose

The **Dish Exploration** experiment is the fundamental baseline assay for studying spontaneous larval locomotion in a circular arena without odors or food.

```{mermaid}
graph TB
    subgraph "Setup"
        Arena[Circular Arena<br/>60-90 mm diameter]
        Larvae[N larvae<br/>Uniform distribution]
        Duration[3-10 minutes<br/>Free movement]
    end

    subgraph "Metrics"
        Dispersion[Dispersion Index<br/>Spatial spread over time]
        Velocity[Velocity Distribution<br/>Speed patterns]
        Tortuosity[Path Tortuosity<br/>Trajectory straightness]
        Activity[Activity Patterns<br/>Runs, turns, pauses]
    end

    subgraph "Applications"
        Baseline[Baseline Behavior<br/>Control experiments]
        Validation[Model Validation<br/>Compare to real data]
        Phenotype[Phenotype Screening<br/>Behavioral differences]
    end

    Arena --> Dispersion
    Larvae --> Velocity
    Duration --> Tortuosity

    Dispersion --> Baseline
    Velocity --> Validation
    Tortuosity --> Phenotype
    Activity --> Phenotype

    style Arena fill:#2196f3,stroke:#1976d2,stroke-width:2px,color:#fff
    style Dispersion fill:#f44336,stroke:#d32f2f,stroke-width:2px,color:#fff
    style Baseline fill:#4caf50,stroke:#388e3c,stroke-width:2px,color:#000
```

### Running via CLI

```bash
larvaworld Exp dish -N 10 -duration 5.0
```

**Parameters**:

- `-N 10`: 10 larvae
- `-duration 5.0`: 5 minutes simulated time

### Running via Python

```python
from larvaworld.lib.sim import ExpRun

# Create experiment
run = ExpRun(
    experiment="dish",
    N=10,
    duration=5.0,
    screen_kws={"show_display": True}
)

# Run simulation
run.simulate()

# Access dataset
dataset = run.datasets[0]
print(f"Tracked {len(dataset.agent_ids)} larvae")
print(f"Duration: {dataset.config.duration} min")
```

### Key Metrics

**Computed automatically** by Larvaworld:

| Metric               | Description                | Unit          |
| -------------------- | -------------------------- | ------------- |
| **Dispersion Index** | Spatial spread over time   | cm            |
| **Linear Velocity**  | Forward speed distribution | mm/s          |
| **Angular Velocity** | Turning rate distribution  | rad/s         |
| **Tortuosity**       | Path straightness          | dimensionless |
| **Activity Index**   | Fraction of time active    | 0-1           |
| **Stride Frequency** | Crawling frequency         | Hz            |
| **Turn Rate**        | Reorientation frequency    | 1/s           |

### Analysis Example

```python
# Process dataset
dataset.preprocess(
    drop_collisions=True,
    interpolate_nans=True,
    filter_f=3.0  # Low-pass filter at 3 Hz
)

dataset.process(
    proc_keys=["angular", "spatial"],
    dsp_starts=[0],
    dsp_stops=[40, 60],
    tor_durs=[5, 10, 20],
)

dataset.annotate(
    anot_keys=[
        "bout_detection",
        "bout_distribution",
        "interference",
    ]
)

# Print summary statistics
print(dataset.e)  # Endpoint metrics
```

---

## Case Study 2: Olfactory Learning

### Purpose

The **Olfactory Learning** experiment demonstrates how larvae associate odors with food rewards through reinforcement learning.

```{mermaid}
sequenceDiagram
    participant L as Larvae
    participant E as Environment
    participant M as Memory System

    Note over L,E: Pre-Test Phase
    L->>E: Naive odor preference
    E-->>L: Odor A & B present
    Note over M: Baseline preference (PI₀)

    Note over L,E: Training Phase
    L->>E: Conditioning
    E-->>L: Odor A + Food (CS+)
    E-->>L: Odor B alone (CS-)
    L->>M: Associate A with food (reward signal)
    Note over M: Q-learning updates gain

    Note over L,E: Test Phase
    L->>E: Trained preference
    E-->>L: Odor A & B present (no food)
    M->>L: Recall association (high gain for A)
    L->>E: Prefer Odor A
    Note over M: Memory expressed (PI > PI₀)

    Note over L,E: Post-Test
    L->>E: Extinction or retention
    Note over M: Memory decay analysis
```

### Experimental Phases

| Phase         | Duration | Odor A  | Odor B  | Food        | Purpose                   |
| ------------- | -------- | ------- | ------- | ----------- | ------------------------- |
| **Pre-Test**  | 3 min    | Present | Present | Absent      | Baseline preference (PI₀) |
| **Training**  | 10 min   | Present | Present | With A only | Conditioning (CS+/CS-)    |
| **Test**      | 3 min    | Present | Present | Absent      | Memory expression (PI)    |
| **Post-Test** | 3 min    | Present | Present | Absent      | Memory retention          |

### Running Training Phase

```bash
larvaworld Exp PItrain -N 30 -duration 10.0
```

### Running Test Phase

```bash
larvaworld Exp PItest_off -N 30 -duration 3.0
```

### Complete Protocol (Python)

```python
from larvaworld.lib.sim import ExpRun
from larvaworld.lib import util

# 1. Pre-Test (baseline)
pretest = ExpRun(
    experiment="PItest_off",
    larva_groups=[{"model": "RL_model", "N": 30}],
    duration=3.0
)
pretest.simulate()
pretest_ds = pretest.datasets[0]

# Compute baseline PI from endpoint x-positions
xs_pre = pretest_ds.e["x"].values
arena_xdim = pretest_ds.c.env_params.arena.dims[0]
PI_baseline = util.comp_PI(arena_xdim=arena_xdim, xs=xs_pre)
print(f"Baseline PI: {PI_baseline:.3f}")

# 2. Training (odor + food conditioning)
training = ExpRun(
    experiment="PItrain",
    larva_groups=[{"model": "RL_model", "N": 30}],
    duration=10.0
)
training.simulate()

# 3. Test (memory expression)
test = ExpRun(
    experiment="PItest_off",
    larva_groups=[{"model": "RL_model", "N": 30}],
    duration=3.0
)
test.simulate()
test_ds = test.datasets[0]

xs_test = test_ds.e["x"].values
PI_trained = util.comp_PI(arena_xdim=arena_xdim, xs=xs_test)
print(f"Trained PI: {PI_trained:.3f}")
print(f"Learning effect: {PI_trained - PI_baseline:.3f}")
```

### Preference Index (PI)

**Definition**: `PI = (N_CS+ - N_CS-) / (N_CS+ + N_CS-)`

Where:

- `N_CS+`: Time spent near odor associated with food
- `N_CS-`: Time spent near control odor

**Interpretation**:

- `PI = 0`: No preference
- `PI > 0`: Preference for CS+ (learning)
- `PI < 0`: Avoidance of CS+

**Computing PI**:

```python
from larvaworld.lib import util

xs = dataset.e["x"].values
arena_xdim = dataset.c.env_params.arena.dims[0]
PI = util.comp_PI(arena_xdim=arena_xdim, xs=xs)

print(f"Preference Index: {PI:.3f}")
```

### Memory Mechanisms

Larvaworld implements two learning algorithms:

#### 1. Q-Learning (Reinforcement Learning)

**Algorithm**: TD-learning with reward-modulated gain adaptation

```python
# Memory module updates sensory gain
if reward > 0:
    gain[odor_A] += α * reward  # Increase gain for rewarded odor
    gain[odor_B] -= α * reward  # Decrease gain for non-rewarded odor
```

**Code Location**: `/lib/model/modules/memory.py` (`RLmemory` class)

#### 2. Mushroom Body (MB) Model

**Algorithm**: Hebbian learning with KC-MBON synaptic plasticity

**Code Location**: `/lib/model/modules/memory.py` (`RemoteBrianModelMemory` class)

### Analysis Example

```python
# Load dataset
dataset = test.datasets[0]

# Compute preference index
PI = comp_PI(dataset)

# Plot odor navigation and trajectories
from larvaworld.lib.plot import bearing, traj

bearing.plot_chunk_Dorient2source(
    source_ID="CS_plus",
    datasets=[dataset],
    chunk="run",
)
traj.traj_1group(dataset)

# Plot memory gain evolution (RL model)
if hasattr(dataset.step_data, 'gain'):
    import matplotlib.pyplot as plt
    plt.plot(dataset.step_data['gain'])
    plt.xlabel('Time (s)')
    plt.ylabel('Sensory Gain')
    plt.title('Memory Gain Evolution')
    plt.show()
```

---

## General Workflow

### 1. Select Experiment

Choose from 57 preconfigured experiments:

```python
from larvaworld.lib import reg

# List available experiments
exp_ids = reg.conf.Exp.confIDs
print(f"Available: {len(exp_ids)} experiments")

# Inspect experiment
exp_conf = reg.conf.Exp.getID("chemotaxis")
print(exp_conf)
```

See {doc}`../concepts/experiment_types` for the full list.

### 2. Customize Parameters

Override default parameters:

```python
run = ExpRun(
    experiment="chemotaxis",

    # Environment customization
    env_params={
        "arena": {"geometry": [0.15, 0.15]},  # 15cm x 15cm
        "odorscape": {
            "odor_layers": [
                {"id": "banana", "peak": [0.075, 0], "intensity": 2.0}
            ]
        }
    },

    # Agent customization
    larva_groups=[
        {"model": "navigator", "N": 30, "age": 72.0}
    ],

    # Simulation settings
    duration=10.0,
    dt=0.1,

    # Visualization
    screen_kws={"vis_mode": "video", "video_name": "chemotaxis.mp4"}
)
```

### 3. Run Simulation

```python
run.simulate()
```

### 4. Access Results

```python
# Get dataset
dataset = run.datasets[0]

# Summary statistics
print(dataset.endpoint_data)

# Time-series data
print(dataset.step_data.keys())

# Configuration
print(dataset.config)
```

### 5. Analyze Data

```python
# Preprocessing
dataset.preprocess(filter_f=3.0)

# Metrics
dataset.process(proc_keys=["angular", "spatial"])

# Annotations
dataset.annotate(
    anot_keys=["bout_detection", "bout_distribution", "interference"]
)

# See :doc:`../visualization/plotting_api` for plotting examples.
```

### 6. Save Results

```python
# Save to HDF5
run.store()

# Location: data/SimGroup/exp_runs/{experiment}/{id}/
print(f"Saved to: {run.dir}")
```

---

## Video Generation

Create videos for presentations or publications:

```python
run = ExpRun(
    experiment="dish",
    Npoints=5,
    duration=3.0,
    screen_kws={
        "vis_mode": "video",
        "video_name": "dish_exploration.mp4",
        "fps": 10,
        "video_speed": 2  # 2x speed
    }
)
run.simulate()
```

For keyboard controls during visualization, see {doc}`../visualization/keyboard_controls`.

---

## Related Documentation

- {doc}`../concepts/simulation_modes` - Simulation mode comparison
- {doc}`../concepts/experiment_types` - All 57 preconfigured experiments
- {doc}`../concepts/experiment_configuration_pipeline` - Configuration system
- {doc}`../agents_environments/arenas_and_substrates` - Environment setup
- {doc}`../data_pipeline/data_processing` - Data processing pipeline
- {doc}`../tutorials/single_simulation` - Step-by-step tutorial
