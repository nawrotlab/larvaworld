# Experiment Configuration Pipeline

Larvaworld provides a **systematic configuration pipeline** that balances ease-of-use with flexibility. You can use predefined templates for quick experiments or customize every parameter for detailed research.

---

## Configuration Flow

```{mermaid}
flowchart LR
    Start([Select Experiment Type]) --> LoadTemplate[Load Experiment Template<br/>Predefined configuration]

    LoadTemplate --> ConfigEnv{Configure<br/>Environment}

    ConfigEnv --> ArenaType[Arena Type<br/>Circular, rectangular, custom]
    ConfigEnv --> OdorSetup[Odorscape Setup<br/>Sources, gradients]
    ConfigEnv --> FoodSetup[Food Setup<br/>Patches, grid, quality]
    ConfigEnv --> Obstacles[Obstacles<br/>Borders, walls]

    ArenaType --> ConfigAgents
    OdorSetup --> ConfigAgents
    FoodSetup --> ConfigAgents
    Obstacles --> ConfigAgents

    ConfigAgents{Configure<br/>Agents} --> ModelSelect[Select Larva Model<br/>Basic to complex]
    ConfigAgents --> NumAgents[Number of Agents<br/>N larvae]
    ConfigAgents --> InitPos[Initial Positions<br/>Distribution]
    ConfigAgents --> InitState[Initial State<br/>Age, hunger, etc.]

    ModelSelect --> ConfigSim
    NumAgents --> ConfigSim
    InitPos --> ConfigSim
    InitState --> ConfigSim

    ConfigSim{Configure<br/>Simulation} --> Duration[Duration<br/>Minutes]
    ConfigSim --> Epochs[Epochs<br/>Pre/test/post]
    ConfigSim --> Timestep[Timestep<br/>dt = 0.1 s]
    ConfigSim --> VisMode[Visualization<br/>video/screen/image/none]

    Duration --> Analysis
    Epochs --> Analysis
    Timestep --> Analysis
    VisMode --> Analysis

    Analysis{Configure<br/>Analysis} --> Metrics[Metrics to Compute<br/>Auto-select by exp type]
    Analysis --> RefData[Reference Data<br/>Compare to real?]
    Analysis --> Output[Output Options<br/>Save location, format]

    Metrics --> Ready
    RefData --> Ready
    Output --> Ready

    Ready[Ready to Run] --> Execute([Execute Simulation])

    style Start fill:#2196f3,stroke:#1976d2,stroke-width:3px,color:#fff
    style ConfigEnv fill:#ff9800,stroke:#f57c00,stroke-width:3px,color:#000
    style ConfigAgents fill:#9c27b0,stroke:#7b1fa2,stroke-width:3px,color:#fff
    style ConfigSim fill:#4caf50,stroke:#388e3c,stroke-width:3px,color:#000
    style Analysis fill:#f44336,stroke:#d32f2f,stroke-width:3px,color:#fff
    style Execute fill:#2196f3,stroke:#1976d2,stroke-width:3px,color:#fff
```

---

## Pipeline Stages

### 1. Select Experiment Type

**Purpose**: Choose from 57 preconfigured experiments.

**Code**:

```python
from larvaworld.lib import reg

# List all available experiments
exp_ids = reg.conf.Exp.confIDs
print(f"Available: {len(exp_ids)} experiments")

# Select an experiment
exp_conf = reg.conf.Exp.getID("chemotaxis")
```

**Predefined Experiments**: See {doc}`experiment_types` for the full list.

---

### 2. Load Template

**Purpose**: Load a complete configuration template.

**Implementation**: Templates are defined in `/lib/reg/stored_confs/sim_conf.py`.

**Template Structure**:

```python
{
    "env_params": {...},      # Environment configuration
    "larva_groups": [...],    # Larva groups
    "trials": {...},          # Trial structure (epochs)
    "vis_kwargs": {...},      # Visualization options
    "enrichment": {...}       # Additional modules
}
```

**Example**:

```python
from larvaworld.lib import reg

exp_conf = reg.conf.Exp.getID("chemotaxis")
print(exp_conf.keys())
# dict_keys(['env_params', 'larva_groups', 'trials', ...])
```

---

### 3. Configure Environment

**Purpose**: Set up the virtual arena and stimuli.

#### Arena Type

Load predefined arenas via the registry:

```python
from larvaworld.lib import reg

# Preconfigured circular arena
env_conf = reg.conf.Env.getID("dish")

# Rectangular arena (200mm x 200mm)
env_conf_rect = reg.conf.Env.getID("arena_200mm")
```

#### Odorscape Setup

**Purpose**: Define odor sources and gradients.

**Example**:

```python
env_params = {
    "odorscape": {
        "odor_layers": [
            {
                "id": "apple",
                "peak": [0.05, 0],  # Position (x, y)
                "spread": 0.02,      # Gaussian width
                "intensity": 1.0
            }
        ]
    }
}
```

#### Food Setup

**Purpose**: Place food sources.

**Options**:

- **Patches**: Discrete food patches
- **Grid**: Regular grid of patches
- **Uniform**: Continuous substrate

**Example**:

```python
env_params = {
    "food_params": {
        "source_groups": [
            {
                "group": "patches",
                "amount": 3,
                "radius": 0.005  # 5mm radius
            }
        ]
    }
}
```

#### Obstacles

**Purpose**: Add borders or walls.

**Example**:

```python
env_params = {
    "borders": [
        {"vertices": [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]}
    ]
}
```

For detailed environment options, see {doc}`../agents_environments/arenas_and_substrates`.

---

### 4. Configure Agents

**Purpose**: Define larva models and initial conditions.

#### Model Selection

Retrieve stored models from the registry:

```python
from larvaworld.lib import reg

model_conf = reg.conf.Model.getID("explorer")
larva_groups = [{"model": "explorer", "N": 10}]
```

#### Number of Agents

**Parameter**: `N` (default: 10)

```python
larva_groups = [
    {"model": "explorer", "N": 20},
    {"model": "navigator", "N": 10}
]
```

#### Initial Positions

Set initial distribution with registry-compatible keys:

```python
larva_groups = [
    {
        "model": "explorer",
        "N": 20,
        "distribution": {
            "mode": "uniform",
            "loc": [0, 0],   # center
            "s": 0.02,       # spread (m)
            "shape": "circle"
        }
    }
]
```

#### Initial State

**Options**:

- `age`: Initial age in hours (default: 72h = 3rd instar)
- `hunger`: Hunger level (0-1, default: 0.5)
- `development_stage`: Larval instar (1, 2, or 3)

**Example**:

```python
larva_groups = [
    {
        "model": "explorer",
        "N": 10,
        "age": 48.0,    # 2nd instar
        "hunger": 0.8   # Very hungry
    }
]
```

For detailed agent options, see {doc}`../agents_environments/larva_agent_architecture`.

---

### 5. Configure Simulation

**Purpose**: Set runtime parameters.

#### Duration

**Parameter**: `duration` (minutes)

```python
run = ExpRun(experiment="chemotaxis", duration=10.0)  # 10 minutes
```

#### Epochs

**Purpose**: Multi-phase experiments (e.g., training + test).

**Example**:

```python
trials = {
    "Ntrials": 2,
    "trial_durations": [5.0, 3.0],  # 5 min train, 3 min test
    "trial_names": ["train", "test"]
}
```

#### Timestep

**Parameter**: `dt` (seconds, default: 0.1s)

```python
run = ExpRun(experiment="chemotaxis", dt=0.05)  # Finer timestep
```

#### Visualization

**Options**:

- `'video'`: Export MP4/AVI
- `'screen'`: Real-time display
- `'image'`: Save snapshots
- `None`: No visualization (faster)

**Example**:

```python
screen_kws = {
    'vis_mode': 'video',
    'video_name': 'chemotaxis.mp4',
    'fps': 10
}

run = ExpRun(experiment="chemotaxis", screen_kws=screen_kws)
```

For keyboard controls, see {doc}`../visualization/keyboard_controls`.

---

### 6. Configure Analysis

**Purpose**: Define what to compute and save.

#### Metrics

**Auto-Selection**: Larvaworld auto-selects metrics based on experiment type.

**Manual Selection**:

```python
# Preprocessing
dataset.preprocess(
    drop_collisions=True,
    interpolate_nans=True,
    filter_f=3.0,           # Low-pass filter at 3 Hz
    rescale_by=0.001,       # mm to m
    transposition="center"  # Center trajectories
)

# Processing
dataset.process(
    proc_keys=["angular", "spatial"],
    dsp_starts=[0],
    dsp_stops=[40, 60],
    tor_durs=[5, 10, 20],
)

# Annotation
dataset.annotate(
    anot_keys=[
        "bout_detection",    # Detect strides, runs, pauses, turns
        "bout_distribution", # Bout distributions
        "interference",      # Crawl–bend interference
    ]
)
```

For details, see {doc}`../data_pipeline/data_processing`.

#### Reference Data

**Purpose**: Compare simulation to experimental data.

**Example**:

```python
from larvaworld.lib.sim import EvalRun

# Compare model against reference
eval_run = EvalRun(
    refID='exploration.30controls',  # Reference dataset
    modelIDs=['explorer', 'navigator'],
    duration=5.0
)
eval_run.simulate()
```

For details, see {doc}`../working_with_larvaworld/model_evaluation`.

#### Output Options

**Storage Location**:

```python
run = ExpRun(
    experiment="chemotaxis",
    dir="/path/to/output"
)
```

**HDF5 Format**: All datasets are saved as HDF5 files for efficient I/O.

---

### 7. Execute Simulation

**Purpose**: Run the configured experiment.

**Example**:

```python
from larvaworld.lib.sim import ExpRun

# Create run with all configurations
run = ExpRun(
    experiment="chemotaxis",
    env_params=env_params,
    larva_groups=larva_groups,
    duration=10.0,
    screen_kws=screen_kws
)

# Execute
run.simulate()

# Access results
dataset = run.datasets[0]
print(dataset.endpoint_data)
```

---

## Configuration Registry

Larvaworld uses a **configuration registry** (`/lib/reg/`) to manage templates:

### Registry Structure

```
lib/reg/
├── stored_confs/         # Predefined configurations
│   ├── sim_conf.py       # Experiments, environments
│   ├── data_conf.py      # Reference datasets
│   └── model_conf.py     # Larva models
├── conf.py               # Configuration classes
└── generators/           # Generators (EnvConf, LabFormat)
```

### Accessing Configurations

```python
from larvaworld.lib import reg

# Experiments
exp_conf = reg.conf.Exp.getID("chemotaxis")

# Environments
env_conf = reg.conf.Env.getID("arena_200mm")

# Models
model_conf = reg.conf.Model.getID("explorer")

# Reference datasets
ref_dataset = reg.loadRef(id="exploration.30controls", load=True)
```

---

## Complete Example

```python
from larvaworld.lib import reg
from larvaworld.lib.sim import ExpRun

# 1. Select experiment
exp_conf = reg.conf.Exp.getID("chemotaxis")

# 2. Customize environment
env_params = {
    "arena": {"geometry": [0.15, 0.15]},  # Larger arena
    "odorscape": {
        "odor_layers": [
            {"id": "banana", "peak": [0.075, 0], "intensity": 2.0}
        ]
    }
}

# 3. Customize agents
larva_groups = [
    {"model": "navigator", "N": 30, "age": 72.0}
]

# 4. Configure simulation
screen_kws = {"vis_mode": "video", "video_name": "custom_chemotaxis.mp4"}

# 5. Execute
run = ExpRun(
    experiment="chemotaxis",
    env_params=env_params,
    larva_groups=larva_groups,
    duration=10.0,
    screen_kws=screen_kws
)
run.simulate()

# 6. Analyze
dataset = run.datasets[0]
dataset.preprocess(filter_f=3.0)
dataset.process(proc_keys=["angular", "spatial"])

# See :doc:`../visualization/plotting_api` for plotting examples.
```

---

## Related Documentation

- {doc}`experiment_types` - All 57 preconfigured experiments
- {doc}`../agents_environments/arenas_and_substrates` - Environment configuration
- {doc}`../agents_environments/larva_agent_architecture` - Agent models
- {doc}`../working_with_larvaworld/single_experiments` - Running experiments
- {doc}`../data_pipeline/data_processing` - Data processing pipeline
