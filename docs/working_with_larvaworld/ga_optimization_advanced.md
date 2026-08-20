# Genetic Algorithm Optimization (Advanced)

The **Ga** simulation mode uses **evolutionary algorithms** to automatically optimize model parameters by matching simulated behavior to experimental reference data.

:::{warning}
`Ga` optimization is an **advanced feature** requiring significant computational resources (hours to days). Ensure you understand the basics (`Exp`, `Eval`) before attempting GA optimization.
:::

---

## Purpose

Use `Ga` mode to:

- ✅ **Automated parameter fitting**: Find optimal parameters without manual tuning
- ✅ **Multi-parameter optimization**: Simultaneously optimize many parameters
- ✅ **Behavioral matching**: Evolve models to match real larval behavior
- ✅ **Sensitivity-guided search**: Discover which parameters matter most

For mode comparison, see {doc}`../concepts/simulation_modes`.

---

## Evolutionary Algorithm Overview

```{mermaid}
graph TB
    Start([Start GA Optimization]) --> Init[Initialize Population<br/>Random genomes]

    Init --> Evaluate[Evaluate Fitness<br/>Run simulations]

    Evaluate --> Fitness{Calculate Fitness<br/>vs Reference Data}

    Fitness --> Compare[Compare to Target<br/>KS tests on metrics]

    Compare --> Rank[Rank Genomes<br/>by Fitness Score]

    Rank --> Converge{Converged?<br/>Max gen reached}

    Converge -->|No| Select[Selection<br/>Keep best genomes]

    Select --> Crossover[Crossover<br/>Combine parent genes]

    Crossover --> Mutate[Mutation<br/>Random perturbations]

    Mutate --> NewGen[New Generation<br/>Next population]

    NewGen --> Evaluate

    Converge -->|Yes| Best[Best Genome<br/>Optimal parameters]

    Best --> Output[Output Results<br/>Best config + history]

    Output --> End([End Optimization])

    style Start fill:#2196f3,stroke:#1976d2,stroke-width:3px,color:#fff
    style Init fill:#4caf50,stroke:#388e3c,stroke-width:2px,color:#000
    style Fitness fill:#ff9800,stroke:#f57c00,stroke-width:2px,color:#000
    style Select fill:#9c27b0,stroke:#7b1fa2,stroke-width:2px,color:#fff
    style Best fill:#e91e63,stroke:#c2185b,stroke-width:3px,color:#fff
    style End fill:#2196f3,stroke:#1976d2,stroke-width:3px,color:#fff
```

---

## Quick Start

### Python API

```python
from larvaworld.lib.sim.genetic_algorithm import GAevaluation, optimize_mID

# 1. Define evaluation against reference dataset
evaluator = GAevaluation(
    refID="exploration.30controls",
    # GA mode currently enriches only `proc_keys=["angular","spatial"]` by default.
    # Keep evaluation metrics aligned with those processed keys (or extend GAlauncher).
    cycle_curve_metrics=[],
    eval_metrics={
        "angular kinematics": ["b", "fov"],
        "spatial displacement": ["v", "a"],
    },
)

# 2. Run genetic algorithm to optimize locomotory model
results = optimize_mID(
    mID0="explorer",                      # Base model to optimize
    mID1="explorer_opt",                  # ID for optimized model (stored in registry)
    ks=["crawler", "turner"],             # Module names to optimize
    evaluator=evaluator,
    Ngenerations=1,                       # Increase for real runs
    Nagents=10,                           # Population size
    duration=0.05,                        # Minutes per agent (increase for real runs)
    screen_kws={"show_display": False},
    store_data=False,
)

# 3. Access optimized configuration
best_conf = results["explorer_opt"]      # Optimized model config (AttrDict)
```

---

## Workflow

### 1. Select Reference Dataset

```python
from larvaworld.lib import reg

# Load reference dataset
ref_dataset = reg.loadRef(id="exploration.30controls", load=True)
print(f"Reference: {ref_dataset.config.refID}")
print(f"Agents: {len(ref_dataset.agent_ids)}")
```

For importing datasets, see {doc}`../data_pipeline/lab_formats_import`.

---

### 2. Define Fitness / Evaluation

The `GAevaluation` class configures how genomes are evaluated against a reference dataset.
In the current codebase, `GAlauncher` uses **reference-based evaluation** (KS/RSS-style errors over selected metrics).

```python
from larvaworld.lib.sim.genetic_algorithm import GAevaluation

evaluator = GAevaluation(
    refID="exploration.30controls",
    cycle_curve_metrics=[],
    eval_metrics={
        "angular kinematics": ["b", "fov"],
        "spatial displacement": ["v", "a"],
    },
)
```

**Fitness Calculation**:

Larvaworld assigns fitness from the aggregated evaluation errors (higher is better). Internally, each genome produces an evaluation dictionary like `{"KS": {...}, "RSS": {...}}` and fitness is computed as a weighted sum of the **negative mean** errors per group (weights currently: `KS=10`, `RSS=1`).

:::{note}
`GAevaluation` also supports robot-based fitness functions (`fitness_func_name=...`), but the current `Ga` implementation expects reference-based evaluation (`fit_func_arg == "s"`).
:::

---

### 3. Define Parameter Space

Specify which modules to optimize (all parameters within each module will be optimized):

```python
# Module names to optimize
ks = [
    "crawler",      # Crawler module (includes freq, stride_dst_mean, etc.)
    "turner",       # Turner module (includes ang_v, freq, etc.)
    "olfactor",     # Olfactor module (includes gain, decay_coef, etc.)
]
```

**Finding available modules**:

```python
from larvaworld.lib import reg
from larvaworld.lib.model.modules.module_modes import moduleDB

# List all available modules
print(moduleDB.AllModules)  # ['crawler', 'turner', 'olfactor', 'intermitter', ...]

# Inspect model configuration
model_conf = reg.conf.Model.getID("explorer")
print(model_conf.brain)  # See nested module structure
```

---

### 4. Run Optimization

#### Option A: Optimize Single Model via `optimize_mID`

```python
from larvaworld.lib.sim.genetic_algorithm import optimize_mID

results = optimize_mID(
    mID0="explorer",           # Base model
    mID1="explorer_opt",       # ID for optimized model (stored in registry)
    ks=["crawler", "turner"],  # Module names to optimize
    evaluator=evaluator,
    Ngenerations=1,            # Increase for real runs
    Nagents=10,                # Population size per generation
    duration=0.05,             # Minutes per agent (increase for real runs)
    screen_kws={"show_display": False},
    store_data=False,
)

best_conf = results["explorer_opt"]   # Optimized model configuration (AttrDict)
```

#### Option B: Custom GA configuration via registry

For more complex GA settings you can use `GAconf` via the registry (`reg.conf.Ga`) and run GA via the CLI (`larvaworld Ga ...`) or Python, instead of configuring `GAlauncher` manually. See {doc}`../concepts/experiment_configuration_pipeline` for how `Ga` configuration entries work.

---

## GA Parameters

**Note**: The defaults listed below match the `optimize_mID(...)` convenience wrapper.

### Population Parameters

| Parameter      | Default | Description                    |
| -------------- | ------- | ------------------------------ |
| `Nagents`      | 10      | Population size per generation |
| `Ngenerations` | 3       | Number of generations          |

**Note**: `Nagents` is the **GA population size**. This is different from `N` used in `ExpRun`/`EvalRun` (number of larvae per simulation run).

:::{note}
Keep `Nagents` high enough so the parent pool is at least 2 genomes, otherwise the GA will fail early with a `ValueError`.
The selection size is computed as `round(Nagents * selection_ratio)` (default `selection_ratio=0.3`).
:::

### Evolution Parameters

| Parameter         | Default   | Description                                                                |
| ----------------- | --------- | -------------------------------------------------------------------------- |
| `Nelits`          | 2         | Number of elite genomes carried over unchanged (when using `optimize_mID`) |
| `selection_ratio` | 0.3       | Fraction used as the parent pool (`round(Nagents * selection_ratio)`)      |
| `Pmutation`       | 0.3       | Per-parameter mutation probability (checked for each gene)                 |
| `Cmutation`       | 0.1       | Mutation scale as a fraction of allowed parameter range                    |
| `init_mode`       | `"model"` | Initial population mode (when using `optimize_mID`)                        |

### Convergence Criteria

| Criterion           | Description               |
| ------------------- | ------------------------- |
| **Max generations** | Stop after `Ngenerations` |

---

## Results

### Accessing Best Genome

```python
results = optimize_mID(...)
best_conf = results["explorer_opt"]   # Optimized model configuration
```

### Comparing Original vs Optimized

```python
from larvaworld.lib.sim import EvalRun

eval_run = EvalRun(
    refID='exploration.30controls',
    modelIDs=['explorer', 'explorer_opt'],  # 'explorer_opt' is created by GA (bestConfID)
    N=3,
    screen_kws={},
)
eval_run.simulate()
print(eval_run.error_dicts["pooled"]["end"])
```

---

## Parameter Encoding

### Genome Structure

A **genome** is a dictionary mapping full configuration paths to values:

```python
genome = {
    "brain.crawler.freq": 1.23,
    "brain.crawler.stride_dst_mean": 0.25,
    "brain.turner.freq": 0.58,      # Turner oscillation frequency
    "brain.turner.amp": 0.45,       # Turner amplitude
    "brain.olfactor.decay_coef": 0.12,
}
```

### Parameter Ranges

Parameter ranges are constrained by the `param` definitions of each module (bounds/step) and are built automatically from the selected model + module modes:

```python
from larvaworld.lib.sim.genetic_algorithm import GAselector

selector = GAselector(
    base_model="explorer",
    space_mkeys=["crawler", "turner", "olfactor"],
    Nagents=10,
    Ngenerations=1,
)

for k, obj in selector.space_objs.items():
    if hasattr(obj, "bounds") and obj.bounds is not None:
        print(k, "bounds:", obj.bounds)
```

**Note**: When you specify module names in `ks` (e.g., `["crawler", "turner"]`), all parameters within those modules are automatically included in the optimization space. The parameter ranges are determined from the module parameter definitions.

---

## Evolution Operators

### 1. Selection

**Strategy**: **Elitism** (keep top N% as parents)

```python
# Top 30% of population becomes the parent pool (and `Nelits` are copied unchanged)
selection_ratio = 0.3
```

### 2. Crossover

**Strategy**: **Uniform crossover** (randomly mix parent genes)

```python
# Example:
# Parent 1: {brain.crawler.freq: 1.2, brain.turner.freq: 0.58, brain.turner.amp: 0.5}
# Parent 2: {brain.crawler.freq: 1.5, brain.turner.freq: 0.60, brain.turner.amp: 0.3}
# Offspring: {brain.crawler.freq: 1.5, brain.turner.freq: 0.60, brain.turner.amp: 0.5}  # Random mix
```

### 3. Mutation

**Strategy**: **Gaussian mutation** (add random noise)

```python
# Per-parameter mutation probability (checked for each gene)
Pmutation = 0.3

# Mutation scale as a fraction of allowed parameter range
Cmutation = 0.1

# Example:
# Original: brain.crawler.freq = 1.2
# Mutated: brain.crawler.freq = 1.2 + N(0, Cmutation * range) = 1.35
```

## Use Case Examples

### 1. Optimize a Model Against a Reference Dataset

```python
from larvaworld.lib.sim.genetic_algorithm import GAevaluation, optimize_mID

# Reference: Use an available reference dataset (replace with your own if needed)
evaluator = GAevaluation(
    refID="exploration.30controls",
    cycle_curve_metrics=[],
    eval_metrics={
        "angular kinematics": ["b", "fov"],
        "spatial displacement": ["v", "a"],
    },
)

# Optimize navigator model (short demo settings)
results = optimize_mID(
    mID0="navigator",
    mID1="navigator_opt",
    ks=["crawler", "turner", "olfactor"],
    evaluator=evaluator,
    Ngenerations=1,
    Nagents=10,
    duration=0.05,
    screen_kws={"show_display": False},
    store_data=False,
)

best_conf = results["navigator_opt"]
print("Optimized navigator config ready")
```

### 2. Match Rovers vs. Sitters

```python
# Optimize foraging phenotypes (rover vs sitter)
evaluator = GAevaluation(
    refID="exploration.30controls",
    cycle_curve_metrics=[],
    eval_metrics={
        "angular kinematics": ["b", "fov"],
        "spatial displacement": ["v", "a"],
    },
)
for phenotype in ["rover", "sitter"]:
    results = optimize_mID(
        mID0=phenotype,
        mID1=f"{phenotype}_opt",
        ks=["crawler", "feeder"],
        evaluator=evaluator,
        Ngenerations=1,
        Nagents=10,
        duration=0.05,
        screen_kws={"show_display": False},
        store_data=False,
    )
    print(f"{phenotype} optimized config ready")  # stored as {phenotype}_opt
```

---

## Related Documentation

- {doc}`../concepts/simulation_modes` - Simulation mode comparison
- {doc}`model_evaluation` - Model evaluation (Eval mode)
- {doc}`../data_pipeline/reference_datasets` - Reference datasets
- {doc}`../tutorials/4_optimization_and_evaluation/genetic_algorithm_optimization` - Step-by-step tutorial
