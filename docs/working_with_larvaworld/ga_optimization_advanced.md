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

    Rank --> Converge{Converged?<br/>Max gen reached<br/>or fitness plateau}

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
    # optionally configure fitness_func_name / exclude_func_name / fit_kws
)

# 2. Run genetic algorithm to optimize locomotory model
results = optimize_mID(
    mID0="explorer",                      # Base model to optimize
    ks=["crawler.f", "turner.ang_v"],     # Parameter keys to vary
    evaluator=evaluator,
    Ngenerations=50,
)

# 3. Access optimized configuration
best_conf = results["explorer"]          # Optimized model config (AttrDict)
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

The `GAevaluation` class configures how genomes are evaluated. You can either:

- use **built-in fitness functions** (e.g. distance-to-source, cumulative distance), ή
- fall back to dataset-based evaluation logic (see `Evaluation` / `DataEvaluation`).

```python
from larvaworld.lib.sim.genetic_algorithm import GAevaluation

evaluator = GAevaluation(
    refID="exploration.30controls",
    fitness_func_name="dst2source",  # or "cum_dst", or use target-based evaluation
    fit_kws={},
)
```

**Fitness Calculation**:

$$
\text{Fitness} = \frac{1}{1 + \overline{D_{KS}}}
$$

Where $\overline{D_{KS}}$ is the mean KS D-statistic across all metrics.

---

### 3. Define Parameter Space

Specify which parameters to optimize:

```python
# Parameter keys to optimize
ks = [
    "crawler.f",            # Crawling frequency
    "turner.ang_v",         # Turning angular velocity
    "olfactor.gain",        # Olfactory gain
    "crawler.stridechain_dist"  # Stride chain parameters
]
```

**Finding parameter keys**:

```python
from larvaworld.lib import reg

# Inspect model configuration
model_conf = reg.conf.Model.getID("explorer")
print(model_conf)  # See nested parameter structure
```

---

### 4. Run Optimization

#### Option A: Optimize Single Model via `optimize_mID`

```python
from larvaworld.lib.sim.genetic_algorithm import optimize_mID

results = optimize_mID(
    mID0="explorer",           # Base model
    ks=["crawler.f", "turner.ang_v"],
    evaluator=evaluator,
    Ngenerations=50,           # Number of generations
    Nagents=20,                # Population size per generation
)

best_conf = results["explorer"]   # Optimized model configuration (AttrDict)
```

#### Option B: Custom GA configuration via registry

Για πιο σύνθετες GA ρυθμίσεις μπορείς να χρησιμοποιήσεις το `GAconf` μέσω
του registry (`reg.conf.Ga`) και να εκτελέσεις GA runs μέσω CLI (`larvaworld Ga ...`)
ή Python, αντί για χειροκίνητο `GAlauncher` με custom selector. Δες και το
{doc}`../concepts/experiment_configuration_pipeline` για το πώς δουλεύουν τα
`Ga` configuration entries.

---

## GA Parameters

### Population Parameters

| Parameter      | Default | Description                    |
| -------------- | ------- | ------------------------------ |
| `Nagents`      | 20      | Population size per generation |
| `Ngenerations` | 50      | Number of generations          |

### Evolution Parameters

| Parameter         | Default | Description                                      |
| ----------------- | ------- | ------------------------------------------------ |
| `selection_ratio` | 0.3     | Fraction of population kept as parents (top 30%) |
| `crossover_rate`  | 0.7     | Probability of crossover (70%)                   |
| `mutation_rate`   | 0.1     | Probability of mutation per gene (10%)           |
| `mutation_scale`  | 0.2     | Mutation magnitude (±20% of parameter range)     |

### Convergence Criteria

| Criterion           | Description                                  |
| ------------------- | -------------------------------------------- |
| **Max generations** | Stop after `Ngenerations`                    |
| **Fitness plateau** | Stop if fitness unchanged for 10 generations |
| **Target fitness**  | Stop if fitness > threshold (e.g., 0.95)     |

---

## Results

### Accessing Best Genome

```python
results = optimize_mID(...)
best_conf = results["explorer"]   # Optimized model configuration
```

### Comparing Original vs Optimized

```python
from larvaworld.lib.sim import EvalRun

# Evaluate both models
eval_run = EvalRun(
    refID='exploration.30controls',
    modelIDs=['explorer', 'explorer_optimized'],  # Add optimized model to registry
    duration=5.0
)
eval_run.simulate()
eval_run.plot_results()
```

---

## Parameter Encoding

### Genome Structure

A **genome** is a dictionary mapping parameter keys to values:

```python
genome = {
    "crawler.f": 1.23,
    "turner.ang_v": 0.45,
    "olfactor.gain": 0.78
}
```

### Parameter Ranges

Parameters are constrained to biologically realistic ranges:

| Parameter       | Min | Max | Unit          |
| --------------- | --- | --- | ------------- |
| `crawler.f`     | 0.5 | 3.0 | Hz            |
| `turner.ang_v`  | 0.1 | 2.0 | rad/s         |
| `olfactor.gain` | 0.0 | 2.0 | dimensionless |

**Custom ranges**:

```python
# Define custom parameter ranges
param_ranges = {
    "crawler.f": (0.8, 2.5),
    "turner.ang_v": (0.2, 1.5)
}

# Pass to optimizer
results = optimize_mID(
    mID0="explorer",
    ks=list(param_ranges.keys()),
    param_ranges=param_ranges,
    evaluator=evaluator
)
```

---

## Evolution Operators

### 1. Selection

**Strategy**: **Elitism** (keep top N% as parents)

```python
# Top 30% of population survives
selection_ratio = 0.3
```

### 2. Crossover

**Strategy**: **Uniform crossover** (randomly mix parent genes)

```python
# 70% probability of crossover
crossover_rate = 0.7

# Example:
# Parent 1: {crawler.f: 1.2, turner.ang_v: 0.5}
# Parent 2: {crawler.f: 1.5, turner.ang_v: 0.3}
# Offspring: {crawler.f: 1.5, turner.ang_v: 0.5}  # Random mix
```

### 3. Mutation

**Strategy**: **Gaussian mutation** (add random noise)

```python
# 10% probability per gene
mutation_rate = 0.1

# ±20% perturbation
mutation_scale = 0.2

# Example:
# Original: crawler.f = 1.2
# Mutated: crawler.f = 1.2 + N(0, 0.2 * 1.2) = 1.35
```

## Use Case Examples

### 1. Optimize Chemotaxis Model

```python
from larvaworld.lib.sim.genetic_algorithm import GAevaluation, optimize_mID

# Reference: Real larvae navigating odor gradient
evaluator = GAevaluation(
    refID="chemotaxis_real_data",
    metric_definition="spatial+angular"
)

# Optimize navigator model
results = optimize_mID(
    mID0="navigator",
    ks=["crawler.f", "turner.ang_v", "olfactor.gain"],
    evaluator=evaluator,
    Ngenerations=100
)

best_conf = results["navigator"]
print(f"Optimized chemotaxis fitness: {best_conf['fitness']:.3f}")
```

### 2. Match Rovers vs. Sitters

```python
# Optimize foraging phenotypes
for phenotype in ["RE", "SI"]:
    evaluator = GAevaluation(
        refID=f"RvsS_{phenotype}_data",
        metric_definition="all"
    )

    results = optimize_mID(
        mID0=phenotype,
        ks=["crawler.f", "feeder.intake_rate"],
        evaluator=evaluator,
        Ngenerations=50
    )

    print(f"{phenotype} optimized: {results[phenotype]['fitness']:.3f}")
```

---

## Related Documentation

- {doc}`../concepts/simulation_modes` - Simulation mode comparison
- {doc}`model_evaluation` - Model evaluation (Eval mode)
- {doc}`../data_pipeline/reference_datasets` - Reference datasets
- {doc}`../tutorials/genetic_algorithm_optimization` - Step-by-step tutorial
