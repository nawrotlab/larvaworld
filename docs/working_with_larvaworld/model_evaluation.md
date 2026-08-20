# Model Evaluation

The **Eval** simulation mode enables you to **compare multiple larva models** against experimental reference datasets using statistical tests. This is essential for model validation and selection.

---

## Purpose

Use `Eval` mode to:

- ✅ **Validate models** against real experimental data
- ✅ **Select best model** from multiple candidates
- ✅ **Behavioral fingerprinting** across 40+ metrics
- ✅ **Hypothesis testing** with statistical rigor

For mode comparison, see {doc}`../concepts/simulation_modes`.

---

## Quick Start

### CLI

```bash
larvaworld Eval -refID exploration.30controls --modelIDs explorer navigator
```

### Python

```python
from larvaworld.lib.sim import EvalRun

eval_run = EvalRun(
    refID='exploration.30controls',
    modelIDs=['explorer', 'navigator', 'forager'],
    N=3,                  # agents per model (use larger N for real runs)
    screen_kws={},        # headless
)
eval_run.simulate()
eval_run.plot_results(show=False)
```

---

## Workflow

### 1. Select Reference Dataset

**Reference datasets** are experimental recordings imported into Larvaworld.

**Available datasets**:

```python
from larvaworld.lib import reg

# List all reference datasets
ref_ids = reg.conf.Ref.confIDs
print(f"Available: {len(ref_ids)} reference datasets")

# Inspect a dataset
ref_conf = reg.conf.Ref.getID("exploration.30controls")
print(ref_conf)
```

**Loading a reference**:

```python
from larvaworld.lib import reg

ref_dataset = reg.loadRef(id="exploration.30controls", load=True)
print(f"Reference: {ref_dataset.config.refID}")
print(f"Agents: {len(ref_dataset.agent_ids)}")
print(f"Duration: {ref_dataset.config.duration} min")
```

For details on importing datasets, see {doc}`../data_pipeline/lab_formats_import`.

---

### 2. Select Models to Compare

**Predefined models**:

| Model ID         | Description                     |
| ---------------- | ------------------------------- |
| `'explorer'`     | Baseline exploration            |
| `'navigator'`    | Odor-guided navigation          |
| `'forager'`      | Feeding/foraging                |
| `'rover'`        | High-activity forager phenotype |
| `'sitter'`       | Low-activity forager phenotype  |
| `'max_forager'`  | Maximal feeding rate            |
| `'max_feeder'`   | Feeder-focused behavior         |
| `'RLnavigator'`  | RL-enhanced navigation          |
| `'OSNnavigator'` | OSN-based navigation            |

**Inspect models**:

```python
from larvaworld.lib import reg

# List all models
model_ids = reg.conf.Model.confIDs
print(f"Available: {len(model_ids)} models")

# Inspect model configuration
model_conf = reg.conf.Model.getID("explorer")
print(model_conf)
```

---

### 3. Run Evaluation

```python
from larvaworld.lib.sim import EvalRun

eval_run = EvalRun(
    refID='exploration.30controls',          # Reference dataset
    modelIDs=['explorer', 'navigator'],      # Models to compare
    N=3,                                     # Agents per model (increase for real)
    screen_kws={},                           # headless
)

# Run simulations
eval_run.simulate()
```

**What happens**:

1. Load reference dataset
2. For each model:
   - Run one simulation with `N` larvae (per model)
   - Compute 40+ behavioral metrics
3. Compare model distributions to reference using **Kolmogorov-Smirnov (KS) tests**

---

### 4. Access Results

```python
# Statistical comparison (endpoint metrics)
print(eval_run.error_dicts["pooled"]["end"])

# Statistical comparison (distribution metrics)
print(eval_run.error_dicts["pooled"]["step"])

# Raw datasets per model
for model_id, datasets in eval_run.model_datasets.items():
    print(f"{model_id}: {len(datasets)} runs")
```

---

### 5. Visualize Results

#### Statistical Comparison Plots

```python
# Aggregate comparison plots
eval_run.plot_results()  # KS D-statistic heatmaps
```

**Generated plots**:

- **KS D-statistic heatmap**: Models × Metrics
- **Box plots**: Metric distributions per model
- **P-value summary**: Statistical significance

#### Model-Specific Visualizations

```python
# Individual model plots
eval_run.plot_models()  # Trajectories, distributions
```

**Generated plots per model**:

- **Trajectories**: Spatial paths
- **Angular distributions**: Orientation, turns
- **Spatial distributions**: Velocity, dispersal
- **Bout distributions**: Stride/turn/pause durations

---

## Evaluation Metrics

Larvaworld computes **40+ behavioral metrics** across three categories:

### Endpoint Metrics (Summary Statistics)

| Metric      | Description              | Unit  |
| ----------- | ------------------------ | ----- |
| **cum_dur** | Total duration           | s     |
| **cum_sd**  | Total distance           | m     |
| **v_mu**    | Mean linear velocity     | mm/s  |
| **a_mu**    | Mean linear acceleration | mm/s² |
| **av_mu**   | Mean angular velocity    | rad/s |
| **fov_mu**  | Mean forward velocity    | mm/s  |
| **pau_N**   | Number of pauses         | count |
| **str_N**   | Number of strides        | count |
| **run_N**   | Number of runs           | count |
| **str_f**   | Stride frequency         | Hz    |
| **run_t**   | Average run duration     | s     |
| **pau_t**   | Average pause duration   | s     |

### Distribution Metrics (Time-Series)

| Metric         | Description                                |
| -------------- | ------------------------------------------ |
| **angular**    | Orientation, angular velocity/acceleration |
| **spatial**    | Linear velocity/acceleration distributions |
| **dispersal**  | Spatial spread over time                   |
| **tortuosity** | Path straightness (sliding windows)        |

### Bout Metrics (Event-Based)

| Metric              | Description                      |
| ------------------- | -------------------------------- |
| **stride_duration** | Distribution of stride durations |
| **turn_amplitude**  | Distribution of turn amplitudes  |
| **pause_duration**  | Distribution of pause durations  |
| **run_distance**    | Distribution of run distances    |

---

## Statistical Testing

### Kolmogorov-Smirnov (KS) Test

**Purpose**: Compare distributions between model and reference.

**Null Hypothesis**: Model and reference are drawn from the same distribution.

**KS D-Statistic**: Maximum difference between cumulative distributions.

- Formula: `D = max_x |F_model(x) - F_ref(x)|`
- Where `F_model(x)` is the cumulative distribution of the model and `F_ref(x)` that of the reference.

**Interpretation**:

- `D = 0`: Perfect match
- `D < 0.2`: Good match
- `D > 0.5`: Poor match

**Computing KS tests manually**:

```python
from larvaworld.lib.process.evaluation import eval_fast

# Compare two datasets
ks_results = eval_fast(
    datasets=[model_dataset],
    refDataset=ref_dataset,
    metric_definition="angular"  # or "spatial", "all"
)

print(ks_results['end'])  # Endpoint metrics
print(ks_results['step'])  # Distribution metrics
```

---

## Example: Rover vs. Sitter Comparison

```python
from larvaworld.lib.sim import EvalRun

# Compare rover vs sitter models (short demo)
eval_run = EvalRun(
    refID='exploration.30controls',
    modelIDs=['rover', 'sitter'],
    N=3,
    screen_kws={},
)

eval_run.simulate()

# Plot comparison
eval_run.plot_results()

# Access D-statistics
ks_end = eval_run.error_dicts["pooled"]['end']
print("Endpoint KS D-statistics:")
for model, metrics in ks_end.items():
    print(f"\n{model}:")
    for metric, d_stat in metrics.items():
        if d_stat < 0.2:
            print(f"  {metric}: {d_stat:.3f} ✅ (good match)")
        else:
            print(f"  {metric}: {d_stat:.3f} ❌ (poor match)")
```

---

## Custom Metric Selection

By default, Larvaworld auto-selects metrics based on experiment type. You can customize:

```python
from larvaworld.lib.sim import EvalRun

eval_run = EvalRun(
    refID='exploration.30controls',
    modelIDs=['explorer'],
    duration=1.0,  # short demo
    N=5,
    screen_kws={},

    # Custom metric selection
    metric_definition="angular"  # Only angular metrics
    # Options: "angular", "spatial", "spatial+angular", "all"
)

eval_run.simulate()
```

---

## Parallelization

Currently `EvalRun.simulate()` runs single-process. For parallel runs, launch multiple `EvalRun` instances via your own batching (e.g., shell/xargs or a task runner) and combine results manually.

---

## Saving Results

```python
# Save evaluation results
eval_run.store()

# Location: DATA/SimGroup/eval_runs/{refID}/
print(f"Saved to: {eval_run.dir}")

# Load later
from larvaworld.lib.sim import EvalRun
eval_run_loaded = EvalRun.load(path=eval_run.dir)
```

---

## Advanced: Custom Reference Data

You can use your own experimental data:

### Step 1: Import Dataset

```python
from larvaworld.lib import reg

lab = reg.gen.LabFormat(labID="Schleyer")
lab.import_dataset(
    parent_dir="exploration",
    merged=True,
    max_Nagents=30,
    min_duration_in_sec=60,
    id="my_experiment",
    refID="my_experiment",
    save_dataset=True,
)
```

For details, see {doc}`../data_pipeline/lab_formats_import`.

### Step 2: Process Dataset

```python
dataset = reg.loadRef(id="my_experiment", load=True)

dataset.preprocess(filter_f=3.0)
dataset.process(proc_keys=["angular", "spatial"])
dataset.annotate(
    anot_keys=["bout_detection", "bout_distribution", "interference"]
)
```

### Step 3: Evaluate Against Custom Reference

```python
eval_run = EvalRun(
    refID='my_experiment',
    modelIDs=['explorer', 'navigator'],
    duration=5.0
)
eval_run.simulate()
```

---

## Related Documentation

- {doc}`../concepts/simulation_modes` - Simulation mode comparison
- {doc}`../data_pipeline/lab_formats_import` - Importing experimental data
- {doc}`../data_pipeline/data_processing` - Data processing pipeline
- {doc}`../data_pipeline/reference_datasets` - Reference dataset management
- {doc}`../tutorials/4_optimization_and_evaluation/model_evaluation` - Step-by-step tutorial
