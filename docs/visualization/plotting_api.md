# Plotting

Larvaworld provides a comprehensive plotting API in `/lib/plot/` for analyzing behavioral data.

---

## Plot Categories

### Trajectory Plots (`traj.py`)

**Purpose**: Spatial visualization

```python
from larvaworld.lib.plot import traj

# Single-dataset trajectories
traj.traj_1group(dataset, unit="mm")

# Grouped trajectories (via graph ID "trajectories")
from larvaworld.lib.process import LarvaDatasetCollection

collection = LarvaDatasetCollection(datasets=[dataset])
collection.plot(ids=["trajectories"])
```

**Key Functions / IDs**:

- `traj.traj_1group(dataset)`: 2D paths for a single dataset
- Graph ID `"trajectories"`: grouped trajectories via `LarvaDatasetCollection.plot`

---

### Time-Series Plots (`time.py`)

**Purpose**: Temporal dynamics

```python
from larvaworld.lib.plot import time as timeplot

# Path length / dispersal over time
timeplot.plot_pathlength(datasets=[dataset])
timeplot.plot_dispersal(datasets=[dataset], range=(0, 60))
```

**Key Functions / IDs**:

- `timeplot.plot_pathlength(datasets=[...])`: Cumulative distance
- `timeplot.plot_dispersal(datasets=[...])`: Dispersal over a time window (graph ID `"dispersal"`)

---

### Distribution Plots (`hist.py`)

**Purpose**: Statistical distributions

```python
from larvaworld.lib.plot import hist

# Kinematic distributions
hist.plot_distros(
    datasets=[dataset],
    ks=["v", "a", "fov"],   # speed, acceleration, forward velocity
    mode="hist",
)
```

**Key Functions / IDs**:

- `hist.plot_distros(datasets=[...])`: Velocity / angular distributions (graph ID `"distros"`)

---

### Bearing Plots (`bearing.py`)

**Purpose**: Navigation analysis

```python
from larvaworld.lib.plot import bearing

bearing.plot_chunk_Dorient2source(
    source_ID="CS_plus",
    datasets=[dataset],
    chunk="run",
)
```

**Key Functions / IDs**:

- `bearing.plot_chunk_Dorient2source(...)`: Bearing to source during behavioral chunks (graph ID `"bearing to source/epoch"`)

---

### Stride Cycle Plots (`stridecycle.py`)

**Purpose**: Crawling kinematics

```python
from larvaworld.lib.plot import stridecycle

stridecycle.plot_stride_Dbend(datasets=[dataset])
stridecycle.plot_vel_during_strides(datasets=[dataset])
```

---

### DEB Plots (`deb.py`)

**Purpose**: Energetics visualization

```python
from larvaworld.lib.plot import deb

deb.plot_debs(datasets=[dataset])
```

---

## Complete Example

```python
from larvaworld.lib.sim import ExpRun
from larvaworld.lib.plot import traj, hist, time as timeplot

# Run experiment
run = ExpRun(experiment="dish", N=10, duration=5.0)
run.simulate()

# Get dataset
dataset = run.datasets[0]

# Preprocess
dataset.preprocess(filter_f=3.0)
dataset.process(proc_keys=["angular", "spatial"])
dataset.annotate(anot_keys=["bout_detection"])

# Plot trajectories
traj.traj_1group(dataset)

# Plot velocity / angular distributions
hist.plot_distros(datasets=[dataset], ks=["v", "fov"])

# Plot dispersal over time
timeplot.plot_dispersal(datasets=[dataset], range=(0, 60))
```

---

## Customization

All plotting functions support matplotlib kwargs:

```python
traj.traj_1group(
    dataset,
    color='blue',
    single_color=True,
)
```

---

## Saving Plots

```python
import matplotlib.pyplot as plt

traj.traj_1group(dataset)
plt.savefig('trajectories.png', dpi=300)
plt.close()
```

---

## Related Documentation

- {doc}`../data_pipeline/data_processing` - Data processing
- {doc}`keyboard_controls` - Real-time visualization
- {doc}`visualization_snapshots` - Visualization examples
