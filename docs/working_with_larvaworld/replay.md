# Dataset Replay

The **Replay** simulation mode allows you to **visualize existing datasets** (simulated or experimental) without re-running simulations. This is useful for creating videos, inspecting trajectories, and quality control.

---

## Purpose

Use `Replay` mode to:

- ✅ **Visualize experimental data** from imported datasets
- ✅ **Create videos** from existing simulations
- ✅ **Quality control** of tracked data
- ✅ **Fast exploration** of saved results (no simulation overhead)

For mode comparison, see {doc}`../concepts/simulation_modes`.

---

## Quick Start

### CLI

```bash
larvaworld Replay -refID exploration.30controls -video_name replay.mp4
```

### Python

```python
from larvaworld.lib.sim import ReplayRun

replay = ReplayRun(
    refID='exploration.30controls',
    screen_kws={
        'vis_mode': 'video',
        'video_name': 'exploration_replay.mp4'
    }
)
replay.run()
```

---

## Replaying Reference Datasets

### List Available Datasets

```python
from larvaworld.lib import reg

# List all reference datasets
ref_ids = reg.conf.Ref.confIDs
print(f"Available: {len(ref_ids)} reference datasets")

for ref_id in ref_ids[:10]:
    print(f"  - {ref_id}")
```

### Load and Replay

```python
from larvaworld.lib import reg
from larvaworld.lib.sim import ReplayRun

# Load reference dataset
ref_dataset = reg.loadRef(id="exploration.30controls", load=True)
print(f"Dataset: {ref_dataset.config.refID}")
print(f"Agents: {len(ref_dataset.agent_ids)}")
print(f"Duration: {ref_dataset.config.env_params.duration} min")

# Replay
replay = ReplayRun(
    refID='exploration.30controls',
    screen_kws={'vis_mode': 'screen'}  # Real-time display
)
replay.run()
```

---

## Replaying Saved Simulations

### Replay from Directory

If you have a saved simulation:

```python
from larvaworld.lib.process.dataset import LarvaDataset
from larvaworld.lib.sim import ReplayRun

# Load dataset from disk
dataset = LarvaDataset(dir="/path/to/simulation", load=True)

# Replay (requires dataset ID to be in registry)
replay = ReplayRun(
    refID=dataset.config.refID,
    screen_kws={'vis_mode': 'screen'}
)
replay.run()
```

---

## Visualization Options

### Real-Time Display

```python
replay = ReplayRun(
    refID='exploration.30controls',
    screen_kws={
        'vis_mode': 'screen',
        'showfps': True  # Show FPS counter
    }
)
replay.run()
```

**Keyboard controls**: See {doc}`../visualization/keyboard_controls`

---

### Video Export

```python
replay = ReplayRun(
    refID='exploration.30controls',
    screen_kws={
        'vis_mode': 'video',
        'video_name': 'exploration.mp4',
        'fps': 10,                # Frames per second
        'video_speed': 2          # 2x speed
    }
)
replay.run()
```

**Supported formats**: MP4, AVI

---

### Image Snapshots

```python
replay = ReplayRun(
    refID='exploration.30controls',
    screen_kws={
        'vis_mode': 'image',
        'image_mode': 'snapshots',
        'snapshot_times': [0, 60, 120, 180]  # Times (seconds) to save
    }
)
replay.run()
```

---

### Headless (No Visualization)

For data processing only:

```python
from larvaworld.lib import reg

# Load dataset
dataset = reg.loadRef(id="exploration.30controls", load=True)

# Process
dataset.preprocess(filter_f=3.0)
dataset.process(proc_keys=["angular", "spatial"])

# See :doc:`../visualization/plotting_api` for plotting functions.
```

---

## Customizing Visualization

### Drawing Options

```python
replay = ReplayRun(
    refID='exploration.30controls',
    screen_kws={
        'vis_mode': 'screen',
        'draw_Nsegs': 12,         # Number of body segments
        'draw_contour': True,     # Draw body contour
        'draw_midline': True,     # Draw midline
        'draw_trails': True,      # Draw trajectory trails
        'trail_dt': 5.0,          # Trail duration (seconds)
        'color_behavior': True    # Color by behavior state
    }
)
replay.run()
```

### Arena Options

```python
replay = ReplayRun(
    refID='exploration.30controls',
    screen_kws={
        'vis_mode': 'screen',
        'draw_odorscape': True,   # Show odor heatmap
        'draw_foodgrid': True,    # Show food patches
        'draw_borders': True      # Show arena borders
    }
)
replay.run()
```

---

## Quality Control

Replay is ideal for **quality control** of imported experimental datasets:

### Check for Issues

```python
from larvaworld.lib import reg

# Load dataset
dataset = reg.loadRef(id="my_imported_data", load=True)

# Replay to visually inspect
replay = ReplayRun(
    refID='my_imported_data',
    screen_kws={'vis_mode': 'screen'}
)
replay.run()
```

**Look for**:

- ❌ Missing data (gaps in trajectories)
- ❌ Tracking errors (jumps, swaps)
- ❌ Collisions or arena boundary issues
- ✅ Smooth, continuous trajectories

---

## Comparing Multiple Datasets

Replay multiple datasets side-by-side:

```python
from larvaworld.lib.sim import ReplayRun

datasets = [
    'exploration.30controls',
    'exploration.mutant_A',
    'exploration.mutant_B'
]

for ref_id in datasets:
    print(f"\nReplaying: {ref_id}")
    replay = ReplayRun(
        refID=ref_id,
        screen_kws={'vis_mode': 'screen'}
    )
    replay.run()
```

---

## Related Documentation

- {doc}`../concepts/simulation_modes` - Simulation mode comparison
- {doc}`../data_pipeline/lab_formats_import` - Importing experimental data
- {doc}`../data_pipeline/reference_datasets` - Reference dataset management
- {doc}`../visualization/keyboard_controls` - Interactive controls
- {doc}`../visualization/visualization_snapshots` - Visualization examples
- {doc}`../tutorials/replay` - Step-by-step tutorial
