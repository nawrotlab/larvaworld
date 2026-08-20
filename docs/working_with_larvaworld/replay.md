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
larvaworld Replay -refID exploration.30controls --vis_mode video --save_video --video_file exploration_replay
```

### Python

```python
from larvaworld.lib import reg
from larvaworld.lib.sim import ReplayRun

params = reg.gen.Replay(refID="exploration.30controls")

replay = ReplayRun(
    parameters=params.nestedConf,
    screen_kws={
        "vis_mode": "video",
        "save_video": True,
        "video_file": "exploration_replay",
        "fps": 10
    },
)
replay.run()  # creates exploration_replay.mp4
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
print(f"Duration: {ref_dataset.config.duration} min")

# Replay
replay = ReplayRun(
    parameters=reg.gen.Replay(refID="exploration.30controls").nestedConf,
    screen_kws={"vis_mode": "video", "show_display": True},  # Real-time display
)
replay.run()
```

---

## Replaying Saved Simulations

### Replay from Directory

If you have a saved simulation:

```python
from larvaworld.lib import reg
from larvaworld.lib.process import LarvaDataset
from larvaworld.lib.sim import ReplayRun

# Load dataset from disk
dataset = LarvaDataset(dir="/path/to/simulation", load=True)

# Replay (pass the loaded dataset directly)
replay = ReplayRun(
    parameters=reg.gen.Replay(refID=dataset.config.refID).nestedConf,
    dataset=dataset,
    screen_kws={"vis_mode": "video", "show_display": True},
)
replay.run()
```

---

## Visualization Options

### Real-Time Display

```python
replay = ReplayRun(
    parameters=reg.gen.Replay(refID="exploration.30controls").nestedConf,
    screen_kws={
        "vis_mode": "video",
        "show_display": True,
    }
)
replay.run()
```

**Keyboard controls**: See {doc}`../visualization/keyboard_controls`

---

### Video Export

```python
replay = ReplayRun(
    parameters=reg.gen.Replay(refID="exploration.30controls").nestedConf,
    screen_kws={
        "vis_mode": "video",
        "save_video": True,
        "video_file": "exploration",
        "fps": 10,  # Frames per second
    }
)
replay.run()
```

**Supported format**: MP4

---

### Image Snapshots

```python
replay = ReplayRun(
    parameters=reg.gen.Replay(refID="exploration.30controls").nestedConf,
    screen_kws={
        "vis_mode": "image",
        "image_mode": "snapshots",
        "snapshot_interval_in_sec": 60,  # seconds between snapshots
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
    parameters=reg.gen.Replay(refID="exploration.30controls", draw_Nsegs=12).nestedConf,
    screen_kws={
        "vis_mode": "video",
        "show_display": True,
        "draw_contour": True,
        "draw_midline": True,
        "visible_trails": True,
    },
)
replay.run()
```

### Arena Options

```python
replay = ReplayRun(
    parameters=reg.gen.Replay(refID="exploration.30controls").nestedConf,
    screen_kws={
        "vis_mode": "video",
        "show_display": True,
        "black_background": True,
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
dataset = reg.loadRef(id="exploration.30controls", load=True)  # replace with your refID

# Replay to visually inspect
replay = ReplayRun(
    parameters=reg.gen.Replay(refID=dataset.config.refID).nestedConf,
    dataset=dataset,
    screen_kws={"vis_mode": "video", "show_display": True},
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
from larvaworld.lib import reg
from larvaworld.lib.sim import ReplayRun

datasets = [
    "exploration.30controls",
    "exploration.dish01",
    "exploration.dish02",
]

for ref_id in datasets:
    print(f"\nReplaying: {ref_id}")
    replay = ReplayRun(
        parameters=reg.gen.Replay(refID=ref_id).nestedConf,
        screen_kws={"vis_mode": "video", "show_display": True},
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
- {doc}`../tutorials/2_experimental_data/replay` - Step-by-step tutorial
