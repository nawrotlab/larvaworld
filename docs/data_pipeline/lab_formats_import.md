# Lab-Specific Data Import

Larvaworld can import experimental datasets from diverse tracking systems. Each lab uses different hardware/software, resulting in different data formats.

---

## Supported Labs

| Lab          | Framerate (Hz) | Midline (#) | Contour (#) | Source                   |
| ------------ | -------------- | ----------- | ----------- | ------------------------ |
| **Schleyer** | 16             | 12          | 22          | Paisios et al. (2017)    |
| **Jovanic**  | 11.27\*        | 11          | 30\*\*      | de Tredern et al. (2024) |
| **Berni**    | 2              | 1           | 0           | Sims et al. (2019)       |
| **Arguello** | 10             | 5           | 0           | Kafle et al. (2025)      |

\*Variable, average framerate
\*\*Variable, convex hull used

---

## Import Workflow

### 1. Create LabFormat

```python
from larvaworld.lib import reg

lab = reg.gen.LabFormat(labID="Schleyer")
```

### 2. Import Single Dataset

```python
dataset = lab.import_dataset(
    parent_dir="exploration",        # Folder under the lab's raw data root
    merged=True,                     # Merge all larvae in this folder
    max_Nagents=30,                  # Optional: limit number of larvae
    min_duration_in_sec=60,          # Optional: minimum track duration
    id="exploration.30controls",     # Dataset ID on disk
    refID="exploration.30controls",  # Reference ID in the registry
    save_dataset=True,               # Store processed dataset
)
```

### 3. Import Multiple Datasets

```python
datasets = lab.import_datasets(
    source_ids=["30controls", "30mutants"],
    ids=["exploration.30controls", "exploration.30mutants"],
    refIDs=["exploration.30controls", "exploration.30mutants"],
    parent_dir="exploration",        # Common parent folder under raw/
    save_dataset=True,
)
```

### 4. Load Imported Dataset

```python
dataset = reg.loadRef(id="my_experiment", load=True)
print(dataset.e)  # Endpoint data
print(dataset.s.head())  # Step-wise data
```

---

## Lab-Specific Details

### Schleyer Lab

**Tracker**: Custom MATLAB tracker

**Data Structure**:

- 12-point midline
- 22-point contour
- 16 Hz framerate

**Example**:

```python
lab = reg.gen.LabFormat(labID="Schleyer")
lab.import_dataset(
    parent_dir="chemotaxis/exp1",
    raw_folder="/data/schleyer/raw",
    id="schleyer_chemotaxis",
    refID="schleyer.chemotaxis",
    save_dataset=True,
)
```

---

### Jovanic Lab

**Tracker**: Custom Python tracker

**Data Structure**:

- 11-point midline
- Convex hull (variable points)
- ~11.27 Hz (variable framerate)

**Example**:

```python
lab = reg.gen.LabFormat(labID="Jovanic")
lab.import_datasets(
    source_ids=["Fed", "Sucrose", "Starved"],  # Folder names under parent_dir
    ids=["Jovanic_Fed", "Jovanic_Sucrose", "Jovanic_Starved"],
    refIDs=["Jovanic.Fed", "Jovanic.Sucrose", "Jovanic.Starved"],
    parent_dir="feeding_state",
    raw_folder="/data/jovanic/raw",
    save_dataset=True,
)
```

---

### Berni Lab

**Tracker**: FIM (Frustrated Total Internal Reflection Microscopy)

**Data Structure**:

- Centroid only (no midline)
- 2 Hz framerate

**Example**:

```python
lab = reg.gen.LabFormat(labID="Berni")
lab.import_dataset(
    parent_dir="exploration/dish",
    raw_folder="/data/berni/raw",
    id="berni_exploration",
    refID="berni.exploration",
    save_dataset=True,
)
```

---

### Arguello Lab

**Tracker**: Custom tracker

**Data Structure**:

- 5-point midline
- 10 Hz framerate

**Example**:

```python
lab = reg.gen.LabFormat(labID="Arguello")
lab.import_dataset(
    parent_dir="thermotaxis/temperature_preference",
    raw_folder="/data/arguello/raw",
    id="arguello_thermotaxis",
    refID="arguello.thermotaxis",
    save_dataset=True,
)
```

---

## Data Processing After Import

```python
# Load dataset
dataset = reg.loadRef(id="my_experiment", load=True)

# Preprocess
dataset.preprocess(
    drop_collisions=True,
    interpolate_nans=True,
    filter_f=3.0,
)

# Process metrics
dataset.process(
    proc_keys=["angular", "spatial"],
    dsp_starts=[0],
    dsp_stops=[40, 60],
    tor_durs=[5, 10, 20],
)

# Annotate bouts
dataset.annotate(
    anot_keys=[
        "bout_detection",
        "bout_distribution",
        "interference",
    ]
)
```

See {doc}`data_processing` for details.

---

## Custom Lab Format

To add a new lab format, create a subclass of `LabFormat`:

```python
from larvaworld.lib.reg.generators import LabFormat

class MyLabFormat(LabFormat):
    def __init__(self):
        super().__init__(
            labID="MyLab",
            Npoints=10,     # Midline points
            Ncontour=0,     # No contour
            fr=15           # 15 Hz
        )

    def read_data(self, filepath):
        # Custom data reading logic
        pass
```

---

## Related Documentation

- {doc}`data_processing` - Data processing pipeline
- {doc}`reference_datasets` - Reference dataset management
- {doc}`../working_with_larvaworld/model_evaluation` - Using imported data for evaluation
