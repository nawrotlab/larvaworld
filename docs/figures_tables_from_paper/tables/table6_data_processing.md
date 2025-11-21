# Table 6: Data Processing Methods

## 📋 Table

### Data Processing Methods

| Category | Function | Description |
|----------|----------|-------------|
| **Preprocessing** | Scaling | x-y scaling by a scalar |
| | Transposition | x-y transposition |
| | Alignment | Trajectory alignment e.g. to common origin |
| | Interpolation | Missing data interpolation |
| | Exclusion | Data exclusion on condition |
| | Filtering | Low-pass filtering at a cut-off frequency |
| **Processing** | Angular analysis | Bend/orientation angle, angular velocity/acceleration |
| | Spatial analysis | Spatial distance/velocity/acceleration |
| | Forward components | Spatial metric components along orientation axis |
| | Dispersal | Larva spatial dispersal during time ranges |
| | Tortuosity | Trajectory tortuosity for sliding temporal windows |
| | Odor preference | Olfactory preference index |
| | Odor concentration | Absolute and perceived odor concentration along trajectory |
| **Annotation** | Strides/Crawl-runs | Individual strides and uninterrupted chains of concatenated strides |
| | Crawl-pauses | Immobility epochs without peristaltic strides |
| | Turns | Turning events based on reorientation amplitude or angular velocity |
| | Bout analysis | Spatial/angular metric change during bouts |
| | Bout distribution | Distribution fitting for bout duration/length |

---

## Detailed Descriptions

### Preprocessing Pipeline

**Purpose**: Clean and standardize raw trajectory data before analysis

**1. Scaling**
- **Function**: Multiply x-y coordinates by a scalar
- **Use**: Convert units (e.g., pixels to meters)
- **Parameters**: Scale factor (float)
- **Example**: `rescale_by=0.001` converts mm to meters

**2. Transposition**
- **Function**: Shift all trajectories by x-y offset
- **Use**: Center trajectories, align to origin
- **Parameters**: `"arena"`, `"origin"`, or `"center"`
- **Example**: `transposition="center"` moves all to arena center

**3. Alignment**
- **Function**: Align trajectories to common starting point/orientation
- **Use**: Compare movement patterns independent of start position
- **Parameters**: Alignment type, reference point
- **Example**: All trajectories start at (0, 0) facing up

**4. Interpolation**
- **Function**: Fill missing data points
- **Methods**: Linear interpolation over NaNs
- **Use**: Handle tracking gaps, ensure continuous data
- **Parameters**: `interpolate_nans=True`

**5. Exclusion**
- **Function**: Remove data based on conditions
- **Use**: Filter out artifacts, low-quality tracks
- **Conditions**: Duration, distance, velocity thresholds
- **Example**: Remove tracks shorter than 10 seconds

**6. Filtering**
- **Function**: Low-pass Butterworth filter
- **Use**: Smooth trajectories, remove high-frequency noise
- **Parameters**: `filter_f` (cutoff frequency in Hz), `recompute`
- **Example**: `filter_f=1.0` removes jitter while preserving behavior

---

### Processing Pipeline

**Purpose**: Compute derived metrics from preprocessed trajectories

**1. Angular Analysis**
- **Computes**:
  - Body bend angle (curvature)
  - Orientation angle (heading)
  - Angular velocity (turning rate)
  - Angular acceleration
- **Units**: Degrees, degrees/second, degrees/second²

**2. Spatial Analysis**
- **Computes**:
  - Distance traveled
  - Linear velocity (speed)
  - Linear acceleration
  - Path length
- **Units**: Meters, meters/second, meters/second²

**3. Forward Components**
- **Computes**:
  - Forward velocity (along body axis)
  - Lateral velocity (perpendicular to body)
  - Forward/lateral decomposition of any metric
- **Use**: Distinguish forward crawling from lateral drift

**4. Dispersal**
- **Computes**:
  - Spatial spread over time
  - Distance from origin/center
  - Dispersion index
- **Use**: Measure exploration, spatial patterns
- **Parameters**: Time ranges, reference point

**5. Tortuosity**
- **Computes**:
  - Path straightness index
  - Sinuosity
  - Computed over sliding windows
- **Use**: Quantify path complexity
- **Parameters**: Window duration

**6. Odor Preference**
- **Computes**:
  - Preference Index (PI)
  - Time spent in odor zones
  - Zone transitions
- **Formula**: PI = (T_odor - T_control) / (T_odor + T_control)
- **Use**: Olfactory learning, chemotaxis experiments

**7. Odor Concentration**
- **Computes**:
  - Absolute concentration at each position
  - Perceived concentration (sensor-weighted)
  - Concentration gradient
- **Use**: Chemotaxis analysis, sensory landscapes

---

### Annotation Pipeline

**Purpose**: Identify and characterize behavioral events

**1. Strides/Crawl-runs**
- **Detects**: Individual peristaltic strides
- **Groups**: Uninterrupted chains of strides (crawl-runs)
- **Metrics**: Stride frequency, amplitude, duration
- **Use**: Locomotor pattern analysis

**2. Crawl-pauses**
- **Detects**: Epochs without peristaltic motion
- **Criteria**: Low velocity, no body waves
- **Metrics**: Pause duration, frequency
- **Use**: Intermittency, behavioral switching

**3. Turns**
- **Detects**: Reorientation events
- **Criteria**: 
  - Amplitude threshold (e.g., >30°)
  - Angular velocity threshold
- **Metrics**: Turn angle, duration, frequency
- **Types**: Head casts vs body bends

**4. Bout Analysis**
- **Function**: Measure changes during behavioral bouts
- **Bouts**: Crawl-runs, pauses, turns
- **Metrics**: Distance traveled, angle changed, duration
- **Use**: Characterize bout structure

**5. Bout Distribution**
- **Function**: Fit distributions to bout durations/lengths
- **Distributions**: Exponential, power-law, log-normal
- **Metrics**: Distribution parameters, goodness-of-fit
- **Use**: Stochastic modeling, behavioral state analysis

---

## Pipeline Configuration

### Example: Complete Processing
```python
from larvaworld.lib import reg

# Load dataset (simulated or imported) via reference ID
dataset = reg.loadRef(id="my_experiment", load=True)

# 1. Preprocessing
dataset.preprocess(
    drop_collisions=True,     # Remove collision frames (optional)
    interpolate_nans=True,    # Fill gaps in xy coordinates
    filter_f=1.0,             # 1 Hz low-pass filter
    rescale_by=0.001,         # mm to m
    transposition="center",   # Align trajectories to arena center
)

# 2. Processing
dataset.process(
    proc_keys=["angular", "spatial"],  # Core metric pipelines
    dsp_starts=[0],                    # Dispersal start times (s)
    dsp_stops=[40, 60],                # Dispersal stop times (s)
    tor_durs=[5, 10, 20],              # Tortuosity window durations (s)
)

# 3. Annotation
dataset.annotate(
    anot_keys=[
        "bout_detection",      # Detect strides, runs, pauses, turns
        "bout_distribution",   # Fit distributions to bout metrics
        "interference",        # Crawl-bend interference metrics
    ],
)

# Access results
print(dataset.s.columns)  # All computed step-wise metrics
print(dataset.e)          # Per-larva endpoint summaries
```

---

## Usage in ReadTheDocs

```rst
Data Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

Larvaworld provides three sequential processing stages for trajectory analysis.

**Stage 1: Preprocessing**

Data cleaning and standardization:

.. list-table::
   :widths: 25 75
   
   * - **Scaling**
     - Unit conversion (e.g., pixels to meters)
   * - **Transposition**
     - Spatial shifting and centering
   * - **Alignment**
     - Trajectory alignment to common origin
   * - **Interpolation**
     - Missing data filling
   * - **Exclusion**
     - Conditional data filtering
   * - **Filtering**
     - Low-pass noise removal

**Stage 2: Processing**

Metric computation:

.. list-table::
   :widths: 25 75
   
   * - **Angular analysis**
     - Bend angle, orientation, angular velocity/acceleration
   * - **Spatial analysis**
     - Distance, velocity, acceleration, path length
   * - **Forward components**
     - Body-axis-aligned velocity decomposition
   * - **Dispersal**
     - Spatial spread and exploration metrics
   * - **Tortuosity**
     - Path straightness and sinuosity
   * - **Odor preference**
     - Preference Index (PI) calculation
   * - **Odor concentration**
     - Sensory exposure along trajectories

**Stage 3: Annotation**

Behavioral event detection:

.. list-table::
   :widths: 25 75
   
   * - **Strides/Crawl-runs**
     - Peristaltic motion segmentation
   * - **Crawl-pauses**
     - Immobility epoch detection
   * - **Turns**
     - Reorientation event identification
   * - **Bout analysis**
     - Within-bout metric computation
   * - **Bout distribution**
     - Statistical distribution fitting

**Configuration Example (Python API):**

.. code-block:: python

   from larvaworld.lib import reg
   
   # Load dataset from reference registry
   ds = reg.loadRef(id="dish_exploration", load=True)
   
   # Configure and run all pipelines
   ds.preprocess(
       drop_collisions=True,
       interpolate_nans=True,
       filter_f=1.0,
       rescale_by=0.001,
       transposition="center",
   )
   
   ds.process(
       proc_keys=["angular", "spatial"],
       dsp_starts=[0],
       dsp_stops=[40, 60],
       tor_durs=[5, 10, 20],
   )
   
   ds.annotate(
       anot_keys=["bout_detection", "bout_distribution", "interference"],
   )
   
   # Access results
   print(ds.s.columns)  # All metrics
   print(ds.e)          # Summary statistics

**Key Features:**

- **Unified pipeline**: Same processing for simulated and real data
- **Configurable**: Select only needed operations
- **Sequential**: Preprocessing → Processing → Annotation
- **Cached**: Results stored in HDF5 for fast re-access
- **Extensible**: Add custom processing functions

For detailed API documentation, see :ref:`api-dataset-processing`.
```

---

## Code Implementation

### Location
`/src/larvaworld/lib/process/dataset.py`

### Key Methods
```python
class LarvaDataset:
    def preprocess(
        self,
        drop_collisions: bool = False,
        interpolate_nans: bool = False,
        filter_f: float | None = None,
        rescale_by: float | None = None,
        transposition: str | None = None,
        recompute: bool = False,
    ):
        """Stage 1: Data cleaning and standardization."""
        ...
    
    def process(
        self,
        proc_keys: list[str] = ["angular", "spatial"],
        dsp_starts: list[float] = [0],
        dsp_stops: list[float] = [40, 60],
        tor_durs: list[float] = [5, 10, 20],
        is_last: bool = False,
        **kwargs,
    ):
        """Stage 2: Compute derived metrics (spatial, angular, dispersal, tortuosity, etc.)."""
        ...
    
    def annotate(
        self,
        anot_keys: list[str] = ["bout_detection", "bout_distribution", "interference"],
        is_last: bool = False,
        **kwargs,
    ):
        """Stage 3: Detect behavioral events and compute bout statistics."""
        ...
```

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **LaTeX**: Lines 897-935 in main.tex
- **Label**: `tab:enrichment`
- **Section**: "Unbiased parameter computation and data analysis" (line 891)
