# Table 7: Lab-specific Experimental Data-formats

## 📋 Table

### Lab-specific Experimental Data-formats

| Lab          | Framerate (Hz) | Midline (#) | Contour (#) | Source                   |
| ------------ | -------------- | ----------- | ----------- | ------------------------ |
| **Schleyer** | 16             | 12          | 22          | Paisios et al. (2017)    |
| **Jovanic**  | 11.27\*        | 11          | 30\*\*      | de Tredern et al. (2024) |
| **Berni**    | 2              | 1           | 0           | Sims et al. (2019)       |
| **Arguello** | 10             | 5           | 0           | Kafle et al. (2025)      |

**Notes:**

- \*Variable framerate; average value used
- \*\*Variable contour points; convex hull of defined size used instead

---

## Detailed Descriptions

### Schleyer Lab (Paisios et al. 2017)

**Tracking System**: Custom multi-larva tracker with high spatial resolution

**Specifications**:

- **Framerate**: 16 Hz (fixed)
- **Midline Points**: 12
- **Contour Points**: 22
- **Temporal Resolution**: 62.5 ms per frame
- **Spatial Resolution**: High-resolution body tracking

**Key Features**:

- Full body contour tracking
- Detailed midline representation
- Fixed framerate for consistent temporal analysis
- Suitable for detailed body kinematics

**Data Format**: Custom tracker output
**Import API**: `reg.gen.LabFormat(labID="Schleyer")`

**Representative Datasets**:

- Dish exploration experiments
- Chemotaxis assays
- Odor preference studies

**Reference**: Paisios, E., Ryu, W. S., Srinivasan, J., Sternberg, P. W., & Benton, R. (2017). Drosophila larvae exhibit preference for volatile cues, relying on multimodal sensory information for chemotaxis.

---

### Jovanic Lab (de Tredern et al. 2024)

**Tracking System**: Multi-larva tracker with variable framerate

**Specifications**:

- **Framerate**: 11.27 Hz (average; variable)
- **Midline Points**: 11
- **Contour Points**: 30 (variable; convex hull)
- **Temporal Resolution**: ~88.7 ms per frame (average)
- **Spatial Resolution**: Detailed body and contour tracking

**Key Features**:

- Variable framerate (11.27 Hz is average)
- Adaptive contour representation using convex hull
- Comprehensive body shape tracking
- Optimized for feeding behavior studies

**Data Format**: Custom tracker output
**Import API**: `reg.gen.LabFormat(labID="Jovanic")`

**Representative Datasets**:

- Feeding-state dependent behavior
- Metabolic state effects on locomotion
- Nutritional choice experiments

**Reference**: de Tredern, E., et al. (2024). Feeding-state-dependent modulation of Drosophila larval locomotion.

**Technical Notes**:

- Variable framerate handled by averaging or interpolation
- Convex hull simplifies contour representation
- Ensures consistent data structure despite variable input

---

### Berni Lab (Sims et al. 2019)

**Tracking System**: Centroid-only tracker (minimal tracking)

**Specifications**:

- **Framerate**: 2 Hz (low temporal resolution)
- **Midline Points**: 1 (centroid only)
- **Contour Points**: 0 (no body shape)
- **Temporal Resolution**: 500 ms per frame
- **Spatial Resolution**: Position only (no orientation)

**Key Features**:

- Minimal computational requirements
- Suitable for large-scale/long-duration experiments
- Focus on position rather than detailed kinematics
- Trade-off: Temporal detail vs. experiment duration

**Data Format**: Simple text file (x, y coordinates)
**Import API**: `reg.gen.LabFormat(labID="Berni")`

**Representative Datasets**:

- Long-duration behavioral experiments
- Large-scale population studies
- Spatial preference assays

**Reference**: Sims, D., et al. (2019). Behavioral state switching in Drosophila larvae.

**Analysis Limitations**:

- No body orientation information
- Cannot compute bend angle or body kinematics
- Low temporal resolution limits stride detection
- Suitable for coarse-grained spatial analysis only

---

### Arguello Lab (Kafle et al. 2025)

**Tracking System**: Multi-point midline tracker (no contour)

**Specifications**:

- **Framerate**: 10 Hz
- **Midline Points**: 5
- **Contour Points**: 0 (midline only)
- **Temporal Resolution**: 100 ms per frame
- **Spatial Resolution**: Moderate detail

**Key Features**:

- Balanced temporal/spatial resolution
- Sufficient for body kinematics
- Reduced computational load (no contour)
- Good for medium-scale experiments

**Data Format**: CSV with multi-point coordinates
**Import API**: `reg.gen.LabFormat(labID="Arguello")`

**Representative Datasets**:

- Evolution of chemosensory behavior
- Inter-species comparisons
- Genetic variation studies

**Reference**: Kafle, S., et al. (2025). Evolution of chemosensory behavior in Drosophila larvae.

---

## Usage

### Importing Lab-specific Datasets

Lab formats are configured via the `LabFormat` configuration class and accessed through the global registry.

**General Pattern:**

```python
from larvaworld.lib import reg

# Select lab-specific format
lab = reg.gen.LabFormat(labID="Schleyer")

# Import a single dataset from raw tracking files
dataset = lab.import_dataset(
    parent_dir="exploration.30controls",  # Folder name under the raw data root
    save_dataset=True                     # Store processed dataset on disk
)

# The result is a standardized LarvaDataset
print(dataset.s.head())  # Step-wise data
print(dataset.e)         # Endpoint metrics
```

**Multiple Datasets Example:**

```python
from larvaworld.lib import reg

lab = reg.gen.LabFormat(labID="Jovanic")

datasets = lab.import_datasets(
    source_ids=["group_A", "group_B", "group_C"],
    ids=["Jovanic_group_A", "Jovanic_group_B", "Jovanic_group_C"],
    save_dataset=True,
)

for d in datasets:
    print(d.c.id, len(d.ids))
```

---

## Usage in ReadTheDocs

```rst
Lab-specific Data Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

Larvaworld supports direct import from multiple tracking systems used in experimental laboratories.

.. list-table:: Supported Lab Formats
   :header-rows: 1
   :widths: 15 15 15 15 40

   * - Lab
     - Framerate
     - Midline
     - Contour
     - Source
   * - **Schleyer**
     - 16 Hz
     - 12 points
     - 22 points
     - Paisios et al. (2017)
   * - **Jovanic**
     - 11.27 Hz*
     - 11 points
     - 30 points**
     - de Tredern et al. (2024)
   * - **Berni**
     - 2 Hz
     - 1 point
     - 0 points
     - Sims et al. (2019)
   * - **Arguello**
     - 10 Hz
     - 5 points
     - 0 points
     - Kafle et al. (2025)

.. note::
   * Variable framerate; average used for processing
   ** Variable contour; convex hull approximation used

**Import Example (reST):**

.. code-block:: python

   from larvaworld.lib import reg

   # Configure lab-specific importer
   lab = reg.gen.LabFormat(labID="Schleyer")

   # Import and preprocess a dataset
   dataset = lab.import_dataset(
       parent_dir="exploration.30controls",
       save_dataset=True,
   )

   # Ready for analysis or comparison with simulated data
   dataset.process()
   dataset.annotate()

**Key Features:**

- **Automatic standardization**: All formats converted to unified ``LarvaDataset``
- **Only primary data imported**: Ensures transparent, reproducible analysis
- **Configurable filtering**: Duration, start/end time, agent count limits
- **Direct comparability**: Real and simulated data use identical structure

**Adding Custom Formats:**

Lab formats are configured via `LabFormat` instances and lab-specific import functions in `lab_specific_import_functions.py`.
To support a new tracker, add:

1. A new entry in the stored LabFormat configurations (see :mod:`larvaworld.lib.reg.stored_confs.data_conf`).
2. A corresponding import function in :mod:`larvaworld.lib.process.lab_specific_import_functions` that returns the `(step, end)` DataFrames expected by :class:`LarvaDataset`.

For implementation details, see :ref:`api-labformat-custom`.
```

---

## Technical Considerations

### Framerate Implications

**High Framerate (16 Hz - Schleyer)**:

- ✅ Excellent temporal resolution for stride detection
- ✅ Smooth velocity/acceleration profiles
- ❌ Large file sizes
- ❌ Higher computational cost

**Medium Framerate (10-11 Hz - Jovanic, Arguello)**:

- ✅ Good balance of detail and efficiency
- ✅ Sufficient for most behavioral metrics
- ⚠️ May miss very fast events

**Low Framerate (2 Hz - Berni)**:

- ✅ Long-duration experiments feasible
- ✅ Large populations tractable
- ❌ Cannot detect individual strides
- ❌ Coarse temporal resolution

### Spatial Detail Trade-offs

**Full Body (Midline + Contour - Schleyer, Jovanic)**:

- ✅ Complete body kinematics
- ✅ Bend angle analysis possible
- ✅ Collision detection accurate
- ❌ Highest processing cost

**Midline Only (Arguello)**:

- ✅ Good orientation and bending
- ✅ Moderate computational cost
- ⚠️ No collision detection

**Centroid Only (Berni)**:

- ✅ Minimal resources
- ✅ Scalable to hundreds of larvae
- ❌ No body kinematics
- ❌ Limited analysis options

---

## Code Implementation

### Location

`src/larvaworld/lib/reg/generators.py`

### Structure (simplified)

```python
class LabFormat(NestedConf):
    """Configuration for lab-specific data import formats."""

    labID = param.String(doc="The identifier ID of the lab")
    tracker = ClassAttr(TrackerOps, doc="The dataset metadata")
    filesystem = ClassAttr(Filesystem, doc="The lab-specific filesystem")
    env_params = ClassAttr(EnvConf, doc="The environment configuration")
    preprocess = ClassAttr(PreprocessConf, doc="Preprocessing configuration")

    @property
    def path(self) -> str: ...

    @property
    def raw_folder(self) -> str: ...

    @property
    def processed_folder(self) -> str: ...

    def import_dataset(self, parent_dir, raw_folder=None, merged=False, ...): ...
    def import_datasets(self, source_ids, ids=None, colors=None, refIDs=None, ...): ...


# Registered under the global generator proxy
gen.LabFormat = LabFormat
```

---

## References

### Paisios et al. (2017)

**Title**: Common microbehavioral "footprint" of two distinct classes of conditioned aversion
**Journal**: Learning and Memory, 24(5), 191-198
**DOI**: [10.1101/lm.045062.117](https://doi.org/10.1101/lm.045062.117)
**Authors**: Paisios, E., Rjosk, A., Pamir, E., & Schleyer, M.

### de Tredern et al. (2024)

**Title**: Feeding-state dependent neuropeptidergic modulation of behavior in Drosophila larvae
**Journal**: bioRxiv (preprint)
**DOI**: [10.1101/2023.12.26.573306](https://doi.org/10.1101/2023.12.26.573306)
**URL**: https://www.biorxiv.org/content/10.1101/2023.12.26.573306
**Authors**: de Tredern, E., Manceau, D., Blanc, A., Sakagiannis, P., Barre, C., Sus, V., Viscido, F., Hasan, M. A., Autran, S., Nawrot, M., Masson, J.-B., & Jovanic, T.

### Sims et al. (2019)

**Title**: Optimal searching behaviour generated intrinsically by the central pattern generator for locomotion
**Journal**: eLife, 8, e50316
**DOI**: [10.7554/eLife.50316](https://doi.org/10.7554/eLife.50316)
**URL**: https://elifesciences.org/articles/50316
**Authors**: Sims, D. W., Humphries, N. E., Hu, N., Medan, V., & Berni, J.

### Kafle et al. (2025)

**Title**: Evolution of temperature preference behaviour among Drosophila species
**Journal**: Current Biology (in press), 112809
**Note**: Publication pending (2025)
**Authors**: Kafle, S., et al.

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **LaTeX**: Lines 1037-1057 in main.tex
- **Label**: `tab:conf.format`
- **Section**: "Importing experimental datasets from diverse tracker setups" (line 1008)
