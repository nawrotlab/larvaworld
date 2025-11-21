# Table 3: Nutritious Arena Substrates

## 📋 Table

### Compound Composition of Established Nutritious Arena Substrates

| Substrate | Glucose (μg/ml) | Dextrose (μg/ml) | Saccharose (μg/ml) | Yeast (μg/ml) | Agar (μg/ml) | Cornmeal (μg/ml) | Literature Source |
|-----------|----------------|------------------|-------------------|---------------|--------------|-----------------|-------------------|
| **standard-medium** | 100 | - | - | 50 | 16 | - | Kaun et al. (2007) |
| **PED-tracker** | - | - | 10 | 187.5 | 5000 | - | Schumann et al. (2020) |
| **cornmeal** | - | 70.3 | - | 14.1 | 6.6 | 65.6 | Wosniack et al. (2021) |
| **sucrose** | 17.1 | - | - | - | 4 | - | Wosniack et al. (2021) |

**Note**: `-` indicates compound not present in substrate. All densities in μg/ml.

---

## Description

These are **established nutritious substrates** used in real *Drosophila* larva experiments, implemented in Larvaworld with their characteristic compound composition.

### Substrate Types:

1. **standard-medium** (Kaun et al. 2007)
   - **Glucose**: 100 μg/ml
   - **Yeast**: 50 μg/ml
   - **Agar**: 16 μg/ml
   - **Use**: Standard rearing and behavioral experiments
   - **Reference**: Rover vs Sitter phenotype studies

2. **PED-tracker** (Schumann et al. 2020)
   - **Saccharose**: 10 μg/ml
   - **Yeast**: 187.5 μg/ml
   - **Agar**: 5000 μg/ml (high agar content)
   - **Use**: Tracking experiments with FIM (Frustrated Total Internal Reflection Microscopy)
   - **Characteristic**: Very high agar content for optical tracking

3. **cornmeal** (Wosniack et al. 2021)
   - **Dextrose**: 70.3 μg/ml
   - **Yeast**: 14.1 μg/ml
   - **Agar**: 6.6 μg/ml
   - **Cornmeal**: 65.6 μg/ml
   - **Use**: More naturalistic substrate composition
   - **Characteristic**: Includes cornmeal component

4. **sucrose** (Wosniack et al. 2021)
   - **Glucose**: 17.1 μg/ml
   - **Agar**: 4 μg/ml
   - **Use**: Simple sugar substrate
   - **Characteristic**: Minimal composition for controlled experiments

---

## Usage in Larvaworld

### Substrate Configuration

Virtual larvae can be **grown, starved, or tested** on these substrates. Each food source in the arena is characterized by:

- **Substrate type**: One of the predefined types above
- **Nutritional quality**: Percentage of full quality (0-100%)
- **Available amount**: Quantity of food available
- **Depletion**: Food can be consumed until depleted

### Food Grid Mode

Alternatively, substrate can be placed as **patches in a grid**:
- Grid covers entire arena
- Each cell independently holds food amount
- Larvae can detect and consume food
- Dynamic depletion as larvae feed

### DEB Integration

Substrate composition affects:
- **Energy intake rate**: During feeding
- **Growth dynamics**: Via DEB model
- **Foraging behavior**: Hunger-driven exploration
- **Rover vs Sitter**: Phenotype expression

---

## Usage in ReadTheDocs

```rst
Nutritious Substrates
~~~~~~~~~~~~~~~~~~~~~

Larvaworld implements several established substrate compositions used in real 
*Drosophila* larva experiments.

.. list-table:: Nutritious Arena Substrates
   :header-rows: 1
   :widths: 15 10 10 12 10 10 10 23

   * - Substrate
     - Glucose
     - Dextrose
     - Saccharose
     - Yeast
     - Agar
     - Cornmeal
     - Reference
   * - standard-medium
     - 100
     - \-
     - \-
     - 50
     - 16
     - \-
     - Kaun et al. (2007)
   * - PED-tracker
     - \-
     - \-
     - 10
     - 187.5
     - 5000
     - \-
     - Schumann et al. (2020)
   * - cornmeal
     - \-
     - 70.3
     - \-
     - 14.1
     - 6.6
     - 65.6
     - Wosniack et al. (2021)
   * - sucrose
     - 17.1
     - \-
     - \-
     - \-
     - 4
     - \-
     - Wosniack et al. (2021)

All values in μg/ml. ``-`` indicates compound not present.

**Configuration Example:**

.. code-block:: python

   from larvaworld import reg
   
   # Create food source with standard-medium substrate
   food_conf = reg.gen.FoodConf(
       substrate='standard-medium',
       quality=1.0,  # 100% quality
       amount=1000.0  # μg
   )
   
   # Food grid with cornmeal substrate
   food_grid_conf = reg.gen.FoodConf(
       substrate='cornmeal',
       quality=0.75,  # 75% quality
       grid_dims=(10, 10),  # 10x10 grid
       amount_per_cell=50.0  # μg per cell
   )

**Substrate Effects:**

Different substrates affect larval behavior through the DEB (Dynamic Energy 
Budget) model:

- **Energy intake rate**: Varies by substrate nutritional content
- **Growth curves**: Different substrates support different growth trajectories
- **Foraging strategies**: Hunger-driven exploration depends on substrate quality
- **Rover/Sitter phenotypes**: Expression modulated by substrate availability

For growth and DEB simulations, see :ref:`growth-experiments` and 
:ref:`deb-model-docs`.
```

---

## Implementation Details

### Code Location

Substrates are defined in: `/src/larvaworld/lib/model/deb/gut.py`

### Substrate Class

```python
class Substrate:
    """Nutritious substrate composition."""
    
    name: str  # One of: 'standard-medium', 'PED-tracker', 'cornmeal', 'sucrose'
    compounds: dict  # Compound densities (μg/ml)
    quality: float  # 0-1, percentage of full nutritional value
    
    def energy_density(self):
        """Calculate energy density based on compound composition."""
        # Glucose, dextrose, saccharose: ~4 kcal/g
        # Yeast: ~3.2 kcal/g
        # Agar, cornmeal: structural, minimal energy
        ...
```

### Usage in Experiments

These substrates are used in:
- **Growth experiments**: `growth`, `RvsS`, `RvsS_on`, etc.
- **Foraging experiments**: `patch_grid`, `random_food`, etc.
- **Feeding experiments**: Single/multiple patch experiments
- **DEB simulations**: All energetics-based experiments

---

## References

1. **Kaun et al. (2007)**: *Drosophila* larvae rover and sitter behavioral phenotypes
2. **Schumann et al. (2020)**: FIM-based tracking with high-agar substrate
3. **Wosniack et al. (2021)**: Naturalistic foraging experiments with multiple substrates

---

## Developer Note

**Priority**: Include in documentation (developer request)  
**Purpose**: Practical information for experiments  
**Audience**: Researchers setting up virtual or real experiments

---

## References

### Kaun et al. (2007)
**Title**: Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase  
**Journal**: Journal of Experimental Biology, 210(20), 3547-3558  
**DOI**: [10.1242/jeb.006924](https://doi.org/10.1242/jeb.006924)  
**Authors**: Kaun, K. R., Riedl, C. A., Chakaborty-Chatterjee, M., Belay, A. T., Douglas, S. J., Gibbs, A. G., & Sokolowski, M. B.

### Schumann et al. (2020)
**Title**: The PEDtracker: An Automatic Staging Approach for Drosophila melanogaster Larvae  
**Journal**: Frontiers in Behavioral Neuroscience, 14  
**DOI**: [10.3389/fnbeh.2020.612313](https://doi.org/10.3389/fnbeh.2020.612313)  
**Authors**: Schumann, I., & Triphan, T.

### Wosniack et al. (2022)
**Title**: Adaptation of Drosophila larva foraging in response to changes in food resources  
**Journal**: eLife, 11, e75826  
**DOI**: [10.7554/eLife.75826](https://doi.org/10.7554/eLife.75826)  
**URL**: https://elifesciences.org/articles/75826  
**Authors**: Wosniack, M. E., Festa, D., Hu, N., Gjorgjieva, J., & Berni, J.

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **LaTeX**: Lines 725-746 in main.tex
- **Label**: `tab:substrate`
- **Section**: Substrate (lines 718-719)

