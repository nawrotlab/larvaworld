# Table 2: Preconfigured Behavioral Experiments

## 📋 Table

### Summary of Preconfigured Behavioral Experiments

| Type/Behavior       | Experiment            | Description                                            | Literature Source          |
| ------------------- | --------------------- | ------------------------------------------------------ | -------------------------- |
| **exploration**     | close-view            | Single larva closely inspected in a tiny arena         | -                          |
|                     | dish                  | Exploration of a non-nutritious Petri-dish             | -                          |
|                     | dispersion            | Larva dispersion from the arena center                 | -                          |
| **chemotaxis**      | navigation            | Navigation up an odor gradient                         | Gomez-Marin et al. (2012)  |
|                     | local search          | Exploration in the vicinity of an odor source          | Gomez-Marin et al. (2012)  |
| **odor preference** | train & test          | Olfactory associative learning (train & test phase)    | The Maggot Learning Manual |
|                     | test on/off food      | Test in the presence/absence of nutritious substrate   | The Maggot Learning Manual |
| **foraging**        | patchy food           | Foraging in arena with one/two/multiple food patches   | -                          |
|                     | uniform food          | Foraging in uniformly distributed nutritious substrate | -                          |
| **growth**          | rearing               | Larva rearing in ad-libitum conditions                 | -                          |
|                     | rovers VS sitters     | Foraging phenotypes compared in diverse conditions     | Kaun et al. (2007)         |
| **imitation**       | realistic bodies      | Multisegment larvae in Box2D physics engine            | -                          |
|                     | dataset imitation     | Experimental dataset imitation                         | -                          |
| **games**           | maze                  | Navigation in a maze towards an odor source            | -                          |
|                     | capture/keep the flag | Larva teams competing for a portable nutritious object | -                          |

---

## Description

This table summarizes all preconfigured behavioral experiments available in Larvaworld. Each experiment is designed to replicate or simulate specific aspects of Drosophila larva behavior, ranging from basic exploration to complex learning tasks and competitive games.

---

## Detailed Descriptions

### Exploration Experiments

#### close-view

**Purpose**: Detailed observation of individual larva behavior
**Setup**: Single larva in a tiny arena (high spatial resolution)
**Use Cases**:

- Detailed kinematic analysis
- High-resolution body tracking
- Individual behavioral profiling
- Video recording for presentations

#### dish

**Purpose**: Standard exploration in Petri dish
**Setup**: Non-nutritious agar substrate, standard Petri dish size
**Use Cases**:

- Baseline locomotor behavior
- Spatial exploration patterns
- Movement statistics

#### dispersion

**Purpose**: Measure spatial spreading from central point
**Setup**: Larvae start clustered at arena center
**Metrics**: Dispersal index, spatial distribution over time
**Use Cases**:

- Population dynamics
- Activity level assessment
- Social spacing behavior

---

### Chemotaxis Experiments

#### navigation

**Purpose**: Test odor-guided navigation
**Setup**: Odor gradient from point source
**Literature**: Gomez-Marin et al. (2012)
**Metrics**: Chemotaxis index, path efficiency
**Use Cases**:

- Olfactory navigation algorithms
- Sensory-motor integration
- Gradient following strategies

#### local search

**Purpose**: Behavior near odor source
**Setup**: Exploration around localized odor
**Literature**: Gomez-Marin et al. (2012)
**Behavior**: Head casting, local search patterns
**Use Cases**:

- Search strategies
- Exploitation vs exploration
- Source localization

---

### Odor Preference Experiments

#### train & test

**Purpose**: Olfactory associative learning
**Setup**: Training phase + test phase
**Protocol**: The Maggot Learning Manual
**Phases**:

- **Training**: Odor-reward pairing
- **Test**: Preference measurement
  **Metrics**: Preference Index (PI), learning score

#### test on/off food

**Purpose**: Test context-dependent preferences
**Conditions**:

- **On food**: Test with nutritious substrate present
- **Off food**: Test without food
  **Protocol**: The Maggot Learning Manual
  **Use Cases**:
- Metabolic state effects
- Memory consolidation
- Context dependency

---

### Foraging Experiments

#### patchy food

**Purpose**: Foraging with discrete food sources
**Variations**:

- Single patch
- Two patches
- Multiple patches (grid)
  **Metrics**: Time per patch, transitions, exploitation efficiency
  **Use Cases**:
- Optimal foraging theory
- Patch residence time
- Food competition

#### uniform food

**Purpose**: Foraging with homogeneous food distribution
**Setup**: Nutritious substrate uniformly distributed
**Metrics**: Total intake, movement patterns
**Use Cases**:

- Baseline feeding behavior
- Satiation dynamics
- Movement under food abundance

---

### Growth Experiments

#### rearing

**Purpose**: Larval development simulation
**Setup**: Ad-libitum food, long duration (hours to days)
**Metrics**: Body size, developmental stage, food consumption
**Use Cases**:

- DEB model validation
- Growth rate analysis
- Metabolic state tracking

#### rovers VS sitters

**Purpose**: Compare foraging phenotypes
**Genotypes**: rover (for^R^) vs sitter (for^s^)
**Literature**: Kaun et al. (2007)
**Conditions**: Various food distributions and quality
**Use Cases**:

- Genetic variation in foraging
- Phenotype-environment interactions
- Natural variation studies

---

### Imitation Experiments

#### realistic bodies

**Purpose**: Physics-based multisegment simulation
**Engine**: Box2D physics
**Features**: Realistic body mechanics, collisions, damping
**Use Cases**:

- High-fidelity simulations
- Body mechanics studies
- Contact interactions

#### dataset imitation

**Purpose**: Replicate experimental datasets
**Method**: Virtual larvae match real trajectories
**Use Cases**:

- Model validation
- Parameter identification
- Reverse engineering behavior

---

### Game Experiments

#### maze

**Purpose**: Navigation problem-solving
**Setup**: Maze structure with odor source at exit
**Challenges**: Dead ends, multiple paths
**Use Cases**:

- Spatial learning
- Problem solving
- Path planning algorithms

#### capture/keep the flag

**Purpose**: Team competition game
**Setup**: Two larva teams, one portable food object
**Objective**: Capture object and bring to home base
**Use Cases**:

- Multi-agent interactions
- Competition dynamics
- Cooperative/competitive behavior

---

## Configuration Access

All experiments can be accessed via:

```python
from larvaworld import reg

# Load preconfigured experiment
exp = reg.conf.Exp.getID('dish')

# List all available experiments
available_exps = reg.conf.Exp.confIDs

# Run experiment
from larvaworld.lib import ExpRun
run = ExpRun(experiment='navigation')
run.simulate()
```

---

## References

### Gomez-Marin et al. (2012)

**Title**: Active sensation during orientation behavior in the Drosophila larva: More sense than luck
**Journal**: Current Opinion in Neurobiology, 22(2)
**DOI**: [10.1016/j.conb.2011.11.008](https://doi.org/10.1016/j.conb.2011.11.008)

### Kaun et al. (2007)

**Title**: Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase
**Journal**: Journal of Experimental Biology, 210(20), 3547-3558
**DOI**: [10.1242/jeb.006924](https://doi.org/10.1242/jeb.006924)

### The Maggot Learning Manual

**Description**: Standardized protocols for Drosophila larva learning experiments
**Reference**: Community-established protocols for olfactory learning assays

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software
- **LaTeX**: Lines 514-555 in main.tex
- **Label**: `tab:experiments`
- **Section**: Implementation

---

## Notes

**Developer Note**: This table is cross-referenced with the experiment types mindmap diagrams (05_a and 05_b) which provide visual representations of how experiments are categorized conceptually vs. in code structure.

**Tutorial References**: Most experiments have corresponding tutorial notebooks in `/docs/tutorials/` demonstrating their usage.
