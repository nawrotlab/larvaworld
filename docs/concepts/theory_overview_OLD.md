# Larvaworld: behavioral simulation and analysis for *Drosophila* larvae

Larvaworld is a virtual laboratory for *Drosophila melanogaster* larval behavior. It combines agent-based simulations, multiscale neural control and standardized data analysis to study how individual larvae move, forage, learn and adapt in complex environments.

The platform was designed with four main aims:

- to integrate modeling principles from neuroscience, ecology and energetics into a single coherent framework,
- to provide a user-friendly interface for both behavioral modeling and data analysis,
- to remain modular and extensible at every level (from single modules to full experiments), and
- to stay computationally efficient while handling large behavioral datasets and long simulations.

Figure 1 summarizes this architecture: virtual larvae (larva models) live in configurable environments, are simulated as agents in an agent-based engine, and generate datasets that are analyzed by the same pipelines as experimental locomotion data. Genetic algorithms and model-evaluation tools close the loop by optimizing and testing models against real experiments.

This page provides a conceptual overview of these components and how they fit together.

---

```{figure} ../figures_tables_from_paper/figures/fig1_architecture.png
:alt: Larvaworld Architecture
:align: center
:width: 90%

**Figure 1**: Schematic of the main components and functionalities of Larvaworld.
```

---

## Larva models: body, sensors and behavioral architecture

At the core of Larvaworld are virtual larvae. Each larva is represented as a 2D body that can be tracked and analyzed exactly like a real animal, together with a modular behavioral controller that links sensory input to motor output.

### Virtual body and sensory modalities

The larval body is modeled as a simple 2D object, typically with one or more coupled segments (torsional spring model). This representation allows:

- realistic angular and linear motion fitted to empirical tracking data,
- explicit midline and contour representation compatible with common tracking setups, and
- optional multi-segment physics using a Box2D engine for more detailed body–arena interactions.

Virtual larvae can carry multiple sensory modalities, including:

- **Olfaction**,
- **Mechanoreception**,
- **Gustation**, and
- **Thermosensation**.

Sensors are placed at defined body locations (e.g. head region), and read out values from the simulated sensory landscapes (odorscapes, thermoscapes, windscapes). This makes it possible to express navigation, local search, and stimulus-dependent behavior using the same kinds of cues that real larvae experience.

### Layered behavioral control

Behavioral control in Larvaworld follows a layered architecture. Each virtual larva is controlled by a hierarchical, behavior-based system with three main layers:

1. **Locomotory layer**  
   Generates the basic locomotor patterns: crawling, turning, bending and feeding. These are implemented as oscillatory processes (crawling and lateral bending oscillators, feeding oscillators), which can be active, coupled, or mutually exclusive depending on the current state.

2. **Reactive layer**  
   Integrates sensory information (e.g. odor gradients, mechanosensory input) and modulates locomotion in a stimulus-dependent way. Typical examples include chemotaxis, local search and avoidance.

3. **Adaptive layer**  
   Governs slower adjustments such as learning and internal-state dependent modulation. This layer can host “memory” or reinforcement-learning-like modules, including a neuron-level mushroom body (MB) model for olfactory associative learning.

Each layer is built from modules with standardized input–output interfaces. Modules can be deterministic, stochastic, rule-based, rate-coded or spiking neural models, as long as they respect these interfaces. This “toolbox-like” modularity allows researchers to:

- swap modules in and out,
- compare competing hypotheses for the same function (e.g. different chemotaxis strategies), and
- extend existing larva models with minimal changes to the surrounding architecture.

### Energetics, metabolism and foraging phenotypes

To bridge fast neural control with slower developmental and homeostatic processes, Larvaworld can couple the behavioral architecture to an energetics model based on Dynamic Energy Budget (DEB) theory.

The DEB model:

- tracks energy reserve dynamics across the larval life stage,
- takes into account rearing substrate, nutrient composition and periods of starvation or partial deprivation, and
- produces realistic growth curves (body length, wet weight, instar durations, time to pupation) under given feeding histories.

This energetics state can feed back onto behavior via a hunger/satiety-like drive that depends on energy reserve density. As a result, foraging strategies such as *rovers* vs *sitters* emerge by differential configuration of nutrient absorption and related parameters, modulating the exploration–exploitation balance at the behavioral level.

In the architectural diagram (Fig. 1), these elements appear as the “Body / Physics / Segmentation / Energetics / Metabolic state” block, linked to locomotory, reactive and adaptive behavior, and ultimately to the larva’s interactions with the environment.

---

## Environments, substrates and larva groups

Larvae in Larvaworld move in explicit arenas, not in abstract state spaces. The environment component of the architecture defines where they live, what they sense and what they can consume.

### Arenas and sensory landscapes

The simulation environment is a 2D arena with configurable shape and size (e.g. Petri dishes or other spatial layouts). Within this arena, the framework allows the placement of:

- odor and food sources,
- impassable borders and internal barriers, and
- other objects such as maze walls or compartment boundaries.

On top of the physical arena, **sensory landscapes** can be defined:

- **Odorscapes** generated by odor sources (static or dynamic),
- **Thermoscapes** representing thermal gradients, and
- **Windscapes** for airflow and wind direction.

These landscapes can be tuned in intensity, spatial distribution and temporal dynamics, allowing direct in silico counterparts of standard chemotaxis, thermotaxis and foraging assays.

### Substrates and food distributions

Experimental rearing and assay conditions often rely on defined substrates with known nutrient composition (e.g. standard medium, PED-tracker medium, cornmeal or sucrose-based substrates). Several such substrates, together with their compound densities, are explicitly implemented in Larvaworld.

Food can be:

- uniformly distributed over the arena,
- arranged in one or more patches, or
- stored in a grid where each cell holds a given amount of food that can be gradually depleted.

Each food source is associated with a substrate type, nutritional quality and quantity, which are read by the DEB model and converted into growth and energy dynamics.

### Larva groups, individuality and life history

Virtual larvae are organized into **groups**, reflecting how real experiments are structured (e.g. different genotypes, feeding states or rearing conditions). Each group is defined by:

- the number of larvae,
- the spatial distribution of initial positions (center, scale, shape and placement within that shape),
- the initial orientation distribution,
- a shared life history (age post-hatch, rearing substrate and deprivation periods), and
- additional traits such as color and odor signature.

Life history is simulated by running the DEB model up to a specified age on a given substrate before the behavioral experiment starts. Periods of food deprivation during rearing or during the assay itself can be imposed to match experimental protocols.

Groups can also be tied to **reference datasets**: parameters may be sampled from empirical distributions, preserving inter-individual variability or constructing “average” individuals, depending on the chosen sampling mode. This supports realistic virtual populations and controlled tests of individuality and variability, as explored in the Results section of the paper.

---

## Agent-based simulation and nested timescales

The dynamic behavior of larvae and their environments is simulated using an agent-based modeling (ABM) approach. Larvae are agents that act, sense and update their internal state at discrete timesteps, interacting with the arena and with each other.

### ABM backbone

Larvaworld builds on the Python ABM library **agentpy**. The framework reuses and extends agentpy’s `Model`, `Space` and `Object` classes, adding nested-dictionary parameterization and simulation workflow structures tailored to modular biological agents.

This ABM backbone provides:

- flexible scheduling of agent actions,
- clear separation between agents and environment, and
- efficient step-wise data retrieval for simulation outputs.

### Multiple timescales

A defining feature of Larvaworld is its **multi-timescale modeling**:

- fast neuronal or synaptic processes (e.g. a spiking mushroom body model) can be simulated at sub-millisecond resolution (e.g. 0.1 ms),
- closed-loop behavioral dynamics (locomotion and navigation) typically run at sub-second resolution (e.g. 0.1 s timestep), and
- energetic and life-history processes (DEB) evolve across circadian timescales in the background.

These nested timescales make it possible to study how slow metabolic constraints and learning processes shape fast behavioral decisions and trajectories.

### Simulation modes and experimental protocols

Simulation modes in Larvaworld correspond to different use cases and appear explicitly in the architectural diagram as “Setup” and “Agent-based simulation” components. The main modes include:

- **Single experiments**: one configuration of environment, larva groups and analysis pipeline, mirroring a single behavioral assay.
- **Genetic algorithm (GA) optimization**: repeated simulations used to optimize model parameters for a given task or dataset.
- **Model evaluation**: standardized comparison of one or more models against a reference experimental dataset.
- **Experiment replay**: visualization and analysis of previously recorded experiments (real or simulated).

Each simulation is fully specified by a nested parameter set (environment, larvae, timescales, termination conditions, analysis options) that can be stored under a unique ID and reused.

---

## Data pipeline and LarvaDataset

A central design decision in Larvaworld is that **simulated and experimental datasets are treated identically**. This is reflected in the “Larva Datasets / Experimental locomotory data / Data processing / Model evaluation” part of Fig. 1.

### Standardized dataset structure

All datasets, whether generated by simulations or imported from tracking software, are stored as instances of a common `LarvaDataset` class with three main components:

1. **Time-series data**  
   A multi-indexed Pandas DataFrame, indexed by timestep and larva ID. Initially it contains only primary tracked parameters – typically 2D coordinates of the centroid, midline points and possibly contour points. Derived metrics are added as new columns during processing.

2. **Endpoint metrics**  
   A per-larva DataFrame indexed by larva ID, storing single-valued measurements such as total distance, mean velocity, bout statistics or dominant frequencies. Like the time series, this table is progressively enriched as analysis proceeds.

3. **Metadata**  
   A nested dictionary describing experimental conditions, tracking parameters, animal groups and storage paths.

The dataframes are saved in HDF5 files under different keys, and metadata are stored as configuration files. Datasets can be registered under unique IDs to facilitate reuse in evaluation, optimization and visualization tasks.

### Unified processing pipelines

To ensure unbiased comparisons, Larvaworld only imports **primary** tracked quantities (2D coordinates) from experimental datasets. All secondary metrics are defined and computed within the platform using shared pipelines that apply equally to simulated and experimental data.

Three sequential stages are available:

1. **Preprocessing**

   - spatial scaling and unit conversion,
   - coordinate transposition and alignment (e.g. to arena center or common origin),
   - interpolation of missing data,
   - conditional exclusion of data segments (e.g. collisions),
   - low-pass filtering at a configurable cut-off frequency.

2. **Processing**

   - angular analysis (bending and orientation angles, angular velocity and acceleration),
   - spatial metrics (distance, velocity, acceleration and forward components),
   - dispersal over time windows,
   - trajectory tortuosity in sliding temporal windows,
   - odorscape navigation metrics (instantaneous concentration, perceived changes, distance and bearing to sources),
   - preference indices for olfactory preference experiments.

3. **Annotation and bout analysis**

   - detection of strides, crawl-runs, crawl-pauses and turns,
   - distribution fitting for bout durations and lengths (e.g. power-law, exponential, log-normal),
   - computation of spatial and angular changes per bout,
   - estimation of crawling frequency.

This standardized pipeline is the basis for all downstream analyses and model-evaluation procedures.

### Importing experimental locomotion data

Larvaworld supports import from several lab- and tracker-specific formats, including datasets from Schleyer, Jovanic, Berni and Arguello laboratories with different frame rates and midline/contour resolutions. For each format, the platform defines conversion rules that map the raw tracker output into the `LarvaDataset` structure.

Import arguments control which tracks are included (e.g. minimum duration, time windows, maximum number of animals). Once imported, these datasets pass through the same preprocessing, processing and annotation stages as simulation outputs.

---

## Visualization and interactive tools

The “Visualization / Interactive display / Media generation / Experiment replay” part of the architecture captures Larvaworld’s real-time and offline visualization capabilities.

Behavioral simulations and replays are visualized using the `pygame` library:

- larvae are shown in a 2D arena with a spatial scale bar and timer,
- midline and contour, head or centroid markers can be toggled on and off,
- trajectories can be displayed with adjustable history length,
- larvae can be colored by ID, randomly or according to behavioral/kinematic quantities, and
- sensory landscapes (e.g. odorscapes) can be rendered on the arena.

Interactive controls allow zooming, panning, selecting and locking onto individuals, adding or removing larvae and objects, and capturing snapshots or videos. Replays of imported experiments can realign tracks to a common origin or arena center, and experimental tracks can be rendered as segmented virtual bodies to visually match simulated larvae.

Beyond the pygame window, Larvaworld offers **web-based applications** launched via `larvaworld-app`. These include:

- an **Experiment Viewer** for browsing and launching preconfigured experiments,
- **Larva Models** and **Locomotory Modules** viewers for inspecting the modular behavioral architecture, and
- a **Track Viewer** for visualizing stored datasets.

These tools are based on the HoloViz ecosystem and expose the param-based configuration of models and environments via dynamic widgets, making exploration and configuration accessible from the browser and Jupyter notebooks.

---

## Model optimization and evaluation

The outer loop of the architecture – “Genetic Algorithm Optimization / Model evaluation” – provides systematic ways to tune models and quantify how well they reproduce real behavior.

### Genetic algorithm optimization

The **GA mode** performs parameter optimization for larva models. It takes three main groups of settings:

- a selection algorithm (population size, number of generations, selection and mutation rules),
- a parameter space (which model parameters are allowed to vary and their ranges), and
- a performance evaluation scheme (fitness function).

Fitness functions can be defined externally or constructed by the GA engine itself, for example by comparing simulation outputs to a reference dataset. This process yields optimized model configurations that can later be used in comparative simulations and evaluation studies.

### Model evaluation against experimental data

The **evaluation mode** compares virtual larva groups to empirical datasets using the unified analysis pipeline described above. Given that simulated and experimental datasets share structure and metrics, the platform can compute distances between their distributions using, for example, Kolmogorov–Smirnov distances over endpoint metrics and time-series derived measures.

Evaluation metrics span:

- angular kinematics (e.g. bending angles, angular velocities),
- spatial displacement (e.g. path length, dispersal, tortuosity), and
- temporal dynamics (e.g. run and pause durations, crawling frequencies).

Rather than collapsing everything into a single “score”, Larvaworld exposes a broad panel of metrics, giving users direct insight into which aspects of behavior a given model captures well and where it diverges from real animals.

---

## From virtual lab to scientific applications

Larvaworld is distributed as an open-source Python package under the **MIT license** and has already been applied in several scientific studies, including:

- analysis of locomotion under different feeding and hydration states,
- development and evaluation of modular locomotory architectures with intermittent behavior,
- integration of a spiking mushroom body model for olfactory learning, and
- comparative thermotaxis simulations across *Drosophila* species.

These applications illustrate the range of use cases supported by the platform: from pure data analysis of imported datasets, through in silico replication of standard assays, to hybrid models that combine detailed neural circuits, realistic energetics and richly structured environments.

The following sections of the documentation describe how to install Larvaworld, explore the tutorial experiments, and build or extend larva models using the tools outlined in this architectural overview.
