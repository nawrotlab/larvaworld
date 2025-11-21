Οκ, πάμε κανονικά. Σου δίνω πλήρες, σφιχτό κείμενο για το `theory_overview.md`. Είναι αυτοτελές, ακολουθεί το Fig. 1 και πατάει **αποκλειστικά** σε ό,τι υπάρχει στο paper + README (χωρίς δικές μου ερμηνείες).

---

````md
# Theoretical Overview

Larvaworld is an open-source Python framework and virtual laboratory for *Drosophila melanogaster* larval behavior. It combines agent-based modeling (ABM) with multiscale neural control and supports analysis of both simulated and experimental motion-tracking data. Virtual larvae are implemented as 2D agents capable of realistic locomotion, guided by multimodal sensory input and constrained by a Dynamic Energy Budget (DEB) model that regulates the exploration–exploitation balance across the larval life stage.

The platform was developed to address two central challenges in behavioral neuroscience and computational modeling. First, it provides a shared “virtual laboratory” in which experimental data analysis and behavioral modeling are integrated within the same software environment. Experimental locomotion datasets can be imported and converted into a standardized internal format that is identical to the format produced by simulations, while all derived kinematic and behavioral metrics are computed within Larvaworld using transparent, configurable analysis pipelines. This ensures that simulated and experimental datasets share the same structure and can be analyzed with identical, unbiased procedures.

Second, Larvaworld aims to bridge a long-standing gap in theory building between sub-individual models in neuroscience and supra-individual models in ecology. It focuses on the behaving individual as the central modeling unit and explicitly links fast neural dynamics, closed-loop behavior, and slower energetic and life-history processes. Simulations can span sub-millisecond neuronal timescales, sub-second behavioral control, and circadian-scale metabolic regulation, within environments that reproduce established larval assays and their extensions.

The design of Larvaworld follows four overarching aims:

- **Integration** of established theoretical and modeling principles (ABM, DEB, layered behavioral control, contemporary motion-tracking analysis),
- **User-friendliness** for both behavioral modeling and data analysis,
- **Modularity and extensibility** at all architectural levels (models, environments, experiments, analysis),
- **Computational efficiency and storage management** for large-scale and long-running simulations.

Figure 1 from the companion paper summarizes the architecture. The following sections provide a conceptual tour of its main components: larva models, larva groups, environments, setup and agent-based simulation, data collection and larva datasets, visualization and experiment replay, analysis and model evaluation, and genetic algorithm (GA) optimization. More practical, step-by-step guides are provided elsewhere in the documentation.

---

```{figure} ../figures_tables_from_paper/figures/fig1_architecture.png
:name: fig-architecture
:align: center

Larvaworld architecture. A schematic of the main components and functionalities of the platform.
````

## Larva Model

The larva model block groups the components that define the virtual animal: its body and physical representation, its metabolic and energetic state, and its sensory and behavioral control architecture. Together, these elements determine how the agent moves, senses, and interacts with its environment across nested timescales.

At the lowest level, the larval body is represented as a 2D object that can be tracked with the same conventions used in real experiments (centroid, midline, and optionally body contour). This ensures that simulated larvae can be recorded, processed, and analyzed using exactly the same pipelines as motion-tracking datasets. On top of this body representation, Larvaworld can optionally incorporate multi-segment physics via a Box2D-based engine, enabling realistic body dynamics and contacts that mimic multi-point tracking setups.

Metabolic and energetic processes are modeled using a Dynamic Energy Budget (DEB) formulation. The DEB model tracks post-hatch age, rearing conditions, and nutritional histories (including starvation or partial food deprivation) across the larval stage, and allocates energy to maintenance and growth. It has been calibrated to reproduce realistic growth curves (body length, wet weight, instar durations, and time to pupation) under defined rearing substrates. The same energetics module allows the introduction of discrete foraging phenotypes (such as rovers and sitters) via differential configuration of nutrient absorption and related parameters. A hunger- or satiety-like drive, derived from energy reserve density, can modulate behavior and thereby shape the exploration–exploitation trade-off.

Sensory input is handled through modular sensory channels, including olfaction, mechanoreception, gustation, and thermosensation. Each sensory modality samples the corresponding landscape in the arena (e.g. odorscapes, thermoscapes, windscapes), providing the information that downstream modules use to guide orientation, stopping, turning, and feeding. The same sensory abstractions apply whether the larva is simulated in a purely virtual arena or replays a real recorded trajectory in a reconstructed experimental environment.

Behavioral control follows a layered, behavior-based architecture. At the lowest layer, locomotory effectors generate basic motor routines such as crawling, turning, feeding, and intermittent pausing. Intermediate layers implement reactive behaviors driven by sensory input, such as chemotaxis and local search. Higher layers support adaptive and memory-related processes, including associative learning, reinforcement-like modulation, and state-dependent behavioral switching. Neural control modules can range from simple linear transfer functions to rate-based or spiking neural network models, including remote models for mushroom body memory or olfactory sensory neuron transduction. Once a modular larva model is assembled from these components, it can be used in single experiments, batch simulations, or optimization procedures.

A more detailed description of the larva agent architecture and brain modules is provided in {doc}`../agents_environments/larva_agent_architecture` and {doc}`../agents_environments/brain_module_architecture`.

## Larva Groups

The larva groups block organizes how individual larva models are instantiated, parameterized, and placed within a given experiment. Conceptually, a larva group represents a population of agents that share a common model template and experimental role, but can differ in their detailed parameters to capture individuality and life-history variation.

A group configuration specifies:

* **Larva models**: which larva model class and parameter set to use (including body, sensory, and control modules),
* **Placement**: how many individuals to instantiate, and how to place them in the arena (initial positions, orientations, and spatial distributions),
* **Age and life history**: rearing conditions, post-hatch age, and nutritional history, as encoded in the DEB model,
* **Individuality**: per-agent parameter variation to capture individual differences (e.g. foraging phenotypes, sensory sensitivity, or controller parameters).

These settings support experiments in which multiple groups (e.g. different genotypes, rearing conditions, or phenotypes) are simulated side by side in a shared environment. They also provide the necessary structure for importing and annotating real experiments that involve multiple groups or conditions.

The practical configuration of larva groups within experiments is described in the experiment configuration guides {doc}`experiment_configuration_pipeline` and {doc}`../working_with_larvaworld/single_experiments`.

## Environment

The environment block covers the arenas, substrates, and sensory landscapes within which larva groups are placed. The arena is a 2D spatial domain whose geometry and scale can reproduce standard experimental setups such as Petri dishes or custom layouts with internal barriers.

Environmental configuration includes:

* **Arena geometry**: shape and size of the arena (e.g. dish radius, rectangular bounds), impassable borders, and internal obstacles,
* **Sensory landscapes**: spatial distributions of sensory quantities, such as odorscapes (odor gradients and sources), thermoscapes (temperature gradients), and windscapes (airflow patterns),
* **Sources and obstacles**: discrete food items, food grids, nutritive substrates with known composition, and other objects that influence movement or feeding.

These elements allow Larvaworld to reproduce established behavioral assays (exploration, chemotaxis, foraging, maze navigation) and to extend them to new configurations. The same abstractions are also used to reconstruct real experimental arenas when importing motion-tracking data.

Further details on arenas, substrates, and sensory landscapes are given in {doc}`../agents_environments/arenas_and_substrates`.

## Setup and Agent-Based Simulation

The setup and agent-based simulation block specifies how larva groups and environments are combined into concrete experiments and how these experiments are executed over time. Larvaworld adopts an ABM approach built on top of the `agentpy` package, whose core `Model`, `Space`, and `Object` classes have been adapted to the needs of modular biological agents and nested parameterization.

A simulation setup defines:

* **Trial protocol**: the temporal structure of the simulated experiment (e.g. number of trials, durations, inter-trial intervals, training vs test phases),
* **Nested timescales**: integration timesteps for neural and behavioral dynamics versus the slower updates of the energetics model (e.g. sub-millisecond neural updates vs circadian-scale DEB updates),
* **Termination conditions**: criteria for stopping a trial or experiment (elapsed time, state changes, or user-defined conditions).

During a simulation, agents execute their control loops in a turn-based fashion determined by the ABM scheduler. At each time step, sensory inputs are sampled from the environment; behavioral modules update their internal state; effectors update the larval body; energetics are advanced at their own timescale; and relevant variables are recorded into the data collection pipeline. The same infrastructure supports single experiments, multi-condition essays, and large batch runs.

An overview of simulation modes and configuration options is provided in {doc}`simulation_modes` and {doc}`experiment_configuration_pipeline`.

## Larva Datasets and Data Collection

The data collection and Larva Datasets block forms the bridge between simulations and motion-tracking experiments. Larvaworld is designed so that both simulated and experimental data are represented by a shared container class, **`LarvaDataset`**, which encapsulates time series, endpoint metrics, and metadata in a standardized way.

Each `LarvaDataset` consists of three main components:

* **Time-series data**: a multi-indexed `pandas.DataFrame` indexed by `(time, larva ID)`. Initially, it contains the primary tracked parameters (2D coordinates of at least the centroid, and often additional midline and contour points). This table is incrementally enriched with derived variables (e.g. velocities, turning angles, curvature, bout-level descriptors) as processing stages are applied.
* **Endpoint metrics**: a per-larva `DataFrame` indexed by larva ID, containing summary metrics computed once per agent (e.g. total path length, time spent on food, growth outcomes). This table is also enriched as the analysis pipeline progresses.
* **Metadata**: a nested configuration dictionary describing experimental conditions, group definitions, tracking parameters, and storage paths.

DataFrames are stored in HDF5 files under different keys (e.g. `step`, `midline`, `contour`, `angular`, `dspNtor`), while metadata are stored in configuration text files. Datasets can be registered under unique IDs as reference datasets for later use in model evaluation, visualization, and replay.

To maximize compatibility and reproducibility, imported experimental datasets use only the original 2D coordinates as input; all further kinematic and behavioral metrics are computed inside Larvaworld with the same code paths used for simulated data. This design ensures that simulated and experimental datasets can be directly compared.

A detailed description of the data pipeline, import formats, and reference datasets is provided in {doc}`../data_pipeline/lab_formats_import`, {doc}`../data_pipeline/reference_datasets`, and {doc}`../data_pipeline/data_processing`.

## Visualization and Experiment Replay

The visualization and experiment replay block comprises tools for interactive inspection of simulations and datasets. Larvaworld offers two complementary visualization layers.

First, a `pygame`-based arena viewer supports real-time rendering of simulations and replays. When visualization is enabled, a pop-up window displays the arena, larval bodies, odor and food sources, and arena boundaries at a realistic spatial scale. Simulations can run at real time, slower, or faster (subject to computational limits). Keyboard shortcuts and mouse actions allow users to zoom in and out, lock onto individual larvae, toggle midline and contour displays, and show or hide auxiliary information such as timers and larva IDs.

Second, a set of web-based applications built with the HoloViz ecosystem (e.g. `param`, `panel`, `hvplot`) provides browser-based dashboards for inspecting experiments, larva models, locomotory modules, and stored tracks. These applications allow users to configure models and environments interactively, to launch simulations, and to visualize both simulated and experimental datasets without writing code, and can be accessed through a single start page.

In addition to online visualization, Larvaworld can generate media files (e.g. videos) from simulations and replays for documentation, publication, or teaching.

Details on visualization tools, keyboard controls, and web applications are described in {doc}`../visualization/visualization_snapshots`, {doc}`../visualization/keyboard_controls`, and {doc}`../visualization/web_applications`.

## Analysis and Model Evaluation

The analysis and model evaluation block implements the standardized pipelines that transform raw trajectories into interpretable behavioral metrics and model comparisons. The core idea is that both simulated and experimental data are processed with the same sequence of steps, thereby reducing analysis bias and making model evaluation more rigorous.

Key stages include:

* **Preprocessing**: cleaning and alignment of raw trajectories, interpolation or trimming of missing samples where appropriate, and basic coordinate transformations;
* **Secondary metrics**: computation of derived kinematic and behavioral variables (e.g. speed, curvature, heading, turning events, bouts, occupancy measures);
* **Epoch annotation**: segmentation of trajectories into behaviorally or experimentally defined epochs (e.g. training vs test, pre- vs post-stimulus intervals, on- vs off-food periods);
* **Distribution fitting and statistical analysis**: fitting of distributions to micro-behavioral variables and group-level comparisons under different conditions;
* **Plotting and visualization**: trajectory plots, dispersal measures, time series plots, frequency and polar plots, and other visual summaries at the individual and group levels;
* **Group comparison and model evaluation**: systematic comparison of models against reference datasets, using standardized metrics and summary statistics.

Model evaluation workflows treat `LarvaDataset` instances as first-class objects and allow competing models to be benchmarked against the same experimental reference under identical analysis settings. This supports both qualitative inspection and quantitative scoring of model performance.

Practical examples and API details for analysis and model evaluation are provided in {doc}`../working_with_larvaworld/model_evaluation`, {doc}`../visualization/plotting_api`, and {doc}`../data_pipeline/data_processing`.

## Genetic Algorithm Optimization

The genetic algorithm (GA) optimization block closes the loop between model specification, simulation, and data analysis. Once a modular larva model, environment, and experiment have been defined, Larvaworld can use GA-based procedures to optimize selected parameters with respect to an objective function.

Typical objectives include:

* matching summary statistics or distributions from a reference experimental dataset,
* optimizing performance in a specific virtual task (e.g. chemotactic efficiency, foraging efficiency),
* tuning parameters that control foraging phenotypes (e.g. rovers vs sitters) under defined environmental conditions.

The GA operates over parameter sets stored in the same nested configuration structures that define larva models and experiments. Each candidate configuration is evaluated by running simulations, collecting the resulting `LarvaDataset`, and computing an objective score from the analysis pipeline. The GA then updates the population of parameter sets accordingly.

This optimization machinery allows researchers to link mechanistic models (neural, sensory, or energetic) to behavioral data in a systematic way, while reusing the same simulation and analysis infrastructure.

Advanced usage of the GA tools is covered in {doc}`../working_with_larvaworld/ga_optimization_advanced`.

## From Theory to Practice

The architectural components described above are instantiated in a set of preconfigured virtual experiments that mirror established larval behavioral paradigms and extend them in controlled ways. These experiments cover, among others:

* free exploration in non-nutritious arenas at different spatial scales,
* chemotaxis and local search around odor sources,
* olfactory learning and odor preference with distinct training and test phases,
* feeding and foraging in patchy or uniform food environments,
* phenotypic comparisons such as rovers vs sitters under varied environmental conditions,
* growth and rearing experiments across the full larval stage under defined DEB parameters and nutritional histories,
* maze-like arenas and a capture-the-flag game where larva groups compete to retrieve a central odor source.

Each experiment can be used as a standalone simulation, combined into multi-condition essays, or embedded in batch runs and optimization procedures. The same experiments can also be run in replay mode on real datasets, enabling direct side-by-side comparison between virtual and real larvae.

For a practical entry point to these experiments, see the tutorials index in {doc}`../tutorials/index` and the working-with-Larvaworld guides {doc}`../working_with_larvaworld/single_experiments` and {doc}`architecture_overview`.

## References

For full methodological and theoretical details, including additional examples and applications, see the companion article:

> Sakagiannis Panagiotis, Rapp Hannes, Jovanic Tihana, Nawrot Martin Paul (2025)
> *Larvaworld: A behavioral simulation and analysis platform for Drosophila larva*. bioRxiv 2025.06.15.659765.

Further references on agent-based modeling, DEB theory, and behavioral architectures are listed in the bibliography of that article and are cited throughout this documentation using the `{cite}` directive.

```
```
