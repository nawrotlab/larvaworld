# larvaworld

<p align="center">
  <a href="https://github.com/nawrotlab/larvaworld/actions/workflows/ci.yml?query=branch%3Amaster">
    <img src="https://img.shields.io/github/actions/workflow/status/nawrotlab/larvaworld/ci.yml?branch=master&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://larvaworld.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/larvaworld.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/nawrotlab/larvaworld">
    <img src="https://img.shields.io/codecov/c/github/nawrotlab/larvaworld.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json" alt="Poetry">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/larvaworld/">
    <img src="https://img.shields.io/pypi/v/larvaworld.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/larvaworld.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/larvaworld.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://larvaworld.readthedocs.io" target="_blank">https://larvaworld.readthedocs.io</a>  
**Source Code**: <a href="https://github.com/nawrotlab/larvaworld" target="_blank">https://github.com/nawrotlab/larvaworld</a>

---

A virtual lab for *Drosophila* larva behavioral modeling and analysis.

## Overview

Larvaworld is an open-source Python framework and virtual laboratory for *Drosophila melanogaster* larval behavior. It combines agent-based modeling with multiscale neural control and supports analysis of both simulated and experimental motion-tracking data.

Virtual larvae are implemented as 2D agents capable of realistic locomotion, guided by multimodal sensory input and constrained by a dynamic energy-budget model that regulates exploration–exploitation trade-offs. Each agent is organized as a hierarchical behavior-based control system with modular layers, allowing competing models to be assembled, optimized and compared under standardized input–output specifications.

Experimental locomotion datasets can be imported and automatically converted into a standardized format that is identical to the format produced by simulations. Only the originally tracked 2D coordinates are taken as input, while all derived kinematic and behavioral metrics are computed within the framework using transparent, configurable analysis pipelines. This design ensures that simulated and experimental datasets share the same structure and can be directly compared.

Simulations operate across sub-millisecond neuronal dynamics, sub-second closed-loop behavior and circadian-scale metabolic regulation. Preconfigured experiments cover a broad spectrum of established larval assays (exploration, chemotaxis, olfactory learning and odor preference, foraging and growth, phenotypic comparisons such as rovers vs sitters, maze navigation and games). The same infrastructure is used to replay and analyze real experiments, enabling standardized and rigorous model evaluation.

## Installation

Install from PyPI:

```shell
pip install larvaworld
```

If you plan to export simulations or replays as video files, install the ffmpeg extension of `imageio`:

```shell
pip install "imageio[ffmpeg]"
```

## Features

### Arena drawing

Larvaworld includes an arena editor that supports detailed configuration of behavioral environments:

1. **Arenas and dishes**
   The simulation environment is designed as a 2D arena with configurable shape and dimensions (e.g. Petri dishes and other layouts). Larva groups, odor sources, food sources and obstacles can be placed at specified locations, including predefined spatial distributions and orientations.

2. **Odorscapes and other sensory landscapes**
   Odor sources can be defined and used to construct arbitrary olfactory landscapes (*odorscapes*). The environment can also host other sensory gradients such as thermal (*thermoscape*) and wind (*windscape*). Virtual larvae themselves can bear an odor label, thereby generating dynamic odorscapes as they move.

3. **Food items and nutritive substrates**
   Food sources can be specified as single items, spatial distributions of defined parameters, or food grids of selected dimensions. Established nutritious substrates with documented compound composition are available and can be used both for rearing and foraging experiments.

4. **Impassable borders and barriers**
   Impassable borders and internal barriers can be added to constrain larval movement and to reproduce typical experimental configurations (e.g. confined dishes, mazes, compartmentalized arenas).

### Larva models

Multiple aspects of real larval behavior are captured in modular larva models that can be configured in detail and directly tested in simulations. The main components are:

* **Virtual body**
  The larval body is represented as a 2D object consisting of one, two (default) or more segments, with viscoelastic coupling (torsional spring model). Olfactory and mechanosensory receptors can be placed at specified locations, and a mouth region is defined for feeding. Exemplary models with angular and linear motion fitted to empirical tracking data are provided, including differential motion of front and rear segments and realistic velocities and accelerations. For multisegment models, the body and arena can optionally be simulated in a Box2D physics engine.

* **Sensorimotor effectors**
  Crawling, lateral bending and feeding are modeled as oscillatory processes. These oscillators can operate independently, be coupled, or be mutually exclusive, with configurable interference depending on body phase. An olfactory sensor enables chemotactic navigation by tracking odor gradients. Feedback from the environment is supported for specific behaviors, such as recurrent feeding motion when encountering food.

* **Intermittent behavior**
  Intermittent operation of the locomotory oscillators can be defined via spatial or temporal distributions, enabling models in which empirically fitted crawling bouts are interspersed with pauses. Behavioral time is effectively quantized at the scale of individual crawling or feeding motions. An intermittent coupled-oscillator model is available as a default locomotory model for many behavioral simulations.

* **Olfactory learning**
  A neuron-level mushroom body (MB) model can be integrated into the behavioral controller to implement olfactory associative learning. The MB is simulated at sub-millisecond resolution (e.g. 0.1 ms) and coupled to the 0.1 s behavioral timestep in parallel simulation, enabling standard olfactory learning protocols (e.g. train & test, tests on or off food).

* **Energetics and life-history**
  A Dynamic Energy Budget (DEB) model governs energy allocation to growth and maintenance across the larval life stage. The model runs in the background at a circadian timescale, coupled to the behavioral simulation, and has been calibrated to reproduce realistic growth curves (body length, wet weight, instar durations, time to pupation) under defined rearing conditions and nutritional histories (including starvation or partial deprivation).

* **Hunger drive and foraging phenotypes**
  The DEB energetics module can be linked to behavior via a hunger/satiety-like drive that depends on energy reserve density. This drive modulates the exploration–exploitation balance and supports discrete foraging phenotypes such as *rovers* and *sitters* by differential configuration of nutrient absorption and related parameters.

### Behavioral simulations

The simulation platform supports virtual experiments that mirror established larval behavioral paradigms and extends them with additional scenarios. Preconfigured experiments include, among others:

* **Exploration**
  Free exploration in non-nutritious arenas, at different spatial scales (single-larva close-up, full-dish exploration, dispersion from the center).

* **Chemotaxis and local search**
  Navigation up odor gradients and exploration around an odor source, implementing assays such as centrally placed larvae and odor sources or opposite-side placements.

* **Olfactory learning and odor preference**
  Associative learning paradigms with distinct training and test phases, including variations with or without food during the test.

* **Feeding and foraging**
  Foraging in patchy or uniform food environments, including arena substrates with known compound composition, and growth/rearing experiments over the entire larval stage.

* **Foraging phenotypes**
  Comparative simulations of rovers vs sitters under diverse environmental conditions, linking foraging behavior to underlying metabolic parameters.

* **Realistic body and dataset imitation**
  Experiments using multisegment larvae in a physics engine for realistic body dynamics, as well as imitation of specific experimental datasets.

* **Maze and games**
  Simulate maze experiments and the capture-the-flag mini-game described in the paper, where two larva groups compete to capture a highly-valenced, centrally placed odor source and carry it back to their respective bases.

Each experiment type can be used as a standalone simulation, combined into essays spanning multiple conditions, or embedded in batch runs and optimization procedures.

### Data import & Behavioral analysis

Larvaworld is designed to both simulate and analyze *Drosophila* larva motion-tracking experiments. Recorded data from different setups can be imported and converted into a standardized internal format. The core container is the `LarvaDataset` class, shared by experimental and simulated data, with three main components:

* **Time-series data**
  Multi-indexed Pandas DataFrame, indexed by (time, larva ID). Initially it contains only the primary tracked parameters (x–y coordinates of at least the centroid, often several midline points and optionally body contour points). The DataFrame is progressively enriched with derived parameters during processing.

* **Endpoint metrics**
  Per-larva summary metrics computed once per agent at the end of the simulation or recording, stored in a separate DataFrame indexed by larva ID. This table is similarly enriched with summary statistics derived from the time series.

* **Metadata**
  A nested dictionary describing experimental conditions, tracking parameters, animal groups, and storage paths.

DataFrames are stored in HDF files under different keys (e.g. `step`, `midline`, `contour`, `angular`, `dspNtor`), and metadata are stored as configuration text files. Datasets can be registered as reference datasets under a unique ID for streamlined reuse in model evaluation, visualization and replay.

To enhance compatibility and reproducibility, only primary tracked quantities (2D coordinates) are imported. All secondary metrics are defined within larvaworld and computed by a standardized pipeline that can be applied identically to simulated and experimental data.

Three main analysis stages are available:

1. **Pre-processing**

   * spatial scaling and unit conversion,
   * transposition and alignment (e.g. to arena center or common origin),
   * interpolation of missing data,
   * conditional exclusion of selected intervals or tracks (e.g. collisions),
   * low-pass filtering at a configurable cut-off frequency.

2. **Processing**

   * **Angular analysis**: bending and orientation angles, angular velocity and acceleration, with options for segment-wise or vector-based definitions (front/rear body vectors).
   * **Spatial analysis**: distance, velocity, acceleration and their components along the forward orientation axis.
   * **Dispersal**: spatial dispersal over specified time windows.
   * **Trajectory tortuosity**: tortuosity measures in sliding windows of configurable duration.
   * **Odorscape navigation**: instantaneous odor concentration, perceived concentration changes, distance and bearing to odor or food sources.
   * **Preference indices**: metrics for olfactory preference experiments.

3. **Annotation & bout analysis**

   * detection of **strides**, **crawl-runs**, **crawl-pauses** and **turns** (based on reorientation amplitude or changes in angular velocity/bending),
   * fitting of distributions (e.g. power-law, exponential, log-normal) to bout durations or lengths,
   * computation of spatial and angular changes during bouts,
   * estimation of crawling frequency.

The same pre-processing, processing and annotation steps are applied to simulated and imported datasets, enabling consistent and transparent comparison.

### Visualization

Both imported experiments and simulations can be visualized in real time at a realistic spatial scale. The visualization system (based on `pygame`) provides:

* a 2D arena view with timer and scale,
* zooming and panning,
* selection and locking on specific individuals,
* toggling of larval IDs, midline and contour, and explicit marking of head or centroid,
* configurable trajectory traces with adjustable duration,
* color schemes for larvae (default, random, or behavior/kinematics-based),
* visualization of sensory landscapes (e.g. odorscapes),
* interactive manipulation of the arena (adding/removing larvae, sources, borders),
* snapshot capture and video export, including collapse of all frames onto a single overlay image.

Replays of simulated or imported datasets can restrict the set of larvae, time ranges and arenas. Tracks can be transposed to the arena center or aligned to a common origin to facilitate visual inspection of dispersal patterns. Experimental tracks can be rendered as segmented virtual bodies, making them visually comparable to simulated larvae.

## Command Line Interface

The platform is mainly accessed via the command line interface (CLI) using the `larvaworld` command. The first positional argument selects the simulation mode, and some modes accept an additional argument (e.g. a predefined experiment ID). Mode-specific arguments can then be used to overrule configuration parameters.

The available simulation modes include:

* `Exp` – single experiment,
* `Batch` – batch runs of an experiment (advanced/experimental feature),
* `Ga` – genetic algorithm optimization,
* `Replay` – replay of existing datasets,
* `Eval` – model evaluation against real data.

### Single Simulation

Run a single simulation of one of multiple available experiments. Optionally run the associated analysis pipeline.

Each of the following commands runs a dish simulation (30 larvae, 3 minutes) and records a video:

```shell
larvaworld Exp dish -N 30 -duration 3.0 -vis_mode video
larvaworld Exp patch_grid -N 30 -duration 3.0 -vis_mode video
```

This command runs a dispersion simulation and compares the results to an existing reference dataset, producing only a final image:

```shell
larvaworld Exp dispersion -N 30 -duration 3.0 -vis_mode image -a
```

### Batch runs

Run multiple trials of a given experiment with different parameters.

Example: batch odor-preference experiments with different valences of the two odor sources:

```shell
larvaworld Batch PItest_off -N 5 -duration 1.0
```

> **Note**: Batch runs are primarily intended for advanced parameter sweeps and are mostly exercised via the Python API (`BatchRun`). CLI support is still evolving, so some use cases may require custom configuration or direct Python use.

### Genetic Algorithm optimization

Run a genetic algorithm (GA) optimization to adjust a model configuration according to a fitness function. This is typically used to obtain optimally parameterized models before comparative studies.

Example: optimize a locomotory model for kinematic realism against a reference experimental dataset:

```shell
larvaworld Ga realism \
  -refID exploration.30controls \
  -Nagents 20 \
  -duration 0.5 \
  -bestConfID GA_test_loco \
  -init_mode model
```

### Experiment replay

Replay real-world experiments or previously stored simulations.

Example: replay a registered reference experimental dataset (as imported by a tutorial such as `import_Schleyer`):

```shell
larvaworld Replay -refID exploration.30controls -vis_mode video
larvaworld Replay -refDir SchleyerGroup/processed/exploration/30controls -vis_mode video
```

### Model evaluation / comparison to real data

Evaluate different model configurations against real data using a standardized set of metrics.

Example: evaluate two models against a reference experimental dataset:

```shell
larvaworld Eval \
  -refID exploration.30controls \
  -modelIDs RE_NEU_PHI_DEF RE_SIN_PHI_DEF \
  -N 10
```

## Web Apps

A number of web-based Larvaworld applications are available to facilitate inspection, configuration and real-time visualization:

* **Experiment Viewer** – inspect and launch preconfigured experiments,
* **Larva Models** – inspect and visualize modular larva models,
* **Locomotory Modules** – inspect and test behavioral modules in isolation,
* **Track Viewer** – visualize stored datasets.

The available applications can be launched from a single start page.

Start the web server with:

```shell
larvaworld-app
```

Then open [http://localhost:5006](http://localhost:5006) in your browser (if it does not open automatically).

## GUI (deprecated)

A desktop graphical user interface (GUI) was originally provided to support data import, inspection and analysis; configuration of models, life history and environments; visualization and data-acquisition setup; and control of simulations, essays and batch runs via dedicated tabs.

The legacy GUI is no longer actively maintained and its entry point is disabled in the current PyPI package. For current workflows we recommend using the command-line interface (`larvaworld`) and the web-based applications (`larvaworld-app`).

## Repository structure

The main components of the repository are organized as follows:

```text
larvaworld/
├── src/larvaworld/        # Main source code
│   ├── cli/               # Command-line interface entry points
│   ├── dashboards/        # Web-based apps (larvaworld-app)
│   ├── gui/               # Legacy desktop GUI (deprecated)
│   └── lib/               # Core library (models, simulation, data, plotting)
├── tests/                 # Test suite (pytest)
├── docs/                  # Sphinx documentation (Read the Docs)
├── pyproject.toml         # Poetry configuration and dependencies
└── .github/workflows/     # CI configuration (GitHub Actions)
```

For details on the public API and module-level organization, see the [online documentation](https://larvaworld.readthedocs.io).

## Development installation

To work on larvaworld locally (e.g. for development, running the full test suite, or building the documentation), it is recommended to use [Poetry](https://python-poetry.org/) and a dedicated virtual environment.

Clone the repository and install all core and development dependencies:

```shell
git clone https://github.com/nawrotlab/larvaworld.git
cd larvaworld

# Install main + development + docs dependencies
poetry install --with dev,docs
```

If you plan to use optional components (e.g. Nengo-based neural modules or Box2D physics), you can install the corresponding extras (names as defined in `pyproject.toml`), for example:

```shell
# Example: install with optional Nengo and Box2D dependencies
poetry install --with dev,docs,nengo,box2d
```

It is recommended to enable the pre-commit hooks that enforce basic formatting and linting:

```shell
poetry run pre-commit install
```

You can then run the application and tools from within the Poetry environment, for example:

```shell
poetry run larvaworld Exp dish -N 30 -duration 3.0 -vis_mode video
poetry run larvaworld-app
```

## Testing

Larvaworld includes an automated test suite (pytest-based) that is exercised in the continuous integration (CI) workflow and can also be run locally.

### Running the tests

From the project root, with the Poetry environment active:

```shell
# Run the full test suite
poetry run pytest
```

Depending on the local environment and installed extras, some test groups may require optional dependencies or external data. Pytest markers are used to distinguish between different classes of tests (e.g. slower tests, tests requiring network access or tests depending on optional libraries). The current marker configuration is documented in `pyproject.toml`.

Typical usage patterns include:

```shell
# Example: run only quick tests (marker configuration-dependent)
poetry run pytest -m "not slow"

# Example: run tests in a specific module
poetry run pytest tests/integration/process/test_import_aux.py
```

For details on the test layout and any additional markers used in CI, please refer to the `tests/` directory and the CI configuration under `.github/workflows/`.

## Supporting resources

Larvaworld builds on a number of established Python libraries:

* **Agent-based modeling** – [agentpy](https://agentpy.readthedocs.io/en/latest/index.html) for core ABM primitives (agents, spaces, experiments).
* **Numerics & data** – `numpy`, `pandas`, `scipy`, `scikit-learn` for numerical computation, data handling and basic statistics.
* **Storage & I/O** – [PyTables](https://www.pytables.org/) (`tables`) for HDF5-based storage of larva datasets (`data.h5`), plus standard CSV/PNG/PDF exports.
* **Visualization** – [pygame](https://pypi.org/project/pygame/) for real-time arena visualization; [Holoviz](https://holoviz.org/) stack (`param`, `panel`, `hvplot`, `holoviews`) for web-based apps and interactive configuration.
* **Energetics** – [Dynamic Energy Budget (DEB) theory](http://www.debtheory.org/wiki/index.php?title=Main_Page) as the conceptual basis for the energetics/homeostasis modules.
* **Physics (optional)** – [Box2D](https://box2d.org/) via [box2d-py](https://pypi.org/project/box2d-py/) for multi-segment body dynamics and realistic contacts.
* **Neural modeling (optional)**
  * [Nengo](https://www.nengo.ai/) for embedded spiking neural modules (e.g. `NengoBrain` and Nengo-based effectors).
  * [Brian2](https://brian2.readthedocs.io/) for **remote** neural simulations of mushroom body (MB) memory and OSN sensory transduction via the `RemoteBrianModelMemory` and `OSNOlfactor` modules (requires a separate Brian2 server; see the examples and documentation).

### Optional dependencies

Some functionality is only available when additional libraries are installed:

* **Nengo integration** – install [`nengo`](https://www.nengo.ai/) to use Nengo-based brain and effector modules.
* **Box2D physics** – install [`box2d-py`](https://pypi.org/project/box2d-py/) to enable multi-segment body physics.
* **Brian2 remote models** – install [`brian2`](https://brian2.readthedocs.io/) and run a Brian2-based remote server to use the remote MB and OSN modules.

With Poetry (development setup), optional groups are configured in `pyproject.toml`, for example:

```shell
poetry install --with dev,docs,nengo,box2d
```

With a plain `pip`-based environment, you can install optional libraries directly, e.g.:

```shell
pip install nengo brian2 box2d-py
```

## Scientific publications

If you use larvaworld in scientific work, please cite:

* **Larvaworld: A behavioral simulation and analysis platform for Drosophila larva**  
  Panagiotis Sakagiannis, Hannes Rapp, Tihana Jovanic, Martin Paul Nawrot  
  *bioRxiv* 2025.06.15.659765; doi: <https://doi.org/10.1101/2025.06.15.659765>  
  *Cite as:* Sakagiannis Panagiotis, Rapp Hannes, Jovanic Tihana, Nawrot Martin Paul (2025) Larvaworld: A behavioral simulation and analysis platform for Drosophila larva. *bioRxiv* 2025.06.15.659765. <https://doi.org/10.1101/2025.06.15.659765>

* **A behavioral architecture for realistic simulations of Drosophila larva locomotion and foraging**  
  Sakagiannis Panagiotis, Jürgensen Anna-Maria, Nawrot Martin Paul (2025)  
  A behavioral architecture for realistic simulations of Drosophila larva locomotion and foraging. *eLife* 14:RP104262.  
  doi: <https://doi.org/10.7554/eLife.104262.1>

## Contributors

![GitHub contributors](https://img.shields.io/github/contributors/nawrotlab/larvaworld?style=flat-square)

<a href="https://github.com/nawrotlab/larvaworld/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nawrotlab/larvaworld" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

Contributions of any kind are welcome!

## Credits

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
