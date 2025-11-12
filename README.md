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

**Documentation**: <a href="https://larvaworld.readthedocs.io" target="_blank">https://larvaworld.readthedocs.io </a>

**Source Code**: <a href="https://github.com/nawrotlab/larvaworld" target="_blank">https://github.com/nawrotlab/larvaworld </a>

---

A virtual lab for Drosophila larva behavioral modeling and analysis.

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install larvaworld
```

If you plan to export simulations as video files you also need to install the ffmpeg extension of imageio:

```shell
pip install 'imageio[ffmpeg]'
```

## Features

### Arena drawing

The platform features an arena editor that supports :

1.  Arenas and dishes
    The arena editor allows defining arena shape and dimensions in detail and placement of larva groups and items at preferred locations in predefined spatial distributions and orientations.
2.  Odorscapes
    Odor sources can be specified and arbitrary odor landscapes can be constructed. The constructed arenas are directly available for modeling simulations. The virtual larvae themselves can bear an odor creating dynamic odorscapes while moving.
3.  Food items
    Food sources are available either as single items, distributions of defined parameters or food grids of defined dimensions.
4.  Impassable borders.

### Larva models

Multiple aspects of real larvae are captured in various models. These can be configured through the GUI at maximum detail and directly tested in simulations. Specifically the components are:

- **Virtual body** : The 2D body consists of 1, 2(default) or more segments, featuring viscoelastic forces (torsional spring model), olfactory and touch sensors at desired locations and a mouth for feeding. Exemplary models with angular and linear motion optimized to fit empirical data are available featuring differential motion of the front and rear segments and realistic velocities and accelerations at both plains. Furthermore, optional use of the Box2D physics engine is available as illustrated in an example of realistic imitation of real larvae with a multi-segment body model.

- **Sensorimotor effectors** : Crawling, lateral bending and feeding are modeled as oscillatory processes, either independent, coupled or mutually exclusive. The individual modules and their interaction are easily configurable through the GUI. Body-dependent phasic interference can be defined as well. An olfactory sensor dynamically tracks odor gradients enabling chemotactic navigation. Feedback from the environment is only partially supported as in the case of recurrent feeding motion at successful food encounter.

- **Intermittent behavior** : Intermittent function of the oscillator modules is available through definition of specific spatial or temporal distributions. Models featuring empirically-fitted intermittent crawling interspersed by brief pauses can be readily tested. Time has been quantized at the scale of single crawling or feeding motions.

- **Olfactory learning** : A neuron-level detailed mushroom-body model has been integrated to the locomotory model, enabling olfactory learning after associative conditioning of novel odorants to food. The short neuron-level temporal scale (0.1 ms) has been coupled to the 0.1 s behavioral timestep in parallel simulation. Detailed implementations of an established olfactory learning behavioral paradigm are supported.

- **Energetics and life-history** : A widely-accepted dynamic energy budget (DEB) model runs in the background and controls energy allocation to growth and biomass maintenance. The model has been fitted to Drosophila and accurately reproduces the larva life stage in terms of body-length, wet-weight, instar duration and time to pupation. The long timescale model (in days) has been coupled to the behavioral timescale as well. Therefore, virtual larvae can be realistically reared in substrates of specified quality before entering the behavioral simulation or can be starved for defined periods during or before being tested.

- **Hunger drive and foraging phenotypes** : The DEB energetics module has been coupled to behavior via a variety of model configurations, each based on different assumptions. For example in one implementation a hunger/satiety homeostatic drive that tracks the energy reserve density deriving from metabolism controls the exploration VS exploitation behavioral balance, boosting consumption after food deprivation and vice versa. The rover and sitter foraging phenotypes have been modeled, integrating differential glucose absorption to differential exploration pathlength and food consumption.

### Behavioral simulations

The simulation platform supports simulations of experiments that implement established behavioral paradigms reported in literature. These can be run as single simulations, grouped in essays for globally testing models over multiple conditions and arenas or as batch-runs that allow parameter search and optimization of defined utility metrics. Specifically the behaviors covered are :

- Free exploration
- Chemotaxis
- Olfactory learning an odor preference
- Feeding
- Foraging in patch environments
- Growth over the whole larva stage

Finally, some games are available for fun where opposite larva groups try to capture the flag or stay at the top of the odorscape hill!!!

### Data import & Behavioral analysis

Experimental datasets from a variety of tracker software can be imported and transformed to a common hdf5 format so that they can be analysed and directly compared to the simulated data. To make datasets compatible and facilitate reproducibility, only the primary tracked x,y coordinates are used, both of the midline points and optionally points around the body contour.Compatible formats are text files, either per individual or per group. All secondary parameters are derived via an identical pipeline that allows parameterization and definition of novel metrics.

### Visualization

Both imported experiments and simulations can be visualized real-time at realistic scale. The pop-up screen allows zooming in and out, locking on specific individuals, bringing up dynamic graphs of selected parameters, coloring of the midline, contour, head and centroid, linear and angular velocity dependent coloring of the larva trajectories and much more. Keyboard and mouse shortcuts enable changing parameters online, adding or deleting agents, food and odor sources and impassable borders.

## Command Line Interface

The platform is mainly accessed through the command line interface via the `larvaworld` command.
Five different modes are available. The mode has to declared after the command as a first positional argument. Mode-specific argumants can be declared afterwards :

### Single Simulation

Run a single simulation of one of multiple available experiments.Optionally run the respective analysis.

This line runs a dish simulation (30 larvae, 3 minutes) without analysis.

`larvaworld Exp dish -N 30 -duration 3.0 -vis_mode video`
`larvaworld Exp patch_grid -N 30 -duration 3.0 -vis_mode video`

This line runs a dispersion simulation and compares the results to the existing reference dataset. We choose to only produce a final image of the simulation.

`larvaworld Exp dispersion -N 30 -duration 3.0 -vis_mode image -a`

### Batch run (needs debugging)

Run multiple trials of a given experiment with different parameters.
This line runs a batch run of odor preference experiments for different valences of the two odor sources.

`larvaworld Batch PItest_off -N 5 -duration 1.0`

### Genetic Algorithm optimization

Run a genetic algorith optimization algorithm to optimize a basic model's configuration set according to a fitness function.
This line optimizes a model for kinematic realism against a reference experimental dataset

`larvaworld Ga realism -refID exploration.30controls -Nagents 20 -duration 0.5 -bestConfID GA_test_loco -init_mode model`

### Experiment replay

Replay a real-world experiment.
This line replays a reference experimental dataset (note that this is imported by the example named : import_Schleyer)

`larvaworld Replay -refID exploration.30controls -vis_mode video`
`larvaworld Replay -refDir SchleyerGroup/processed/exploration/30controls -vis_mode video`

### Model evaluation / comparison to real data

Evaluate diverse model configurations against real data.
This line evaluates two models against a reference experimental dataset

`larvaworld Eval -refID exploration.30controls -modelIDs RE_NEU_PHI_DEF RE_SIN_PHI_DEF -N 10`

## Web Apps

A number of web-based applications are available to inspect larva models, test isolated behavioral modules, view replays of stored datasets and launch simulations of behavioral experiments. The apps can be launched via a single webpage by clicking on their respective icons.
Launch the web server :

`larvaworld-app`

Then open http://localhost:5006 in your browser (if not automatically opened).

## GUI (deprecated)

A user-friendly GUI allows easy importation, inspection and analysis of data, model, life-history and environment configuration, visualization and data-acquisition setup and control over simulations, essays and batch-runs. Videos and tutorials are also available. In principle the user shouldn't have to mess with the code at all.
All functionalities are available via the respective tabs.
Launch the GUI :

`larvaworld-gui`

## Supporting resources

- Agent and simulation classes extend on the agent-based modeling library [agentpy](https://agentpy.readthedocs.io/en/latest/index.html).

- The homeostasis/energetics module is based on the [DEB](http://www.debtheory.org/wiki/index.php?title=Main_Page) (Dynamic Energy Budget) Theory

- Optionally, for multi-segment larvae the spatial environment and bodies are simulated through [Box2D](https://box2d.org/) physics engine based on [box2d-py](https://pypi.org/project/box2d-py/) package.

- Optionally neural modules can be implemented using the [Nengo](https://www.nengo.ai/) neural simulator

## Scientific publications

- **A behavioral architecture for realistic simulations of Drosophila larva locomotion and foraging**
  Panagiotis Sakagiannis, Anna-Maria Jürgensen, Martin Paul Nawrot
  bioRxiv 2021.07.07.451470; doi: https://doi.org/10.1101/2021.07.07.451470

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
