==================LARVAWORLD==================

Drosophila larva behavioral analysis and simulation platform

================================================

Publication :

A realistic locomotory model of Drosophila larva for behavioral simulations

Panagiotis Sakagiannis, Anna-Maria Jürgensen, Martin Paul Nawrot

doi: https://doi.org/10.1101/2021.07.07.451470 

GUI
===================
A user-friendly GUI allows easy importation, inspection and analysis of data, model, life-history and environment configuration, visualization and data-acquisition setup and control over simulations, essays and batch-runs. Videos and tutorials are also available. In principle the user shouldn't have to mess with the code at all.

Visualization
===================
Both imported experiments and simulations can be visualized real-time at realistic scale. The pop-up screen allows zooming in and out, locking on specific individuals, bringing up dynamic graphs of selected parameters, coloring of the midline, contour, head and centroid, linear and angular velocity dependent coloring of the larva trajectories and much more. Keyboard and mouse shortcuts enable changing parameters online, adding or deleting agents, food and odor sources and impassable borders.

Arena drawing
===================
The GUI features an arena editor that supports :

   1. Arenas and dishes
      The arena editor allows defining arena shape and dimensions in detail and placement of larva groups and items at preferred locations in predefined spatial         distributions and orientations.
   2. Odorscapes
      Odor sources can be specified and arbitrary odor landscapes can be constructed. The constructed arenas are directly available for modeling simulations. The       virtual larvae themselves can bear an odor creating dynamic odorscapes while moving.
   3. Food items
      Food sources are available either as single items, distributions of defined parameters or food grids of defined dimensions. 
   4. Impassable borders.



Larva models
=====================
Multiple aspects of real larvae are captured in various models. These can be configured through the GUI at maximum detail and directly tested in simulations.  Specifically the components are:


   1. Virtual body
            The 2D body consists of 1, 2(default) or more segments, featuring viscoelastic forces (torsional spring model), olfactory and touch sensors at                     desired locations and a mouth for feeding. Exemplary models with angular and linear motion optimized to fit empirical data are available                           featuring differential motion of the front and rear segments and realistic velocities and accelerations at both plains. Furthermore, optional use                   of the Box2D physics engine is available as illustrated in an example of realistic imitation of real larvae with a multi-segment body model.
   2. Sensorimotor effectors
            Crawling, lateral bending and feeding are modeled as oscillatory processes, either independent, coupled or mutually exclusive. The individual                     modules and their interaction are easily configurable through the GUI. Body-dependent phasic interference can be defined as well. An olfactory                     sensor dynamically tracks odor gradients enabling chemotactic navigation. Feedback from the environment is only partially supported as in the case                 of recurrent feeding motion at successful food encounter.
   3. Intermittent behavior
            Intermittent function of the oscillator modules is available through definition of specific spatial or temporal distributions. Models featuring                   empirically-fitted intermittent crawling interspersed by brief pauses can be readily tested. Time has been quantized at the scale of single                       crawling or feeding motions.
   4. Olfactory learning
            A neuron-level detailed mushroom-body model has been integrated to the locomotory model, enabling olfactory learning after associative                             conditioning of novel odorants to food. The short neuron-level temporal scale (0.1 ms) has been coupled to the 0.1 s behavioral timestep in parallel               simulation. Detailed implementations of an established olfactory learning behavioral paradigm are supported.
   5. Energetics and life-history
            A widely-accepted dynamic energy budget (DEB) model runs in the background and controls energy allocation to growth and biomass maintenance. The model             has been fitted to Drosophila and accurately reproduces the larva life stage in terms of body-length, wet-weight, instar duration and time to                     pupation. The long timescale model (in days) has been coupled to the behavioral timescale as well. Therefore, virtual larvae can be realistically                   reared in substrates of specified quality before entering the behavioral simulation or can be starved for defined periods during or before being                   tested.
   6. Hunger drive and foraging phenotypes
            The DEB energetics module has been coupled to behavior via a variety of model configurations, each based on different assumptions. For example in one             implementation a hunger/satiety homeostatic drive that tracks the energy reserve density deriving from metabolism controls the exploration VS                       exploitation behavioral balance, boosting consumption after food deprivation and vice versa. The rover and sitter foraging phenotypes have been                   modeled, integrating differential glucose absorption to differential exploration pathlength and food consumption. 

            
         
Data import & Behavioral analysis
========================================
Experimental datasets from a variety of tracker software can be imported and transformed to a common hdf5 format so that they can be analysed and directly compared to the simulated data. To make datasets compatible and facilitate reproducibility, only the primary tracked x,y coordinates are used, both of the midline points and optionally points around the body contour.Compatible formats are text files, either per individual or per group. All secondary parameters are derived via an identical pipeline that allows parameterization and definition of novel metrics. 
            


Behavioral simulation
=====================
The simulation platform supports simulations of experiments that implement established behavioral paradigms reported in literature. These can be run as single simulations, grouped in essays for globally testing models over multiple conditions and arenas or as batch-runs that allow parameter search and optimization of defined utility metrics. Specifically the behaviors covered are :

   1. Free exploration
   2. Chemotaxis
   3. Olfactory learning an odor preference
   4. Feeding
   5. Foraging in patch environments
   6. Growth over the whole larva stage

Finally, some games are available for fun where opposite larva groups try to capture the flag or stay at the top of the odorscape hill!!!

-----------------------------------------------------------------------------------------------------------------

Supporting resources
=====================

- Agent and simulation classes extend on the agent-based modeling library [agentpy](https://agentpy.readthedocs.io/en/latest/index.html).

- The homeostasis/energetics module is based on the [DEB](http://www.debtheory.org/wiki/index.php?title=Main_Page) (Dynamic Energy Budget) Theory
 
- Optionally, for multi-segment larvae the spatial environment and bodies are simulated through [Box2D](https://box2d.org/) physics engine based on [box2d-py](https://pypi.org/project/box2d-py/) package.

- Optionally neural modules can be implemented using the [Nengo](https://www.nengo.ai/) neural simulator

-----------------------------------------------------------------------------------------------------------------

**Installation**

The platform comes as a Pypi package. Install easily using pip. This will additionally install all package dependencies. Note that a python version >=3.8 is required :

    pip install larvaworld
    

**Run Larvaworld**

The platform is mainly accessed through the command line interface via the *larvaworld-cli* command.
Five different modes are available. The mode has to declared after the command as a first positional argument. Mode-specific argumants can be declared afterwards :

1. Single Simulation

    Run a single simulation of one of multiple available experiments. 
    Optionally run the respective analysis.

    This line runs a dish simulation (30 larvae, 3 minutes) without analysis. 

        larvaworld-cli Exp dish -N 30 -t 3.0 -m video
        larvaworld-cli Exp patch_grid -N 30 -t 3.0 -m video

    This line runs a dispersion simulation and compares the results to the existing reference dataset (`larvaworld/data/reference`)
    We choose to only produce a final image of the simulation.

        larvaworld-cli Exp dispersion -N 30 -t 3.0 -m image -a

2. Batch run
    (needs debugging)
    Run multiple trials of a given experiment with different parameters.
    This line runs a batch run of odor preference experiments for different valences of the two odor sources.

        larvaworld-cli Batch PItest_off -N 5 -t 1.0

3. Genetic Algorithm optimization

    Run a genetic algorith optimization algorithm to optimize a basic model's configuration set according to a fitness function.
    This line optimizes a model for kinematic realism against a reference experimental dataset

        larvaworld-cli Ga realism -N 20 -t 0.5 -mID1 GA_test_loco -mGA model

4. Experiment replay

   Replay a real-world experiment.
   This line replays a reference experimental dataset (note that this is imported by the tests/data_import/Schleyer/import_Schleyer.py)

        larvaworld-cli Replay -refID exploration.dish03
        larvaworld-cli Replay -dir SchleyerGroup/processed/exploration/dish03

5. Model evaluation / comparison to real data

   Evaluate diverse model configurations against real data.
   This line evaluates two models against a reference experimental dataset

        larvaworld-cli Eval -refID exploration.merged_dishes -mIDs RE_NEU_PHI_DEF RE_SIN_PHI_DEF -N 3

   
A graphical user interface is also available. All functionalities are available via the respective tabs.
Launch the GUI :

      larvaworld-gui
