==================LARVAWORLD==================

Drosophila larva behavioral analysis and simulation platform

================================================

GUI
===================
A user-friendly GUI allows easy importation, inspection and analysis of data, model, life-history and environment configuration, visualization and data-acquisition setup and control over simulations, essays and batch-runs. Videos and tutorials are also available. In principal the user shouldn't have to mess with the code at all.

Visualization
===================
Both imported experiments and simulations can be visualized real-time at realistic scale. The pop-up screen allows zooming in and out, locking on specific individuals, bringing up dynamic graphs of selected parameters, coloring of the midline, contour, head and centroid, linear and angular velocity dependent coloring of the larva trajectories and much more. Keyboard and mouse shortcuts enable changing parameters online, adding or deleting agents, food and odor sources and impassable borders.

Arena drawing
===================
The GUI features an arena editor where larva groups and items can be placed at prefered locations in predefined spatial distributions and orientations. Odor sources can be specified and arbitrary odor landscapes (odorscapes) can be constructed. The constructed arenas are directly available for modeling simulations.

Behavioral analysis
===================
Experimental datasets from a variety of tracker software can be imported and transformed to a common hdf5 format so that they can be analysed and directly compared to the simulated data. To make datasets compatible and facilitate reproducibility, only the primary tracked x,y coordinates are used, both of the midline points and optionally points around the body contour.Compatible formats are text files, either per individual or per group. All secondary parameters are derived via an identical pipeline that allows parameterization and definition of novel metrics. 

Larva models
=====================
Multiple aspects of real larvae are captured in various models. These can be configured through the GUI at maximum detail and directly tested in simulations.  Specifically the components are:

            1. Virtual body
                The 2D body consists of 1, 2(default) or more segments, featuring viscoelastic forces (torsional spring model), olfactory and touch sensors at                     desired locations and a mouth for feeding. Exemplary models with angular and linear motion optimized to fit empirical data are available                           featuring differential motion of the front and rear segments and realistic velocities and accelerations at both plains. Furthermore optional use                   of the Box2D physics engine is available as illustrated in an example of realistic imitation of real larvae with a multi-segment body model.
            2. Sensorimotor effectors
                Crawling, lateral bending and feeding are modeled as oscillatory processes, either independent, coupled or mutually exclusive. The individual                     modules and their interaction are easily configurable through the GUI. Body-dependent phasic interference can be defined as well. An olfactory                     sensor dynamically tracks odor gradients enabling chemotactic navigation. Feedback from the environment is only partially supported as in the case                 of reoccurent feeding motion at succesfull food encounter.
            3. Intermittent behavior
                Intermittent function of the oscillator modules is available through definition of specific spatial or temporal distributions. Models featuring                   empirically-fitted intermittent crawling interspersed by brief pauses can be readily tested. Time has been quantized at the scale of single                       crawling or feeding motions.
            4. Olfactory learning
                A neuron-level detailed mushroom-body model has been integrated to the locomotory model, enabling olfactory learning after associative                             conditioning of novel odorants to food. 
            5. Energetics and homeostatic drive
            
            


Behavioral simulation
=====================
The simulation platform implements virtual larvae for use in behavioral experiments. 
Specifically the components are :
1. Environment
    1. Spatial arena
    2. Odorscape (Olfactory landscape)
    3. Food sources

2. Larva model
    1. Body
        1. Two-segment
        2. Multi-segment
    2. Sensorimotor effectors
        1. Crawler
        2. Turner
        3. Feeder
    3. Nervous system
        1. Olfaction
        2. Olfactory learning
        3. Intermittent behavior
    4. Energetics
    5. Homeostatic Drive

Scheduling is managed by the [mesa](https://mesa.readthedocs.io/en/master/) agent-based modeling library

For multi-segment larvae the spatial environment and bodies are simulated through Box2D physics engine, 
based on the [kilobots gym](https://github.com/gregorgebhardt/gym-kilobots).

Optionally neural modules can be implemented using the [Nengo](https://www.nengo.ai/) neural simulator

The homeostasis/energetics module is based on the [DEB](http://www.debtheory.org/wiki/index.php?title=Main_Page) (Dynamic Energy Budget) Theory

-----------------------------------------------------------------------------------------------------------------

**Installation**

Open linux terminal.
Navigate to a directory of your choice.
Download or clone the repository to your local drive :

    git clone https://github.com/nawrotlab/larvaworld.git

Make sure python 3.7 is your default python interpreter.
Optionally create a python 3.7 virtual environment, for example in folder `larvaworld_venv`, and activate it:

    apt-get install python-virtualenv

    virtualenv -p /usr/bin/python3 larvaworld_venv

    source larvaworld_venv/bin/activate


Install package dependencies :

    cd larvaworld

    pip install -r requirements.txt
    
Add the virtual environment to jupyter so that you can run the notebooks

    python -m ipykernel install --user --name=larvaworld_venv
    

**Walkthrough**

Visit the [tutorial notebook](tutorial/walkthrough.ipynb) for a complete walkthrough to Larvaworld.

**Run Larvaworld**

Larvaworld can be run directly from linux terminal.
The executable files are in `larvaworld/run` directory. Navigate there.

    cd run


Three modes are available :

1. Analysis 

    Run analysis on the existing sample empirical data (3 dishes of freely exploring larvae).
    First build a larvaworld dataset for each of the raw dishes, selecting tracks longer than 160''.
    Then enrich the datasets computing derived parameters and annotating epochs and analyse them creating comparative plots.
    
        python process.py TestGroup build -each -t 160
        python process.py TestGroup enrich anal -nam dish_0 dish_1 dish_2
        
    Check the comparative plots in `larvaworld/data/TestGroup/plots`.

    Visualize one of the dishes (dish 1) you have created by generating a video.

        python process.py TestGroup vis -nam dish_1 -vid
    
    Check the generated video in `larvaworld/data/TestGroup/processed/dish_1/visuals`.

2. Simulation

    Run a single simulation of one of multiple available experiments. 
    Optionally run the respetive analysis.
   
    This line runs a dish simulation (30 larvae, 3 minutes) without analysis. 
    We choose to also see the simulation at a speed x6 as it unfolds.
    
        python exp_run.py dish -N 30 -t 3.0 -vid 6
    
    This line runs a dispersion simulation and compares the results to the existing reference dataset (`larvaworld/data/reference`)
    We choose to only produce a final image of the simulation.
    
        python exp_run.py dispersion -N 30 -t 3.0 -img -a
        
    Check the plots comparing simulated to empirical data in `larvaworld/data/SimGroup/single_runs/dispersion`.
    
3. Batch run

    Run multiple trials of a given experiment with different parameters.
    This line runs a batch run of odor preference experiments for different valences of the two odor sources.
    
        python batch_run.py odor_pref -N 25 -t 3.0 -rng -200.0 200.0 -Ngrd 5
        
    Check the heatmap of preference indexes in `larvaworld/data/SimGroup/batch_runs`.
