==================LARVAWORLD==================

Drosophila larva behavioral analysis and simulation platform

================================================

Behavioral analysis
===================
Data from any tracker software can be analysed. Compatible file formats are csv and dat. 
The only required parameters are x,y coordinates of midline points and optionally contourpoints.

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
    

**Walkthrough**

Visit the [tutorial notebook](tutorial/walkthrough.ipynb) for a complete walkthrough to Larvaworld.

**Run Larvaworld**

Larvaworld can be run directly from linux terminal.
The executable files are in `larvaworld/run` directory. Navigate there.
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
    We choose to also see the simulation at a framerate of 6 fps as it unfolds.
    
        python exp_run.py dish -N 30 -t 3.0 -vid
    
    This line runs a dispersion simulation and compares the results to the existing reference dataset (`larvaworld/data/reference`)
    We choose to only produce a final image of the simulation.
    
        python exp_run.py dispersion -N 30 -t 3.0 -img -a
        
    Check the plots comparing simulated to empirical data in `larvaworld/data/SimGroup/single_runs/dispersion`.
    
3. Batch run

    Run multiple trials of a given experiment with different parameters.
    This line runs a batch run of odor preference experiments for different valences of the two odor sources.
    
        python batch_run.py odor_pref -N 25 -t 3.0 -rng -200.0 200.0 -Ngrd 5
        
    Check the heatmap of preference indexes in `larvaworld/data/SimGroup/batch_runs`.
