# Tutorials

Step-by-step Jupyter notebooks demonstrating Larvaworld workflows from basic to advanced.

```{toctree}
:maxdepth: 2
:caption: Tutorials

Theoretical Background <theoretical_background.ipynb>
Single Simulation <single_simulation.ipynb>
CLI Interface <cli.ipynb>
Environment Configuration <environment_configuration.ipynb>
Import Datasets <import_datasets.ipynb>
Dataset Replay <replay.ipynb>
Model Evaluation <model_evaluation.ipynb>
Genetic Algorithm Optimization <genetic_algorithm_optimization.ipynb>
Library Interface <library_interface.ipynb>
Custom Module <custom_module.ipynb>
Sensorscapes <sensorscapes.ipynb>
Remote Model Interface <remote_model_interface.ipynb>
Configuration Types <CONFTYPES.ipynb>
```

---

## Getting Started

1. **Single Simulation**
   **File**: `single_simulation.ipynb`

   **Topics**:
   - Running your first experiment
   - `ExpRun` basics
   - Accessing datasets
   - Basic plotting

2. **CLI Interface**
   **File**: `cli.ipynb`

   **Topics**:
   - Command-line usage
   - Common CLI options
   - Batch execution from terminal
   - Scripting experiments

## Configuration

3. **Environment Configuration**
   **File**: `environment_configuration.ipynb`

   **Topics**:
   - Arena geometry
   - Food sources and substrates
   - Odorscapes
   - Obstacles and borders

## Data Management

4. **Importing Datasets**
   **File**: `import_datasets.ipynb`

   **Topics**:
   - Lab-specific formats (Schleyer, Jovanic, Berni, Arguello)
   - `LabFormat` usage
   - Data preprocessing
   - Registering reference datasets

5. **Dataset Replay**
   **File**: `replay.ipynb`

   **Topics**:
   - Replay mode (`ReplayRun`)
   - Visualization options
   - Video export
   - Quality control

## Model Evaluation

6. **Model Evaluation**
   **File**: `model_evaluation.ipynb`

   **Topics**:
   - `EvalRun` usage
   - Comparing models against real data
   - Kolmogorov-Smirnov tests
   - Statistical analysis

7. **Genetic Algorithm Optimization**
   **File**: `genetic_algorithm_optimization.ipynb`

   **Topics**:
   - `optimize_mID` function
   - `GAevaluation` setup
   - Parameter space definition
   - Fitness convergence analysis

## Advanced Usage

8. **Library Interface**
   **File**: `library_interface.ipynb`

   **Topics**:
   - Direct library API usage
   - Custom simulation loops
   - Advanced configuration
   - Integration with external tools

9. **Custom Modules**
   **File**: `custom_module.ipynb`

   **Topics**:
   - Creating custom behavioral modules
   - Subclassing `Effector`
   - Implementing `update()` methods
   - Registering custom modules

## Specialized Topics

10. **Sensorscapes**
    **File**: `sensorscapes.ipynb`

    **Topics**:
    - Thermal landscapes
    - Wind fields
    - Odor plumes
    - Multi-modal sensing

11. **Remote Model Interface**
    **File**: `remote_model_interface.ipynb`

    **Topics**:
    - Remote integration
    - External controller interface
    - Advanced scripting

12. **Configuration Types**
    **File**: `CONFTYPES.ipynb`

    **Topics**:
    - `EnvConf`, `ModelConf`, `ExpConf`, `BatchConf`, `GaConf`
    - Configuration registry system
    - Creating custom configurations
    - Configuration inheritance

---

## Tutorial Path

### Recommended Learning Path

**Beginners** (New to Larvaworld):
1. Single Simulation
2. CLI Interface
3. Environment Configuration

**Intermediate** (Basic understanding):
4. Importing Datasets
5. Dataset Replay
6. Model Evaluation

**Advanced** (Ready for research):
7. Genetic Algorithm Optimization
8. Library Interface
9. Custom Modules

**Experts** (Platform development):
10. Sensorscapes
11. Remote Model Interface
12. Configuration Types

---

## Running Tutorials

### Launch Jupyter

```bash
cd /path/to/larvaworld/docs/tutorials
jupyter notebook
```

### Dependencies

Ensure you have Jupyter installed:

```bash
pip install jupyter notebook
```

---

## Contributing Tutorials

If you've developed a workflow that could help others, consider contributing a tutorial!

**Guidelines**:
- Clear learning objectives
- Step-by-step code cells
- Markdown explanations
- Expected outputs shown
- Error handling examples

**Submit**: Open a pull request on [GitHub](https://github.com/nawrotlab/larvaworld)

