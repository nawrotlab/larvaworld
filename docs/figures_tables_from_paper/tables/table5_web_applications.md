# Table 5: Web-based Applications

## 📋 Table

### Web-based Applications

| Application | Description |
|-------------|-------------|
| **Experiment Viewer** | Inspect/launch preconfigured experiments |
| **Larva Models** | Inspect/visualize modular larva-models |
| **Locomotory Modules** | Inspect/test behavioral modules |
| **Track Viewer** | Visualize stored datasets |

---

## Detailed Descriptions

### Experiment Viewer
**Purpose**: Browse and launch preconfigured experiments

**Features**:
- View all available experiment configurations
- Filter by category (chemotaxis, foraging, learning, etc.)
- Inspect experiment parameters before launching
- Launch experiments directly from browser
- Modify parameters on-the-fly

**Use Cases**:
- Quick experiment exploration
- Parameter inspection
- Educational demonstrations
- Rapid prototyping

---

### Larva Models (Model Inspector)
**Purpose**: Inspect and visualize modular composition of larva models

**Features**:
- Select from all preconfigured locomotory models
- View configuration of 4 basic locomotor modules:
  - **Crawler**: Peristaltic crawling parameters
  - **Turner**: Body bending/turning parameters
  - **Feeder**: Feeding behavior parameters
  - **Intermitter**: Run/pause switching parameters
- Real-time simulation with dynamic plotting
- Input/output variable visualization
- Module parameter adjustment

**Use Cases**:
- Understanding model architecture
- Comparing different models
- Educational tool for modular design
- Parameter sensitivity exploration

**Related**: See **Figure 8** for screenshot

---

### Locomotory Modules
**Purpose**: Inspect and test individual behavioral modules

**Features**:
- Standalone module testing
- Parameter configuration interface
- Real-time oscillator visualization
- Input-response curves
- Module output plotting

**Use Cases**:
- Module development and debugging
- Understanding module behavior in isolation
- Parameter tuning
- Educational demonstrations of neural oscillators

---

### Track Viewer
**Purpose**: Visualize and analyze stored datasets

**Features**:
- Load any stored or imported dataset
- Interactive trajectory visualization
- Time-series plotting
- Multi-larva comparison
- Statistical summaries
- Export visualizations

**Use Cases**:
- Quick dataset inspection
- Presentation-ready visualizations
- Comparative analysis
- Quality control for imported data

---

## Developer Note

**Status**: 🟡 Under Development

The applications are **functional but may have bugs or incomplete features**. User feedback is welcome for identifying issues and prioritizing improvements.

---

## Usage

### Launching Applications

All web applications are served from a single dashboard process:

```bash
larvaworld-app
```

**Browser Access:**
Applications open at `http://localhost:5006` (or different port if specified), where you can select between the available apps (experiment viewer, larva models, locomotory modules, lateral oscillator, track viewer) from the Panel index page.

---

## Usage in ReadTheDocs

```rst
Web-based Applications
~~~~~~~~~~~~~~~~~~~~~~

Larvaworld provides interactive web applications for exploration and configuration.

.. list-table:: Available Applications
   :header-rows: 1
   :widths: 30 70

   * - Application
     - Description
   * - **Experiment Viewer**
     - Inspect and launch preconfigured experiments from a browsable catalog
   * - **Larva Models**
     - Visualize modular composition and behavior of locomotory models
   * - **Locomotory Modules**
     - Test and configure individual behavioral modules in isolation
   * - **Track Viewer**
     - Visualize and analyze stored experimental or simulated datasets

**Launching Applications:**

All applications can be launched from the command line:

.. code-block:: bash

   # Launch main dashboard (serves all apps)
   larvaworld-app

Applications will open in your default web browser at ``http://localhost:5006`` with multiple named apps (e.g. ``larva_models``, ``locomotory_modules``, ``lateral_oscillator``, ``track_viewer``, ``experiment_viewer``) available from the Panel index page.

**Model Inspector Example:**

.. figure:: _static/images/fig8_web_app.png
   :alt: Model Inspector Application
   :align: center
   :width: 90%
   
   Web-based application for inspecting locomotory models. Select any model
   from the dropdown to view its modular configuration and real-time behavior.

.. note::
   **Development Status**: Web applications are currently under active development.
   Some features may not work as expected and will be improved in future releases.
   
   Please report issues at: https://github.com/nawrotlab/larvaworld/issues

**Technical Stack:**

- **Framework**: Panel (Holoviz ecosystem)
- **Backend**: Bokeh server
- **Frontend**: JavaScript/HTML5
- **Plotting**: HoloViews, Matplotlib
```

---

## Technical Details

### Technology Stack

**Web Framework**: Panel (https://panel.holoviz.org/)
- Part of the Holoviz ecosystem
- Python-based dashboard framework
- Supports reactive programming
- Jupyter-compatible widgets

**Server**: Bokeh Server
- Real-time data streaming
- WebSocket communication
- Interactive callbacks

**Visualization**: 
- HoloViews for declarative plotting
- Bokeh for interactive plots
- Matplotlib for static figures

### Code Location

Applications are defined in: ``src/larvaworld/dashboards/``

**File Structure**:

```
dashboards/
├── __init__.py
├── main.py                         # Dashboard launcher (larvaworld-app entry point)
├── experiment_viewer.py            # Experiment browser/launcher
├── model_inspector.py              # Larva model visualization
├── module_inspector.py             # Behavioral module tester
├── lateral_oscillator_inspector.py # Lateral oscillator inspector
└── track_viewer.py                 # Dataset visualization
```

### Dependencies

Required packages (from `pyproject.toml`):
- `panel`: Dashboard framework
- `holoviews`: Declarative plotting
- `hvplot`: High-level plotting API
- `bokeh`: Interactive visualization
- `param`: Parameter definitions

---

## Future Improvements

**Planned Enhancements** (next development cycle):

1. **Experiment Viewer**:
   - Better filtering/search
   - Experiment comparison mode
   - Batch launch interface

2. **Model Inspector**:
   - Side-by-side model comparison
   - Parameter optimization interface
   - Export model configurations

3. **Module Tester**:
   - Module composition playground
   - Custom module integration
   - Advanced parameter sweeps

4. **Track Viewer**:
   - Enhanced statistical analysis
   - Custom metric calculation
   - Video export functionality

5. **General**:
   - Improved error handling
   - Better performance for large datasets
   - Mobile-responsive design
   - User authentication for deployment

---

## Related Content

- **Figure 8**: Screenshot of Model Inspector application
- **Tutorial**: Using web applications (to be added)
- **API Docs**: Dashboard module documentation

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **LaTeX**: Lines 845-863 in main.tex
- **Label**: `tab:apps`
- **Section**: Web-based applications (lines 837-838)
- **Related Figure**: Fig 7 (labeled `fig:app.models`)
