# Web Applications

Larvaworld provides interactive web dashboards built with **Panel** (Holoviz stack) for exploration, configuration, and analysis.

---

## Launching the App

```bash
larvaworld-app
```

**Access**: `http://localhost:5006`

---

## Available Dashboards

| Dashboard | Purpose |
|-----------|---------|
| **Experiment Viewer** | View experiment results interactively |
| **Track Viewer** | Inspect trajectories |
| **Model Inspector** | Explore locomotory models |
| **Module Inspector** | Inspect behavioral modules |
| **Neural Oscillator Inspector** | Visualize neural oscillators |

---

## Experiment Viewer

**Purpose**: Interactive exploration of simulation results

**Features**:
- Load saved experiments
- Plot trajectories, metrics, distributions
- Filter by time window, agent ID
- Export plots as PNG/SVG

**Access**: Main dashboard landing page

---

## Track Viewer

**Purpose**: Detailed trajectory inspection

**Features**:
- 2D trajectory plots
- Velocity/acceleration profiles
- Zoom and pan
- Multi-agent comparison

---

## Model Inspector

**Purpose**: Explore model parameters

**Features**:
- Browse available models
- View parameter values
- Compare model configurations
- Test parameter combinations

---

## Module Inspector

**Purpose**: Inspect behavioral modules

**Features**:
- Crawler, Turner, Feeder modules
- Real-time parameter adjustment
- Behavior visualization

---

## Neural Oscillator Inspector

**Purpose**: Visualize neural oscillators (CPG)

**Features**:
- Phase plots
- Oscillation frequency analysis
- Coupling visualization

---

## Web App Architecture

![Web App](../figures_tables_from_paper/figures/fig8_web_app.png)

**Figure 8**: Screenshot of Larvaworld web application showing interactive visualization and control panels.

---

## Status

:::{note}
Web applications are **functional but under active development**. Some features may change in future releases. For production use, prefer CLI/Python API.
:::

---

## Related Documentation

- {doc}`keyboard_controls` - Interactive controls
- {doc}`visualization_snapshots` - Visualization examples
- {doc}`../concepts/architecture_overview` - Platform architecture

