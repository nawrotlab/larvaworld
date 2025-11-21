# Dependencies

Larvaworld leverages **40 carefully selected Python packages** spanning scientific computing, visualization, agent-based modeling, and development tools. This page provides a comprehensive overview of the dependency ecosystem.

---

## Dependency Overview

```{mermaid}
mindmap
    root((Larvaworld<br/>40 deps))
        sci
            numpy
            pandas
            scipy
            matplotlib
            seaborn
            scikit-learn
            powerlaw
            statannot
        abm
            agentpy
            box2d-py
            nengo
        geo
            geopandas
            shapely
            movingpandas
        viz
            holoviews
            hvplot
            panel
            param
            pygame
            imageio
        io
            tables
            pypdf
        cli
            typer
            rich
            argparse
            docopt
            progressbar
        util
            pint
            pint_pandas
            typing-extensions
            filelock
        test
            pytest
            pytest-cov
            pytest-xdist
        docs
            sphinx
            sphinx-rtd-theme
            sphinx-autoapi
            sphinx-autobuild
            furo
            myst-parser
```

---

## Core Dependencies

### Scientific Computing (`sci`)

The foundation of all numerical operations:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **numpy** | N-dimensional arrays | All numerical operations, trajectory data |
| **pandas** | Data manipulation | Time-series data (`step_data`), tabular results |
| **scipy** | Scientific algorithms | Statistical tests (KS), signal processing, optimization |
| **matplotlib** | Base plotting | All static plots (trajectories, distributions, etc.) |
| **seaborn** | Statistical visualization | Box plots, heatmaps, distribution comparisons |
| **scikit-learn** | Machine learning | Clustering, dimensionality reduction |
| **powerlaw** | Power-law distributions | Analysis of movement patterns |
| **statannot** | Statistical annotations | P-value annotations on plots |

---

### Agent-Based Modeling (`abm`)

Specialized libraries for simulation and physics:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **agentpy** | Agent-based modeling | Core ABM framework (`ABModel` class), parameter sweeps |
| **box2d-py** | 2D physics engine | Realistic multisegment larva bodies with physics |
| **nengo** | Neural simulation | Neural network-based brain controllers (`NengoBrain`) |

:::{note}
`nengo` and `box2d-py` are **optional dependencies**. Install them if you need neural controllers or physics-based body simulation:

```bash
pip install larvaworld[nengo,box2d]
```
:::

---

### Geospatial (`geo`)

For spatial operations and trajectory analysis:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **geopandas** | Geospatial data frames | Spatial analysis of trajectories |
| **shapely** | Geometric objects | Arena boundaries, collision detection |
| **movingpandas** | Trajectory analysis | Movement pattern analysis |

---

### Visualization (`viz`)

Rich visualization tools for interactive and static outputs:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **holoviews** | Declarative visualization | Interactive plots |
| **hvplot** | Pandas plotting | High-level interactive plotting API |
| **panel** | Web dashboards | Interactive web applications (`larvaworld-app`) |
| **param** | Parameter management | Dashboard widgets and validation |
| **pygame** | Game engine | Real-time 2D rendering (`screen` module) |
| **imageio** | Image/video I/O | Video export (MP4, AVI) |

---

### I/O (`io`)

Data persistence and file handling:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **tables** (PyTables) | HDF5 interface | Dataset storage (`LarvaDataset` persistence) |
| **pypdf** | PDF handling | PDF figure import/export |

---

### CLI (`cli`)

Command-line interface utilities:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **argparse** | Argument parsing | Primary CLI argument parser |
| **typer** | Modern CLI | Optional CLI utilities (legacy) |
| **rich** | Terminal formatting | Pretty printing, progress bars |
| **docopt** | CLI from docstrings | Alternative CLI interface |
| **progressbar** | Progress bars | Long-running task feedback |

---

### Utilities (`util`)

General-purpose utilities:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **pint** | Physical units | Unit conversions (mm, cm, seconds, etc.) |
| **pint_pandas** | Pint + Pandas | Units in data frames |
| **typing-extensions** | Type hints | Python 3.10+ type annotations |
| **filelock** | File locking | Multi-process safe file access |

---

### Testing (`test`)

Test framework for quality assurance:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **pytest** | Testing framework | Test suite runner |
| **pytest-cov** | Coverage reporting | Code coverage analysis |
| **pytest-xdist** | Parallel testing | Speed up test execution |

---

### Documentation (`docs`)

Sphinx-based documentation generation:

| Package | Purpose | Use in Larvaworld |
|---------|---------|-------------------|
| **sphinx** | Documentation generator | ReadTheDocs build |
| **sphinx-rtd-theme** | ReadTheDocs theme | Classic RTD theme (alternative) |
| **sphinx-autoapi** | API documentation | Auto-generate API docs from docstrings |
| **sphinx-autobuild** | Live rebuild | Development server with auto-reload |
| **furo** | Modern Sphinx theme | Primary documentation theme |
| **myst-parser** | Markdown support | Write docs in Markdown instead of RST |

---

## Dependency Groups

Dependencies are organized into **optional groups** in `pyproject.toml`:

### Main Dependencies (Always Installed)

Core scientific stack + agent modeling:

```bash
pip install larvaworld
```

Includes: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `agentpy`, `shapely`, `pygame`, `tables`, and more.

---

### Optional: Neural Simulators

```bash
pip install larvaworld[nengo]   # Nengo neural simulator
pip install larvaworld[brian2]  # Brian2 neural simulator
```

Use case: Neural network-based brain controllers.

---

### Optional: Physics Engine

```bash
pip install larvaworld[box2d]
```

Use case: Realistic multisegment body simulation.

---

### Optional: Development

```bash
pip install larvaworld[dev]
```

Includes: `pytest`, `ruff`, `pre-commit`, `commitizen`, and more.

Use case: Contributing to Larvaworld.

---

### Install All

```bash
pip install larvaworld[all]
```

Installs **all** optional dependencies.

---

## Python Version Requirements

Larvaworld supports **Python 3.10 and 3.11**:

```toml
[tool.poetry.dependencies]
python = ">=3.10,<3.13"
```

:::{warning}
Python 3.12 is not yet supported due to some dependency incompatibilities (notably `tables` and `box2d-py`). Support for Python 3.12+ will be added once all dependencies are compatible.
:::

## Dependency Updates

### Current Strategy

- **Major updates**: Tested manually before merging
- **Minor updates**: Automated via Dependabot
- **Security patches**: Applied immediately

### Pinning Policy

- **Core deps**: Pinned to major versions (e.g., `numpy >=1.20,<2.0`)
- **Dev deps**: Flexible (e.g., `pytest >=7.0`)
- **Docs deps**: Python 3.11+ only (e.g., `sphinx >=4.0, python >=3.11`)


---

## Related Documentation

- {doc}`../installation` - Installation instructions
- {doc}`code_structure` - Codebase organization
- {doc}`architecture_overview` - Platform architecture
