(installation)=

# Installation

Larvaworld is published on [PyPI](https://pypi.org/project/larvaworld/) and can be installed with `pip` or `poetry`.

## System Requirements

- **Python**: 3.10 or 3.11
- **Operating Systems**: Linux, macOS, Windows
- **RAM**: Minimum 4 GB (8+ GB recommended for large simulations)
- **Disk Space**: ~100 MB for core installation

## Basic Installation

### Using pip

The simplest way to install Larvaworld:

```bash
pip install larvaworld
```

### Using Poetry

If you're managing your project with [Poetry](https://python-poetry.org/):

```bash
poetry add larvaworld
```

## Optional Dependencies

Web dashboards (`larvaworld-app`) are included in the default install. The options below are truly optional and enable specific features:

### Neural Simulators

For neural network-based brain models:

```bash
# Nengo (spiking neural networks)
pip install larvaworld[nengo]

# Brian2 (alternative neural simulator)
pip install larvaworld[brian2]
```

**Use case**: Required for experiments using `NengoBrain` or `Brian2Brain` controllers.

### Physics Engine

For realistic multisegment body simulation:

```bash
pip install larvaworld[box2d]
```

**Use case**: Required for experiments with `physics_model=True` (multisegment larvae with Box2D).

### Development Tools

For contributors and developers:

```bash
pip install larvaworld[dev]
```

Includes: `pytest`, `pre-commit`, documentation and linting utilities.

### All Extras

To install **everything** at once:

```bash
pip install larvaworld[all]
```

## Development Installation

If you want to modify the source code or contribute to Larvaworld:

### 1. Clone the Repository

```bash
git clone https://github.com/nawrotlab/larvaworld.git
cd larvaworld
```

### 2. Install with Poetry (Recommended)

```bash
poetry install --with dev,docs
```

This installs the package in editable mode with development and documentation dependencies.

### 3. Activate the Virtual Environment

```bash
poetry shell
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically before each commit.

## Verifying Installation

Check that Larvaworld is correctly installed:

```bash
larvaworld --version
```

Run a quick test simulation:

```bash
larvaworld Exp dish -N 1 -duration 1.0
```

If this runs without errors, your installation is successful.

## Troubleshooting

### Issue: `larvaworld: command not found`

**Solution**: Ensure your Python environment's `bin/` directory is in your `PATH`. With Poetry:

```bash
poetry shell
larvaworld --version
```

### Issue: `ImportError: No module named 'nengo'`

**Solution**: Install the optional dependency:

```bash
pip install larvaworld[nengo]
```

### Issue: Box2D installation fails on macOS

**Solution**: Install dependencies via Homebrew first:

```bash
brew install swig
pip install larvaworld[box2d]
```

### Issue: Permission errors during `pip install`

**Solution**: Use a virtual environment (recommended) or install for the user only:

```bash
pip install --user larvaworld
```

### Issue: PyTables on Windows

**Issue**: PyTables (HDF5 backend) may require pre-built binaries.

**Solution**: Install from conda-forge if pip fails:

```bash
conda install -c conda-forge pytables
pip install larvaworld
```
