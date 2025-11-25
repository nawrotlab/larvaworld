(installation)=

# Installation

Larvaworld is published on [PyPI](https://pypi.org/project/larvaworld/) and can be installed with `pip` (recommended) or `poetry`.

## System Requirements

- **Python**: 3.10 or 3.11
- **Operating Systems**: Linux, macOS, Windows
- **RAM**: Minimum 4 GB (8+ GB recommended for large simulations)
- **Disk Space**: ~100 MB for core installation

## Recommended Installation Setup

**It is strongly recommended to install Larvaworld in a virtual environment** to avoid conflicts with other Python packages and ensure a clean installation.

### Create a Virtual Environment

First, ensure you have Python 3.10 or 3.11 installed. Then create and activate a virtual environment:

**On Linux/macOS:**

```bash
python3.10 -m venv larvaworld_env
# or
python3.11 -m venv larvaworld_env

source larvaworld_env/bin/activate
```

**On Windows:**

```bash
python -m venv larvaworld_env
# or
py -3.10 -m venv larvaworld_env
# or
py -3.11 -m venv larvaworld_env

larvaworld_env\Scripts\activate
```

### Upgrade pip, setuptools, and wheel

Before installing Larvaworld, upgrade the build tools (especially important on Windows):

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Basic Installation

### Using pip (Recommended)

Once your virtual environment is activated, install Larvaworld:

```bash
pip install larvaworld
```

This is the recommended method for most users as it's simple and doesn't require additional tools.

### Using Poetry

If you're managing your project with [Poetry](https://python-poetry.org/):

```bash
poetry add larvaworld
```

Poetry is recommended for developers and contributors who need advanced dependency management.

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

For development and contributing, see the {doc}`contributing` guide which includes:

- Development installation instructions
- Contribution guidelines
- Pull request process
- CI/CD workflow
- Release process
- Documentation build instructions

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

### Issue: `BackendUnavailable: Cannot import 'setuptools.build_meta'` on Windows

**Issue**: When installing on Windows, pip may fail to build dependencies from source if setuptools and wheel are not up to date.

**Solution**: Upgrade pip, setuptools, and wheel before installing:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install larvaworld
```

If the issue persists, try installing build tools explicitly:

```bash
pip install --upgrade pip
pip install setuptools wheel
pip install larvaworld
```

### Issue: PyTables on Windows

**Issue**: PyTables (HDF5 backend) may require pre-built binaries.

**Solution**: Install from conda-forge if pip fails:

```bash
conda install -c conda-forge pytables
pip install larvaworld
```
