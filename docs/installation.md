(installation)=

# Installation

Larvaworld is published on [PyPI](https://pypi.org/project/larvaworld/) and can be installed with `pip` (recommended) or `poetry`.

## System Requirements

- **Python**: 3.10–3.13
- **Operating Systems**: Linux, macOS, Windows
- **RAM**: Minimum 4 GB (8+ GB recommended for large simulations)
- **Disk Space**: ~100 MB for core installation

## Recommended Installation Setup

**It is strongly recommended to install Larvaworld in a virtual environment** to avoid conflicts with other Python packages and ensure a clean installation.

### Create a Virtual Environment

First, ensure you have Python 3.10–3.13 installed. Then create and activate a virtual environment:

**On Linux/macOS:**

```bash
# Replace 3.12 with any supported version (3.10–3.13)
python3.12 -m venv larvaworld_env

source larvaworld_env/bin/activate
```

**On Windows:**

```bash
python -m venv larvaworld_env
# or
# Replace 3.12 with any supported version (3.10–3.13)
py -3.12 -m venv larvaworld_env

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

### Installing optional features (PyPI / `pip`)

If you installed Larvaworld from PyPI using `pip install larvaworld`, install optional features by installing the relevant _third-party packages_ into the same environment (Larvaworld does not currently expose `pip` “extras” like `larvaworld[nengo]`).

```bash
# Neural simulators
pip install nengo
pip install brian2

# Physics engine (multisegment body simulation)
pip install box2d-py
```

**Use cases**:

- Neural simulators: required for experiments using `NengoBrain` or `Brian2Brain` controllers.
- Box2D: required for experiments with `physics_model=True` (multisegment larvae with Box2D).

**Note (Linux/WSL/macOS)**: Installing `box2d-py` from source may require system build tools and `swig`. On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y swig build-essential
```

### Installing optional dependency groups (source / `poetry`)

If you're working from source, Larvaworld uses Poetry dependency groups (including optional groups). From the Poetry project root (folder containing `pyproject.toml`), install any combination of groups:

```bash
# Common developer setup (tests + all optional feature groups)
poetry install --with dev,docs,nengo,brian2,box2d
```

## Development Installation

### Editable install from source (recommended for repo changes)

If you want to modify the repo and have changes reflected immediately, use a dedicated virtual environment and install Larvaworld from the local source tree.

#### Option 1: Poetry-managed environment (recommended for contributors)

From the Poetry project root (folder containing `pyproject.toml`):

```bash
# Installs the project + selected dependency groups into Poetry's virtual environment
poetry install --with dev,docs,nengo,brian2,box2d
```

Run commands inside that environment using `poetry run`:

```bash
poetry run larvaworld --version
poetry run larvaworld-app
poetry run pytest
```

If you need Poetry to use a specific Python interpreter version (e.g., 3.13), select it explicitly:

```bash
poetry env use python3.13
poetry install --with dev,docs,nengo,brian2,box2d
```

#### Option 2: Your own virtual environment + Poetry (advanced)

Create and activate a virtual environment, then point Poetry at that environment's Python before installing dependency groups.

**On Linux/macOS (example uses Python 3.13):**

```bash
python3.13 -m venv /path/to/venvs/larvaworld_p313
source /path/to/venvs/larvaworld_p313/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

**On Windows (PowerShell; example uses Python 3.13):**

```powershell
py -3.13 -m venv C:\path\to\venvs\larvaworld_p313
C:\path\to\venvs\larvaworld_p313\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Then, from the Poetry project root (folder containing `pyproject.toml`):

```bash
# Install into the *currently active* environment (disable Poetry's own venv management)
POETRY_VIRTUALENVS_CREATE=false poetry install --with dev,docs,nengo,brian2,box2d
```

If you specifically require a strict `pip` editable install, install dependencies without the root project and then install the project in editable mode:

```bash
POETRY_VIRTUALENVS_CREATE=false poetry install --no-root --with dev,docs,nengo,brian2,box2d
python -m pip install -e .
```

If an optional group fails to install on your platform/Python version, omit it (the project targets Python 3.10–3.13 where feasible, subject to third-party availability).

For general development and contributing, see the {doc}`contributing` guide which includes:

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
pip install nengo
```

### Issue: Box2D installation fails on macOS

**Solution**: Install dependencies via Homebrew first:

```bash
brew install swig
pip install box2d-py
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
