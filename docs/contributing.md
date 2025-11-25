(contributing)=

# Contributing & Development

Contributions are welcome, and they are greatly appreciated! Every little helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs to [our issue page][gh-issues]. If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

Larvaworld could always use more documentation, whether as part of the official larvaworld docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is via [our issue page][gh-issues] on GitHub. If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome 😊

---

## Development Installation

If you want to modify the source code or contribute to Larvaworld, follow these steps:

### 1. Fork and Clone the Repository

Fork the repository on GitHub, then clone your fork locally:

```bash
git clone git@github.com:your_name_here/larvaworld.git
cd larvaworld
```

### 2. Install with Poetry (Recommended)

Install the package in editable mode with development and documentation dependencies:

```bash
poetry install --with dev,docs
```

This installs:

- The package in editable mode (changes are immediately available)
- Development dependencies (pytest, pre-commit, linting tools)
- Documentation dependencies (Sphinx, extensions)

### 3. Activate the Virtual Environment

```bash
poetry shell
```

### 4. Install Pre-commit Hooks

Pre-commit hooks ensure code quality checks run automatically before each commit:

```bash
pre-commit install
```

This will run:

- Code formatting (ruff, black)
- Linting (ruff, flake8)
- File checks (end of files, trailing whitespace)
- Other quality checks

You can also run all checks manually:

```bash
pre-commit run --all-files
```

---

## Development Workflow

### 1. Create a Branch

Create a branch for your changes:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

**Branch naming conventions:**

- `fix/` for bug fixes
- `feat/` for new features
- `docs/` for documentation changes
- `refactor/` for code refactoring

### 2. Make Your Changes

Make your changes locally. Remember to:

- Follow the existing code style
- Add docstrings for new functions/classes
- Update documentation if needed
- Write tests for new features or bug fixes

### 3. Run Tests Locally

Before committing, ensure your changes pass all tests:

```bash
# Run all tests
poetry run pytest

# Run a specific test file
poetry run pytest tests/path/to/test_file.py

# Run tests with coverage
poetry run pytest --cov=larvaworld --cov-report=term
```

### 4. Run Linting and Formatting

Pre-commit hooks will run automatically, but you can also run them manually:

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run specific checks
pre-commit run ruff --all-files
pre-commit run black --all-files
```

### 5. Commit Your Changes

Commit your changes following [Conventional Commits](https://www.conventionalcommits.org):

```bash
git add .
git commit -m "feat(module): your detailed description of your changes"
```

**Commit message format:**

- `feat(module): description` - New feature
- `fix(module): description` - Bug fix
- `docs(module): description` - Documentation changes
- `refactor(module): description` - Code refactoring
- `test(module): description` - Test additions/changes

The commit message will be validated by CI using `commitlint`. If you've installed pre-commit hooks, it will be checked at commit time.

### 6. Push and Create a Pull Request

Push your branch to GitHub:

```bash
git push origin name-of-your-bugfix-or-feature
```

Then create a pull request:

**Via GitHub website:**

1. Go to the repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill in the PR description

**Via GitHub CLI:**

```bash
gh pr create --fill
```

---

## Pull Request Process

### Opening a PR

We encourage opening pull requests as early as possible, even for work in progress. Use **draft pull requests** if your work is still incomplete. This allows for early feedback and discussion.

### PR Guidelines

1. **Include tests**: Add tests for new features or bug fixes
2. **Update documentation**: Update relevant documentation for significant features
3. **Follow code style**: Ensure your code follows the project's style (enforced by pre-commit)
4. **Write clear commit messages**: Follow Conventional Commits format
5. **Keep PRs focused**: One feature or fix per PR when possible

### PR Review Process

1. **Automated checks**: CI will automatically run:

   - Linting and formatting checks
   - Commit message validation
   - Full test suite on multiple platforms (Linux, macOS, Windows)
   - Code coverage reporting

2. **Review**: Maintainers will review your PR. Address any feedback by:

   - Making requested changes
   - Pushing new commits to the same branch
   - Discussing any concerns in the PR comments

3. **Approval**: Once approved, a maintainer will merge your PR.

---

## Continuous Integration (CI)

Larvaworld uses GitHub Actions for continuous integration. The CI pipeline runs automatically on:

- **Push to `master`**: Full test suite and release checks
- **Pull requests**: Full test suite and linting
- **Scheduled runs**: Weekly full test suite (Mondays at 3:00 UTC)
- **Manual trigger**: Can be triggered manually via `workflow_dispatch`

### CI Jobs

The CI pipeline includes:

1. **Lint**: Code formatting and linting checks

   - Runs `pre-commit` with all hooks
   - Automatically commits formatting fixes if needed

2. **Commitlint**: Validates commit messages follow Conventional Commits

3. **Full Test Suite** (Ubuntu, Python 3.11):

   - Installs dependencies with Poetry
   - Runs full test suite with coverage
   - Uploads coverage to Codecov

4. **Smoke Tests** (Matrix):
   - Tests on multiple platforms: Ubuntu, macOS, Windows
   - Tests on Python 3.10 and 3.11
   - Runs full test suite with coverage

### Viewing CI Results

- **GitHub Actions tab**: View all CI runs and their status
- **PR checks**: See CI status directly in your pull request
- **Codecov**: View code coverage reports and changes

### Triggering Tests Manually

You can trigger CI runs manually:

1. Go to the "Actions" tab on GitHub
2. Select the "CI" workflow
3. Click "Run workflow"
4. Choose the branch and click "Run workflow"

You can also trigger tests by pushing an empty commit:

```bash
git commit --allow-empty -m "ci: trigger tests"
git push
```

---

## Release Process

Releases are automated using [python-semantic-release](https://python-semantic-release.readthedocs.io/) and triggered automatically when changes are merged to the `master` branch.

### How Releases Work

1. **Version Detection**: The next version is determined automatically based on commit messages:

   - `feat:` commits → minor version bump
   - `fix:` commits → patch version bump
   - `BREAKING CHANGE:` → major version bump

2. **Release Job**: The release job runs after all CI checks pass:

   - Creates a new Git tag
   - Updates version in `pyproject.toml`
   - Creates a GitHub Release
   - Publishes to PyPI

3. **Release Environment**: The release job requires the `release` environment to be configured with appropriate permissions.

### Manual Release

If needed, releases can be triggered manually:

1. Go to the "Actions" tab on GitHub
2. Select the "CI" workflow
3. Find the "release" job
4. Click "Run workflow" (if available)

Or use semantic-release locally (dry run):

```bash
poetry run semantic-release version --dry-run
```

---

## Documentation Build

Documentation is built automatically on ReadTheDocs when changes are pushed to `master`. The documentation is also built during CI for validation.

### Building Documentation Locally

To build and preview documentation locally:

```bash
# Install documentation dependencies (if not already installed)
poetry install --with docs

# Build documentation
cd docs
poetry run sphinx-build -b html . _build/html

# View the documentation
# Open _build/html/index.html in your browser
```

### Documentation Structure

- **Source files**: `docs/` directory
- **Configuration**: `docs/conf.py`
- **Build output**: `docs/_build/`
- **ReadTheDocs**: Automatically builds from `master` branch

### Documentation Guidelines

- Use reStructuredText (`.rst`) or Markdown (`.md`) for documentation
- Follow the existing documentation style
- Include code examples where appropriate
- Update the table of contents in `docs/index.rst` if adding new pages

---

## Tips

### Running a Subset of Tests

```bash
# Run tests in a specific directory
poetry run pytest tests/specific/directory

# Run tests matching a pattern
poetry run pytest -k "test_pattern"

# Run tests with verbose output
poetry run pytest -v
```

### Debugging

```bash
# Run tests with debugger
poetry run pytest --pdb

# Run with print statements visible
poetry run pytest -s
```

### Code Coverage

```bash
# Generate coverage report
poetry run pytest --cov=larvaworld --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Getting Help

If you need help or have questions:

- **GitHub Issues**: [Open an issue][gh-issues] for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the maintainers at `p.sakagiannis@uni-koeln.de`

[gh-issues]: https://github.com/nawrotlab/larvaworld/issues
