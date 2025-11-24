# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information
project = "larvaworld"
author = "Panagiotis Sakagiannis"
copyright = "2024, Panagiotis Sakagiannis"
release = "1.0.0"

# General configuration
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinxcontrib.mermaid",  # Mermaid diagrams
    "myst_nb",  # Jupyter notebook support
]

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",  # ::: directives
    "deflist",  # Definition lists
    "html_image",  # HTML images
]

# Mermaid configuration
mermaid_output_format = "raw"
mermaid_version = "10.6.1"  # Latest stable

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# The suffix of source filenames.
source_suffix = [
    ".rst",
    ".md",
    ".ipynb",  # Jupyter notebooks
]

# MyST-NB configuration
nb_execution_mode = "off"  # Don't execute notebooks, just render them
nb_execution_timeout = 60  # Timeout in seconds (not used when mode=off)
nb_execution_allow_errors = True  # Continue even if there are errors
myst_footnote_transition = False
templates_path = [
    "_templates",
]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Options for HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# html_css_files = ["custom.css"]  # Disabled to use default RTD theme styling

# AutoAPI configuration
autoapi_dirs = ["../src/larvaworld"]
autoapi_type = "python"
autoapi_ignore = ["*/gui/*", "*/tests/*"]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
