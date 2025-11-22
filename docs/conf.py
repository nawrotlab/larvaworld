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
    "nbsphinx",  # Jupyter notebook support
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

# The suffix of source filenames.
source_suffix = [
    ".rst",
    ".md",
    ".ipynb",  # Jupyter notebooks
]

# nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks, just render them
nbsphinx_timeout = 60  # Timeout in seconds (not used when execute=never)
nbsphinx_allow_errors = True  # Continue even if there are errors
# Use first H1 or H2 header as title for notebooks
nbsphinx_requirejs_path = ""
templates_path = [
    "_templates",
]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Options for HTML output
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

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
