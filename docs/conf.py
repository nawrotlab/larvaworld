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
]

# The suffix of source filenames.
source_suffix = [
    ".rst",
    ".md",
]
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

# AutoAPI configuration
autoapi_dirs = ["../src/larvaworld"]
autoapi_type = "python"
autoapi_ignore = ["*/gui/*", "*/tests/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
