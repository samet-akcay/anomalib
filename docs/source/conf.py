"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Define paths
project_root = Path(__file__).parent.parent.parent
module_path = project_root / "src"
examples_path = project_root / "examples"

# Insert paths to sys.path
sys.path.insert(0, str(module_path.resolve()))
sys.path.insert(0, str(project_root.resolve()))

project = "Anomalib"
copyright = "Intel Corporation"  # noqa: A001
author = "Intel Corporation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "myst_nb",  # myst_nb includes myst_parser functionality
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
]

# MyST-NB Configuration
nb_execution_mode = "off"  # "auto", "force", "cache", or "off" - temporarily off for setup
nb_execution_cache_path = "_build/.jupyter_cache"
nb_execution_excludepatterns = [
    "*checkpoint*",
    "*/.ipynb_checkpoints/*",
]
nb_execution_timeout = 300
nb_execution_allow_errors = False
nb_execution_show_tb = True

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "linkify",
    "substitution",
    "tasklist",
    "deflist",
    "fieldlist",
    "amsmath",
    "dollarmath",
    "attrs_inline",
    "attrs_block",
]

# MyST-NB specific configuration
nb_render_markdown_format = "myst"  # Ensure MyST rendering in notebooks
nb_custom_formats = {".md": ["jupytext.reads", {"fmt": "mystnb"}]}

# Add separate setting for eval-rst
myst_enable_eval_rst = True

# Templates and patterns
templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "**.ipynb_checkpoints",
    "**/.pytest_cache",
    "**/.git",
    "**/.github",
    "**/.venv",
    "**/*.egg-info",
    "**/build",
    "**/dist",
]

# Automatic exclusion of prompts from the copies
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#automatic-exclusion-of-prompts-from-the-copies
copybutton_exclude = ".linenos, .gp, .go"

# Enable section anchors for cross-referencing
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/images/logos/anomalib-icon.png"
html_favicon = "_static/images/logos/anomalib-favicon.png"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "text": "Anomalib",
    },
}

# Add references to example files
html_context = {"examples_path": str(examples_path)}

# External documentation references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}
