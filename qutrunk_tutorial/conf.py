# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from os import path

qutrunk_path = path.dirname(path.dirname(path.abspath(__file__)))
print("qutrunk_path=", qutrunk_path)
sys.path.insert(0, qutrunk_path)

# -- Project information -----------------------------------------------------

project = 'QuTrunk'
copyright = '2022, qudoor'
author = 'qudoor'
release = 'v0.2.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "sphinx_design",
]

# 按源码顺序，不自动排序
autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]