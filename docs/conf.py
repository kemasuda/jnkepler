# docs/conf.py
import os
import sys

# src-layout
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(os.path.expanduser('~/jnkepler')))

project = "jnkepler"
author = "Kento Masuda"
copyright = "jnkepler"
master_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# version from setuptools_scm
try:
    from jnkepler.jnkepler_version import __version__
    release = __version__
except Exception:
    release = "unknown"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_css_files = ["header.css"]
