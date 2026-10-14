# docs/conf.py
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import sys

# src-layout
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "jnkepler"
author = "Kento Masuda"
copyright = "jnkepler"
master_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

# The build installs the package, whose version comes from setuptools_scm.
try:
    release = package_version("jnkepler")
except PackageNotFoundError:
    release = "unknown"

myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
