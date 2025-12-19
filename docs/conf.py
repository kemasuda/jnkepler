# docs/conf.py
import os
import sys

# src-layout: RTD/ローカル共通でimport可能にする
sys.path.insert(0, os.path.abspath("../src"))

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

# バージョンは setuptools_scm が write_to したものを表示
try:
    from jnkepler.jnkepler_version import __version__
    release = __version__
except Exception:
    release = "unknown"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_css_files = ["header.css"]
