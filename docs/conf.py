# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the python source directory so autodoc can import ferro_ta
# Only add if ferro_ta is not already installed (e.g. from a wheel in CI)
try:
    import ferro_ta  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.abspath("../python"))

# -- Project information -------------------------------------------------------
project = "ferro-ta"
copyright = "2024, pratikbhadane24"
author = "pratikbhadane24"
# Version from env (e.g. set in CI from git tag) or default
release = os.environ.get("FERRO_TA_VERSION", "1.0.0")
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Google / NumPy-style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output --------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "ferro-ta Documentation"
html_short_title = "ferro-ta"

# -- autodoc ------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Suppress autodoc import warnings for modules that can't be loaded without
# the compiled Rust extension (_ferro_ta). These are expected when building
# docs without the wheel; the documented API is still accurate.
# Also suppress duplicate object descriptions that arise when Rust-backed
# streaming classes (defined in ferro_ta._ferro_ta) are re-exported through
# ferro_ta.streaming — autodoc sees them in both modules.
suppress_warnings = [
    "autodoc.import_object",
    "ref.doc",
    "py.duplicate",
]
