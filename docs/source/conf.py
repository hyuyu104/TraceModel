import os
import sys

# irrelevant for local build
# required for readthedocs
# from unittest.mock import MagicMock
# sys.modules['matplotlib'] = MagicMock()
# sys.modules['scipy'] = MagicMock()
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "pybind11",
    "pytest"
]

sys.path.insert(0, os.path.abspath('../../traceHMM'))

# -- Project information ----------------------------------------

project = 'TraceModel'
copyright = '2024, Hongyu'
author = 'Hongyu Yu'

release = '0.0'
version = '0.0.0'

# -- General configuration --------------------------------------

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

extensions = [
    'myst_parser',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon' # numpy style
]

# for README.md
myst_enable_extensions = [
    "html_image",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_custom_sections = ['Attributes']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ------------------------------------

html_theme = "furo"

html_context = {"default_mode": "light"}
