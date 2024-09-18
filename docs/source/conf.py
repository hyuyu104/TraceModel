import os
import sys

# sys.path.append(os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath("../../traceHMM/update"))
# sys.path.insert(0, os.path.abspath("../../traceHMM/utils"))
sys.path.append(os.path.abspath("../../"))

# -- Project information ----------------------------------------

project = 'TraceModel'
copyright = '2024, Hongyu'
author = 'Hongyu Yu'

release = '0.0'
version = '0.0.0'

# -- General configuration --------------------------------------

# update is a cpp module, cannot be imported properly
# temporary fix, raised a warning but won't affect build
autodoc_mock_imports = [ "traceHMM.update" ]

source_suffix = {
    ".rst": "restructuredtext", 
    ".md": "markdown"
}

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon', # numpy style
    'sphinx.ext.mathjax', # math symbols
    'myst_parser' # markdown files
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
