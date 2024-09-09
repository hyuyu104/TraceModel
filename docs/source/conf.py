# Configuration file for the Sphinx documentation builder.

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
]

# for README.md
myst_enable_extensions = [
    "html_image",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output ------------------------------------

html_theme = 'sphinx_rtd_theme'