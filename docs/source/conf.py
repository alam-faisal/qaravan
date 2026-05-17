import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Qaravan'
author = 'Faisal Alam'
release = '0.1.57'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

html_theme = 'furo'
html_static_path = ['_static']
templates_path = ['_templates']
exclude_patterns = []
