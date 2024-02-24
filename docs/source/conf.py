# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'conservation_laws'
copyright = '2024, sibyllema'
author = 'sibyllema'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    #'pydata_sphinx_theme',
]
#autodoc_mock_imports = ["sage"]


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'furo'
html_static_path = ['_static']

from docutils import nodes
from docutils.parsers.rst import roles

def arxiv_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    url = f"https://arxiv.org/abs/{text}"
    node = nodes.reference(rawtext, text, refuri=url, **options)
    return [node], []

def setup(app):
    roles.register_local_role('arxiv', arxiv_role)
