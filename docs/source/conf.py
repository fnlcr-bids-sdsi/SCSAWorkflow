# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spac'
copyright = '2025, Fang Liu, Rui He, and George Zaki'
author = 'Fang Liu, Rui He, and George Zaki'

version = '0.8.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
import os
import os
import sys
path = os.path.abspath('../../src')
sys.path.insert(0,path)
extensions = [
	'sphinx.ext.napoleon',
	'sphinx.ext.autodoc',
	'sphinx.ext.autosectionlabel',
	'sphinx.ext.todo',
	'sphinx.ext.viewcode',
	'sphinx.ext.githubpages',
	'sphinx.ext.autosummary',
	'm2r']
autosummary_generate = True
source_suffix = ['.rst', '.md']
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
    'sticky_navigation': True,
    'titles_only': False,
    'style_external_links': True,
}
