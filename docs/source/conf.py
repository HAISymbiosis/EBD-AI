# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
from datetime import datetime

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../..'))

version_ns = {}
with open(os.path.abspath('../../ex_fuzzy/_version.py'), encoding='utf-8') as f:
    exec(f.read(), version_ns)

# -- Project information -----------------------------------------------------
project = 'BDI'
copyright = (
    f'2026-{datetime.now().year}, Prof. Javier Andreu-Perez. '
    'Rebase of Ex-Fuzzy by Javier Fumanal Idocin and Javier Andreu-Perez.'
)
author = 'Prof. Javier Andreu-Perez'
release = version_ns['__version__']
version = version_ns['__version__']

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.extlinks',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
    'nbsphinx',
]

# Templates and patterns
templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]

# -- Extension configuration -------------------------------------------------

# Autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary
# API pages are maintained directly under docs/source/api.
autosummary_generate = False
autosummary_generate_overwrite = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {}

# External links
extlinks = {
    'issue': ('https://github.com/HAISymbiosis/EBD-AI/issues/%s', 'issue %s'),
    'pr': ('https://github.com/HAISymbiosis/EBD-AI/pull/%s', 'PR %s'),
}

# Todo configuration
todo_include_todos = True

# Reduce external-reference noise from inherited third-party docstrings.
suppress_warnings = ['ref.ref', 'ref.term']

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_title = f"BDI {version}"

html_theme_options = {
    "logo": {
        "text": "BDI",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/HAISymbiosis/EBD-AI",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "show_prev_next": False,
    "search_bar_text": "Search the docs...",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "announcement": None,
}

html_context = {
    "github_user": "HAISymbiosis",
    "github_repo": "EBD-AI",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Favicon
html_favicon = None

# Show source links
html_show_sourcelink = True
html_copy_source = True

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    ('index', 'bdi.tex', 'BDI: The Explainable By Design AI Toolbox',
     'Prof. Javier Andreu-Perez', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ('index', 'bdi', 'BDI: The Explainable By Design AI Toolbox',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    ('index', 'bdi', 'BDI: The Explainable By Design AI Toolbox',
     author, 'bdi', 'The Explainable By Design AI Toolbox, a rebase of Ex-Fuzzy.',
     'Miscellaneous'),
]

# -- Options for epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ['search.html']
