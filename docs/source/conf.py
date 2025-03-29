# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "diffsptk"
copyright = "2022, SPTK Working Group"
author = "SPTK Working Group"
exec(open(f"../../{project}/version.py").read())
version = __version__
release = __version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",
]
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
]


# -- Options for HTML output -------------------------------------------------

html_title = "diffsptk"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_with_keys": False,
    "switcher": {
        "json_url": "https://sp-nitech.github.io/diffsptk/switcher.json",
        "version_match": "master" if "dev" in version else version,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sp-nitech/diffsptk",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://x.com/SPTK_DSP",
            "icon": "fab fa-twitter-square",
        },
    ],
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
}
html_static_path = []
html_show_sourcelink = False
numpydoc_show_class_members = False
