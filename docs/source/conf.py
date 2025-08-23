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

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_with_keys": False,
    "switcher": {
        "json_url": "https://sp-nitech.github.io/diffsptk/switcher.json",
        "version_match": "master" if "dev" in version else version,
    },
    "logo": {
        "text": project,
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
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
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/diffsptk",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_start": ["navbar-logo", "version-switcher"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
numpydoc_show_class_members = False
