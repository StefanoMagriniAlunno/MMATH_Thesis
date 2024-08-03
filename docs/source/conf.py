# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MMATH thesis"
copyright = "2024, Stefano Magrini Alunno"
author = "Stefano Magrini Alunno"
release = "v0.0.0"
manpages_url = "https://manpages.debian.org/{path}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../../source"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]  # type: ignore

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


def setup(app):
    doxygen_src = os.path.abspath("docs/source/doxygen")
    doxygen_dst = os.path.abspath("docs/build/html/doxygen")

    # Verifica se la sorgente esiste
    if not os.path.exists(doxygen_src):
        print(f"Error: Source directory {doxygen_src} does not exist.")
        return

    # Rimuovi la directory di destinazione se esiste
    if os.path.exists(doxygen_dst):
        shutil.rmtree(doxygen_dst)

    # Copia la directory
    try:
        shutil.copytree(doxygen_src, doxygen_dst)
        print(f"Copied Doxygen documentation from {doxygen_src} to {doxygen_dst}")
    except Exception as e:
        print(f"Error copying Doxygen documentation: {e}")


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "top",
}
html_static_path = ["_static"]
