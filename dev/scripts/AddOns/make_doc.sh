#!/bin/bash


# Parameters
venv_dir=$(grep "venv_dir" "$1" | cut -d'=' -f2)
config_dir=$(grep "sphinx__config_dir" "$1" | cut -d'=' -f2)
req_doxygen_version=$(grep "sphinx__doxygen_version" "$1" | cut -d'=' -f2)

sphinx_cmd="$(pwd)/$venv_dir/bin/sphinx-build"


doxygen_version=$(doxygen --version | cut -d " " -f 2)
if [ "$doxygen_version" != "$req_doxygen_version" ]; then
    echo
    echo -e "\e[31mERROR\e[0m doxygen version must be $req_doxygen_version (read $doxygen_version)"
    echo
    echo
    exit 1
fi
if ! dot -V; then
    echo
    echo -e "\e[31mERROR\e[0m graphviz is not installed"
    echo
    echo
    exit 1
fi


if ! doxygen Doxyfile; then
    echo
    echo -e "\e[31mERROR\e[0m doxygen command failed"
    echo
    echo
    exit 1
fi
mkdir -p docs/_build
mkdir -p docs/source/_static
if ! "$sphinx_cmd" -b html "$config_dir" docs/build/html; then
    echo
    echo -e "\e[31mERROR\e[0m: An error occurred while making documentation"
    echo
    echo
    exit 1
fi
