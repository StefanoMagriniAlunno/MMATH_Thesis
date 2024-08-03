#!/bin/bash

logfile=$1
python3_cmd=$2
sphinx_cmd=$3
sphinx_config_dir=$4

if ! doxygen Doxyfile >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to generate Doxygen documentation"
    "$python3_cmd" assets/error.py
    exit 1
fi
mkdir -p docs/_build
mkdir -p docs/source/_static
if ! "$sphinx_cmd" -b html "$sphinx_config_dir" docs/build/html >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m: An error occurred while making documentation"
    "$python3_cmd" assets/error.py
    exit 1
fi
