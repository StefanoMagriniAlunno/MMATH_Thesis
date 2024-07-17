#!/bin/bash

# Read inputs
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python3_executable>"
    exit 1
fi
python3_executable=$1

# Delete log file
rm -f temp/installation.log

# Delete virtual environment
rm -rf .venv

# Delete temp
rm -rf temp

# Delete documentation build
rm -rf docs/_build
rm -rf docs/source/_static

# Delete lib in source
rm -rf source/lib

# Installation
bash install.sh "$python3_executable"
