#!/bin/bash

# Delete log file
rm -f temp/installation.log
rm -f temp/build.log

# Delete virtual environment
rm -rf .venv

# Delete temp
rm -rf temp

# Delete documentation build
rm -rf docs/_build
rm -rf docs/source/_static

# Delete .cache in source
rm -rf source/.cache

# Installation
bash install.sh
