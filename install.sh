#!/bin/bash

# Params
logfile="temp/installation.log"
venv=".venv"
venv_pip_version="24.1.2"
sphinx_config_dir="docs/source"

# Read inputs
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python3_executable>"
    exit 1
fi
python3_executable=$1
# controllo se python3_executable è effettivamente un eseguibile di python3
if ! [ -x "$(command -v $python3_executable)" ]; then
    echo -e "\e[31mERROR\e[0m $1 is not an executable"
    exit 1
fi

# Make log file
mkdir -p temp
> "$logfile"

# System prerequisites
echo -e "\e[33mChecking prerequisites...\e[0m"
if ! scripts/check_prerequisites.sh "$python3_executable"; then
    echo -e "\e[31mERROR\e[0m Prerequisites check failed"
    exit 1
fi
echo -e "\e[32mSUCCESS\e[0m Prerequisites check completed"

# Make the environment with virtualenv
echo -e "\e[33mCreating virtual environment...\e[0m"
if ! "$python3_executable" -m virtualenv "$venv" --no-download --always-copy --prompt="MMATH_thesis" >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to create virtual environment or save logs"
    exit 1
fi
echo -e "\e[32mSUCCESS\e[0m Virtual environment created"

# New params
python3_cmd="$(pwd)/$venv/bin/python3"
pre_commit_cmd="$(pwd)/$venv/bin/pre-commit"
invoke_cmd="$(pwd)/$venv/bin/invoke"
sphinx_cmd="$(pwd)/$venv/bin/sphinx-build"

# Install packages for the virtual environment
echo -e "\e[33mPreparing virtual environment...\e[0m"
echo "pip upgrade..."
if ! "$python3_cmd" -m pip install --upgrade pip=="$venv_pip_version" >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to upgrade pip"
    "$python3_cmd" assets/error.py
    exit 1
fi
echo "install requirements.txt..."
if ! "$python3_cmd" -m pip install -r requirements.txt >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install packages"
    "$python3_cmd" assets/error.py
    exit 1
fi
echo "invoke install..."
if ! "$invoke_cmd" install >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install packages"
    "$python3_cmd" assets/error.py
    exit 1
fi
echo "invoke build..."
if ! "$invoke_cmd" build >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to build packages"
    exit 1
fi
echo -e "\e[32mSUCCESS\e[0m Virtual environment prepared"

# Prepare repository
echo -e "\e[33mPreparing repository...\e[0m"
echo "pre-commit install..."
if ! "$pre_commit_cmd" install >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install pre-commit"
    "$python3_cmd" assets/error.py
    exit 1
fi
echo "pre-commit install-hooks..."
if ! "$pre_commit_cmd" install-hooks >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install pre-commit hooks"
    "$python3_cmd" assets/error.py
    exit 1
fi
if ! doxygen Doxyfile >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to generate Doxygen documentation"
    "$python3_cmd" assets/error.py
    exit 1
fi
echo "sphinx build..."
mkdir -p docs/_build
mkdir -p docs/source/_static
if ! "$sphinx_cmd" -b html "$sphinx_config_dir" docs/_build/html >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m: An error occurred while making documentation"
    "$python3_cmd" assets/error.py
    exit 1
fi
echo "create data directory..."
mkdir -p data
echo "invoke directories..."
if ! "$invoke_cmd" directories >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to create directories"
    exit 1
fi
echo "invoke download..."
if ! "$invoke_cmd" download >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to download data"
    exit 1
fi
echo -e "\e[32mSUCCESS\e[0m Repository prepared"

# Done
$python3_cmd assets/done.py

echo ""
echo "Please activate the virtual environment with the following command:"
echo "$ source $venv/bin/activate"
echo ""
