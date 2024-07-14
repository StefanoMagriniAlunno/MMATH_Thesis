#!/bin/bash

# Params
req_python3_version="3.10.12"
req_python3_pip_version="24.1.2"
req_python3_virtualenv_version="20.26.3"
req_CC="gcc"
req_CC_version="11.4.0"
req_nvcc_version="11.5"
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
if ! [ -f /etc/debian_version ]; then
    echo -e "\e[31mERROR\e[0m This script is only for Ubuntu/Debian systems"
    exit 1
fi
python3_version=$($python3_executable --version | cut -d " " -f 2)
if [ "$python3_version" != "$req_python3_version" ]; then
    echo -e "\e[31mERROR\e[0m Python3 version must be 3.10.12"
    exit 1
fi
if ! [ "$python3_executable -m pip --version" ]; then
    echo -e "\e[31mERROR\e[0m pip3 is not installed"
    exit 1
fi
python3_pip_version=$($python3_executable -m pip --version | cut -d " " -f 2)
if [ "$python3_pip_version" != "$req_python3_pip_version" ]; then
    echo -e "\e[31mERROR\e[0m pip3 version must be 24.1.2"
    exit 1
fi
if ! [ "$python3_executable -m virtualenv --version" ]; then
    echo -e "\e[31mERROR\e[0m virtualenv is not installed"
    exit 1
fi
python3_virtualenv_version=$($python3_executable -m virtualenv --version | cut -d " " -f 2)
if [ "$python3_virtualenv_version" != "$req_python3_virtualenv_version" ]; then
    echo -e "\e[31mERROR\e[0m virtualenv version must be 20.26.3"
    exit 1
fi
if ! [ "$req_CC --version" ]; then
    echo -e "\e[31mERROR\e[0m gcc is not installed"
    exit 1
fi
req_CC_version=$($req_CC --version | head -n 1 | cut -d " " -f 3)
if [ "$req_CC_version" != "$req_CC_version" ]; then
    echo -e "\e[31mERROR\e[0m gcc version must be 11.4.0"
    exit 1
fi
if ! [ "$(nvcc --version)" ]; then
    echo -e "\e[31mERROR\e[0m nvcc is not installed"
    exit 1
fi
nvcc_version=$(nvcc --version | grep release | cut -d " " -f 5)
if [ "$nvcc_version" != "$req_nvcc_version," ]; then
    echo -e "\e[31mERROR\e[0m nvcc version must be 11.5"
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
    "$python3_cmd" assets/finish_error.py
    exit 1
fi
echo "install requirements.txt..."
if ! "$python3_cmd" -m pip install -r requirements.txt >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install packages"
    "$python3_cmd" assets/finish_error.py
    exit 1
fi
echo "invoke install..."
if ! "$invoke_cmd" install >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install packages"
    "$python3_cmd" assets/finish_error.py
    exit 1
fi
echo -e "\e[32mSUCCESS\e[0m Virtual environment prepared"

# Prepare repository
echo -e "\e[33mPreparing repository...\e[0m"
echo "pre-commit install..."
if ! "$pre_commit_cmd" install >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install pre-commit"
    "$python3_cmd" assets/finish_error.py
    exit 1
fi
echo "pre-commit install-hooks..."
if ! "$pre_commit_cmd" install-hooks >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to install pre-commit hooks"
    "$python3_cmd" assets/finish_error.py
    exit 1
fi
echo "sphinx build..."
if ! "$sphinx_cmd" -b html "$sphinx_config_dir" docs/_build/html; then
    echo -e "\e[31mERROR\e[0m: An error occurred while making documentation"
    "$python3_cmd" assets/finish_error.py
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
echo "invoke make..."
if ! "$invoke_cmd" build >> "$logfile" 2>&1; then
    echo -e "\e[31mERROR\e[0m Failed to build packages"
    exit 1
fi
echo -e "\e[32mSUCCESS\e[0m Repository prepared"

# Done
$python3_cmd assets/done.py

echo ""
echo "Please activate the virtual environment with the following command:"
echo "$ source $venv/bin/activate"
echo ""
