#!/bin/bash

venv_pip_version="24.1.2"
python3_executable=$1

# System prerequisites
scripts/check_prerequisites.sh "$python3_executable"
"$python3_executable" -m virtualenv ".venv" --no-download --always-copy --prompt="MMATH_thesis"
python3_cmd="$(pwd)/.venv/bin/python3"

# Install base packages
"$python3_cmd" -m pip install --upgrade pip=="$venv_pip_version"
"$python3_cmd" -m pip install invoke pre-commit pytest jupyter sphinx esbonio breathe
"$python3_cmd" -m pip install flake8 doc8 mypy black autoflake isort shellcheck-py
"$python3_cmd" -m pip freeze > requirements.txt

# remove .venv
rm -rf .venv
