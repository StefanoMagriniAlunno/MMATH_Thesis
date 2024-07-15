# MMATH Thesis Project

Autori e finalità della repository...

## Prerequisites
- `Linux Ubuntu` 22.04
- `Python` 3.10.12 with:
  - `pip` 24.1.2
  - `virtualenv` 20.26.3
  - `python3-dev` 3.10.6-1~22.04
- `GCC` 11.4.0 with:
  - `doxygen` 1.9.1
  - `graphviz`
- `CUDA 12.2` with `nvcc` 11.5 (required NVIDIA drivers)

## Download and Install the repository
```bash
# Download repository
cd /path/where/download/repository
git clone https://github.com/StefanoMagriniAlunno/MMATH_thesis
cd MMATH_thesis
# Install repository
./install.sh /path/of/python/executable
```

# Use the repository

## Contents
### Documentation
This repository uses:
- `Sphinx` to create documentation for Python scripts.
- `Doxygen` to create documentation for C/C++ and Java programs.
Parlare di docs per i dettagli del codice
### Base packages for repository
```bash
pip3 install invoke pre-commit pytest jupyter sphinx esbonio breathe
pip3 install flake8 doc8 mypy black autoflake isort shellcheck-py
pip3 freeze > requirements.txt
```
### How to add own packages and contents
Open **tasks.py**:
- `install` task adds packages
- `download` task adds dataset, models and other media.

## Pre-commit


# Issue
