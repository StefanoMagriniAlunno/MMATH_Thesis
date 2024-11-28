# MMATH Thesis Project

## Prerequisites
- `Linux Ubuntu` 22.04
- `Python` 3.10.12 with:
  - `pip` 24.3.1
  - `virtualenv` 20.26.6
  - `python3-dev` 3.10.6-1~22.04.1
  - `python3-dbg` 3.10.12
- `gcc` & `gxx` 12.3.0 with:
  - `doxygen` 1.9.1
  - `graphviz`
- `CUDA 12.4` with `nvcc` 12.4 (required NVIDIA drivers)

## Download and Install the repository
```bash
# Download repository
cd /path/where/download/repository
git clone https://github.com/StefanoMagriniAlunno/MMATH_thesis
cd MMATH_thesis
# Install repository
./repo.sh -i
./repo.sh -a start build.sh
./repo.sh -a start pre_commit.sh
./repo.sh -a start make_doc.sh
./repo.sh -a start cuda_check.sh
```

# Use the repository

## Contents
### Documentation
This repository uses:
- `Sphinx` to create documentation for Python scripts.
- `Doxygen` to create documentation for C/C++, Fortran and Java programs.

Open complete documentation in `docs/_build/html/index.html` with your browser.
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

When you add a project, insert the path of the project in the Doxyfile in the `INPUTS` key:
```bash
INPUT                  = source/projects/your_project/main.c \
                         source/projects/your_project/inc \
                         source/projects/your_project/src
```

## Pre-commit

This repository use pre-commit:
- pre-commit-hooks
  - trailing-whitespace
  - end-of-file-fixer
  - mixed-line-ending
  - check-yaml
  - check-json
  - check-docstring-first
  - sort-simple-yaml
  - pretty-format-json --autofix
- flake8 --ignore=E203,E501,W503
- doc8 --ignore=D001
- autoflake
- isort
- mypy
- shellcheck
- black

# Usage
Start the script
```bash
  python3 source/main.py
```

Start a debug session
```bash
```
