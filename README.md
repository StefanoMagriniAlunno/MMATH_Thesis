# MMATH Thesis Project

Autori e finalità della repository...

## Prerequisites
- Linux Ubuntu/Debian system
- Python 3.10.12 with:
  -  pip 24.1.2
  -  virtualenv 20.26.3
- GCC 11.4.0
- CUDA 12.2 with `nvcc` 11.5 (required NVIDIA drivers)

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
Parlare della licenza
Parlare di docs per i dettagli del codice
### installazioni per la repo
```bash
pip3 install invoke pre-commit pytest jupyter sphinx esbonio
pip3 install flake8 doc8 mypy black autoflake isort shellcheck-py
pip3 freeze > requirements.txt
```

- installazioni per il codice

## Pre-commit
