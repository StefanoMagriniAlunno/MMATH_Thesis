#!/bin/bash

# Read input parameters:
# $1: mode of installation
# $2: python3 command of the system
# $3: environment directory
# $4: pip version of environment
# $5: cache directory
# $6: requirements file
# $7: prompt of the virtual environment

# if mode is --venv we create the virtual environment with all packages
if [ "$1" == "--venv" ]; then
    if [ -d "$3" ]; then
        echo
        echo -e "\e[33mWARNING\e[0m Nothing to do"
        echo "If you want to reinstall the environment try:"
        echo "./repo.sh --reinstall venv"
        echo
        echo
        exit 0
    else
        if ! "$2" -m virtualenv "$3" --prompt="$7" --pip "$4"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to create virtual environment"
            echo
            echo
            exit 1
        fi
        venv_python3_cmd="$3/bin/python3"
        # Install packages
        if ! "$venv_python3_cmd" -m pip install --compile --no-index --find-links="$5" -r "$6" ; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to install packages"
            echo
            echo
            exit 1
        fi
    fi
    echo
    echo -e "\e[32mSUCCESS\e[0m Virtual environment created"
    echo "To activate the environment run:"
    echo "source .venv/bin/activate"
    echo
    echo
    exit 0
# If mode is --base we install the base packages
elif [ "$1" == "--base" ]; then
    if [ -d "$3" ]; then
        echo
        echo -e "\e[33mWARNING\e[0m Nothing to do"
        echo "If you want to reinstall the base and the environment try:"
        echo "./repo.sh --reinstall base"
        echo
        echo
        exit 0
    fi
    # se esiste la cache
    if [ -d "$5" ]; then
        echo
        echo -e "\e[33mWARNING\e[0m Nothing to do"
        echo "If you want to reinstall the base:"
        echo "./repo.sh --reinstall base"
        echo
        echo
        exit 0
    else
        if ! mkdir -p "$5"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to create cache directory"
            echo
            echo
            exit 1
        fi
    fi
    # Download base packages

    if ! "$2" -m pip download --no-cache-dir --dest "$5" -r "$6" --timeout 99999; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to download base packages"
        echo
        echo
        exit 1
    fi
    echo
    echo -e "\e[32mSUCCESS\e[0m Base packages installed"
    echo
    echo
    exit 0
else
    echo
    echo -e "\e[31mERROR\e[0m Invalid option"
    echo
    echo
    exit 1
fi
