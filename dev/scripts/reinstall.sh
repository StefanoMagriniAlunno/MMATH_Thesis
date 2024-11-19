#!/bin/bash

# Read input parameters:
# $1: mode of installation
# $2: python3 command of the system
# $3: environment directory
# $4: pip version of environment
# $5: cache directory
# $6: requirements file
# $7: prompt of the virtual environment

# if mode is --venv we recreate the virtual environment with all packages
if [ "$1" == "--venv" ]; then
    if [ -d "$3" ]; then
        if ! rm -rf "$3"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to remove virtual environment"
            echo
            echo
            exit 1
        fi
        # create the virtual environment
        if ! "$2" -m virtualenv "$3" --prompt="$7" --pip "$4"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to create virtual environment"
            echo
            echo
            exit 1
        fi
        venv_python3_cmd="$3"/bin/python3
        # Install packages
        if ! "$venv_python3_cmd" -m pip install --compile --no-index --find-links="$5" -r "$6"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to install packages"
            echo
            echo
            exit 1
        fi
    else
        echo
        echo -e "\e[33mWARNING\e[0m Nothing to do"
        echo "If you want to install the environment try:"
        echo "./repo.sh --install venv"
        echo
        echo
        exit 0
    fi
    echo
    echo -e "\e[32mSUCCESS\e[0m Virtual environment created"
    echo "To activate the environment run:"
    echo "source .venv/bin/activate"
    echo
    echo
    echo 0
# If mode is --base we install the base packages
elif [ "$1" == "--base" ]; then
    # se esiste l'environment lo rimuovo
    if [ -d "$3" ]; then
        if ! rm -rf "$3"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to remove virtual environment"
            echo
            echo
            exit 1
        fi
    fi
    if [ -d "$5" ]; then
        if ! rm -rf "$5"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to remove cache directory"
            echo
            echo
            exit 1
        fi
        # ricreo la cache
        if ! mkdir -p "$5"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to create cache directory"
            echo
            echo
            exit 1
        fi
        # Download base packages
        if ! "$2" -m pip download --no-cache-dir --dest "$5" -r "$6"; then
            echo
            echo -e "\e[31mERROR\e[0m Failed to download base packages"
            echo
            echo
            exit 1
        fi
    else
        echo
        echo -e "\e[33mWARNING\e[0m Nothing to do"
        echo "If you want to install the base try:"
        echo "./repo.sh --install base"
        echo
        echo
        exit 0
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
