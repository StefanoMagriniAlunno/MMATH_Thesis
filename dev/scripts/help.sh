#!/bin/bash

echo "Usage: ${0##*/} [-h|--help] [-b|--base] [-i|--install] [-r|--reinstall] [-u|--update] [package [package ...]]"
echo
echo "This script initialize the repository."
echo "Check conf.ini to set parameters of the installation."
echo
echo "Options:"
echo "  -h, --help            Show this help message and exit"
echo "  -i, --install         Download and install all the contents of the repository"
echo "  -r, --reinstall       Reinstall all the contents of the repository from the base"
echo "  -a, --addons          Start a personalization of the repository"
echo
echo
echo
echo "Install"
echo "  -i, --install         Install all the contents of the repository"
echo "                        -i base                       Download packages"
echo "                        -i venv                       Install virtual environment"
echo
echo "Reinstall"
echo "  -r, --reinstall       Reinstall all the contents of the repository"
echo "                        -r base             Redownload the base packages"
echo "                                            so reinstall the virtual environment"
echo "                        -r venv             Reinstall the virtual environment"
echo
echo
echo "AddOns"
echo "  -a, --addons          Start a personalization of the repository"
echo "                        -a show             Show the list of addons"
echo "                        -a start addon      Start the addon with a configuration file"
echo
echo
