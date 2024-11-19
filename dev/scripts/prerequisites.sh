#!/bin/bash

# Read input from user
# $1 is the configuration file

req_linux_release=$(grep "linux_release" "$1" | cut -d'=' -f2)
req_linux_version=$(grep "linux_version" "$1" | cut -d'=' -f2)
req_python3_version=$(grep "python3_version" "$1" | cut -d'=' -f2)
req_python3_pip_version=$(grep "python3_pip_version" "$1" | cut -d'=' -f2)
req_python3_virtualenv_version=$(grep "python3_virtualenv_version" "$1" | cut -d'=' -f2)
req_python3_dev_version=$(grep "python3_dev_version" "$1" | cut -d'=' -f2)

python3_cmd=$(grep "local_python3" "$1" | cut -d'=' -f2)

if [ "$(uname)" != "Linux" ]; then
    echo -e "\e[31mERROR\e[0m This script is for Linux only"
    exit 1
fi
linux_release=$(lsb_release -i | cut -d ":" -f 2 | tr -d "[:space:]")
if [ "$linux_release" != "$req_linux_release" ]; then
    echo -e "\e[31mERROR\e[0m Linux release must be Ubuntu (read $linux_release)"
    exit 1
fi
linux_version=$(lsb_release -r | cut -d ":" -f 2 | tr -d "[:space:]")
if [ "$linux_version" != "$req_linux_version" ]; then
    echo -e "\e[31mERROR\e[0m Linux version must be $req_linux_version (read $linux_version)"
    exit 1
fi
python3_version=$($python3_cmd --version | cut -d " " -f 2)
if [ "$python3_version" != "$req_python3_version" ]; then
    echo -e "\e[31mERROR\e[0m Python3 version must be $req_python3_version (read $python3_version)"
    exit 1
fi
if ! "$python3_cmd" -m pip --version > /dev/null; then
    echo -e "\e[31mERROR\e[0m pip3 is not installed"
    exit 1
fi
python3_pip_version=$($python3_cmd -m pip --version | cut -d " " -f 2)
if [ "$python3_pip_version" != "$req_python3_pip_version" ]; then
    echo -e "\e[31mERROR\e[0m pip3 version must be $req_python3_pip_version (read $python3_pip_version)"
    exit 1
fi
if ! "$python3_cmd" -m virtualenv --version > /dev/null; then
    echo -e "\e[31mERROR\e[0m virtualenv is not installed"
    exit 1
fi
python3_virtualenv_version=$($python3_cmd -m virtualenv --version | cut -d " " -f 2)
if [ "$python3_virtualenv_version" != "$req_python3_virtualenv_version" ]; then
    echo -e "\e[31mERROR\e[0m virtualenv version must be $req_python3_virtualenv_version (read $python3_virtualenv_version)"
    exit 1
fi
python3_dev_version=$(dpkg -s python3-dev | grep Version | cut -d " " -f 2)
if [ "$python3_dev_version" != "$req_python3_dev_version" ]; then
    echo -e "\e[31mERROR\e[0m python3-dev version must be $req_python3_dev_version (read $python3_dev_version)"
    exit 1
fi

exit 0
