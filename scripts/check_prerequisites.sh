#!/bin/bash

# Params
req_linux_release="Ubuntu"
req_linux_version="22.04"
req_python3_version="3.10.12"
req_python3_pip_version="24.2"
req_python3_virtualenv_version="20.26.3"
req_python3_dev_version="3.10.6-1~22.04"
req_CC="gcc"
req_CC_version="11.4.0"
req_CXX="g++"
req_CXX_version="11.4.0"
req_CC_doxygen_version="1.9.1"
req_nvcc_version="12.5"

python3_executable=$1

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
python3_version=$($python3_executable --version | cut -d " " -f 2)
if [ "$python3_version" != "$req_python3_version" ]; then
    echo -e "\e[31mERROR\e[0m Python3 version must be $req_python3_version (read $python3_version)"
    exit 1
fi
if ! "$python3_executable" -m pip --version > /dev/null; then
    echo -e "\e[31mERROR\e[0m pip3 is not installed"
    exit 1
fi
python3_pip_version=$($python3_executable -m pip --version | cut -d " " -f 2)
if [ "$python3_pip_version" != "$req_python3_pip_version" ]; then
    echo -e "\e[31mERROR\e[0m pip3 version must be $req_python3_pip_version (read $python3_pip_version)"
    exit 1
fi
if ! "$python3_executable" -m virtualenv --version > /dev/null; then
    echo -e "\e[31mERROR\e[0m virtualenv is not installed"
    exit 1
fi
python3_virtualenv_version=$($python3_executable -m virtualenv --version | cut -d " " -f 2)
if [ "$python3_virtualenv_version" != "$req_python3_virtualenv_version" ]; then
    echo -e "\e[31mERROR\e[0m virtualenv version must be $req_python3_virtualenv_version (read $python3_virtualenv_version)"
    exit 1
fi
python3_dev_version=$(dpkg -s python3-dev | grep Version | cut -d " " -f 2)
if [ "$python3_dev_version" != "$req_python3_dev_version" ]; then
    echo -e "\e[31mERROR\e[0m python3-dev version must be $req_python3_dev_version (read $python3_dev_version)"
    exit 1
fi
if ! "$req_CC" --version > /dev/null; then
    echo -e "\e[31mERROR\e[0m gcc is not installed"
    exit 1
fi
CC_version=$($req_CC --version | head -n 1 | cut -d " " -f 4)
if [ "$CC_version" != "$req_CC_version" ]; then
    echo -e "\e[31mERROR\e[0m gcc version must be $req_CC_version (read $CC_version)"
    exit 1
fi
if ! "$req_CXX" --version > /dev/null; then
    echo -e "\e[31mERROR\e[0m g++ is not installed"
    exit 1
fi
CXX_version=$($req_CXX --version | head -n 1 | cut -d " " -f 4)
if [ "$CXX_version" != "$req_CXX_version" ]; then
    echo -e "\e[31mERROR\e[0m g++ version must be $req_CXX_version (read $CXX_version)"
    exit 1
fi
CC_doxygen_version=$(doxygen --version | cut -d " " -f 2)
if [ "$CC_doxygen_version" != "$req_CC_doxygen_version" ]; then
    echo -e "\e[31mERROR\e[0m doxygen version must be $req_CC_doxygen_version (read $CC_doxygen_version)"
    exit 1
fi
if ! dot -V; then
    echo -e "\e[31mERROR\e[0m graphviz is not installed"
    exit 1
fi
if ! [ "$(nvcc --version)" ]; then
    echo -e "\e[31mERROR\e[0m nvcc is not installed"
    exit 1
fi
nvcc_version=$(nvcc --version | grep release | cut -d " " -f 5)
if [ "$nvcc_version" != "$req_nvcc_version," ]; then
    echo -e "\e[31mERROR\e[0m nvcc version must be $req_nvcc_version (read $nvcc_version)"
    exit 1
fi

exit 0
