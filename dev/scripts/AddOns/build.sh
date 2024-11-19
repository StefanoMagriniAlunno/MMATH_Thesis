#!/bin/bash

# Parameters
venv_dir=$(grep "venv_dir" "$1" | cut -d'=' -f2)
venv_dir=$(pwd)/"$venv_dir"
local_python=$(grep "local_python" "$1" | cut -d'=' -f2)
source_dir=$(grep "build__source" "$1" | cut -d'=' -f2)
source_dir=$(pwd)/"$source_dir"
gcc_cmd=$(grep "build__gcc_cmd" "$1" | cut -d'=' -f2)
gxx_cmd=$(grep "build__gxx_cmd" "$1" | cut -d'=' -f2)
make_cmd=$(grep "build__make_cmd" "$1" | cut -d'=' -f2)
cmake_cmd=$(grep "build__cmake_cmd" "$1" | cut -d'=' -f2)
nvcc_cmd=$(grep "build__nvcc_cmd" "$1" | cut -d'=' -f2)
cache_dir=$(grep "build__cache" "$1" | cut -d'=' -f2)
cache_dir=$(pwd)/"$cache_dir"
log_file=$(grep "build__log_file" "$1" | cut -d'=' -f2)
log_file=$(pwd)/"$log_file"
req_gcc_version=$(grep "build__gcc_version" "$1" | cut -d'=' -f2)
req_gxx_version=$(grep "build__gxx_version" "$1" | cut -d'=' -f2)
req_cmake_version=$(grep "build__cmake_version" "$1" | cut -d'=' -f2)
req_make_version=$(grep "build__make_version" "$1" | cut -d'=' -f2)
req_nvcc_version=$(grep "build__nvcc_version" "$1" | cut -d'=' -f2)

if ! "$gcc_cmd" --version > /dev/null; then
    echo
    echo -e "\e[31mERROR\e[0m gcc is not installed"
    echo
    echo
    exit 1
fi
gcc_version=$($gcc_cmd --version | head -n 1 | cut -d " " -f 4)
if [ "$gcc_version" != "$req_gcc_version" ]; then
    echo
    echo -e "\e[31mERROR\e[0m gcc version must be $req_gcc_version (read $gcc_version)"
    echo
    echo
    exit 1
fi
if ! "$gxx_cmd" --version > /dev/null; then
    echo
    echo -e "\e[31mERROR\e[0m g++ is not installed"
    echo
    echo
    exit 1
fi
gxx_version=$($gxx_cmd --version | head -n 1 | cut -d " " -f 4)
if [ "$gxx_version" != "$req_gxx_version" ]; then
    echo
    echo -e "\e[31mERROR\e[0m g++ version must be $req_gxx_version (read $gxx_version)"
    echo
    echo
    exit 1
fi
if ! [ "$($nvcc_cmd --version)" ]; then
    echo
    echo -e "\e[31mERROR\e[0m nvcc is not installed"
    echo
    echo
    exit 1
fi
nvcc_version=$($nvcc_cmd --version | grep release | cut -d " " -f 5)
if [ "$nvcc_version" != "$req_nvcc_version," ]; then
    echo
    echo -e "\e[31mERROR\e[0m nvcc version must be $req_nvcc_version (read $nvcc_version)"
    echo
    echo
    exit 1
fi
if ! [ "$($cmake_cmd --version)" ]; then
    echo
    echo -e "\e[31mERROR\e[0m cmake is not installed"
    echo
    echo
    exit 1
fi
cmake_version=$($cmake_cmd --version | grep version | cut -d " " -f 3)
if [ "$cmake_version" != "$req_cmake_version" ]; then
    echo
    echo -e "\e[31mERROR\e[0m cmake version must be $req_cmake_version (read $cmake_version)"
    echo
    echo
    exit 1
fi
if ! [ "$($make_cmd --version)" ]; then
    echo
    echo -e "\e[31mERROR\e[0m cmake is not installed"
    echo
    echo
    exit 1
fi
make_version=$("$make_cmd" --version | head -n 1 | cut -d " " -f 3)
if [ "$make_version" != "$req_make_version" ]; then
    echo
    echo -e "\e[31mERROR\e[0m make version must be $req_make_version (read $make_version)"
    echo
    echo
    exit 1
fi

# creo il file vuoto $log_file
mkdir -p "$(dirname "$log_file")"
touch "$log_file"

mkdir -p "$cache_dir"
"$make_cmd" -C "$source_dir" PY="$local_python" CC="$gcc_cmd" CXX="$gxx_cmd" CU="$nvcc_cmd" > "$log_file" 2>&1
cp "$cache_dir"/* "$venv_dir"/lib/python3.10/site-packages
