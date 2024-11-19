#!/bin/bash

# Parameters
driver_nvidia_minversion=$(grep "cuda_check__driver_nvidia_minversion" "$1" | cut -d "=" -f 2)
gpu_uuid=$(grep "cuda_check__gpu_uuid" "$1" | cut -d "=" -f 2)
cuda_check__linux_kernel_version=$(grep "cuda_check__linux_kernel_version" "$1" | cut -d "=" -f 2)

# check linux kernel version
if [ "$(uname -r)" != "$cuda_check__linux_kernel_version" ]; then
    echo
    echo -e "\e[31mERROR\e[0m Linux kernel version is not correct"
    echo "Expected version: $cuda_check__linux_kernel_version"
    echo "Installed version: $(uname -r)"
    echo "Please check the kernel version"
    echo
    echo
    exit 1
fi

# check nvidia-smi command
if [ -z "$(command -v nvidia-smi)" ]; then
    echo
    echo -e "\e[31mERROR\e[0m nvidia-smi command not found"
    echo "Please install nvidia driver"
    echo
    echo
    exit 1
fi

# check gpu
if [ -z "$(nvidia-smi --query-gpu=name --id="$gpu_uuid" --format=csv,noheader)" ]; then
    echo
    echo -e "\e[31mERROR\e[0m GPU with UUID $gpu_uuid not found"
    echo "Please check the UUID in the config file"
    echo
    echo
    exit 1
fi

# check driver version
driver_version=$(nvidia-smi --query-gpu=driver_version --id="$gpu_uuid" --format=csv,noheader)
if ! dpkg --compare-versions "$driver_version" ge "$driver_nvidia_minversion"; then
    echo
    echo -e "\e[31mERROR\e[0m Nvidia driver version is not correct"
    echo "Expected version: >=$driver_nvidia_minversion"
    echo "Installed version: $driver_version"
    echo "Please check the UUID or the driver version"
    echo
    echo
    exit 1
fi

exit 0
