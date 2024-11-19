#!/bin/bash

# Read input from user
# if the user not provide any input, then we write a message to the user
if [ $# -eq 0 ]; then
    echo "No arguments provided, try -h or --help"
    exit 1
fi

# READ PARAMETERS FROM CONF.INI
conf_ini="./dev/conf.ini"
if [ ! -f "$conf_ini" ]; then
    echo
    echo -e "\e[31mERROR\e[0m The file conf.ini does not exist"
    echo
    echo
    exit 1
fi

# If the user provide -h or --help, then we show the help message
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    ./dev/scripts/help.sh
    exit 0
# If the user provide -p or --prerequisites
elif [ "$1" == "-p" ] || [ "$1" == "--prerequisites" ]; then
    ./dev/scripts/prerequisites.sh "$conf_ini"
# If the option is -i or --install, then we show the install message
elif [ "$1" == "-i" ] || [ "$1" == "--install" ]; then
    # get from conf.ini parameters
    local_python3=$(grep "local_python3" "$conf_ini" | cut -d "=" -f 2)
    venv_dir=$(grep "venv_dir" "$conf_ini" | cut -d "=" -f 2)
    venv_dir=$(pwd)/"$venv_dir"
    venv_pip_version=$(grep "venv_pip_version" "$conf_ini" | cut -d "=" -f 2)
    cache_dir=$(grep "cache_dir" "$conf_ini" | cut -d "=" -f 2)
    cache_dir=$(pwd)/"$cache_dir"
    requirements_file=$(grep "requirements_file" "$conf_ini" | cut -d "=" -f 2)
    requirements_file=$(pwd)/"$requirements_file"
    venv_name=$(grep "venv_name" "$conf_ini" | cut -d "=" -f 2)
    # If the option does not have any argument, then we use the default option --base
    if [ $# -eq 1 ]; then
        ./dev/scripts/install.sh --base "$local_python3" "$venv_dir" "$venv_pip_version" "$cache_dir" "$requirements_file"
        ./dev/scripts/install.sh --venv "$local_python3" "$venv_dir" "$venv_pip_version" "$cache_dir" "$requirements_file" "$venv_name"
    elif [ "$2" == "base" ]; then
        ./dev/scripts/install.sh --base "$local_python3" "$venv_dir" "$venv_pip_version" "$cache_dir" "$requirements_file" "$venv_name"
    elif [ "$2" == "venv" ]; then
        ./dev/scripts/install.sh --venv "$local_python3" "$venv_dir" "$venv_pip_version" "$cache_dir" "$requirements_file" "$venv_name"
    else
        echo
        echo -e "\e[31mERROR\e[0m Invalid option"
        echo
        echo
        exit 1
    fi
# If the option is -r or --reinstall, then we show the reinstall message
elif [ "$1" == "-r" ] || [ "$1" == "--reinstall" ]; then
    # get from conf.ini parameters
    local_python3=$(grep "local_python3" "$conf_ini" | cut -d "=" -f 2)
    venv_dir=$(grep "venv_dir" "$conf_ini" | cut -d "=" -f 2)
    venv_pip_version=$(grep "venv_pip_version" "$conf_ini" | cut -d "=" -f 2)
    cache_dir=$(grep "cache_dir" "$conf_ini" | cut -d "=" -f 2)
    requirements_file=$(grep "requirements_file" "$conf_ini" | cut -d "=" -f 2)
    venv_name=$(grep "venv_name" "$conf_ini" | cut -d "=" -f 2)
    if [ $# -eq 1 ]; then
        echo
        echo -e "\e[31mERROR\e[0m Invalid option"
        echo
        echo
        exit 1
    elif [ "$2" == "base" ]; then
        ./dev/scripts/reinstall.sh --base "$local_python3" "$venv_dir" "$venv_pip_version" "$cache_dir" "$requirements_file" "$venv_name"
    elif [ "$2" == "venv" ]; then
        ./dev/scripts/reinstall.sh --venv "$local_python3" "$venv_dir" "$venv_pip_version" "$cache_dir" "$requirements_file" "$venv_name"
    else
        echo
        echo -e "\e[31mERROR\e[0m Invalid option"
        echo
        echo
        exit 1
    fi
# If the option is -a or -addOns, then we show the addons message
elif [ "$1" == "-a" ] || [ "$1" == "--addons" ]; then
    if [ $# -eq 1 ]; then
        ./dev/scripts/addons.sh --show
    elif [ "$2" == "show" ]; then
        ./dev/scripts/addons.sh --show
    elif [ "$2" == "start" ]; then
        ./dev/scripts/addons.sh --start "$3" "$conf_ini"
    else
        echo
        echo -e "\e[31mERROR\e[0m Invalid option"
        echo
        echo
        exit 1
    fi
else
    echo
    echo -e "\e[31mERROR\e[0m Invalid option"
    echo
    echo
    exit 1
fi
