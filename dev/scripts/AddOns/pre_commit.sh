#!/bin/bash

# Parameters
venv_dir=$(grep "venv_dir" "$1" | cut -d "=" -f 2)

pre_commit_cmd="./$venv_dir/bin/pre-commit"
pre_commit_cmd=$(realpath "$pre_commit_cmd")

# check pre-commit command
if [ -z "$(command -v "$pre_commit_cmd")" ]; then
    echo
    echo -e "\e[31mERROR\e[0m pre-commit command not found"
    echo "Please install pre-commit"
    echo "open ./dev/requirements.txt and insert the line:"
    echo "pre-commit"
    echo "then run the command:"
    echo "./repo.sh -r base"
    echo "./repo.sh -r venv"
    echo
    echo
    exit 1
fi

if ! "$pre_commit_cmd" install; then
    echo
    echo -e "\e[31mERROR\e[0m Failed to install pre-commit"
    echo
    echo
    exit 1
fi

if ! "$pre_commit_cmd" install-hooks; then
    echo
    echo -e "\e[31mERROR\e[0m Failed to install pre-commit hooks"
    echo
    echo
    exit 1
fi

exit 0
