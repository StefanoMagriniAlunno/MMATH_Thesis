#!/bin/bash

# Read input from user
# $1 is the starting argument
# $2 is an optional argument with the choosen addOn
# $3 is an optional argument with the configuration file

# if the user not provide any input, then we write a message to the user
if [ $# -eq 0 ]; then
    echo "No arguments provided, try -h or --help"
    exit 1
fi

# if the user pass --show
if [ "$1" == "--show" ]; then
    # Mostro la lista di tutti gli addon nella cartella AddOns
    echo "List of AddOns:"
    ls -1 ./dev/scripts/AddOns
    echo
    # indico come avviare un addOn
    echo "To start an AddOn, use the command:"
    echo "./repo.sh -a start myAddOn.sh"
    echo
    echo
    exit 0
# if the user pass --start
elif [ "$1" == "--start" ]; then
    # eseguo lo script dell'addon scelto, quindi:
    # cerco lo script $2 nella cartella AddOns
    if [ -f "./dev/scripts/AddOns/$2" ]; then
        # eseguo lo script
        ./dev/scripts/AddOns/"$2" "$3"
    else
        # se non esiste lo script, allora scrivo un messaggio di errore
        echo
        echo -e "\e[31mERROR\e[0m The AddOn $2 does not exist"
        echo "Please check the name of the AddOn"
        echo "(note: include the extension .sh)"
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
