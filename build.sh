#!/bin/bash

cmds=(
    "jupyter-book clean docs"
    "jupyter-book build docs"
    "rm -rf \"${JUPYTER_SERVER_ROOT}/www/cs5483jupyter\" && mkdir --parents \$_ && cp -rf docs/_build/html/* \$_"
    ) 

if [ -z ${JUPYTER_SERVER_ROOT} ]
then

    echo "[Error] Must run in a jupyter terminal with \$JUPYTER_SERVER_ROOT defined." 1>&2
    exit 1

else

    for cmd in "${cmds[@]}"
    do
        read -r -p "${cmd}?[Y/n] " input

        case $input in
            [yY][eE][sS]|[yY]|'')
        echo "Executing..."
        eval $cmd
        ;;
            [nN][oO]|[nN])
        echo "Skipped..."
            ;;
            *)
        echo "Invalid input..."
        exit 1
        ;;
        esac
    done

fi