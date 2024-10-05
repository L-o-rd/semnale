#!/bin/bash

LAB_EX=0
if [ -z "$1" ]; then
    LAB_NO=01
else
    case "$1" in
        "01") LAB_NO=$1;;
        *) 
            echo "usage: $0 <lab-no> (01 - 01)"
            exit 0;;
    esac
fi

if [ ! -z "$2" ]; then
    case "$2" in
        "01") LAB_EX=$2;;
        "02") LAB_EX=$2;;
        "03") LAB_EX=$2;;
        *) LAB_EX=0;;
    esac
fi

if [ "$LAB_EX" -eq "0" ]; then
    echo "Running labs/$LAB_NO/*.py"
    for entry in `ls ./labs/$LAB_NO/*.py`
    do
        echo "  - $entry.."
        python3 "$entry"
    done
else
    echo "Running labs/$LAB_NO/$LAB_EX.py"
    python3 ./labs/$LAB_NO/$LAB_EX.py
fi
