#!/bin/bash

if [ ! -d output ]; then
    mkdir output
fi

cd output

if [ ! -d task1 ]; then
    mkdir task1
fi

if [ ! -d task2 ]; then
    mkdir task2
fi

cd ..

python3 run_task1.py

