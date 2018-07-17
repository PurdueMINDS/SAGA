#!/bin/bash

if [ $# -eq 1 ]; then
    python main.py output output --phone galaxy \
        --batch-size 16 --keyword $1 --trig --freq haar \
        --epochs 100 --lr 0.01 --print-freq 5
elif [ $# -eq 2 ]; then
    python main.py output output --phone galaxy \
        --batch-size 16 --keyword $1 --trig --freq haar \
        --epochs 100 --lr 0.01 --print-freq 5 \
        --cuda --device $2
fi
