#!/bin/bash

if [ $(hostname) != 'ml01.cs.purdue.edu' ]; then
    echo "Unexpected Hostname"
    exit
fi

if [ $# -ne 1 ]; then
    echo "Missing #Epochs"
    exit
fi

target=roll
gpu=1
lr=0.01
N=$1

# smart phone
for phone in galaxy pixel; do
    python main.py \
        output output --phone ${phone} \
        --batch-size 16 --model FC3 \
        --win 32 --rate 0.4 \
        --trig --keyword ${target} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/${phone}.${target}.win-32.FC3.log
done

# stratux
for lv in 0 1 2; do
    python main.py \
        output output --phone stratux --stratux ${lv} \
        --batch-size 16 --model FC3 \
        --win 32 --rate 0.4 \
        --trig --keyword ${target} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/stratux-${lv}.${target}.win-32.FC3.log
done
