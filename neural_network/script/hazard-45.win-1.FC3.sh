#!/bin/bash

if [ $(hostname) != 'ml06.cs.purdue.edu' ]; then
    echo "Unexpected Hostname"
    exit
fi

if [ $# -ne 1 ]; then
    echo "Missing #Epochs"
    exit
fi

target=hazard
thres=45
gpu=0
lr=0.01
N=$1

# smart phone
for phone in galaxy pixel; do
    python main.py \
        output output --phone ${phone} \
        --batch-size 16 --model FC3 \
        --win 1 --offset 1 \
        --direct --no-normal --keyword ${target} --threshold ${thres} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/${phone}.${target}-${thres}.win-1.FC3.log
done

# stratux
for lv in 0 1 2; do
    python main.py \
        output output --phone stratux --stratux ${lv} \
        --batch-size 16 --model FC3 \
        --win 1 --offset 1 \
        --direct --no-normal --keyword ${target} --threshold ${thres} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/stratux-${lv}.${target}-${thres}.win-1.FC3.log
done
