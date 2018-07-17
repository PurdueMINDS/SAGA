#!/bin/bash

if [ $(hostname) != 'ml02.cs.purdue.edu' ]; then
    echo "Unexpected Hostname"
    exit
fi

if [ $# -ne 1 ]; then
    echo "Missing #Epochs"
    exit
fi

target=pitch
gpu=2
lr=0.01
N=$1

# smart phone
for phone in galaxy pixel; do
    python main.py \
        output output --phone ${phone} \
        --batch-size 5 --model LSTM3 \
        --win 1 --offset 1 \
        --trig --keyword ${target} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/${phone}.${target}.win-1.LSTM3.log
done

# stratux
for lv in 0 1 2; do
    python main.py \
        output output --phone stratux --stratux ${lv} \
        --batch-size 5 --model LSTM3 \
        --win 1 --offset 1 \
        --trig --keyword ${target} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/stratux-${lv}.${target}.win-1.LSTM3.log
done
