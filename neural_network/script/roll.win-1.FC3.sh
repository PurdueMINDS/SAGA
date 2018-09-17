#!/bin/bash
#   Copyright 2018 Jianfei Gao, Leonardo Teixeira, Bruno Ribeiro.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

if [ $(hostname) != 'ml01.cs.purdue.edu' ]; then
    echo "Unexpected Hostname"
    exit
fi

if [ $# -ne 1 ]; then
    echo "Missing #Epochs"
    exit
fi

target=roll
gpu=0
lr=0.01
N=$1

# smart phone
for phone in galaxy pixel; do
    python main.py \
        output output --phone ${phone} \
        --batch-size 16 --model FC3 \
        --win 1 --offset 1 \
        --trig --keyword ${target} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/${phone}.${target}.win-1.FC3.log
done

# stratux
for lv in 0 1 2; do
    python main.py \
        output output --phone stratux --stratux ${lv} \
        --batch-size 16 --model FC3 \
        --win 1 --offset 1 \
        --trig --keyword ${target} \
        --lr ${lr} --epochs ${N} \
        --cuda --device ${gpu} \
        | tee log/stratux-${lv}.${target}.win-1.FC3.log
done
