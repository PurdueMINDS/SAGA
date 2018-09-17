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
