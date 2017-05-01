#!/usr/bin/env bash

source /mfs/jiaxin/start_env.sh
mkdir -p logs/avb_bnn/$1
cp examples/avb_bnn.py logs/avb_bnn/$1/
CUDA_VISIBLE_DEVICES=$2 python -u examples/avb_bnn.py --dataset=$1
