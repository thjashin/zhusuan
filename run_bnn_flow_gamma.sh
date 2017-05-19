#!/usr/bin/env bash

source /mfs/jiaxin/start_env.sh
mkdir -p logs/bnn_flow_gamma/$1
CUDA_VISIBLE_DEVICES=$2 python -u examples/bnn_flow_gamma.py --dataset=$1
