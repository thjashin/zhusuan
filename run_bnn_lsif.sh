#!/usr/bin/env bash

source /mfs/jiaxin/start_env.sh
mkdir -p logs/bnn_lsif/$1
cp examples/bnn_lsif.py logs/bnn_lsif/$1/
CUDA_VISIBLE_DEVICES=$2 python -u examples/bnn_lsif.py --dataset=$1
