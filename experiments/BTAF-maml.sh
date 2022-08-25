#!/bin/sh

export GPU_ID=$1
echo shell: GPU_ID $GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 5-way 1-shot
# maml
# train
#python -m experiments.maml --dataset BTAF --seed 9 --order 1 --n 1 --k 5 --q 15 --meta-batch-size 4 \
#    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --meta-lr 0.001 --eval-batches 80 \
#    --epochs 50 --epoch-len 1000
python -m experiments.maml --dataset BTAF --seed 10 --order 2 --n 1 --k 5 --q 15 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.1 --meta-lr 0.001 --eval-batches 80 \
    --epochs 50 --epoch-len 1000

# test
