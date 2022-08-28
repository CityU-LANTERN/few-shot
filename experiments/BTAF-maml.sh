#!/bin/sh

export GPU_ID=$1
echo shell: GPU_ID $GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

# maml
# train 1-shot
#python -m experiments.maml --dataset BTAF --seed 10 --order 2 --n 1 --k 5 --q 15 --meta-batch-size 4 \
#    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.1 --meta-lr 0.001 --eval-batches 80 \
#    --epochs 50 --epoch-len 1000

# train 5-shot
python -m experiments.maml --dataset BTAF --seed 10 --order 2 --n 5 --k 5 --q 15 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.1 --meta-lr 0.001 --eval-batches 80 \
    --epochs 50 --epoch-len 1000

# test
