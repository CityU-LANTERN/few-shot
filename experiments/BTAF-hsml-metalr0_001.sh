#!/bin/sh

export GPU_ID=$1
echo shell: GPU_ID $GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

# use to test a fast meta lr.
# train
python -m experiments.hsml --experiment-name BTAF-hsml-metalr0_001 --seed 10 --dataset BTAF --use-warm-start \
    --num-classes-per-set 5 --num-samples-per-class 1 --num-target-samples 15 --batch-size 4 \
    --inner-train-steps 3 --inner-val-steps 15 --order 2 \
    --meta-learning-rate 0.001 --inner-learning-rate 0.01 \
    --epochs 50 --epoch-len 2000 --eval-batches 80 \
    --use-pool --pool-start-epoch 50 \


# test
