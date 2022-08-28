#!/bin/sh

export GPU_ID=$1
echo shell: GPU_ID $GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 5-way 1-shot
# hsml
experiment_name="BTAF-poolmo-0_03"
seed=10
way=5
shot=1
quey=15
innerlr=0.01
HVw=0.03

# train
python -m experiments.hsml --experiment-name $experiment_name --seed $seed --dataset BTAF --use-warm-start \
    --num-classes-per-set $way --num-samples-per-class $shot --num-target-samples $quey --batch-size 4 \
    --inner-train-steps 3 --inner-val-steps 15 --order 2 \
    --meta-learning-rate 0.0001 --inner-learning-rate $innerlr \
    --epochs 50 --epoch-len 2000 --eval-batches 80 \
    --use-pool --pool-start-epoch 2 \
    --use-conflict-loss --HV-weight $HVw \
    > ../few-shot-experiments/out/$experiment_name.out 2>&1

# test
datasets="Omniglot_84 FGVC_Aircraft CUB_Bird DTD_Texture FGVCx_Fungi VGG_Flower_84 traffic_sign_84 mscoco_84 mini_84
clipart_84 infograph_84 painting_84 quickdraw_84 real_84 sketch_84 CIFAR10_84 CIFAR100_84 cars_84 pets_head_84 dogs_84"
for dataset in $datasets
do
python -m experiments.hsml_test --experiment-name $experiment_name --seed $seed --test-dataset $dataset \
    --num-classes-per-set $way --num-samples-per-class $shot --num-target-samples $quey \
    --inner-val-steps 15 --order 2 \
    --inner-learning-rate $innerlr \
     >> ../few-shot-experiments/out/$experiment_name.out 2>&1
done
