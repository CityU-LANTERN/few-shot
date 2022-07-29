#!/bin/sh

## reproduction: 2nd order MAML
#python -m experiments.maml --device 1 --dataset omniglot --order 2 --n 1 --k 5 --eval-batches 10 --epoch-len 50
#
#python -m experiments.maml --device 2 --dataset miniImageNet --order 2 --n 1 --k 5 --q 5 --meta-batch-size 4 \
#    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 80 --epoch-len 400

## proto_nets, BTAF
# train
# python -m experiments.proto_nets --dataset BTAF --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15

# test
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset Omniglot_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset FGVC_Aircraft --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset CUB_Bird --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset DTD_Texture --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset FGVCx_Fungi --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset VGG_Flower_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset traffic_sign_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset mscoco_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
python -m experiments.proto_nets_test --dataset BTAF --test-dataset mini_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset clipart_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset infograph_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset painting_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset quickdraw_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset real_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset sketch_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset CIFAR10_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset CIFAR100_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset cars_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset pets_head_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
#python -m experiments.proto_nets_test --dataset BTAF --test-dataset dogs_84 --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15

