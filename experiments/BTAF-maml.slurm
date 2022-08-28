#!/bin/bash
#SBATCH --error=~/output/job.%j.err
#SBATCH --output ~/output/job.%j.out
#SBATCH -p v100
#SBATCH --qos=v100
#SBATCH -J hsml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1

cd ..
singularity exec --bind ~/datasets:/liaoweiduo --nv ~/pool-mo.sif \
python -m experiments.maml --dataset BTAF --seed 10 --order 2 --n 1 --k 5 --q 15 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.1 --meta-lr 0.001 --eval-batches 80 \
    --epochs 50 --epoch-len 1000
