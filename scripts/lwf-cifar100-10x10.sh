d#!/bin/bash
for SEED in 1
do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach lwf --seed $SEED --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 \
    --lr 0.1  --use-test-as-val --network convnet --exp-name /10x10
done
