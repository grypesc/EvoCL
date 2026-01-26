#!/bin/bash
for SEED in 1
do
  for VAL in 15
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach lwf --seed $SEED --batch-size 256 --num-workers 4 --nepochs 100 --datasets mnist --num-tasks 3 --nc-first-task 4 \
    --lr 0.1 --use-test-as-val --network convnet --exp-name /3x3
  done
done
