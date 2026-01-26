#!/bin/bash
for SEED in 1
do
  for VAL in 15
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach lwf --seed $SEED --batch-size 256 --num-workers 4 --nepochs 100 --datasets fashion --num-tasks 5 --nc-first-task 2 \
    --lr 0.1 --use-test-as-val --network convnet --exp-name /5x2
  done
done
