#!/bin/bash
for SEED in 1
do
  for VAL in 100
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach gradient --seed $SEED --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 \
    --lr 0.1 --S 32 --weight-decay 1e-4  --dump --alpha $VAL  --use-test-as-val --exp-name /10x10/alpha=${VAL}
  done
done
