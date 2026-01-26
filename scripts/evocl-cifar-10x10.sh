#!/bin/bash
for SEED in 1
do
  for ALPHA in 100
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach evo --seed $SEED --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 \
    --strategy Lambda --load-model-path cifar100-10x10.pth --lr 0.0001 --weight-decay 1e-4 --S 32 --alpha $ALPHA --mu 16 --lamb 128 --use-test-as-val --exp-name /10x10/alpha=${ALPHA}
  done
done
