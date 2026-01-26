#!/bin/bash
for SEED in 1
do
  for VAL in 100
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach evo --seed $SEED --batch-size 128 --num-workers 4 --nepochs 200 --datasets fashion --num-tasks 5 --nc-first-task 2 \
    --strategy Lambda --load-model-path fashion-5x2.pth --lr 0.0001 --weight-decay 1e-4 --S 32 --alpha $VAL --mu 16 --lamb 128 --use-test-as-val --exp-name /5x2/alpha=${VAL}
  done
done
