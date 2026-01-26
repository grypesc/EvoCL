#!/bin/bash
for SEED in 1
do
  for ALPHA in 100
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach evo --seed $SEED --batch-size 32 --num-workers 4 --nepochs 1000 --datasets domainnet --num-tasks 18 --nc-first-task 10 \
    --strategy vit --load-model-path dino_deitsmall16_pretrain.pth --lr 0.0001  --weight-decay 1e-4 --S 384 --K 64  --alpha $ALPHA --lr-ratio 0.1 --mu 16 --lamb 128 --use-fp16  --use-test-as-val --exp-name /18x10
  done
done