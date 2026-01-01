#!/usr/bin/env bash
python train.py \
  --seed 1 \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/cifar100n.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/cifar100n_nlprompt_seed1 \
  DATASET.NUM_SHOTS 16 \
  DATASET.NOISE_LABEL False \
  DATASET.USE_OT False \
  OPTIM.MAX_EPOCH 50 \
  TRAIN.CHECKPOINT_FREQ 10 \
  TEST.NO_TEST False
