#!/usr/bin/env bash

python train.py \
  --seed 1 \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/cifar100n.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/cifar100n_nlprompt_full_noise_seed1 \
  DATASET.NUM_SHOTS -1 \
  DATASET.NOISE_LABEL True \
  DATASET.NOISE_RATE 0.5 \
  DATASET.NOISE_TYPE sym \
  DATASET.USE_OT False \
  OPTIM.MAX_EPOCH 50 \
  TRAIN.CHECKPOINT_FREQ 10 \
  TEST.NO_TEST False
