python train.py \
  --seed 1 \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/cifar100n.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/cifar100n_nlprompt_noise_safeaug_seed1 \
  DATASET.NUM_SHOTS -1 \
  DATASET.NOISE_LABEL True \
  DATASET.NOISE_RATE 0.5 \
  DATASET.NOISE_TYPE sym \
  DATASET.USE_OT False \
  INPUT.TRANSFORMS random_flip normalize \
  OPTIM.LR 0.005 \
  OPTIM.MAX_EPOCH 100 \
  TRAIN.CHECKPOINT_FREQ 10 \
  TEST.NO_TEST False
