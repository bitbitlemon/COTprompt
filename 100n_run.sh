python train.py --seed 1 --trainer NLPrompt \
  --dataset-config-file configs/datasets/cifar100n.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/cifar100n_safe_smoke_seed1 \
  DATASET.NUM_SHOTS 16 DATASET.USE_OT True OPTIM.MAX_EPOCH 2 TRAIN.CHECKPOINT_FREQ 0
