# COTprompt
run:
python train.py \
  --seed 1 \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/caltech101_nlprompt_16shot_seed1 \
  DATASET.NUM_SHOTS 16 \
  DATASET.USE_OT True \
  OPTIM.MAX_EPOCH 50 \
  TRAIN.CHECKPOINT_FREQ 10

