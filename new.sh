python dividemix_cifar100n.py \
  --root ./datasets/DATA \
  --output-dir ./output/cifar100n_dividemix_wrn \
  --arch-type backbone \
  --backbone wide_resnet_28_2 \
  --epochs 200 \
  --warmup-epochs 10 \
  --batch-size 64 \
  --lr 0.02 \
  --p-threshold 0.5 \
  --T 0.5 \
  --alpha 4.0 \
  --lambda-u 25.0 \
  --lambda-u-ramp 16

# python dividemix_cifar100n.py \
#   --root ./datasets/DATA \
#   --output-dir ./output/cifar100n_dividemix_nlprompt \
#   --arch-type nlprompt \
#   --backbone RN50 \
#   --epochs 200 \
#   --warmup-epochs 10 \
#   --batch-size 64 \
#   --lr 0.005 \
#   --p-threshold 0.5 \
#   --T 0.5 \
#   --alpha 4.0 \
#   --lambda-u 25.0 \
#   --lambda-u-ramp 16
