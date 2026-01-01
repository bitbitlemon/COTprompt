import argparse
import os
import sys
import torch

repo_dir = os.path.dirname(__file__)
dassl_dir = os.path.join(repo_dir, "Dassl.pytorch")
if dassl_dir not in sys.path:
    sys.path.insert(0, dassl_dir)

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from datasets import cifar100n


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def _merge_cfg_from_file(cfg, file_path):
    from yacs.config import CfgNode as CN

    last_err = None
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                cfg_str = f.read()
            loaded_cfg = CN._load_cfg_from_yaml_str(cfg_str)
            cfg.merge_from_other_cfg(loaded_cfg)
            return
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def extend_cfg(cfg):
    cfg.DATASET.NOISE_LABEL = True
    cfg.DATASET.NOISE_RATE = 0.0
    cfg.DATASET.NOISE_TYPE = "sym"
    cfg.DATASET.num_class = 100

    cfg.DATASET.USE_OT = False
    cfg.DATASET.REG_FEAT = 1.0
    cfg.DATASET.REG_LAB = 1.0
    cfg.DATASET.CURRICLUM_EPOCH = 0
    cfg.DATASET.BEGIN_RATE = 0.3
    cfg.DATASET.CURRICLUM_MODE = "linear"
    cfg.DATASET.PMODE = "logP"
    cfg.DATASET.REG_E = 0.01


def set_baseline_defaults(cfg):
    cfg.OUTPUT_DIR = "output/cifar100n_vanilla"

    cfg.TRAINER.NAME = "Vanilla"

    cfg.DATASET.NAME = "CIFAR100N"
    cfg.DATASET.NUM_SHOTS = -1
    cfg.DATASET.NOISE_LABEL = False

    cfg.INPUT.SIZE = (32, 32)
    cfg.INPUT.TRANSFORMS = ("random_flip", "random_crop", "normalize")
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.INPUT.CROP_PADDING = 4

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 128
    cfg.DATALOADER.TEST.BATCH_SIZE = 256

    cfg.MODEL.BACKBONE.NAME = "wide_resnet_28_2"
    cfg.MODEL.BACKBONE.PRETRAINED = False
    cfg.MODEL.HEAD.NAME = ""

    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.1
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.MOMENTUM = 0.9
    cfg.OPTIM.MAX_EPOCH = 200
    cfg.OPTIM.LR_SCHEDULER = "cosine"

    cfg.TRAIN.PRINT_FREQ = 50
    cfg.TRAIN.CHECKPOINT_FREQ = 10
    cfg.TEST.NO_TEST = False
    cfg.TEST.FINAL_MODEL = "last_step"


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed is not None:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    set_baseline_defaults(cfg)

    if args.dataset_config_file:
        _merge_cfg_from_file(cfg, args.dataset_config_file)

    if args.config_file:
        _merge_cfg_from_file(cfg, args.config_file)

    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    assert cifar100n is not None
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(repo_dir, "datasets", "DATA"),
        help="path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/cifar100n_vanilla",
        help="output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="only positive value enables a fixed seed",
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="path to config file (optional)",
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default=os.path.join("configs", "datasets", "cifar100n.yaml"),
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--trainer", type=str, default="Vanilla", help="name of trainer"
    )
    parser.add_argument(
        "--backbone", type=str, default="wide_resnet_28_2", help="name of backbone"
    )
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        default=None,
        help="load model weights at this epoch for evaluation",
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    main(parser.parse_args())
