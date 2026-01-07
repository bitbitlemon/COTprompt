import argparse
import math
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


repo_dir = os.path.dirname(__file__)
dassl_dir = os.path.join(repo_dir, "Dassl.pytorch")
if dassl_dir not in sys.path:
    sys.path.insert(0, dassl_dir)

from dassl.modeling import build_backbone

from datasets.cifar100n import _load_noisy_labels, _scan_indexed_images, _get_classnames

from trainers.nlprompt import load_clip_to_cpu, CustomCLIP


class _CfgNode:
    pass


class NLPromptCfg:
    def __init__(
        self,
        backbone_name: str,
        n_ctx: int = 16,
        ctx_init: str = "",
        prec: str = "fp32",
        class_token_position: str = "end",
        prompt_style: str = "coop",
    ):
        self.MODEL = _CfgNode()
        self.MODEL.BACKBONE = _CfgNode()
        self.MODEL.BACKBONE.NAME = backbone_name
        self.TRAINER = _CfgNode()
        self.TRAINER.NLPROMPT = _CfgNode()
        self.TRAINER.NLPROMPT.N_CTX = n_ctx
        self.TRAINER.NLPROMPT.CTX_INIT = ctx_init
        self.TRAINER.NLPROMPT.PREC = prec
        self.TRAINER.NLPROMPT.CLASS_TOKEN_POSITION = class_token_position
        self.TRAINER.NLPROMPT.PROMPT_STYLE = prompt_style


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(labels: torch.Tensor, num_classes: int):
    return F.one_hot(labels, num_classes=num_classes).float()


def sharpen(p: torch.Tensor, T: float):
    p = p.pow(1.0 / T)
    return p / p.sum(dim=1, keepdim=True)


def linear_rampup(current: float, rampup_length: float):
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current / rampup_length, 0.0, 1.0))
    return current


def neg_entropy(probs: torch.Tensor):
    return torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    mixed_y = lam * y + (1.0 - lam) * y[idx]
    return mixed_x, mixed_y, lam


class Cifar100NTrain(Dataset):
    def __init__(
        self,
        paths,
        noisy_labels,
        transform,
        indices=None,
        probs=None,
    ):
        self.paths = paths
        self.noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
        self.transform = transform
        self.indices = np.asarray(indices, dtype=np.int64) if indices is not None else None
        self.probs = probs

    def __len__(self):
        return int(self.indices.shape[0]) if self.indices is not None else int(len(self.paths))

    def __getitem__(self, i):
        gi = int(self.indices[i]) if self.indices is not None else int(i)
        path = self.paths[gi]
        img = Image.open(path).convert("RGB")
        img1 = self.transform(img)
        img2 = self.transform(img)
        img.close()
        y = int(self.noisy_labels[gi])
        if self.probs is None:
            return img1, img2, y, gi
        p = float(self.probs[gi])
        return img1, img2, y, p, gi


class Cifar100NTest(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return int(len(self.paths))

    def __getitem__(self, i):
        path = self.paths[int(i)]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        img.close()
        y = int(self.labels[int(i)])
        return x, y


class ClsNet(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=False, verbose=False)
        self.classifier = nn.Linear(self.backbone.out_features, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f)


class NLPromptNet(nn.Module):
    def __init__(self, backbone_name: str, classnames, device: torch.device):
        super().__init__()
        cfg = NLPromptCfg(backbone_name=backbone_name)
        clip_model = load_clip_to_cpu(cfg)
        if device.type != "cuda":
            clip_model.float()
        self.model = CustomCLIP(cfg, classnames, clip_model)
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, x):
        return self.model(x)


@dataclass
class SplitResult:
    probs: np.ndarray
    labeled_idx: np.ndarray
    unlabeled_idx: np.ndarray


def eval_train_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
    p_threshold: float,
    gmm_max_iter: int,
):
    model.eval()
    losses = np.zeros((num_samples,), dtype=np.float64)
    ce = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for img1, _, y, gi in loader:
            img1 = img1.to(device)
            y = y.to(device)
            gi = gi.numpy().astype(np.int64)
            logits = model(img1)
            loss = ce(logits, y).detach().cpu().numpy()
            losses[gi] = loss

    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-12)
    gmm = GaussianMixture(
        n_components=2,
        max_iter=gmm_max_iter,
        tol=1e-2,
        reg_covar=1e-4,
        covariance_type="full",
        random_state=0,
    )
    gmm.fit(losses.reshape(-1, 1))
    prob = gmm.predict_proba(losses.reshape(-1, 1))
    means = gmm.means_.reshape(-1)
    clean_comp = int(np.argmin(means))
    probs = prob[:, clean_comp]

    labeled_idx = np.where(probs > p_threshold)[0].astype(np.int64)
    unlabeled_idx = np.where(probs <= p_threshold)[0].astype(np.int64)
    return SplitResult(probs=probs.astype(np.float32), labeled_idx=labeled_idx, unlabeled_idx=unlabeled_idx)


def warmup_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ce_weight: float,
    penalty_weight: float,
):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    for img1, _, y, _ in loader:
        img1 = img1.to(device)
        y = y.to(device)
        logits = model(img1)
        loss = ce_weight * ce(logits, y)
        if penalty_weight > 0:
            loss = loss + penalty_weight * neg_entropy(F.softmax(logits, dim=1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = int(img1.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(1, total_n)


def train_epoch_dividemix(
    model: nn.Module,
    model_peer: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    T: float,
    alpha: float,
    lambda_u: float,
    penalty_weight: float,
):
    model.train()
    model_peer.eval()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    steps = min(len(labeled_loader), len(unlabeled_loader))
    total_loss = 0.0
    total_n = 0

    for _ in range(steps):
        lx1, lx2, y, w, _ = next(labeled_iter)
        ux1, ux2, _, _ = next(unlabeled_iter)

        lx1 = lx1.to(device)
        lx2 = lx2.to(device)
        y = y.to(device)
        w = w.to(device).float().view(-1, 1)

        ux1 = ux1.to(device)
        ux2 = ux2.to(device)

        with torch.no_grad():
            py = one_hot(y, num_classes)

            logits_m_lx1 = model(lx1)
            logits_m_lx2 = model(lx2)
            logits_p_lx1 = model_peer(lx1)
            logits_p_lx2 = model_peer(lx2)
            p_l = (
                F.softmax(logits_m_lx1, dim=1)
                + F.softmax(logits_m_lx2, dim=1)
                + F.softmax(logits_p_lx1, dim=1)
                + F.softmax(logits_p_lx2, dim=1)
            ) / 4.0
            refined = w * py + (1.0 - w) * p_l
            refined = sharpen(refined, T)

            logits_m_ux1 = model(ux1)
            logits_m_ux2 = model(ux2)
            logits_p_ux1 = model_peer(ux1)
            logits_p_ux2 = model_peer(ux2)
            p_u = (
                F.softmax(logits_m_ux1, dim=1)
                + F.softmax(logits_m_ux2, dim=1)
                + F.softmax(logits_p_ux1, dim=1)
                + F.softmax(logits_p_ux2, dim=1)
            ) / 4.0
            guessed = sharpen(p_u, T)

        all_inputs = torch.cat([lx1, lx2, ux1, ux2], dim=0)
        all_targets = torch.cat([refined, refined, guessed, guessed], dim=0)

        mixed_x, mixed_y, _ = mixup(all_inputs, all_targets, alpha)

        logits = model(mixed_x)
        logits_x = logits[: 2 * lx1.size(0)]
        logits_u = logits[2 * lx1.size(0) :]

        targets_x = mixed_y[: 2 * lx1.size(0)]
        targets_u = mixed_y[2 * lx1.size(0) :]

        Lx = -(F.log_softmax(logits_x, dim=1) * targets_x).sum(dim=1).mean()
        probs_u = F.softmax(logits_u, dim=1)
        Lu = (probs_u - targets_u).pow(2).mean()

        penalty = 0.0
        if penalty_weight > 0:
            p_avg = F.softmax(logits, dim=1).mean(dim=0)
            penalty = penalty_weight * torch.sum(p_avg * torch.log(p_avg + 1e-12))

        loss = Lx + lambda_u * Lu + penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = int(lx1.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)


def evaluate_ensemble(model1: nn.Module, model2: nn.Module, loader: DataLoader, device: torch.device):
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model1(x) + model2(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=os.path.join(repo_dir, "datasets", "DATA"))
    p.add_argument("--output-dir", type=str, default=os.path.join(repo_dir, "output", "cifar100n_dividemix"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backbone", type=str, default="wide_resnet_28_2")
    p.add_argument(
        "--arch-type",
        type=str,
        default="backbone",
        choices=["backbone", "nlprompt"],
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--p-threshold", type=float, default=0.5)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--lambda-u", type=float, default=25.0)
    p.add_argument("--lambda-u-ramp", type=int, default=16)
    p.add_argument("--penalty-weight", type=float, default=0.1)
    p.add_argument("--gmm-max-iter", type=int, default=10)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = os.path.abspath(os.path.expanduser(args.root))
    train_dir = os.path.join(data_root, "cifar100", "train")
    test_dir = os.path.join(data_root, "cifar100", "test")
    labels_dir = os.path.join(data_root, "cifar-100n", "data")
    if not os.path.isdir(os.path.join(data_root, "cifar-100n")):
        alt = os.path.join(data_root, "cifar100n", "data")
        if os.path.isdir(alt):
            labels_dir = alt

    train_paths, _ = _scan_indexed_images(train_dir)
    test_paths, test_labels = _scan_indexed_images(test_dir)
    _, noisy_labels = _load_noisy_labels(labels_dir)
    num_classes = 100
    if args.arch_type == "nlprompt":
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=clip_mean, std=clip_std),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=clip_mean, std=clip_std),
            ]
        )
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    train_ds_all = Cifar100NTrain(train_paths, noisy_labels, transform_train)
    train_loader_warm = DataLoader(
        train_ds_all,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    train_loader_eval = DataLoader(
        train_ds_all,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        Cifar100NTest(test_paths, test_labels, transform_test),
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if args.arch_type == "nlprompt":
        classnames = _get_classnames(data_root)
        model1 = NLPromptNet(args.backbone, classnames, device).to(device)
        model2 = NLPromptNet(args.backbone, classnames, device).to(device)
    else:
        model1 = ClsNet(args.backbone, num_classes).to(device)
        model2 = ClsNet(args.backbone, num_classes).to(device)

    opt1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    opt2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs)

    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            loss1 = warmup_epoch(
                model1,
                train_loader_warm,
                opt1,
                device,
                ce_weight=1.0,
                penalty_weight=args.penalty_weight,
            )
            loss2 = warmup_epoch(
                model2,
                train_loader_warm,
                opt2,
                device,
                ce_weight=1.0,
                penalty_weight=args.penalty_weight,
            )
        else:
            split1 = eval_train_split(
                model1,
                train_loader_eval,
                device,
                num_samples=len(train_ds_all),
                p_threshold=args.p_threshold,
                gmm_max_iter=args.gmm_max_iter,
            )
            split2 = eval_train_split(
                model2,
                train_loader_eval,
                device,
                num_samples=len(train_ds_all),
                p_threshold=args.p_threshold,
                gmm_max_iter=args.gmm_max_iter,
            )

            ds_l1 = Cifar100NTrain(
                train_paths,
                noisy_labels,
                transform_train,
                indices=split2.labeled_idx,
                probs=split2.probs,
            )
            ds_u1 = Cifar100NTrain(
                train_paths,
                noisy_labels,
                transform_train,
                indices=split2.unlabeled_idx,
            )

            ds_l2 = Cifar100NTrain(
                train_paths,
                noisy_labels,
                transform_train,
                indices=split1.labeled_idx,
                probs=split1.probs,
            )
            ds_u2 = Cifar100NTrain(
                train_paths,
                noisy_labels,
                transform_train,
                indices=split1.unlabeled_idx,
            )

            labeled_loader_1 = DataLoader(
                ds_l1,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )
            unlabeled_loader_1 = DataLoader(
                ds_u1,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )
            labeled_loader_2 = DataLoader(
                ds_l2,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )
            unlabeled_loader_2 = DataLoader(
                ds_u2,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )

            lam_u = args.lambda_u * linear_rampup(epoch - args.warmup_epochs, args.lambda_u_ramp)
            loss1 = train_epoch_dividemix(
                model1,
                model2,
                labeled_loader_1,
                unlabeled_loader_1,
                opt1,
                device,
                num_classes=num_classes,
                T=args.T,
                alpha=args.alpha,
                lambda_u=lam_u,
                penalty_weight=args.penalty_weight,
            )
            loss2 = train_epoch_dividemix(
                model2,
                model1,
                labeled_loader_2,
                unlabeled_loader_2,
                opt2,
                device,
                num_classes=num_classes,
                T=args.T,
                alpha=args.alpha,
                lambda_u=lam_u,
                penalty_weight=args.penalty_weight,
            )

        sch1.step()
        sch2.step()

        acc = evaluate_ensemble(model1, model2, test_loader, device)
        line = f"epoch={epoch+1}/{args.epochs} loss1={loss1:.4f} loss2={loss2:.4f} acc={acc*100:.2f}%"
        print(line)
        with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "epoch": epoch,
                    "acc": acc,
                    "model1": model1.state_dict(),
                    "model2": model2.state_dict(),
                    "opt1": opt1.state_dict(),
                    "opt2": opt2.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )

    print(f"best_acc={best_acc*100:.2f}% saved={best_path}")


if __name__ == "__main__":
    main()

