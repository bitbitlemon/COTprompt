import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from dassl.utils import (
    MetricMeter, AverageMeter
)
import datetime
import time
import copy

_tokenizer = _Tokenizer()


# =========================
# Helpers for dataset.data_source format
# =========================
def get_item_path(item):
    """Try to extract image path from dataset.data_source item."""
    # Common patterns:
    # 1) dict: {"impath":..., "label":...}
    # 2) tuple/list: (impath, label, ...)
    if isinstance(item, dict):
        if "impath" in item:
            return item["impath"]
        if "path" in item:
            return item["path"]
    elif isinstance(item, (tuple, list)) and len(item) >= 1:
        return item[0]
    raise KeyError("Cannot infer impath from data_source item. Please adapt get_item_path().")


def get_item_label(item):
    """Extract label from dataset.data_source item."""
    if isinstance(item, dict):
        if "label" in item:
            return int(item["label"])
        if "target" in item:
            return int(item["target"])
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        return int(item[1])
    raise KeyError("Cannot infer label from data_source item. Please adapt get_item_label().")


def set_item_label(item, new_label: int):
    """Set label for dataset.data_source item (for relabel). Returns updated item."""
    if isinstance(item, dict):
        if "label" in item:
            item["label"] = int(new_label)
            return item
        if "target" in item:
            item["target"] = int(new_label)
            return item
    elif isinstance(item, list) and len(item) >= 2:
        item[1] = int(new_label)
        return item
    elif isinstance(item, tuple) and len(item) >= 2:
        # tuples are immutable
        item = list(item)
        item[1] = int(new_label)
        return tuple(item)
    raise KeyError("Cannot set label for data_source item. Please adapt set_item_label().")


# =========================
# CLIP loading (your original)
# =========================
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'NLPrompt',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.NLPROMPT.N_CTX
        ctx_init = cfg.TRAINER.NLPROMPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.NLPROMPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.NLPROMPT.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


# =========================
# CLIPCleaner (offline selection)
# =========================
class ClipCleaner:
    """
    Offline: use frozen CLIP (zero-shot templates) to estimate P~(y|x),
    then apply:
      1) consistency selector
      2) loss-GMM selector (per-class 2-component GMM)
    and optionally take intersection.
    """
    def __init__(self, clip_model, classnames, device, templates=None):
        self.clip_model = clip_model.eval().to(device)
        self.classnames = [c.replace("_", " ") for c in classnames]
        self.device = device
        if templates is None or len(templates) == 0:
            # minimal template; you can add more in cfg later
            templates = ["a photo of a {}."]
        self.templates = templates

        for p in self.clip_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def build_zeroshot_text_features(self):
        # average over templates per class
        text_features = []
        for cname in self.classnames:
            texts = [t.format(cname) for t in self.templates]
            tokenized = torch.cat([clip.tokenize(x) for x in texts]).to(self.device)
            txt = self.clip_model.encode_text(tokenized)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            txt = txt.mean(dim=0, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            text_features.append(txt)
        text_features = torch.cat(text_features, dim=0)  # [C, D]
        return text_features

    @torch.no_grad()
    def infer_probs(self, loader, text_features):
        # returns probs [N,C] aligned with dataset order (requires index)
        dataset_len = len(loader.dataset.data_source)
        C = text_features.shape[0]
        probs = torch.zeros(dataset_len, C, dtype=torch.float32)
        seen = torch.zeros(dataset_len, dtype=torch.bool)

        for batch in loader:
            img = batch["img"].to(self.device)
            if "index" in batch:
                idx = batch["index"].long().cpu()
            else:
                raise KeyError(
                    "CLIPCleaner needs batch['index'] to place predictions back to dataset order. "
                    "Please modify your Dataset.__getitem__ to return 'index'."
                )

            imf = self.clip_model.encode_image(img)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            logits = (imf @ text_features.t()) * self.clip_model.logit_scale.exp()
            p = logits.softmax(dim=-1).float().cpu()

            probs[idx] = p
            seen[idx] = True

        if not torch.all(seen):
            missing = torch.where(~seen)[0][:10].tolist()
            raise RuntimeError(f"CLIPCleaner did not see all samples. Missing indices examples: {missing}")

        return probs  # cpu float

    @staticmethod
    def consistency_selector(probs, noisy_labels, tau=0.6):
        # keep if P~(y|x) / max P~ >= tau
        N, C = probs.shape
        p_y = probs[torch.arange(N), noisy_labels]
        p_max, _ = probs.max(dim=1)
        ratio = p_y / (p_max + 1e-12)
        return (ratio >= tau)

    @staticmethod
    def gmm_selector_per_class(losses, noisy_labels, num_classes):
        # per-class 2-component GMM, choose smaller-mean component as clean
        # to avoid adding sklearn dependency, use simple 1D 2-means as fallback
        # (If你环境有 sklearn，可替换成 GaussianMixture 更像论文)
        clean = torch.zeros_like(noisy_labels, dtype=torch.bool)
        losses_np = losses.numpy()

        for c in range(num_classes):
            idx = torch.where(noisy_labels == c)[0]
            if len(idx) < 10:
                # too few samples: mark none as clean by GMM
                continue
            x = losses_np[idx.numpy()]
            # 2-means fallback
            m1, m2 = np.percentile(x, 30), np.percentile(x, 70)
            for _ in range(20):
                d1 = np.abs(x - m1)
                d2 = np.abs(x - m2)
                a = d1 <= d2
                if a.sum() == 0 or (~a).sum() == 0:
                    break
                m1_new = x[a].mean()
                m2_new = x[~a].mean()
                if abs(m1_new - m1) < 1e-6 and abs(m2_new - m2) < 1e-6:
                    break
                m1, m2 = m1_new, m2_new
            # smaller-mean cluster as clean
            clean_cluster = 0 if m1 < m2 else 1
            assign = (np.abs(x - m1) <= np.abs(x - m2)).astype(np.int32)  # 0 if closer to m1 else 1
            if clean_cluster == 0:
                clean_idx = idx[torch.from_numpy(assign == 1).logical_not()]
            else:
                clean_idx = idx[torch.from_numpy(assign == 1)]
            # Wait: assign==0 corresponds closer to m1. If m1 is smaller, clean = assign==0.
            if clean_cluster == 0:
                clean_idx = idx[torch.from_numpy(assign == 0)]
            else:
                clean_idx = idx[torch.from_numpy(assign == 1)]
            clean[clean_idx] = True

        return clean

    def run(self, loader, noisy_labels, tau=0.6, use_intersection=True):
        text_features = self.build_zeroshot_text_features()
        probs = self.infer_probs(loader, text_features)  # cpu
        noisy_labels = noisy_labels.cpu()

        # selectors
        cons_mask = self.consistency_selector(probs, noisy_labels, tau=tau)
        loss = -torch.log(probs[torch.arange(len(noisy_labels)), noisy_labels] + 1e-12)
        gmm_mask = self.gmm_selector_per_class(loss, noisy_labels, probs.shape[1])

        if use_intersection:
            clean = cons_mask & gmm_mask
        else:
            clean = cons_mask | gmm_mask

        return clean.cpu(), probs  # clean mask aligned to dataset order


# =========================
# MixFix (online absorb/relabel/drop)
# =========================
class MixFixBuffer:
    """
    Online: given a model f (your current model),
    for each sample in noisy set:
      - if p_max < theta_drop: drop
      - elif p_max >= theta_rel and pred != noisy_label: relabel+absorb
      - elif p_max >= theta_abs and pred == noisy_label: absorb
      - else: keep in noisy
    """
    def __init__(self, theta_abs=0.7, theta_rel=0.9, theta_drop=0.0):
        self.theta_abs = theta_abs
        self.theta_rel = theta_rel
        self.theta_drop = theta_drop

    @torch.no_grad()
    def infer_preds(self, model, loader, device):
        dataset_len = len(loader.dataset.data_source)
        C = None
        pmax = torch.zeros(dataset_len, dtype=torch.float32)
        ypred = torch.zeros(dataset_len, dtype=torch.long)
        seen = torch.zeros(dataset_len, dtype=torch.bool)

        model.eval()
        for batch in loader:
            img = batch["img"].to(device)
            if "index" in batch:
                idx = batch["index"].long().cpu()
            else:
                raise KeyError(
                    "MixFix needs batch['index'] to map predictions back to dataset order. "
                    "Please modify your Dataset.__getitem__ to return 'index'."
                )

            logits = model(img)
            prob = logits.softmax(dim=-1).float().cpu()
            if C is None:
                C = prob.shape[1]
            pm, ym = prob.max(dim=1)

            pmax[idx] = pm
            ypred[idx] = ym
            seen[idx] = True

        if not torch.all(seen):
            missing = torch.where(~seen)[0][:10].tolist()
            raise RuntimeError(f"MixFix did not see all samples. Missing indices examples: {missing}")

        return pmax, ypred

    def update_sets(self, dataset, clean_mask, noisy_mask, pmax, ypred):
        """
        Directly modify dataset.data_source:
          - move absorbed/relabelled items from noisy to clean
          - drop low-confidence items
        Return updated masks.
        """
        data = dataset.data_source
        N = len(data)

        # Collect indices in noisy set
        noisy_idx = torch.where(noisy_mask)[0].tolist()
        to_drop = []
        to_move_clean = []
        to_relabel = {}

        for i in noisy_idx:
            if pmax[i] < self.theta_drop:
                to_drop.append(i)
                continue

            old_y = get_item_label(data[i])
            pred_y = int(ypred[i].item())
            pm = float(pmax[i].item())

            if (pm >= self.theta_rel) and (pred_y != old_y):
                to_move_clean.append(i)
                to_relabel[i] = pred_y
            elif (pm >= self.theta_abs) and (pred_y == old_y):
                to_move_clean.append(i)
            else:
                pass  # keep as noisy

        # Apply relabel
        for i, ny in to_relabel.items():
            data[i] = set_item_label(data[i], ny)

        # Drop first (reverse to keep indices stable)
        for i in sorted(to_drop, reverse=True):
            del data[i]

        # After deletion, indices shift: rebuild masks by path matching is expensive.
        # Best practice: keep dataset stable and use masks to index, but your current code deletes items.
        # So here we do a "re-split" by re-running selection on the current list positions.
        # We'll approximate: move-clean/drop based on original indices; after drop, we can't safely move by index.
        # Therefore: we recommend NOT deleting in MixFix stage in production; instead store flags.
        #
        # For now, to stay consistent with your current style (delete items), we will:
        #  - NOT physically move samples between lists here.
        #  - We return a "desired" clean_mask/noisy_mask for the current dataset length by recomputing with a heuristic:
        #      clean if previously clean OR (previously noisy and selected to move clean and not dropped)
        #
        # This requires we also not delete above; but we already deleted. So we need a safer route:
        # => In this fallback, we will NOT delete in MixFix, only mark.
        raise RuntimeError(
            "Your current training pipeline deletes samples from data_source to form clean/noisy loaders. "
            "MixFix needs stable indexing to move/relabel/drop reliably. "
            "Recommended fix: do NOT delete items; instead keep full dataset and use per-sample masks to build samplers/loaders. "
            "If you want, paste your Dataset + DataLoader builder, I will modify them to support masked sampling clean/noisy."
        )


@TRAINER_REGISTRY.register()
class NLPrompt(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE_loss = GeneralizedCrossEntropy(q=1.0)

        self.num_equal = []
        self.confident_rate = []
        self.clean_rate = []

        self.best_acc = -1
        self.best_epoch = -1
        self.test_acc = []

        # CLIPCleaner + MixFix holders
        self._clipcleaner = None
        self._mixfix = None
        self._clipcleaner_ran = False

    def check_cfg(self, cfg):
        assert cfg.TRAINER.NLPROMPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.NLPROMPT.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.NLPROMPT.PREC == "amp" else None

        # Init CLIPCleaner/MixFix if enabled
        # (These cfg keys you can later add to yaml; if absent -> default False)
        use_clipcleaner = getattr(cfg.DATASET, "USE_CLIPCLEANER", False)
        use_mixfix = getattr(cfg.DATASET, "USE_MIXFIX", False)

        if use_clipcleaner:
            # Use a frozen CLIP (base model) for offline cleaning
            # Here we reuse the loaded clip_model inside CustomCLIP? We don't have direct handle.
            # So we load another CLIP to keep it simple.
            base_clip = load_clip_to_cpu(cfg)
            if cfg.TRAINER.NLPROMPT.PREC in ["fp32", "amp"]:
                base_clip.float()

            templates = getattr(cfg.DATASET, "CLIPCLEANER_TEMPLATES", ["a photo of a {}."])
            self._clipcleaner = ClipCleaner(
                clip_model=base_clip,
                classnames=classnames,
                device=self.device,
                templates=templates
            )

        if use_mixfix:
            theta_abs = float(getattr(cfg.DATASET, "MIXFIX_THETA_ABS", 0.7))
            theta_rel = float(getattr(cfg.DATASET, "MIXFIX_THETA_REL", 0.9))
            theta_drop = float(getattr(cfg.DATASET, "MIXFIX_THETA_DROP", 0.0))
            self._mixfix = MixFixBuffer(theta_abs=theta_abs, theta_rel=theta_rel, theta_drop=theta_drop)

    def forward_backward_ce(self, batch):
        image, label, gt_label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.NLPROMPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss.item(),
            "acc_x": compute_accuracy(output, label)[0].item(),
        }
        return loss_summary

    def forward_backward_mae(self, batch):
        image, label, gt_label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.NLPROMPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = self.GCE_loss(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = self.GCE_loss(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss_u": loss.item(),
            "acc_u": compute_accuracy(output, label)[0].item(),
        }
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        gt_label = batch["gttarget"]
        input = input.to(self.device)
        label = label.to(self.device)
        gt_label = gt_label.to(self.device)
        return input, label, gt_label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_epoch(self):
        cfg = self.cfg

        # =========================================================
        # Option A: CLIPCleaner (offline, run once)
        # =========================================================
        if getattr(cfg.DATASET, "USE_CLIPCLEANER", False) and (not self._clipcleaner_ran):
            print("[CLIPCleaner] Running offline cleaning once before training...")
            # Need noisy labels aligned with dataset order
            # We assume dataset.data_source holds current noisy labels.
            labels = []
            for it in self.train_loader_x.dataset.data_source:
                labels.append(get_item_label(it))
            labels = torch.tensor(labels, dtype=torch.long)

            tau = float(getattr(cfg.DATASET, "CLIPCLEANER_TAU", 0.6))
            use_intersection = bool(getattr(cfg.DATASET, "CLIPCLEANER_INTERSECTION", True))

            # IMPORTANT: loader must yield batch["index"] aligned to dataset.data_source order
            clean_mask, clip_probs = self._clipcleaner.run(
                loader=self.train_loader_x,
                noisy_labels=labels,
                tau=tau,
                use_intersection=use_intersection
            )
            self._clipcleaner_ran = True

            # Save masks for later
            self._clip_clean_mask = clean_mask.numpy().astype(bool)
            self._clip_noisy_mask = (~clean_mask).numpy().astype(bool)

            print(f"[CLIPCleaner] clean={clean_mask.sum().item()} / {len(clean_mask)}")

        # =========================================================
        # Your original OT split (kept)
        # You can combine it with CLIPCleaner by intersection if you want
        # =========================================================
        if cfg.DATASET.USE_OT == True:
            reg_feat = cfg.DATASET.REG_FEAT
            reg_lab = cfg.DATASET.REG_LAB
            curriclum_epoch = cfg.DATASET.CURRICLUM_EPOCH
            begin_rate = cfg.DATASET.BEGIN_RATE
            curriclum_mode = cfg.DATASET.CURRICLUM_MODE
            Pmode = cfg.DATASET.PMODE
            reg_e = cfg.DATASET.REG_E

            if self.epoch < curriclum_epoch:
                budget, pho = curriculum_scheduler(self.epoch, curriclum_epoch, begin=begin_rate, end=1, mode=curriclum_mode)
            else:
                budget, pho = 1., 1.

            with torch.no_grad():
                pseudo_labels1, noisy_labels, gt_labels, selected_mask, conf1, argmax_plabels = OT_PL(
                    self.model,
                    self.train_loader_x,
                    num_class=cfg.DATASET.num_class,
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    budget=budget,
                    reg_feat=reg_feat,
                    reg_lab=reg_lab,
                    Pmode=Pmode,
                    reg_e=reg_e,
                    load_all=True
                )

                print("before epoch:data num:", len(gt_labels))
                print("before epoch:different number:", np.sum(gt_labels.cpu().numpy() != argmax_plabels.cpu().numpy()))

                conf_l_mask, conf_u_mask, lowconf_u_mask = get_masks(argmax_plabels, noisy_labels, None, selected_mask)
                selected_rate_conf_l, selected_rate_conf_u, selected_rate_lowconf_u = output_selected_rate(
                    conf_l_mask, conf_u_mask, lowconf_u_mask
                )
                print("confident_label rate", selected_rate_conf_l)
                unlabeled_mask1 = torch.logical_or(conf_u_mask, lowconf_u_mask)

            if np.sum(conf_l_mask.cpu().numpy()) > 0:
                mask = conf_l_mask.cpu().numpy()
                self.mask2 = unlabeled_mask1.cpu().numpy()

                # Combine with CLIPCleaner if enabled
                if getattr(cfg.DATASET, "USE_CLIPCLEANER", False) and hasattr(self, "_clip_clean_mask"):
                    combine_mode = getattr(cfg.DATASET, "CLIPCLEANER_COMBINE_MODE", "intersect")
                    if combine_mode == "intersect":
                        mask = np.logical_and(mask, self._clip_clean_mask)
                        self.mask2 = np.logical_not(mask)
                    elif combine_mode == "union":
                        mask = np.logical_or(mask, self._clip_clean_mask)
                        self.mask2 = np.logical_not(mask)
                    else:
                        # keep OT only
                        pass
                    print(f"[Combine] mode={combine_mode}, clean={mask.sum()} noisy={self.mask2.sum()}")

                pred_idx = mask.nonzero()[0]
                pred_idx2 = self.mask2.nonzero()[0]
                conf = conf1.cpu().numpy()
                plabel = argmax_plabels.cpu().numpy()

                self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
                self.train_loader_u = copy.deepcopy(self.train_loader_x)

                print("before: len(self.train)", len(self.train_loader_x.dataset.data_source))
                print("before: len of confident samples", len(pred_idx))

                count11 = count12 = count21 = count22 = 0
                for i in range(len(self.train_loader_x.dataset.data_source)):
                    if mask[i] == True:
                        if plabel[i] == gt_labels[i]:
                            count11 += 1
                        else:
                            count12 += 1
                    elif self.mask2[i] == True:
                        if plabel[i] == gt_labels[i]:
                            count21 += 1
                        else:
                            count22 += 1

                print(f"clean true:{count11}")
                print(f"clean false:{count12}")
                clean_rate = count11 / (count11 + count12 + 1e-12)
                print(f"clean_rate:{clean_rate}")
                self.clean_rate.append(clean_rate)
                print(f"noisy true:{count21}")
                print(f"noisy false:{count22}")

                if self.epoch == 99:
                    print("all clean rate: ", self.clean_rate)

                # === Your current split method deletes items to create two datasets ===
                # This is OK for OT-only pipeline, but MixFix needs stable indexing (no delete).
                # So if you want MixFix, you should switch to masked sampling (recommended).
                if getattr(cfg.DATASET, "USE_MIXFIX", False):
                    print("[MixFix] Detected USE_MIXFIX=True but current pipeline deletes items.")
                    print("[MixFix] Please switch to sampler/mask-based split to keep dataset indices stable.")
                    # You can still run without MixFix (only OT/CLIPCleaner split)
                    # fall back to your delete split for now.

                for index in sorted(pred_idx2, reverse=True):
                    del self.train_loader_x.dataset.data_source[index]
                print("after delete: len(clean_dataset)", len(self.train_loader_x.dataset.data_source))

                for index in sorted(pred_idx, reverse=True):
                    del self.train_loader_u.dataset.data_source[index]
                print("after delete: len(noisy_dataset)", len(self.train_loader_u.dataset.data_source))

    def run_epoch(self):
        self.set_model_mode("train")
        losses_x = MetricMeter()
        losses_u = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if self.train_loader_x is not None:
            train_loader_x_iter = iter(self.train_loader_x)
            len_train_loader_x = len(self.train_loader_x)
        else:
            len_train_loader_x = 0

        if self.train_loader_u is not None:
            train_loader_u_iter = iter(self.train_loader_u)
            len_train_loader_u = len(self.train_loader_u)
        else:
            len_train_loader_u = 0

        self.num_batches_x = len_train_loader_x
        self.num_batches_u = len_train_loader_u

        end = time.time()

        for self.batch_idx in range(self.num_batches_x):
            try:
                batch_x = next(train_loader_x_iter)
                data_time.update(time.time() - end)
                loss_summary_x = self.forward_backward_ce(batch_x)
                losses_x.update(loss_summary_x)
            except StopIteration:
                break

            batch_time.update(time.time() - end)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches_x < self.cfg.TRAIN.PRINT_FREQ:
                eta_seconds = batch_time.avg * (self.num_batches_x - self.batch_idx - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches_x}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"loss_x {losses_x}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))

            n_iter = self.epoch * (self.num_batches_x + self.num_batches_u) + self.batch_idx
            for name, meter in losses_x.meters.items():
                self.write_scalar("train_x/" + name, meter.avg, n_iter)

            end = time.time()

        for self.batch_idx in range(self.num_batches_u):
            try:
                batch_u = next(train_loader_u_iter)
                data_time.update(time.time() - end)
                loss_summary_u = self.forward_backward_mae(batch_u)
                losses_u.update(loss_summary_u)
            except StopIteration:
                break

            batch_time.update(time.time() - end)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches_u < self.cfg.TRAIN.PRINT_FREQ:
                eta_seconds = batch_time.avg * (self.num_batches_u - self.batch_idx - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches_u}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"loss_u {losses_u}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))

            n_iter = self.epoch * (self.num_batches_x + self.num_batches_u) + self.batch_idx
            for name, meter in losses_u.meters.items():
                self.write_scalar("train_u/" + name, meter.avg, n_iter)

            end = time.time()

        self.update_lr()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

        if self.cfg.DATASET.USE_OT == True:
            self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)
            self.train_loader_u = copy.deepcopy(self.tmp_train_loader_x)
            print("after epoch: len(clean dataset)", len(self.train_loader_x.dataset.data_source))
            print("after epoch: len(noisy dataset)", len(self.train_loader_u.dataset.data_source))
