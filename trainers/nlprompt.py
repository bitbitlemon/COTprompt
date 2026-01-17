import os.path as osp
import os
import copy
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import MetricMeter, AverageMeter

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils import *  # 你的 OT_PL / get_masks 等

_tokenizer = _Tokenizer()

# ------------------------------ helpers ------------------------------
def _safe_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _impath_to_index(impath: str):
    # cifar100n.py 的扫描方式：stem 为数字且 idx = int(stem)-1
    base = os.path.basename(impath)
    stem, _ = os.path.splitext(base)
    if stem.isdigit():
        return int(stem) - 1
    return None

def _get_score(prediction: torch.Tensor, labels: torch.Tensor, mode='celoss'):
    """
    prediction: [N,C] prob
    labels: [N]
    return: [N] score (higher usually means "cleaner" for consistency; for celoss it returns log p(y) which is <=0)
    """
    pred_safe = prediction.clamp(min=1e-12)
    if mode == 'celoss':
        loss = torch.log(pred_safe)
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == 'perclass_celoss':
        loss = torch.log(pred_safe)
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        labels_np = labels.detach().cpu().numpy()
        num_classes = int(labels_np.max()) + 1 if labels_np.size > 0 else int(prediction.size(1))
        for c in range(num_classes):
            ids = np.where(labels_np == c)[0]
            if len(ids) == 0:
                continue
            s = score[ids]
            s_min = s.min()
            s_max = s.max()
            denom = (s_max - s_min)
            if torch.isfinite(denom) and denom > 0:
                score[ids] = (s - s_min) / (denom + 1e-12)
            else:
                score[ids] = 0.0
        return score

    # consistency: p(y)/pmax
    vote_y = torch.gather(prediction, 1, labels.view(-1, 1)).squeeze()
    vote_max = prediction.max(dim=1)[0]
    score = vote_y / (vote_max + 1e-12)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score

def _logistic_regression_prob(features_cpu: torch.Tensor, labels_cpu: torch.Tensor, max_samples=20000):
    """
    features_cpu: [N,D] on CPU
    labels_cpu: [N] on CPU
    return prob: [N,C] torch float
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        print(f"[warn] sklearn not available for LogisticRegression: {e}")
        return None

    X = features_cpu.numpy()
    y = labels_cpu.numpy()

    N = X.shape[0]
    if N > max_samples:
        # stratified-ish subsample
        rng = np.random.RandomState(0)
        idx = rng.choice(N, size=max_samples, replace=False)
        X_fit = X[idx]
        y_fit = y[idx]
    else:
        X_fit, y_fit = X, y

    clf = LogisticRegression(
        random_state=0,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1
    ).fit(X_fit, y_fit)

    prob = clf.predict_proba(X)  # [N,C]
    return torch.tensor(prob, dtype=torch.float32)

def _perclass_select(score: torch.Tensor, labels: torch.Tensor, mode: str, theta_gmm=0.5, theta_cons=0.5, num_classes=100):
    """
    score: [N] (cpu tensor)
    labels: [N] (cpu tensor)
    mode: 'loss' or 'consistency'
    returns: np.ndarray selected indices
    """
    labels_np = labels.cpu().numpy()
    score_np = score.cpu().numpy()

    id_by_label = [np.where(labels_np == i)[0] for i in range(num_classes)]
    selected_all = []

    if mode == "loss":
        # per-class GMM on score
        try:
            from sklearn.mixture import GaussianMixture
        except Exception as e:
            raise RuntimeError(f"Need sklearn for GaussianMixture: {e}")

        for c in range(num_classes):
            ids = id_by_label[c]
            if len(ids) == 0:
                continue
            s = score_np[ids].reshape(-1, 1)
            if len(ids) < 4:
                # too few, keep all
                selected_all.append(ids)
                continue
            gmm = GaussianMixture(2, random_state=0)
            gmm.fit(s)
            # choose component with larger mean (cleaner for perclass_celoss normalized)
            comp = int(np.argmax(gmm.means_.reshape(-1)))
            prob = gmm.predict_proba(s)[:, comp]
            keep = ids[np.where(prob >= theta_gmm)[0]]
            selected_all.append(keep)
    else:
        # threshold consistency
        for c in range(num_classes):
            ids = id_by_label[c]
            if len(ids) == 0:
                continue
            s = score_np[ids]
            keep = ids[np.where(s >= theta_cons)[0]]
            selected_all.append(keep)

    if len(selected_all) == 0:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(selected_all)).astype(np.int64)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'NLPrompt', "vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0}
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

        prompt_style = getattr(cfg.TRAINER.NLPROMPT, "PROMPT_STYLE", "coop")
        prompt_style = str(prompt_style).lower()
        if prompt_style == "cot":
            prefix_str = "Let's think step by step. This image contains visual features of"
            suffix_str = ", which implies it is a"
        else:
            prefix_str = "a photo of a"
            suffix_str = ""

        prefix_len = len(_tokenizer.encode(prefix_str))

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print(f"Initializing generic context for {n_cls} classes")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]

        if suffix_str:
            prompts = [f"{prefix_str} {prompt_prefix}{suffix_str} {name}." for name in classnames]
        else:
            prompts = [f"{prefix_str} {prompt_prefix} {name}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1 + prefix_len, :])
        self.register_buffer("token_suffix", embedding[:, 1 + prefix_len + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        self.class_token_position = cfg.TRAINER.NLPROMPT.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix, suffix = self.token_prefix, self.token_suffix
        return torch.cat([prefix, ctx, suffix], dim=1)

class FixedPrompt(nn.Module):
    """
    固定 prompt 分支：不训练，只提供 token embedding + tokenized_prompts
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        dtype = clip_model.dtype

        prompt_style = getattr(cfg.TRAINER.NLPROMPT, "PROMPT_STYLE", "coop")
        prompt_style = str(prompt_style).lower()
        if prompt_style == "cot":
            prefix_str = "Let's think step by step. This image contains visual features of"
            suffix_str = ", which implies it is a"
        else:
            prefix_str = "a photo of a"
            suffix_str = ""

        classnames = [name.replace("_", " ") for name in classnames]
        if suffix_str:
            prompts = [f"{prefix_str}{suffix_str} {name}." for name in classnames]
        else:
            prompts = [f"{prefix_str} {name}." for name in classnames]

        tokenized = torch.cat([clip.tokenize(p) for p in prompts])
        self.register_buffer("tokenized_prompts", tokenized)

        with torch.no_grad():
            emb = clip_model.token_embedding(tokenized).type(dtype)
        self.register_buffer("prompts_emb", emb)

    def forward(self):
        return self.prompts_emb, self.tokenized_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.fixed_prompt = None
        if getattr(cfg.TRAINER.NLPROMPT, "USE_FIXED_PROMPT", True):
            self.fixed_prompt = FixedPrompt(cfg, classnames, clip_model)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_prob=False):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # learnable
        prompts = self.prompt_learner()
        text_features_learn = self.text_encoder(prompts, self.tokenized_prompts)
        text_features_learn = text_features_learn / text_features_learn.norm(dim=-1, keepdim=True)
        logits_learn = self.logit_scale.exp() * image_features @ text_features_learn.t()

        use_hybrid = bool(getattr(self.cfg.TRAINER.NLPROMPT, "USE_HYBRID", True))
        if (self.fixed_prompt is None) or (not use_hybrid):
            return torch.softmax(logits_learn, dim=1) if return_prob else logits_learn

        # fixed
        fixed_emb, fixed_tok = self.fixed_prompt()
        text_features_fixed = self.text_encoder(fixed_emb, fixed_tok)
        text_features_fixed = text_features_fixed / text_features_fixed.norm(dim=-1, keepdim=True)
        logits_fixed = self.logit_scale.exp() * image_features @ text_features_fixed.t()

        alpha = float(getattr(self.cfg.TRAINER.NLPROMPT, "HYBRID_ALPHA", 0.5))
        alpha = max(0.0, min(1.0, alpha))
        logits = (1 - alpha) * logits_fixed + alpha * logits_learn
        return torch.softmax(logits, dim=1) if return_prob else logits

@TRAINER_REGISTRY.register()
class NLPrompt(TrainerX):
    def __init__(self, cfg):
        self.prec = cfg.TRAINER.NLPROMPT.PREC
        super().__init__(cfg)
        self.GCE_loss = nn.CrossEntropyLoss()
        self.clean_rate = []

        # 备份原始 loader（用于动态筛选时重置）
        self.tmp_train_loader_x = None
        self.train_loader_u = None

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = load_clip_to_cpu(cfg)

        if self.device.type != "cuda":
            clip_model.float()
            self.prec = "fp32"
        elif self.prec in ["fp32", "amp"]:
            clip_model.float()

        print("Building Custom CLIP with Hybrid Prompt + Dynamic Selection")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # freeze non-prompt
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if (self.prec == "amp" and self.device.type == "cuda") else None

    def forward_backward_ce(self, batch):
        image, label, _, _ = self.parse_batch_train(batch)
        if self.prec == "amp" and self.device.type == "cuda":
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
        return {"loss_x": loss.item(), "acc_x": compute_accuracy(output, label)[0].item()}

    def forward_backward_mae(self, batch):
        image, label, _, _ = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.GCE_loss(output, label)
        self.model_backward_and_update(loss)
        return {"loss_u": loss.item(), "acc_u": compute_accuracy(output, label)[0].item()}

    def parse_batch_train(self, batch):
        # Dassl 默认 batch 往往包含 impath；如果没有，返回 None
        img = batch["img"].to(self.device)
        lab = batch["label"].to(self.device)
        gtt = batch["gttarget"].to(self.device) if "gttarget" in batch else None
        imp = batch["impath"] if "impath" in batch else None
        return img, lab, gtt, imp

    def load_model(self, directory, epoch=None):
        if not directory:
            return
        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                continue
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            for k in ["token_prefix", "token_suffix"]:
                if k in state_dict:
                    del state_dict[k]
            self._models[name].load_state_dict(state_dict, strict=False)

    # ===================== Dynamic Selection Core =====================
    @torch.no_grad()
    def _collect_features_probs(self):
        """
        在“当前 train_loader_x 的 dataset”上，shuffle=False 遍历，收集：
          - features [N,D] (cpu)
          - probs    [N,C] (cpu)  # 使用 hybrid 输出的 softmax
          - labels   [N] (cpu)    # noisy label
          - impaths  [N] list[str] 或 None
        """
        base_loader = self.train_loader_x
        if base_loader is None:
            return None, None, None, None

        # 重新构造一个 shuffle=False 的 loader，保证顺序稳定
        ds = base_loader.dataset
        bs = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        nw = self.cfg.DATALOADER.NUM_WORKERS if hasattr(self.cfg.DATALOADER, "NUM_WORKERS") else 8

        eval_loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False)

        self.model.eval()

        feats = []
        probs = []
        labels = []
        impaths = []

        for batch in eval_loader:
            img, lab, _, imp = self.parse_batch_train(batch)

            # hybrid prob
            p = self.model(img, return_prob=True).detach().float().cpu()

            # image feature
            f = self.model.image_encoder(img.type(self.model.dtype)).detach().float()
            f = F.normalize(f, dim=1).cpu()

            feats.append(f)
            probs.append(p)
            labels.append(lab.detach().cpu())

            if imp is None:
                impaths.extend([None] * lab.size(0))
            else:
                # imp 可能是 list[str]
                if isinstance(imp, (list, tuple)):
                    impaths.extend(list(imp))
                else:
                    # 有些 collate 会变成 list
                    try:
                        impaths.extend(list(imp))
                    except Exception:
                        impaths.extend([None] * lab.size(0))

        feats = torch.cat(feats, dim=0)
        probs = torch.cat(probs, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, probs, labels, impaths

    @torch.no_grad()
    def _run_hybrid_fourway_selection(self):
        cfg = self.cfg
        num_classes = int(getattr(cfg.DATASET, "num_class", 100))

        feats_cpu, probs_cpu, labels_cpu, impaths = self._collect_features_probs()
        if feats_cpu is None:
            return None, None

        # LR branch prob
        prob_lr = None
        if bool(getattr(cfg.TRAINER.NLPROMPT, "USE_LR_BRANCH", True)):
            prob_lr = _logistic_regression_prob(
                feats_cpu,
                labels_cpu,
                max_samples=int(getattr(cfg.TRAINER.NLPROMPT, "LR_MAX_SAMPLES", 20000))
            )

        # 4 scores
        s1 = _get_score(probs_cpu, labels_cpu, "perclass_celoss")
        s2 = _get_score(probs_cpu, labels_cpu, "consistency")

        if prob_lr is not None:
            s3 = _get_score(prob_lr, labels_cpu, "perclass_celoss")
            s4 = _get_score(prob_lr, labels_cpu, "consistency")
        else:
            # fallback：只用两路
            s3, s4 = None, None

        theta_gmm = float(getattr(cfg.TRAINER.NLPROMPT, "THETA_GMM", 0.5))
        theta_cons = float(getattr(cfg.TRAINER.NLPROMPT, "THETA_CONS", 0.5))

        # per-class select
        sel1 = _perclass_select(s1, labels_cpu, mode="loss", theta_gmm=theta_gmm, theta_cons=theta_cons, num_classes=num_classes)
        sel2 = _perclass_select(s2, labels_cpu, mode="consistency", theta_gmm=theta_gmm, theta_cons=theta_cons, num_classes=num_classes)

        if s3 is not None and s4 is not None:
            sel3 = _perclass_select(s3, labels_cpu, mode="loss", theta_gmm=theta_gmm, theta_cons=theta_cons, num_classes=num_classes)
            sel4 = _perclass_select(s4, labels_cpu, mode="consistency", theta_gmm=theta_gmm, theta_cons=theta_cons, num_classes=num_classes)
        else:
            sel3, sel4 = None, None

        use_intersect = bool(getattr(cfg.TRAINER.NLPROMPT, "SELECTION_INTERSECT", True))
        if use_intersect:
            sel = sel1
            sel = np.intersect1d(sel, sel2)
            if sel3 is not None:
                sel = np.intersect1d(sel, sel3)
            if sel4 is not None:
                sel = np.intersect1d(sel, sel4)
        else:
            sel = np.unique(np.concatenate([x for x in [sel1, sel2, sel3, sel4] if x is not None]))

        sel = np.unique(sel).astype(np.int64)

        # 防止空类：每类至少 MIN_PER_CLASS
        min_k = int(getattr(cfg.TRAINER.NLPROMPT, "MIN_PER_CLASS", 1))
        if min_k > 0:
            labels_np = labels_cpu.numpy()
            # 用 s1（perclass_celoss）做补齐排序（越大越“好”）
            sfill = s1.cpu().numpy()
            for c in range(num_classes):
                have = np.sum(labels_np[sel] == c)
                if have >= min_k:
                    continue
                all_c = np.where(labels_np == c)[0]
                if len(all_c) == 0:
                    continue
                # 补齐 top-k
                order = np.argsort(-sfill[all_c])  # desc
                need = min_k - have
                add = all_c[order[:need]]
                sel = np.unique(np.concatenate([sel, add])).astype(np.int64)

        N = labels_cpu.numel()
        mask_clean = np.zeros(N, dtype=bool)
        mask_clean[sel] = True
        clean_idx = np.where(mask_clean)[0].astype(np.int64)
        noisy_idx = np.where(~mask_clean)[0].astype(np.int64)

        print(f"[DynSel] selected clean={len(clean_idx)}/{N} ({len(clean_idx)/max(N,1):.3f})")
        return clean_idx, noisy_idx

    def _rebuild_loaders_by_indices(self, clean_idx: np.ndarray, noisy_idx: np.ndarray):
        """
        用你现在的“删除 data_source”方式重建 train_loader_x / train_loader_u
        """
        if self.tmp_train_loader_x is None:
            self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)

        # 基于 tmp 的全量，分别 deep copy
        self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)
        self.train_loader_u = copy.deepcopy(self.tmp_train_loader_x)

        # 注意：这里假设 dataset 有 data_source（Dassl 常见）
        ds_x = self.train_loader_x.dataset
        ds_u = self.train_loader_u.dataset

        # 删除不需要的样本：逆序 del
        keep_clean = set(clean_idx.tolist())
        keep_noisy = set(noisy_idx.tolist())

        # ds_x：只保留 clean
        for i in range(len(ds_x.data_source) - 1, -1, -1):
            if i not in keep_clean:
                del ds_x.data_source[i]

        # ds_u：只保留 noisy
        for i in range(len(ds_u.data_source) - 1, -1, -1):
            if i not in keep_noisy:
                del ds_u.data_source[i]

    # ===================== Training loop hooks =====================
    def before_epoch(self):
        cfg = self.cfg

        # 先做一次备份
        if self.tmp_train_loader_x is None:
            self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)

        # --- 动态筛选优先（如果开了） ---
        if bool(getattr(cfg.TRAINER.NLPROMPT, "DYN_SEL", False)):
            start_ep = int(getattr(cfg.TRAINER.NLPROMPT, "DYN_SEL_START_EPOCH", 0))
            freq = int(getattr(cfg.TRAINER.NLPROMPT, "DYN_SEL_FREQ", 2))
            if self.epoch >= start_ep and (self.epoch % max(freq, 1) == 0):
                try:
                    clean_idx, noisy_idx = self._run_hybrid_fourway_selection()
                    if clean_idx is not None:
                        self._rebuild_loaders_by_indices(clean_idx, noisy_idx)
                        return
                except Exception as e:
                    print(f"[warn] DynSel failed, fallback to OT if enabled. err={e}")

        # --- fallback：你原来的 OT ---
        if cfg.DATASET.USE_OT:
            budget, _ = curriculum_scheduler(
                self.epoch, cfg.DATASET.CURRICLUM_EPOCH,
                begin=cfg.DATASET.BEGIN_RATE, end=1.0
            ) if self.epoch < cfg.DATASET.CURRICLUM_EPOCH else (1.0, 1.0)

            with torch.no_grad():
                _, noisy_labels, gt_labels, selected_mask, _, argmax_plabels = OT_PL(
                    self.model, self.train_loader_x,
                    num_class=cfg.DATASET.num_class,
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    budget=budget,
                    reg_feat=cfg.DATASET.REG_FEAT,
                    reg_lab=cfg.DATASET.REG_LAB,
                    Pmode=cfg.DATASET.PMODE,
                    reg_e=cfg.DATASET.REG_E,
                    load_all=True
                )
                conf_l, conf_u, low_u = get_masks(argmax_plabels, noisy_labels, None, selected_mask)

            mask = conf_l.cpu().numpy()
            if np.sum(mask) > 0:
                self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
                self.train_loader_u = copy.deepcopy(self.train_loader_x)
                u_mask = torch.logical_or(conf_u, low_u).cpu().numpy()
                idx_x, idx_u = u_mask.nonzero()[0], mask.nonzero()[0]
                for i in sorted(idx_x, reverse=True):
                    del self.train_loader_x.dataset.data_source[i]
                for i in sorted(idx_u, reverse=True):
                    del self.train_loader_u.dataset.data_source[i]

    def run_epoch(self):
        self.set_model_mode("train")
        losses_x, losses_u = MetricMeter(), MetricMeter()
        batch_time, data_time = AverageMeter(), AverageMeter()

        end = time.time()

        if self.train_loader_x:
            iter_x = iter(self.train_loader_x)
            for self.batch_idx in range(len(self.train_loader_x)):
                data_time.update(time.time() - end)
                losses_x.update(self.forward_backward_ce(next(iter_x)))
                batch_time.update(time.time() - end)

                meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
                only_few_batches = len(self.train_loader_x) < self.cfg.TRAIN.PRINT_FREQ
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += len(self.train_loader_x) - self.batch_idx - 1
                    nb_remain += (self.max_epoch - self.epoch - 1) * len(self.train_loader_x)
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                    info += [f"batch [{self.batch_idx + 1}/{len(self.train_loader_x)}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"eta {eta}"]
                    info += [f"{losses_x}"]
                    print(" ".join(info))

                end = time.time()

        if self.train_loader_u:
            iter_u = iter(self.train_loader_u)
            for self.batch_idx in range(len(self.train_loader_u)):
                losses_u.update(self.forward_backward_mae(next(iter_u)))

        self.update_lr()
        print(f"epoch [{self.epoch + 1}/{self.max_epoch}] train_x {losses_x}")
        if self.train_loader_u:
            print(f"epoch [{self.epoch + 1}/{self.max_epoch}] train_u {losses_u}")

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            if curr_result > self.best_result:
                self.best_result = curr_result
                self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
