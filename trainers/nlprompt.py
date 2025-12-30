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
        
        # --- 优化后的 CoT 模板：针对细粒度数据集 CIFAR-100N ---
        # 引导模型先提取视觉属性 (Context)，再根据逻辑桥梁 (therefore) 识别类别
        COT_PREFIX = "Let's think step by step. This image contains visual features of"
        LOGIC_BRIDGE = ", which implies it is a"
        
        cot_len = len(_tokenizer.encode(COT_PREFIX))

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print(f"Initializing generic context for {n_cls} classes")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        
        # 复合 Prompts 构造 [cite: 193]
        prompts = [f"{COT_PREFIX} {prompt_prefix}{LOGIC_BRIDGE} {name}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 注册 Buffer，确保切片与推理模版完全对齐
        self.register_buffer("token_prefix", embedding[:, :1 + cot_len, :]) 
        self.register_buffer("token_suffix", embedding[:, 1 + cot_len + n_ctx :, :]) 

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

class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q, self.epsilon = q, 1e-6
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target] + self.epsilon
        return torch.mean((1 - p ** self.q) / self.q)

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
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ text_features.t()

@TRAINER_REGISTRY.register()
class NLPrompt(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE_loss = GeneralizedCrossEntropy(q=1.0)
        self.clean_rate = []

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.NLPROMPT.PREC in ["fp32", "amp"]: clip_model.float()
        
        print("Building Custom CLIP with Fine-grained CoT Reasoning")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name: param.requires_grad_(False)
            
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.NLPROMPT.PREC == "amp" else None

    def forward_backward_ce(self, batch):
        image, label, _ = self.parse_batch_train(batch)
        if self.cfg.TRAINER.NLPROMPT.PREC == "amp":
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
        image, label, _ = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.GCE_loss(output, label)
        self.model_backward_and_update(loss)
        return {"loss_u": loss.item(), "acc_u": compute_accuracy(output, label)[0].item()}

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device), batch["gttarget"].to(self.device)

    def load_model(self, directory, epoch=None):
        if not directory: return
        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path): continue
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # 自动忽略不匹配的旧 Buffer
            for k in ["token_prefix", "token_suffix"]:
                if k in state_dict: del state_dict[k]
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_epoch(self):
        cfg = self.cfg
        if cfg.DATASET.USE_OT:
            budget, _ = curriculum_scheduler(self.epoch, cfg.DATASET.CURRICLUM_EPOCH, begin=cfg.DATASET.BEGIN_RATE, end=1.0) if self.epoch < cfg.DATASET.CURRICLUM_EPOCH else (1.0, 1.0)
            with torch.no_grad():
                _, noisy_labels, gt_labels, selected_mask, _, argmax_plabels = OT_PL(self.model, self.train_loader_x, num_class=cfg.DATASET.num_class, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, budget=budget, reg_feat=cfg.DATASET.REG_FEAT, reg_lab=cfg.DATASET.REG_LAB, Pmode=cfg.DATASET.PMODE, reg_e=cfg.DATASET.REG_E, load_all=True)
                conf_l, conf_u, low_u = get_masks(argmax_plabels, noisy_labels, None, selected_mask)
            
            mask = conf_l.cpu().numpy()
            if np.sum(mask) > 0:
                self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
                self.train_loader_u = copy.deepcopy(self.train_loader_x)
                u_mask = torch.logical_or(conf_u, low_u).cpu().numpy()
                idx_x, idx_u = u_mask.nonzero()[0], mask.nonzero()[0]
                for i in sorted(idx_x, reverse=True): del self.train_loader_x.dataset.data_source[i]
                for i in sorted(idx_u, reverse=True): del self.train_loader_u.dataset.data_source[i]

    def run_epoch(self):
        self.set_model_mode("train")
        losses_x, losses_u = MetricMeter(), MetricMeter()
        if self.train_loader_x:
            iter_x = iter(self.train_loader_x)
            for self.batch_idx in range(len(self.train_loader_x)):
                losses_x.update(self.forward_backward_ce(next(iter_x)))
        if self.train_loader_u:
            iter_u = iter(self.train_loader_u)
            for self.batch_idx in range(len(self.train_loader_u)):
                losses_u.update(self.forward_backward_mae(next(iter_u)))
        self.update_lr()

    def after_epoch(self):
        if not self.cfg.TEST.NO_TEST and self.test(split="val") > self.best_result:
            self.best_result = self.test(split="val")
            self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
        if self.cfg.DATASET.USE_OT:
            self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)