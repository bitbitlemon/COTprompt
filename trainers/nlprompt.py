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
        # loading JIT archive
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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
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

        # --- COT (Chain of Thought) 思维链配置 ---
        # 增加逻辑引导词，强制模型按推理逻辑处理特征
        COT_PREFIX = "Let's think step by step. This image shows"
        INFERENCE_BRIDGE = ", therefore it is a"
        
        # 计算引导词的 token 长度以用于后续切片
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
            if cfg.TRAINER.NLPROMPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"COT strategy applied: {COT_PREFIX} [CTX] {INFERENCE_BRIDGE} [CLS]")

        self.ctx = nn.Parameter(ctx_vectors)  # 优化的上下文向量

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        # 构建带有思维链逻辑的完整 Prompt 模板
        prompts = [f"{COT_PREFIX} {prompt_prefix}{INFERENCE_BRIDGE} {name}." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 重新计算 Buffer 切片位置
        # 现在的结构: [SOS] [COT_TOKENS] [Learned_CTX_TOKENS] [BRIDGE_TOKENS] [CLASS_TOKENS] [EOS]
        # self.token_prefix 包含 SOS 和 COT 引导语
        self.register_buffer("token_prefix", embedding[:, :1 + cot_len, :])  
        # self.token_suffix 包含 "therefore..."、类名、句号和 EOS
        self.register_buffer("token_suffix", embedding[:, 1 + cot_len + n_ctx :, :])  

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
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
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
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

@TRAINER_REGISTRY.register()
class NLPrompt(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE_loss = GeneralizedCrossEntropy(q=1.0)
        self.num_equal = []
        self.confident_rate = []
        self.clean_rate  = []
        self.best_acc = -1
        self.best_epoch = -1
        self.test_acc = []

    def check_cfg(self, cfg):
        assert cfg.TRAINER.NLPROMPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.NLPROMPT.PREC == "fp32" or cfg.TRAINER.NLPROMPT.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP with COT reasoning learner")
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

        return {"loss_x": loss.item(), "acc_x": compute_accuracy(output, label)[0].item()}
    
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

        return {"loss_u": loss.item(), "acc_u": compute_accuracy(output, label)[0].item()}

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device), batch["gttarget"].to(self.device)

    def load_model(self, directory, epoch=None):
        if not directory: return
        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path): raise FileNotFoundError(f'Model not found at "{model_path}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            if "token_prefix" in state_dict: del state_dict["token_prefix"]
            if "token_suffix" in state_dict: del state_dict["token_suffix"]
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_epoch(self):
        cfg = self.cfg
        if cfg.DATASET.USE_OT == True:
            budget, pho = curriculum_scheduler(self.epoch, cfg.DATASET.CURRICLUM_EPOCH, begin=cfg.DATASET.BEGIN_RATE, end=1, mode=cfg.DATASET.CURRICLUM_MODE) if self.epoch < cfg.DATASET.CURRICLUM_EPOCH else (1., 1.)
            
            with torch.no_grad():
                pseudo_labels1, noisy_labels, gt_labels, selected_mask, conf1, argmax_plabels = OT_PL(
                    self.model, self.train_loader_x, num_class=cfg.DATASET.num_class, 
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, budget=budget, 
                    reg_feat=cfg.DATASET.REG_FEAT, reg_lab=cfg.DATASET.REG_LAB,
                    Pmode=cfg.DATASET.PMODE, reg_e=cfg.DATASET.REG_E, load_all=True
                )
                conf_l_mask, conf_u_mask, lowconf_u_mask = get_masks(argmax_plabels, noisy_labels, None, selected_mask)
                unlabeled_mask1 = torch.logical_or(conf_u_mask, lowconf_u_mask)

            if np.sum(conf_l_mask.cpu().numpy()) > 0:
                mask = conf_l_mask.cpu().numpy() 
                self.mask2 = unlabeled_mask1.cpu().numpy()
                plabel = argmax_plabels.cpu().numpy()
                self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
                self.train_loader_u = copy.deepcopy(self.train_loader_x)

                # 计算并保存指标
                c11 = np.sum((mask == True) & (plabel == gt_labels.cpu().numpy()))
                c12 = np.sum((mask == True) & (plabel != gt_labels.cpu().numpy()))
                clean_rate = c11 / (c11 + c12) if (c11 + c12) > 0 else 0
                self.clean_rate.append(clean_rate)
                
                pred_idx = mask.nonzero()[0]
                pred_idx2 = self.mask2.nonzero()[0]
                for index in sorted(pred_idx2, reverse=True):
                    del self.train_loader_x.dataset.data_source[index]
                for index in sorted(pred_idx, reverse=True):
                    del self.train_loader_u.dataset.data_source[index]

    def run_epoch(self):
        self.set_model_mode("train")
        losses_x, losses_u = MetricMeter(), MetricMeter()
        batch_time, data_time = AverageMeter(), AverageMeter()

        train_loader_x_iter = iter(self.train_loader_x) if self.train_loader_x else None
        len_x = len(self.train_loader_x) if self.train_loader_x else 0
        train_loader_u_iter = iter(self.train_loader_u) if self.train_loader_u else None
        len_u = len(self.train_loader_u) if self.train_loader_u else 0

        self.num_batches_x, self.num_batches_u = len_x, len_u
        end = time.time()
        
        for self.batch_idx in range(self.num_batches_x):
            try:
                batch_x = next(train_loader_x_iter)
                data_time.update(time.time() - end)
                losses_x.update(self.forward_backward_ce(batch_x))
                batch_time.update(time.time() - end)
                end = time.time()
            except StopIteration: break

        for self.batch_idx in range(self.num_batches_u):
            try:
                batch_u = next(train_loader_u_iter)
                losses_u.update(self.forward_backward_mae(batch_u))
            except StopIteration: break

        self.update_lr()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        if not self.cfg.TEST.NO_TEST and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            if curr_result > self.best_result:
                self.best_result = curr_result
                self.save_model(self.epoch, self.output_dir, val_result=curr_result, model_name="model-best.pth.tar")
        
        if last_epoch or ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False):
            self.save_model(self.epoch, self.output_dir)
        
        if self.cfg.DATASET.USE_OT == True:
            self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)
            self.train_loader_u = copy.deepcopy(self.tmp_train_loader_x)