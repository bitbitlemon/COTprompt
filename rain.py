#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创新点验证消融脚本（按“创新点”组织）
- 13个配置
- 仅在3个数据集上跑（默认 eth,hotel,univ，可通过环境变量 SPLITS 覆盖）
- 早停：连续4个评估轮次无提升 -> 终止该配置训练（更快）
- 自动解析 best ADE/FDE 并汇总

用法示例：
  python ablation_innovations.py

  # 自己指定3个数据集（只会按你给的跑）
  SPLITS="eth,hotel,nba50k" python ablation_innovations.py

  # 如果你想更激进一点（可选）
  EARLY_STOP_DELTA="0.01" python ablation_innovations.py
"""

import os
import sys
import re
import time
import csv
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# =================== 基本配置（按需改） ===================
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
MAIN_FILE = os.environ.get("MAIN_FILE", "main.py")

MODEL = os.environ.get("MODEL", "ran")

# 只跑3个数据集：默认 eth,hotel,univ（你可用 SPLITS 覆盖）
SPLITS = os.environ.get("SPLITS", "eth,hotel,univ").split(",")

GPU = os.environ.get("GPU", "0")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "64")
LR = os.environ.get("LR", "1e-4")
PARTITIONS = os.environ.get("PARTITIONS", "8")

# ============== 日志解析（覆盖多种输出格式） ==============
BEST_PAIR_PATTERN = re.compile(r"\bbest=\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")
ADE_PATTERN = re.compile(r"\bADE\b(?:\s*\(\s*Metrics\s*\))?\s*[:=]\s*([0-9]*\.?[0-9]+)")
FDE_PATTERN = re.compile(r"\bFDE\b(?:\s*\(\s*Metrics\s*\))?\s*[:=]\s*([0-9]*\.?[0-9]+)")

# ============== 早停（你要求：超过4个轮次无提升就停止） ==============
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "4"))   # 固定默认4
EARLY_STOP_DELTA = float(os.environ.get("EARLY_STOP_DELTA", "0.005"))
# 为了“更快”，不强制最小运行时间；一旦达到patience就停
MIN_RUNTIME_SEC = int(os.environ.get("MIN_RUNTIME_SEC", "0"))
DUPLICATE_TIME_WINDOW_SEC = float(os.environ.get("DUPLICATE_TIME_WINDOW_SEC", "1.5"))


@dataclass
class ExpConfig:
    """
    13个配置（与你原始脚本一致）
    """
    use_signature: int = 1
    use_sig_tokens: int = 1
    use_sig_attn: int = 1
    use_struct_ctrl: int = 1
    use_dynamic_intent: int = 1
    use_expert_norm: int = 1
    use_diversity_gate: int = 1
    use_score_head: int = 1
    cache_scores: int = 1

    use_softmin: int = 1
    use_score_align: int = 1

    tau: float = 0.7
    score_align_w: float = 0.2
    score_gt_tau: float = 0.5
    score_pred_tau: float = 1.0

    K: Optional[int] = None
    tag: str = ""

    def name(self, idx: int, split: str) -> str:
        parts = [
            f"I{idx:02d}",
            f"sig{self.use_signature}",
            f"tok{self.use_sig_tokens}",
            f"attn{self.use_sig_attn}",
            f"struct{self.use_struct_ctrl}",
            f"dyn{self.use_dynamic_intent}",
            f"en{self.use_expert_norm}",
            f"div{self.use_diversity_gate}",
            f"score{self.use_score_head}",
            f"cache{self.cache_scores}",
            f"soft{self.use_softmin}",
            f"salign{self.use_score_align}",
            f"split-{split}",
        ]
        if self.K is not None:
            parts.append(f"K{self.K}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def build_cmd(exp_name: str, split: str) -> List[str]:
    return [
        PYTHON_BIN, MAIN_FILE,
        "--model", MODEL,
        "--split", split,
        "--gpu", GPU,
        "--batch_size", BATCH_SIZE,
        "--lr", LR,
        "--partitions", PARTITIONS,
        "--exp_name", exp_name,
    ]


def build_env(cfg: ExpConfig) -> Dict[str, str]:
    env = os.environ.copy()

    # -------- model side --------
    env["RAN_USE_SIGNATURE"] = str(cfg.use_signature)
    env["RAN_USE_SIG_TOKENS"] = str(cfg.use_sig_tokens)
    env["RAN_USE_SIG_ATTN"] = str(cfg.use_sig_attn)
    env["RAN_USE_STRUCT_CTRL"] = str(cfg.use_struct_ctrl)
    env["RAN_USE_DYNAMIC_INTENT"] = str(cfg.use_dynamic_intent)
    env["RAN_USE_EXPERT_NORM"] = str(cfg.use_expert_norm)
    env["RAN_USE_DIVERSITY_GATE"] = str(cfg.use_diversity_gate)
    env["RAN_USE_SCORE_HEAD"] = str(cfg.use_score_head)
    env["RAN_CACHE_SCORES"] = str(cfg.cache_scores)

    if cfg.K is not None:
        env["RAN_K"] = str(cfg.K)

    # -------- loss side --------
    env["RAN_USE_SOFTMIN"] = str(cfg.use_softmin)
    env["RAN_USE_SCORE_ALIGN"] = str(cfg.use_score_align)

    env["RAN_SOFTMIN_TAU"] = str(cfg.tau)
    env["RAN_SCORE_ALIGN_W"] = str(cfg.score_align_w)
    env["RAN_SCORE_GT_TAU"] = str(cfg.score_gt_tau)
    env["RAN_SCORE_PRED_TAU"] = str(cfg.score_pred_tau)

    return env


def parse_best_from_line(line: str) -> Optional[Tuple[float, float]]:
    m = BEST_PAIR_PATTERN.search(line)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            return None
    return None


def run_one(
    cmd: List[str],
    env: Dict[str, str],
    cwd: str,
    log_path: str,
) -> Tuple[float, float, bool, int]:
    best_ade = float("inf")
    best_fde = float("inf")
    last_fde = None

    last_metric_time = 0.0
    patience = 0
    early_stopped = False
    start_time = time.time()

    tracked = [
        "RAN_USE_SIGNATURE", "RAN_USE_SIG_TOKENS", "RAN_USE_SIG_ATTN",
        "RAN_USE_STRUCT_CTRL", "RAN_USE_DYNAMIC_INTENT",
        "RAN_USE_EXPERT_NORM", "RAN_USE_DIVERSITY_GATE",
        "RAN_USE_SCORE_HEAD", "RAN_CACHE_SCORES",
        "RAN_USE_SOFTMIN", "RAN_USE_SCORE_ALIGN",
        "RAN_SOFTMIN_TAU", "RAN_SCORE_ALIGN_W", "RAN_SCORE_GT_TAU", "RAN_SCORE_PRED_TAU",
        "RAN_K",
    ]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("ENV:\n")
        for k in tracked:
            if k in env:
                f.write(f"  {k}={env.get(k)}\n")
        f.write("\nCMD:\n" + " ".join(cmd) + "\n")
        f.write("=" * 110 + "\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env
        )

        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)

            # 1) 优先抓 best=[a,b]
            pair = parse_best_from_line(line)
            if pair:
                ade, fde = pair
                if ade < best_ade:
                    best_ade, best_fde = ade, fde
                    patience = 0
                else:
                    patience += 1

                now = time.time()
                if (now - start_time) >= MIN_RUNTIME_SEC and patience >= EARLY_STOP_PATIENCE:
                    early_stopped = True
                    msg = "\n[EarlyStop] no improvement for 4 rounds(best=), terminate.\n"
                    sys.stdout.write(msg)
                    f.write(msg)
                    f.flush()
                    p.terminate()
                    break
                continue

            # 2) 否则用 ADE/FDE pattern 近似抓
            m_fde = FDE_PATTERN.search(line)
            if m_fde:
                try:
                    last_fde = float(m_fde.group(1))
                except Exception:
                    pass

            m_ade = ADE_PATTERN.search(line)
            if not m_ade:
                continue

            try:
                ade = float(m_ade.group(1))
            except Exception:
                continue

            now = time.time()
            if now - last_metric_time < DUPLICATE_TIME_WINDOW_SEC:
                continue
            last_metric_time = now

            improved = ade < (best_ade - EARLY_STOP_DELTA)
            if improved:
                best_ade = ade
                if last_fde is not None:
                    best_fde = last_fde
                patience = 0
            else:
                patience += 1

            if (now - start_time) >= MIN_RUNTIME_SEC and patience >= EARLY_STOP_PATIENCE:
                early_stopped = True
                msg = "\n[EarlyStop] no improvement for 4 rounds(ADE), terminate.\n"
                sys.stdout.write(msg)
                f.write(msg)
                f.flush()
                p.terminate()
                break

        rc = p.wait()

    if best_fde == float("inf"):
        best_fde = -1.0
    return best_ade, best_fde, early_stopped, rc


def generate_innovation_ablation() -> List[ExpConfig]:
    """
    固定13组配置（与你原脚本一致）
    """
    exps: List[ExpConfig] = []

    exps.append(ExpConfig(tag="FULL"))

    exps.append(ExpConfig(use_signature=0, use_sig_tokens=0, use_sig_attn=0, tag="I1_no_signature"))
    exps.append(ExpConfig(use_sig_tokens=0, use_sig_attn=0, tag="I1_no_tokens_no_attn"))
    exps.append(ExpConfig(use_sig_attn=0, tag="I1_no_attn_only"))

    exps.append(ExpConfig(use_struct_ctrl=0, tag="I2_no_struct_control"))
    exps.append(ExpConfig(use_expert_norm=0, tag="I2_no_expert_norm"))

    exps.append(ExpConfig(use_dynamic_intent=0, tag="I3_no_dynamic_intent"))
    exps.append(ExpConfig(use_diversity_gate=0, tag="I3_no_diversity_gate"))

    exps.append(ExpConfig(use_softmin=0, tag="I4_no_softmin"))

    exps.append(ExpConfig(use_score_align=0, tag="I5_no_score_align"))
    exps.append(ExpConfig(cache_scores=0, tag="I5_no_score_cache"))
    exps.append(ExpConfig(use_score_head=0, tag="I5_no_score_head"))

    exps.append(ExpConfig(K=1, tag="K1_sanity"))

    # 保险：确保就是13个
    assert len(exps) == 13, f"Expected 13 configs, got {len(exps)}"
    return exps


def write_summary_header_txt(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("创新点验证消融实验汇总（best ADE / best FDE）\n")
        f.write(f"SPLITS={','.join(SPLITS)} | EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE} | DELTA={EARLY_STOP_DELTA}\n")
        f.write("=" * 110 + "\n\n")


def append_summary_txt(path: str, exp_name: str, split: str, cfg: ExpConfig,
                       best_ade: float, best_fde: float,
                       early_stopped: bool, log_path: str, rc: int):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{exp_name}\n")
        f.write(f"  split: {split}\n")
        f.write(f"  tag: {cfg.tag}\n")
        f.write(f"  best_ADE: {best_ade:.6f}\n")
        f.write(f"  best_FDE: {best_fde:.6f}\n")
        f.write(f"  early_stopped: {early_stopped}\n")
        f.write(f"  return_code: {rc}\n")
        f.write(f"  log: {log_path}\n")
        f.write("-" * 110 + "\n")


def append_summary_csv(csv_path: str, row: Dict[str, Any]):
    file_exists = os.path.exists(csv_path)
    need_header = (not file_exists) or (os.path.getsize(csv_path) == 0)
    fieldnames = list(row.keys())
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            w.writeheader()
        w.writerow(row)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, "logs_ablation_innovations_ran_3splits")
    os.makedirs(log_dir, exist_ok=True)

    summary_txt = os.path.join(log_dir, "summary.txt")
    summary_csv = os.path.join(log_dir, "summary.csv")

    write_summary_header_txt(summary_txt)
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        f.write("")

    exps = generate_innovation_ablation()

    for split in SPLITS:
        split = split.strip()
        if not split:
            continue

        for i, cfg in enumerate(exps):
            exp_name = cfg.name(i, split)
            log_path = os.path.join(log_dir, exp_name + ".log")

            print("\n" + "=" * 120)
            print(f"[RUN] {exp_name}")
            print(f"      split={split} tag={cfg.tag}")
            print(f"      log={log_path}")
            print("=" * 120 + "\n")

            cmd = build_cmd(exp_name, split)
            env = build_env(cfg)

            best_ade, best_fde, early_stopped, rc = run_one(
                cmd=cmd, env=env, cwd=project_root, log_path=log_path
            )

            append_summary_txt(summary_txt, exp_name, split, cfg, best_ade, best_fde, early_stopped, log_path, rc)

            row = {
                "exp_name": exp_name,
                "split": split,
                "tag": cfg.tag,
                "best_ADE": f"{best_ade:.6f}",
                "best_FDE": f"{best_fde:.6f}",
                "early_stopped": str(early_stopped),
                "return_code": str(rc),
                "log_path": log_path,

                "use_signature": cfg.use_signature,
                "use_sig_tokens": cfg.use_sig_tokens,
                "use_sig_attn": cfg.use_sig_attn,
                "use_struct_ctrl": cfg.use_struct_ctrl,
                "use_dynamic_intent": cfg.use_dynamic_intent,
                "use_expert_norm": cfg.use_expert_norm,
                "use_diversity_gate": cfg.use_diversity_gate,
                "use_score_head": cfg.use_score_head,
                "cache_scores": cfg.cache_scores,

                "use_softmin": cfg.use_softmin,
                "use_score_align": cfg.use_score_align,
                "tau": cfg.tau,
                "score_align_w": cfg.score_align_w,
                "score_gt_tau": cfg.score_gt_tau,
                "score_pred_tau": cfg.score_pred_tau,

                "K_override": (cfg.K if cfg.K is not None else ""),
                "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
                "EARLY_STOP_DELTA": EARLY_STOP_DELTA,
            }
            append_summary_csv(summary_csv, row)

    print("\n[DONE] 消融实验完成（13配置 x 3数据集）")
    print(f"summary txt: {summary_txt}")
    print(f"summary csv: {summary_csv}")
    print(f"logs dir   : {log_dir}")


if __name__ == "__main__":
    main()
