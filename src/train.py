"""
* 一个样本 = 一个窗口 (T=15 帧), 内含 P 个 target ped;
* 模型一次 forward 输出 [P] 维 logits, loss 在 P 维上加权平均;
* batch_size=1 + 梯度累积 (图大小不一致);
* intention/action 各自 BCEWithLogits + pos_weight + per-ped sample_weight.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from dataset import PIEDatasetB, collate_fn, get_static_ctx_dim
from graph_builder import PED_DIM_BASE, PED_DIM_WITH_LOOK, VEH_DIM, CW_DIM, EGO_DIM
from models.gru import GAT2_GRU_Model
from models.temporal_gat import GAT2_TempGAT_Model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weights(label_stats: dict, split: str = "train") -> Dict[str, float]:
    s = label_stats[split]
    return {
        "intention": max(s["int_neg"], 1) / max(s["int_pos"], 1),
        "action":    max(s["act_neg"], 1) / max(s["act_pos"], 1),
    }


def _masked_weighted_bce_sum(logits: torch.Tensor, targets: torch.Tensor,
                             valid: torch.Tensor, weights: torch.Tensor,
                             pos_weight: float) -> Tuple[torch.Tensor, float]:
    """返回 (sum(w * BCE), sum(w)).

    上层 step 累加 sum_loss / total_weight 进行梯度缩放 (等价 per-ped 权重平均).
    这样 window 内 P 个 ped 会被独立贡献梯度, 避免被 window-mean 稀释.
    """
    if not valid.any():
        return logits.new_zeros(()), 0.0
    pw = torch.tensor(pos_weight, device=logits.device)
    raw = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw,
                                             reduction="none")  # [P]
    w = weights * valid.float()
    return (raw * w).sum(), float(w.sum().item())


def run_epoch(model, loader, device, cfg, pw, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    lw = cfg["training"]["loss_weights"]
    accum = cfg["training"]["accumulation_steps"]
    ls_int = float(cfg["training"].get("label_smoothing_int", 0.0))

    total_loss, n_batch = 0.0, 0
    ys = {"int": [], "act": []}
    ps = {"int": [], "act": []}

    # accum-window 内的总权重, 用于 "per-ped sum" 进行梯度归一
    accum_int_w = 0.0
    accum_act_w = 0.0
    accum_int_loss = None
    accum_act_loss = None

    if is_train:
        optimizer.zero_grad()

    for step, sample in enumerate(loader):
        if sample is None:
            continue

        idx_TP = sample["ped_index_per_frame"]
        P = idx_TP.size(1)
        if P == 0:
            continue

        with torch.set_grad_enabled(is_train):
            int_log, act_log = model(sample["frame_graphs"], idx_TP,
                                     sample["static_ctx"])  # [P], [P]

            int_t_raw = sample["labels"]["intention"].to(device)        # [P] 0/1
            act_t = sample["labels"]["action"].to(device)               # [P]
            int_m = sample["label_masks"]["intention"].to(device)       # [P]
            act_m = sample["label_masks"]["action"].to(device)          # [P]
            int_sw = sample["int_sample_weight"].to(device)             # [P]
            act_sw = sample["act_sample_weight"].to(device)             # [P]

            # label smoothing (intention only)
            if ls_int > 0:
                int_t = int_t_raw * (1.0 - 2 * ls_int) + ls_int
            else:
                int_t = int_t_raw

            li_sum, li_w = _masked_weighted_bce_sum(
                int_log, int_t, int_m, int_sw, pos_weight=pw["intention"])
            la_sum, la_w = _masked_weighted_bce_sum(
                act_log, act_t, act_m, act_sw, pos_weight=pw["action"])

            # 累加到 accum-window
            if accum_int_loss is None:
                accum_int_loss = li_sum
                accum_act_loss = la_sum
            else:
                accum_int_loss = accum_int_loss + li_sum
                accum_act_loss = accum_act_loss + la_sum
            accum_int_w += li_w
            accum_act_w += la_w

            # 记录 "平均 per-ped loss" 用于日志
            denom = max(li_w + la_w, 1e-6)
            loss_for_log = (li_sum + la_sum).detach().item() / denom

            # F1 收集
            with torch.no_grad():
                int_pred = (torch.sigmoid(int_log) > 0.5).int().cpu().tolist()
                act_pred = (torch.sigmoid(act_log) > 0.5).int().cpu().tolist()
                int_targ = int_t_raw.int().cpu().tolist()
                act_targ = act_t.int().cpu().tolist()
                int_mask = int_m.cpu().tolist()
                act_mask = act_m.cpu().tolist()
                for p in range(P):
                    if int_mask[p]:
                        ys["int"].append(int_targ[p])
                        ps["int"].append(int_pred[p])
                    if act_mask[p]:
                        ys["act"].append(act_targ[p])
                        ps["act"].append(act_pred[p])

        if is_train and (step + 1) % accum == 0:
            # accum-window 内 per-ped 权重平均 作为该 step 的 loss
            int_part = (accum_int_loss / max(accum_int_w, 1e-6)) if accum_int_w > 0 else accum_int_loss * 0
            act_part = (accum_act_loss / max(accum_act_w, 1e-6)) if accum_act_w > 0 else accum_act_loss * 0
            step_loss = lw["intention"] * int_part + lw["action"] * act_part
            step_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accum_int_loss = None
            accum_act_loss = None
            accum_int_w = 0.0
            accum_act_w = 0.0

        total_loss += loss_for_log
        n_batch += 1

        if is_train and (step + 1) % 500 == 0:
            print(f"    step {step+1}/{len(loader)} | running_loss={total_loss / n_batch:.4f}")

    # 收尾: 剩余未反向的 accum
    if is_train and accum_int_loss is not None and (accum_int_w + accum_act_w) > 0:
        int_part = (accum_int_loss / max(accum_int_w, 1e-6)) if accum_int_w > 0 else accum_int_loss * 0
        act_part = (accum_act_loss / max(accum_act_w, 1e-6)) if accum_act_w > 0 else accum_act_loss * 0
        step_loss = lw["intention"] * int_part + lw["action"] * act_part
        step_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    detail = {}
    for k in ("int", "act"):
        if ys[k]:
            detail[f"{k}_f1"] = f1_score(ys[k], ps[k], zero_division=0)
            detail[f"{k}_prec"] = precision_score(ys[k], ps[k], zero_division=0)
            detail[f"{k}_recall"] = recall_score(ys[k], ps[k], zero_division=0)
            detail[f"{k}_acc"] = accuracy_score(ys[k], ps[k])
        else:
            for suf in ("f1", "prec", "recall", "acc"):
                detail[f"{k}_{suf}"] = 0.0

    return total_loss / max(n_batch, 1), detail


def main():
    parser = argparse.ArgumentParser(description="Train one final-report ablation configuration.")
    parser.add_argument("--config", required=True, help="Path to a YAML file in configs/.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dc, mc, tc = cfg["data"], cfg["model"], cfg["training"]

    device = pick_device()
    print(f"[train] device = {device}")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = (repo_root / tc.get("data_dir", dc["output_dir"])).resolve()
    ckpt_dir = (repo_root / tc["checkpoint_dir"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("[train] loading preprocessed data ...")
    video_dbs = torch.load(data_dir / "video_databases.pt", weights_only=False)
    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    ds_kwargs = dict(
        video_dbs=video_dbs,
        t_obs=dc["t_obs"], knn_k=cfg["graph"]["knn_k"],
        veh_dist=cfg["graph"]["veh_dist"],
        img_w=dc["img_w"], img_h=dc["img_h"],
        include_look=bool(cfg["experiment"]["include_look"]),
    )
    val_windows = split_info["windows"]["val"]
    val_sub = float(tc.get("val_subsample", 1.0))
    if 0.0 < val_sub < 1.0:
        import random as _rnd
        _rng = _rnd.Random(42)
        k = max(1, int(len(val_windows) * val_sub))
        val_windows = _rng.sample(val_windows, k)
        print(f"[train] val_subsample={val_sub} -> {k} windows")

    train_ds = PIEDatasetB(windows=split_info["windows"]["train"], **ds_kwargs)
    val_ds   = PIEDatasetB(windows=val_windows, **ds_kwargs)

    print(f"[train] train windows = {len(train_ds):,}")
    print(f"[train] val   windows = {len(val_ds):,}")

    nw = tc.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=nw,
                              persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=nw,
                            persistent_workers=(nw > 0))

    ped_dim = PED_DIM_WITH_LOOK if cfg["experiment"]["include_look"] else PED_DIM_BASE
    model_name = cfg["experiment"]["temporal_encoder"]
    ModelCls = GAT2_GRU_Model if model_name == "gru" else GAT2_TempGAT_Model
    print(f"[train] encoder={model_name} include_look={cfg['experiment']['include_look']}")
    print(f"[train] dims: ped={ped_dim} veh={VEH_DIM} cw={CW_DIM} ego={EGO_DIM} static={get_static_ctx_dim()}")

    model = ModelCls(
        ped_dim=ped_dim, veh_dim=VEH_DIM, cw_dim=CW_DIM, ego_dim=EGO_DIM,
        static_dim=get_static_ctx_dim(),
        hidden_dim=mc["hidden_dim"],
        n_spatial_layers=mc["n_spatial_layers"],
        n_spatial_heads=mc["n_spatial_heads"],
        n_temporal_layers=mc["n_temporal_layers"],
        n_temporal_heads=mc["n_temporal_heads"],
        dropout=mc["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params = {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(),
                              lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=tc["epochs"])
    pw = compute_pos_weights(split_info["label_stats"], "train")
    print(f"[train] pos_weight = {pw}")

    best = dict(int_f1=0.0, act_f1=0.0, avg_f1=0.0)
    es_metric = tc.get("early_stop_metric", "avg_f1")
    print(f"[train] early_stop_metric = {es_metric}")
    patience_cnt = 0
    for ep in range(1, tc["epochs"] + 1):
        print(f"\n[epoch {ep}/{tc['epochs']}]")
        tr_loss, tr_det = run_epoch(model, train_loader, device, cfg, pw,
                                    optimizer=optim)
        va_loss, va_det = run_epoch(model, val_loader, device, cfg, pw,
                                    optimizer=None)
        sched.step()

        print(f"  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        for k in ("int", "act"):
            print(f"  val {k:<4} | f1={va_det[k+'_f1']:.4f}  "
                  f"prec={va_det[k+'_prec']:.4f}  recall={va_det[k+'_recall']:.4f}  "
                  f"acc={va_det[k+'_acc']:.4f}")

        avg_f1 = (va_det["int_f1"] + va_det["act_f1"]) / 2.0
        cur = {"int_f1": va_det["int_f1"], "act_f1": va_det["act_f1"], "avg_f1": avg_f1}
        prev_best_es = best[es_metric]

        if va_det["int_f1"] > best["int_f1"]:
            best["int_f1"] = va_det["int_f1"]
            torch.save(model.state_dict(), ckpt_dir / "best_int.pt")
            print(f"  -> saved best_int.pt (f1={best['int_f1']:.4f})")
        if va_det["act_f1"] > best["act_f1"]:
            best["act_f1"] = va_det["act_f1"]
            torch.save(model.state_dict(), ckpt_dir / "best_action.pt")
            print(f"  -> saved best_action.pt (f1={best['act_f1']:.4f})")
        if avg_f1 > best["avg_f1"]:
            best["avg_f1"] = avg_f1
            torch.save(model.state_dict(), ckpt_dir / "best_avg.pt")
            print(f"  -> saved best_avg.pt (avg_f1={avg_f1:.4f})")

        improved = cur[es_metric] > prev_best_es + 1e-6
        if improved:
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= tc["patience"]:
                print(f"[train] early stop at epoch {ep} (metric={es_metric})")
                break

    print(f"\n[train] best = {best}")


if __name__ == "__main__":
    main()
