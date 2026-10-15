"""
一个样本 = 一个滑窗 (T=15 帧).

返回字段::
    {
        "frame_graphs"   : List[Data]   长度 T (graph_builder 多 ped 图)
        "target_pids"    : List[str]    P 个待预测行人 (整 T 帧都出现)
        "ped_index_per_frame": LongTensor [T, P]
                            target_pids[p] 在第 t 帧 graph.ped_ids 中的下标
                            (没出现则 -1; 因 full_obs 通常都 >=0)
        "labels"         : {
            "intention"  : FloatTensor [P]   软标签 ∈ [0,1]
            "action"     : FloatTensor [P]   0/1
        }
        "label_masks"    : {
            "intention"  : BoolTensor  [P]
            "action"     : BoolTensor  [P]   全 True
        }
        "sample_weight"  : FloatTensor [P]   per-ped occlusion-aware 权重
    }
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from graph_builder import build_frame_graph


# --------------------------------------------------------------------------- 
# scene-level static context (3 维): 该 ped 全程稳定的场景先验
#   * is_intersection: 1 if intersection != midblock else 0
#   * has_signal:      1 if signalized in {S, CS} else 0
#   * num_lanes:       num_lanes / 4, clip to [0, 1.5]
# --------------------------------------------------------------------------- #
_STATIC_DIM = 3


def encode_static_ctx(ped_attr: dict) -> torch.Tensor:
    inter = ped_attr.get("intersection", "")
    sig = ped_attr.get("signalized", "n/a") or "n/a"
    is_inter = 0.0 if inter == "midblock" else (1.0 if inter else 0.0)
    has_signal = 1.0 if sig in ("S", "CS") else 0.0
    try:
        lanes = float(ped_attr.get("num_lanes", 0)) / 4.0
    except (TypeError, ValueError):
        lanes = 0.0
    lanes = max(0.0, min(1.5, lanes))
    return torch.tensor([is_inter, has_signal, lanes], dtype=torch.float)


def get_static_ctx_dim() -> int:
    return _STATIC_DIM


class PIEDatasetB(Dataset):
    def __init__(self, video_dbs: Dict[str, dict], windows: List[dict],
                 vocab: Dict[str, Dict[str, int]] | None = None,  # 兼容签名, 不再使用
                 t_obs: int = 15, knn_k: int = 4, veh_dist: float = 0.20,
                 img_w: float = 1920.0, img_h: float = 1080.0,
                 require_full_obs: bool = True, include_look: bool = True):
        self.video_dbs = video_dbs
        self.t_obs = t_obs
        self.knn_k = knn_k
        self.veh_dist = veh_dist
        self.img_w = img_w
        self.img_h = img_h
        self.include_look = include_look

        # 预枚举窗口 + 该窗口的 target_pids (整 T 帧都出现的 ped)
        self.items: List[Tuple[dict, List[str]]] = []
        for w in windows:
            key = f"{w['set_id']}/{w['vid_id']}"
            db = self.video_dbs[key]
            fo = db["frame_objects"]
            f0, f1 = w["frame_start"], w["frame_end"]
            common = None
            ok = True
            for fid in range(f0, f1 + 1):
                fd = fo.get(fid)
                if fd is None:
                    ok = False
                    break
                pids = {p["id"] for p in fd["pedestrians"]}
                common = pids if common is None else (common & pids)
                if not common:
                    ok = False
                    break
            if not ok or not common:
                continue
            self.items.append((w, sorted(common)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        w, target_pids = self.items[idx]
        key = f"{w['set_id']}/{w['vid_id']}"
        db = self.video_dbs[key]
        fo = db["frame_objects"]
        ego = db["ego_data"]
        pa = db["ped_attributes"]

        f0, f1 = w["frame_start"], w["frame_end"]
        empty = {"pedestrians": [], "vehicles": [], "traffic_lights": [],
                 "signs": [], "crosswalks": [], "transit_stations": []}

        T = self.t_obs
        P = len(target_pids)

        frame_graphs = []
        prev_bboxes: Dict[str, Tuple[float, float]] = {}
        idx_TP = torch.full((T, P), -1, dtype=torch.long)

        for t, fid in enumerate(range(f0, f1 + 1)):
            g = build_frame_graph(
                fo.get(fid, empty), ego.get(fid, {}),
                prev_bboxes,
                knn_k=self.knn_k, veh_dist=self.veh_dist,
                img_w=self.img_w, img_h=self.img_h,
                include_look=self.include_look,
            )
            frame_graphs.append(g)
            prev_bboxes = g._ped_bboxes
            id2li = {pid: i for i, pid in enumerate(g.ped_ids)}
            for p_idx, pid in enumerate(target_pids):
                idx_TP[t, p_idx] = id2li.get(pid, -1)

        # 构造 per-ped 标签
        last_g = frame_graphs[-1]
        last_id2li = {pid: i for i, pid in enumerate(last_g.ped_ids)}

        action_lbls = torch.zeros(P, dtype=torch.float)
        int_lbls = torch.zeros(P, dtype=torch.float)
        int_masks = torch.zeros(P, dtype=torch.bool)
        # sample_weight 拆分: intention 是人工全局标注 与帧遇遮无关 -> 恒 1.0
        # action 是帧级动作 遇遮严重时 该帧标签不准 -> 降权
        act_sample_weights = torch.ones(P, dtype=torch.float)
        static_ctx = torch.zeros(P, _STATIC_DIM, dtype=torch.float)

        for p_idx, pid in enumerate(target_pids):
            li = last_id2li.get(pid, -1)
            if li >= 0:
                action_lbls[p_idx] = float(last_g.ped_actions[li].item())
                occ = last_g.ped_occlusions[li]
                if occ == "full":
                    act_sample_weights[p_idx] = 0.3
                elif occ == "part":
                    act_sample_weights[p_idx] = 0.7
                else:
                    act_sample_weights[p_idx] = 1.0

            ped_attr = pa.get(pid, {})
            ip = ped_attr.get("intention_prob", -1.0)
            if ip is not None and ip >= 0:
                # 二值化: prob > 0.5 为正例 (label_smoothing 由 train.py 负责)
                int_lbls[p_idx] = 1.0 if ip > 0.5 else 0.0
                int_masks[p_idx] = True

            static_ctx[p_idx] = encode_static_ctx(ped_attr)

        return {
            "frame_graphs": frame_graphs,
            "target_pids": target_pids,
            "ped_index_per_frame": idx_TP,                          # [T, P]
            "static_ctx": static_ctx,                               # [P, _STATIC_DIM]
            "labels": {
                "intention": int_lbls,                              # [P] 0/1
                "action": action_lbls,                              # [P]
            },
            "label_masks": {
                "intention": int_masks,                             # [P]
                "action": torch.ones(P, dtype=torch.bool),
            },
            "int_sample_weight": torch.ones(P, dtype=torch.float),  # [P] 恒 1
            "act_sample_weight": act_sample_weights,                # [P] occ-aware
        }


def collate_fn(batch):
    """batch_size=1 友好 collate, 自动过滤 None."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return batch[0]
