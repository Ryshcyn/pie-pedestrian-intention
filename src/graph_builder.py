"""
每帧 = 一张图. 设计目标:
  * 图中所有行人都是预测对象 (no target);
  * 所有节点维度尽量精简, 与 analysis/results/report.md 对齐;
  * 边只用于把 "上下文" (其它 ped/车/人行横道) 信息汇聚到行人节点.

节点维度::
    ped_x : 7   = [cx, cy, w, h, vx, vy, is_looking]   # look: looking=1, else 0 (PIE 标注, 仅 9.49% 为 looking)
    veh_x : 5   = [cx, cy, w, h, is_car]            # 与最终报告一致
    cw_x  : 7   = [cx, cy, w, h, has_ped_tl, tl_stop, tl_go]  # effective TL 折进 cw
    ego_x : 1   = [OBD_speed_norm]                  # FiLM 全局调制

边类型 (4 种, edge_attr 末尾 4 维 one-hot)::
    0: ped -> ped  (KNN-k, 双向)
    1: veh -> ped  (半径内, 单向)
    2: cw  -> ped  (cw 节点 -> 全部 ped)
    3: self-loop

注意: edge_attr = [dx, dy, dist, *type_oh(4)] 共 7 维.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data


IMG_W_DEFAULT = 1920.0
IMG_H_DEFAULT = 1080.0

PED_DIM_BASE = 6
PED_DIM_WITH_LOOK = 7
VEH_DIM = 5
CW_DIM = 7
EGO_DIM = 1
EDGE_DIM = 7  # [dx, dy, dist] + type_oh(4)

TL_STOP = {"red", "yellow"}
TL_GO = {"green"}

EDGE_TYPE_PP = 0   # ped -> ped
EDGE_TYPE_VP = 1   # veh -> ped
EDGE_TYPE_CP = 2   # cw  -> ped
EDGE_TYPE_SELF = 3

PED_TL_PREFER_DIST = 0.30  # crosswalk 选 effective TL 时的最大优先距离 (归一化坐标)
OBD_SPEED_NORM = 80.0      # OBD_speed 归一化常数 (km/h)

MAX_VEHS = 6               # 全图最多保留几辆车 (按到 ped 群质心的距离取近)
MAX_CWS = 2                # 全图最多保留几个 crosswalk


def _bbox_geom(b: dict, img_w: float, img_h: float):
    cx = (b["xtl"] + b["xbr"]) * 0.5 / img_w
    cy = (b["ytl"] + b["ybr"]) * 0.5 / img_h
    w  = (b["xbr"] - b["xtl"]) / img_w
    h  = (b["ybr"] - b["ytl"]) / img_h
    return cx, cy, w, h


def _effective_tl_for_cw(cw_xy: Tuple[float, float], tls: List[dict],
                         img_w: float, img_h: float) -> Tuple[float, float, float]:
    """为单个 crosswalk 选 effective TL 状态.

    优先选 PED_TL_PREFER_DIST 半径内的 pedestrian 信号灯, 否则用最近 regular 信号灯.
    Returns: (has_ped_tl, tl_stop, tl_go)
    """
    cx, cy = cw_xy
    best_ped, best_ped_d = None, float("inf")
    best_reg, best_reg_d = None, float("inf")
    for tl in tls:
        ttype = tl.get("type", "regular")
        tx = (tl["xtl"] + tl["xbr"]) * 0.5 / img_w
        ty = (tl["ytl"] + tl["ybr"]) * 0.5 / img_h
        d2 = (tx - cx) ** 2 + (ty - cy) ** 2
        if ttype == "pedestrian":
            if d2 < best_ped_d:
                best_ped, best_ped_d = tl, d2
        elif ttype == "regular":
            if d2 < best_reg_d:
                best_reg, best_reg_d = tl, d2

    chosen, has_ped = None, 0.0
    if best_ped is not None and best_ped_d < PED_TL_PREFER_DIST ** 2:
        chosen, has_ped = best_ped, 1.0
    elif best_reg is not None:
        chosen = best_reg
    if chosen is None:
        return 0.0, 0.0, 0.0
    state = chosen.get("state", "")
    return has_ped, (1.0 if state in TL_STOP else 0.0), (1.0 if state in TL_GO else 0.0)


def _edge_type_oh(t: int) -> List[float]:
    oh = [0.0, 0.0, 0.0, 0.0]
    oh[t] = 1.0
    return oh


def build_frame_graph(
    frame_data: dict,
    ego_frame: dict,
    prev_ped_bboxes: Dict[str, Tuple[float, float]],
    vocab: Dict[str, Dict[str, int]] | None = None,  # 兼容签名, 不再使用
    knn_k: int = 4,
    veh_dist: float = 0.20,
    img_w: float = IMG_W_DEFAULT,
    img_h: float = IMG_H_DEFAULT,
    include_look: bool = True,
) -> Data:
    """构建一帧多行人图.

    所有 pedestrian 都是节点; 不再有 "target" 概念.
    每个 ped 节点是预测对象, 上层 dataset/model 按 id 跨帧对齐.
    """
    peds = frame_data.get("pedestrians", [])
    vehs = frame_data.get("vehicles", [])
    cws  = frame_data.get("crosswalks", [])
    tls  = frame_data.get("traffic_lights", [])

    cur_ped_bboxes: Dict[str, Tuple[float, float]] = {}

    # ---- 1) ped 节点 ---- #
    ped_feats: List[List[float]] = []
    ped_xy: List[Tuple[float, float]] = []
    ped_ids: List[str] = []
    ped_actions: List[float] = []
    ped_occlusions: List[str] = []
    for p in peds:
        cx, cy, w, h = _bbox_geom(p, img_w, img_h)
        cur_ped_bboxes[p["id"]] = (cx, cy)
        prev = prev_ped_bboxes.get(p["id"])
        if prev is not None:
            vx, vy = cx - prev[0], cy - prev[1]
        else:
            vx, vy = 0.0, 0.0
        feat = [cx, cy, w, h, vx, vy]
        if include_look:
            feat.append(1.0 if p.get("look") == "looking" else 0.0)
        ped_feats.append(feat)
        ped_xy.append((cx, cy))
        ped_ids.append(p["id"])
        ped_actions.append(1.0 if p.get("action") == "walking" else 0.0)
        ped_occlusions.append(p.get("occlusion", "none"))
    n_ped = len(ped_feats)

    # 没有行人 -> 空图占位
    if n_ped == 0:
        data = Data()
        data.edge_index = torch.zeros(2, 0, dtype=torch.long)
        data.edge_attr  = torch.zeros(0, EDGE_DIM, dtype=torch.float)
        data.num_nodes  = 1
        ped_dim = PED_DIM_WITH_LOOK if include_look else PED_DIM_BASE
        data.ped_x = torch.zeros(0, ped_dim)
        data.veh_x = torch.zeros(0, VEH_DIM)
        data.cw_x  = torch.zeros(0, CW_DIM)
        data.ego_x = torch.zeros(EGO_DIM)
        data.ped_ids = []
        data.ped_actions = torch.zeros(0)
        data.ped_occlusions = []
        data._ped_bboxes = cur_ped_bboxes
        data._edge_stats = {}
        return data

    # 计算 ped 群质心 (用于挑车辆)
    pc_x = sum(c for c, _ in ped_xy) / n_ped
    pc_y = sum(c for _, c in ped_xy) / n_ped

    # ---- 2) 车辆节点 (按到 ped 群质心距离取最近 MAX_VEHS) ---- #
    veh_feats: List[List[float]] = []
    veh_xy: List[Tuple[float, float]] = []
    if vehs:
        veh_cands = []
        for v in vehs:
            cx, cy, w, h = _bbox_geom(v, img_w, img_h)
            d2 = (cx - pc_x) ** 2 + (cy - pc_y) ** 2
            is_car = 1.0 if v.get("type", "car") == "car" else 0.0
            veh_cands.append((d2, [cx, cy, w, h, is_car], (cx, cy)))
        veh_cands.sort(key=lambda x: x[0])
        for d2, feat, xy in veh_cands[:MAX_VEHS]:
            veh_feats.append(feat)
            veh_xy.append(xy)
    n_veh = len(veh_feats)

    # ---- 3) crosswalk 节点 (按到 ped 群质心距离取最近 MAX_CWS) + effective TL ---- #
    cw_feats: List[List[float]] = []
    cw_xy: List[Tuple[float, float]] = []
    if cws:
        cw_cands = []
        for c in cws:
            cx, cy, w, h = _bbox_geom(c, img_w, img_h)
            d2 = (cx - pc_x) ** 2 + (cy - pc_y) ** 2
            cw_cands.append((d2, [cx, cy, w, h], (cx, cy)))
        cw_cands.sort(key=lambda x: x[0])
        for d2, geom, xy in cw_cands[:MAX_CWS]:
            has_ped_tl, tl_stop, tl_go = _effective_tl_for_cw(xy, tls, img_w, img_h)
            cw_feats.append(geom + [has_ped_tl, tl_stop, tl_go])
            cw_xy.append(xy)
    n_cw = len(cw_feats)

    N = n_ped + n_veh + n_cw

    # ---- 4) ego (1 维: OBD_speed 归一化 + clip) ---- #
    ef = ego_frame or {}
    obd = float(ef.get("OBD_speed", 0.0)) / OBD_SPEED_NORM
    obd = max(0.0, min(1.0, obd))
    ego_x = torch.tensor([obd], dtype=torch.float)

    # ---- 5) 建边 ---- #
    srcs: List[int] = []
    dsts: List[int] = []
    feats: List[List[float]] = []
    stats: Counter = Counter()

    veh_off = n_ped
    cw_off = n_ped + n_veh

    # 5a) ped -> ped (KNN-k 双向); n_ped<=1 时跳过
    if n_ped > 1:
        ped_xy_t = torch.tensor(ped_xy, dtype=torch.float)             # [P, 2]
        diff = ped_xy_t.unsqueeze(0) - ped_xy_t.unsqueeze(1)             # [P, P, 2] 形成 i->j 时 j-i
        d2 = (diff ** 2).sum(-1)                                          # [P, P]
        d2.fill_diagonal_(float("inf"))
        k = min(knn_k, n_ped - 1)
        topk = torch.topk(d2, k, dim=-1, largest=False).indices           # [P, k]
        for i in range(n_ped):
            for j in topk[i].tolist():
                dx = ped_xy[j][0] - ped_xy[i][0]
                dy = ped_xy[j][1] - ped_xy[i][1]
                dist = (dx * dx + dy * dy) ** 0.5
                srcs.append(j); dsts.append(i)  # j -> i (邻居影响 i)
                feats.append([dx, dy, dist] + _edge_type_oh(EDGE_TYPE_PP))
                stats["ped->ped"] += 1

    # 5b) veh -> ped  (半径 veh_dist 内)
    veh_dist2 = veh_dist ** 2
    for j in range(n_veh):
        vx, vy = veh_xy[j]
        for i in range(n_ped):
            px, py = ped_xy[i]
            dx, dy = vx - px, vy - py
            if dx * dx + dy * dy > veh_dist2:
                continue
            dist = (dx * dx + dy * dy) ** 0.5
            srcs.append(veh_off + j); dsts.append(i)
            feats.append([dx, dy, dist] + _edge_type_oh(EDGE_TYPE_VP))
            stats["veh->ped"] += 1

    # 5c) cw -> ped (全连接)
    for j in range(n_cw):
        cwx, cwy = cw_xy[j]
        for i in range(n_ped):
            px, py = ped_xy[i]
            dx, dy = cwx - px, cwy - py
            dist = (dx * dx + dy * dy) ** 0.5
            srcs.append(cw_off + j); dsts.append(i)
            feats.append([dx, dy, dist] + _edge_type_oh(EDGE_TYPE_CP))
            stats["cw->ped"] += 1

    # 5d) self-loop 全节点
    for i in range(N):
        srcs.append(i); dsts.append(i)
        feats.append([0.0, 0.0, 0.0] + _edge_type_oh(EDGE_TYPE_SELF))
    stats["self"] += N

    if srcs:
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_attr  = torch.tensor(feats, dtype=torch.float)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr  = torch.zeros(0, EDGE_DIM, dtype=torch.float)

    data = Data()
    data.edge_index = edge_index
    data.edge_attr  = edge_attr
    data.num_nodes  = N if N > 0 else 1

    data.ped_x = torch.tensor(ped_feats, dtype=torch.float)
    data.veh_x = torch.tensor(veh_feats, dtype=torch.float) if veh_feats else torch.zeros(0, VEH_DIM)
    data.cw_x  = torch.tensor(cw_feats,  dtype=torch.float) if cw_feats  else torch.zeros(0, CW_DIM)
    data.ego_x = ego_x

    data.ped_ids = ped_ids
    data.ped_actions = torch.tensor(ped_actions, dtype=torch.float)
    data.ped_occlusions = ped_occlusions

    data._ped_bboxes = cur_ped_bboxes
    data._edge_stats = dict(stats)
    return data
