"""
把每个 target ped 的 T 帧 embedding 视为一个 "时间链图":
    * 节点: T 个时间步, 每节点是该帧空间编码后的 ped emb (d 维);
    * 边  : 因果聚合 — 对每个 t 创建 (t' -> t) for all t' <= t
            (含 self-loop t -> t). 这样 1-2 层 GATv2 即可让位置 t 看到完整历史.
    * edge_attr 3 维: [Δt / T, is_self, is_past]
P 个 ped 一次前向: 把它们的链图拼成一个大图 (PyG 风格), 节点偏移 p*T.

数据流::
    每帧空间编码 (与 GAT2 一致) -> H_seq [P, T, d]
    展开为 [P*T, d] + 因果链 edge_index (P 个独立连通分量)
    GATv2Conv ×2 (edge_dim=3) -> [P*T, d]
    取每个 ped 的 last frame (索引 p*T + (T-1)) -> [P, d]
    intention/action heads (与 GAT2 一致)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


# --------------------------------------------------------------------------- #
# FiLM
# --------------------------------------------------------------------------- #
class EgoFiLM(nn.Module):
    def __init__(self, ego_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ego_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, h: torch.Tensor, ego_x: torch.Tensor) -> torch.Tensor:
        gb = self.net(ego_x)
        gamma, beta = gb.chunk(2, dim=-1)
        return h * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0)


# --------------------------------------------------------------------------- #
# 空间编码器 (与 GAT2 相同)
# --------------------------------------------------------------------------- #
class SpatialEncoder(nn.Module):
    def __init__(self, ped_dim: int, veh_dim: int, cw_dim: int, ego_dim: int,
                 edge_dim: int = 7, hidden_dim: int = 128,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ped_proj = nn.Linear(ped_dim, hidden_dim)
        self.veh_proj = nn.Linear(veh_dim, hidden_dim)
        self.cw_proj  = nn.Linear(cw_dim,  hidden_dim)
        self.film = EgoFiLM(ego_dim, hidden_dim)

        per_head = hidden_dim // n_heads
        self.gats = nn.ModuleList([
            GATv2Conv(hidden_dim, per_head, heads=n_heads, concat=True,
                      edge_dim=edge_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, g: Data, device: torch.device) -> torch.Tensor:
        parts = []
        n_ped = g.ped_x.size(0)
        n_veh = g.veh_x.size(0)
        n_cw  = g.cw_x.size(0)
        if n_ped > 0:
            parts.append(self.ped_proj(g.ped_x.to(device)))
        if n_veh > 0:
            parts.append(self.veh_proj(g.veh_x.to(device)))
        if n_cw > 0:
            parts.append(self.cw_proj(g.cw_x.to(device)))
        if not parts:
            return torch.zeros(1, self.hidden_dim, device=device)
        h = torch.cat(parts, dim=0)
        h = self.film(h, g.ego_x.to(device))

        edge_index = g.edge_index.to(device)
        edge_attr = g.edge_attr.to(device) if getattr(g, "edge_attr", None) is not None else None
        for gat, ln in zip(self.gats, self.norms):
            h_new = gat(h, edge_index, edge_attr=edge_attr)
            h = ln(h + F.elu(h_new))
        return h


# --------------------------------------------------------------------------- #
# Temporal GAT 编码器
# --------------------------------------------------------------------------- #
def build_causal_chain(T: int, P: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 P 个 ped 的因果时间链合并图.

    Returns:
        edge_index : LongTensor [2, E]
        edge_attr  : FloatTensor [E, 3]   [Δt/T, is_self, is_past]
    """
    srcs: list[int] = []
    dsts: list[int] = []
    feats: list[list[float]] = []
    for p in range(P):
        off = p * T
        for t in range(T):
            for tp in range(t + 1):  # t' <= t
                src = off + tp
                dst = off + t
                dt = t - tp
                if dt == 0:
                    feats.append([0.0, 1.0, 0.0])
                else:
                    feats.append([dt / max(T, 1), 0.0, 1.0])
                srcs.append(src); dsts.append(dst)
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
    edge_attr = torch.tensor(feats, dtype=torch.float, device=device)
    return edge_index, edge_attr


class TemporalGAT(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        per_head = hidden_dim // n_heads
        self.gats = nn.ModuleList([
            GATv2Conv(hidden_dim, per_head, heads=n_heads, concat=True,
                      edge_dim=3, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, H_seq: torch.Tensor) -> torch.Tensor:
        """H_seq: [P, T, d] -> [P, T, d]"""
        P, T, d = H_seq.shape
        device = H_seq.device
        # flatten 成 [P*T, d]
        x = H_seq.reshape(P * T, d)
        edge_index, edge_attr = build_causal_chain(T, P, device)
        for gat, ln in zip(self.gats, self.norms):
            h_new = gat(x, edge_index, edge_attr=edge_attr)
            x = ln(x + F.elu(h_new))
        return x.reshape(P, T, d)


# --------------------------------------------------------------------------- #
# 主模型 (spatial GAT + temporal GAT)
# --------------------------------------------------------------------------- #
class GAT2_TempGAT_Model(nn.Module):
    def __init__(self, ped_dim: int, veh_dim: int, cw_dim: int,
                 ego_dim: int, static_dim: int = 0,
                 hidden_dim: int = 128,
                 n_spatial_layers: int = 2, n_spatial_heads: int = 4,
                 n_temporal_layers: int = 2, n_temporal_heads: int = 4,
                 dropout: float = 0.1, **_unused):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim

        self.spatial = SpatialEncoder(
            ped_dim=ped_dim, veh_dim=veh_dim, cw_dim=cw_dim,
            ego_dim=ego_dim, edge_dim=7, hidden_dim=hidden_dim,
            n_layers=n_spatial_layers, n_heads=n_spatial_heads,
            dropout=dropout,
        )

        self.temporal = TemporalGAT(
            hidden_dim=hidden_dim, n_layers=n_temporal_layers,
            n_heads=n_temporal_heads, dropout=dropout,
        )

        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            int_in = hidden_dim + hidden_dim // 2
        else:
            self.static_proj = None
            int_in = hidden_dim

        self.int_head = nn.Sequential(
            nn.Linear(int_in, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, frame_graphs: List[Data],
                ped_index_per_frame: torch.Tensor,
                static_ctx: torch.Tensor | None = None):
        device = next(self.parameters()).device
        T = len(frame_graphs)
        idx_TP = ped_index_per_frame.to(device)
        P = idx_TP.size(1)

        if P == 0:
            return (torch.zeros(0, device=device),
                    torch.zeros(0, device=device))

        H_seq = torch.zeros(P, T, self.hidden_dim, device=device)
        for t in range(T):
            g = frame_graphs[t]
            h_all = self.spatial(g, device)
            n_ped = g.ped_x.size(0)
            for p in range(P):
                li = int(idx_TP[t, p].item())
                if 0 <= li < n_ped:
                    H_seq[p, t] = h_all[li]

        H_out = self.temporal(H_seq)             # [P, T, d]
        last = H_out[:, -1, :]                   # [P, d]

        if self.static_proj is not None and static_ctx is not None:
            sh = self.static_proj(static_ctx.to(device))
            int_in = torch.cat([last, sh], dim=-1)
        else:
            int_in = last
        int_logits = self.int_head(int_in).squeeze(-1)
        act_logits = self.act_head(last).squeeze(-1)
        return int_logits, act_logits
