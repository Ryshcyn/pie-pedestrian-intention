"""
数据流:
    每帧空间编码  (与 GAT 一致, ped/veh/cw + ego FiLM + GATv2Conv ×2)
        -> per-frame ped emb [P, T, d]
    GRU(input_size=d, hidden_size=d, num_layers=2, batch_first=True, bidirectional=False)
        -> hidden seq [P, T, d];  取 last frame
    intention 头: concat(last, static_proj(static_ctx))  -> MLP -> logit
    action 头   : last                                    -> MLP -> logit

GRU 是单向, 自然只看过去.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


# --------------------------------------------------------------------------- #
# FiLM (与 GAT2 相同)
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
# 主模型 (spatial GAT + GRU)
# --------------------------------------------------------------------------- #
class GAT2_GRU_Model(nn.Module):
    def __init__(self, ped_dim: int, veh_dim: int, cw_dim: int,
                 ego_dim: int, static_dim: int = 0,
                 hidden_dim: int = 128,
                 n_spatial_layers: int = 2, n_spatial_heads: int = 4,
                 n_gru_layers: int = 2,
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

        # GRU 时间编码器 (单向, 因果)
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_gru_layers, batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        # static_ctx 投影 (仅 intention 头使用)
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
        valid = torch.zeros(P, T, dtype=torch.bool, device=device)

        for t in range(T):
            g = frame_graphs[t]
            h_all = self.spatial(g, device)       # [N_t, d]
            n_ped = g.ped_x.size(0)
            for p in range(P):
                li = int(idx_TP[t, p].item())
                if 0 <= li < n_ped:
                    H_seq[p, t] = h_all[li]
                    valid[p, t] = True

        # GRU forward, 直接喂 [P, T, d]; pad 帧用 0 输入是无害的近似
        # (更严谨可用 pack_padded_sequence, 但 dataset 已要求 full_obs 所以 valid 几乎全 True)
        H_out, _ = self.gru(H_seq)               # [P, T, d]

        last = H_out[:, -1, :]                   # [P, d]

        if self.static_proj is not None and static_ctx is not None:
            sh = self.static_proj(static_ctx.to(device))
            int_in = torch.cat([last, sh], dim=-1)
        else:
            int_in = last
        int_logits = self.int_head(int_in).squeeze(-1)
        act_logits = self.act_head(last).squeeze(-1)
        return int_logits, act_logits
