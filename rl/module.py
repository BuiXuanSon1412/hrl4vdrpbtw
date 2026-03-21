"""
networks/base_network.py
------------------------
Abstract network interface that all policy networks must implement.

Design principles
-----------------
- BaseNetwork defines the contract that agents call.  Agents never import
  concrete network classes; they hold a BaseNetwork reference.
- obs_shape and action_space_size are NOT properties of the network.
  They belong to the problem and are used to build the network, not stored on it.
- The three abstract methods (forward, get_action_and_log_prob,
  evaluate_actions) are the ONLY interface agents use.
- Device management is centralised here so concrete classes don't repeat it.

Concrete networks must inherit BOTH nn.Module and BaseNetwork.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn.functional as F

from config import NetworkConfig
from problem import NODE_FEAT_DIM, VEH_FEAT_DIM, EDGE_FEAT_DIM


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base for all policy + value networks.

    Subclassing
    -----------
    class MyNetwork(BaseNetwork):
        def __init__(self, obs_shape, action_space_size, cfg):
            super().__init__()
            # build layers here

        def forward(self, obs, action_mask=None):
            ...
            return logits, value

        def get_action_and_log_prob(self, obs, action_mask=None, deterministic=False):
            ...
            return action, log_prob, value

        def evaluate_actions(self, obs, actions, action_mask=None):
            ...
            return log_probs, values, entropy
    """

    # ------------------------------------------------------------------
    # Abstract methods — the agent/algorithm interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        obs         : (B, *obs_shape) float32
        action_mask : (B, action_space_size) bool, True=feasible.  None = no masking.

        Returns
        -------
        logits : (B, action_space_size) — raw, masked, tanh-clipped scores
        value  : (B,) — critic estimate
        """
        ...

    @abstractmethod
    def get_action_and_log_prob(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or deterministically select an action.

        Returns
        -------
        action   : (B,) int64
        log_prob : (B,) float32
        value    : (B,) float32
        """
        ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate stored actions under the current policy.  Used in PPO update.

        Returns
        -------
        log_probs : (B,) float32
        values    : (B,) float32
        entropy   : (B,) float32
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def to_device(self, device: str) -> "BaseNetwork":
        """Move network to device and return self for chaining."""
        return self.to(torch.device(device))

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Zero-out infeasible logits with -inf."""
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    @staticmethod
    def _ortho_init(module: nn.Module, gain: float = 1.414) -> None:
        """Apply orthogonal initialisation to all Linear layers."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


"""
networks/hacn_network.py
------------------------
Heterogeneous Attention Construction Network (HACN) for VRPBTW.

Architecture
────────────
ENCODER  (runs once per instance, output Z_node cached)
  Node features (N+1, NODE_FEAT_DIM)
  → MLP projection
  → L × [Self-attention + Heterogeneous cross-attention (linehaul↔backhaul)
          + FF + InstanceNorm]
  → Z_node: (N+1, D)

VEHICLE GNN  (re-runs every step on dynamic partial-solution graph G_t)
  Graph G_t nodes  = visited nodes, features:
      Z_node[v]              (D,)   cached encoder output
      visit_time[v]/T_max    (1,)
      visiting_fleet[v]      (K,)   one-hot
      visiting_vehicle[v]    (1,)   0=truck 1=drone
  Graph G_t edges  (E, EDGE_FEAT_DIM=6):
      [edge_type, travel_time, travel_dist, depart_time, arrive_time, tardiness]
  → L' × edge-conditioned message passing
  → Z_graph: (|V_t|, D)

  Vehicle readout:
    base[k,type]  = Z_graph[current_node(k,type)]   (or zeros if graph empty)
    props[k,type] = MLP(vehicle_feature_row)         (D,)
    Z_veh[v]      = base[v] + props[v]               (D,)   shape (2K, D)

HIERARCHICAL DECODER  (runs every step)
  graph_node = mean_pool(Z_node)     (D,)
  graph_veh  = mean_pool(Z_veh)      (D,)

  Upper policy (node selection):
    context_U = MLP([graph_node ; graph_veh])   (D,)
    score[i]  = clip · tanh(dot(W_q·context_U, W_k·Z_node[i]) / sqrt(D))
    → masked softmax → node n*

  Lower policy (vehicle selection):
    context_L = MLP([graph_veh ; Z_node[n*]])   (D,)
    score[v]  = MLP([Z_veh[v] ; context_L])     scalar
    → masked softmax → vehicle v*

  Joint log-prob = log π_U(n*) + log π_L(v* | n*)

Observation contract
────────────────────
  obs["node_features"]    : (B, N+1, NODE_FEAT_DIM)
  obs["vehicle_features"] : (B, 2K,  VEH_FEAT_DIM)
  obs["edge_index"]       : list[B] of (2, E_b)  int   (variable per sample)
  obs["edge_attr"]        : list[B] of (E_b, 6)  float
  obs["edge_fleet"]       : list[B] of (E_b,)    int

Action encoding
────────────────
  flat = node * 2K + vehicle_idx
  Matches VRPBTWProblem.encode_action / decode_action
"""


# ---------------------------------------------------------------------------
# Shared sub-modules
# ---------------------------------------------------------------------------


def _make_norm(use_in: bool, dim: int) -> nn.Module:
    return _InstanceNormWrapper(dim) if use_in else nn.LayerNorm(dim)


class _InstanceNormWrapper(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _MHA(nn.Module):
    """Multi-head attention supporting self and cross variants."""

    def __init__(self, D: int, H: int, dropout: float = 0.0):
        super().__init__()
        assert D % H == 0
        self.H, self.Dh, self.D = H, D // H, D
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, Tq, D)
        k: torch.Tensor,  # (B, Tk, D)
        v: torch.Tensor,  # (B, Tk, D)
        mask: Optional[torch.Tensor] = None,  # (B, Tq, Tk) bool True=ignore
    ) -> torch.Tensor:
        B, Tq, _ = q.shape
        _, Tk, _ = k.shape
        H, Dh = self.H, self.Dh

        def reshape(t, T):
            return t.view(B, T, H, Dh).transpose(1, 2)

        Q = reshape(self.Wq(q), Tq)
        K = reshape(self.Wk(k), Tk)
        V = reshape(self.Wv(v), Tk)
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            sc = sc.masked_fill(mask.unsqueeze(1), float("-inf"))
        at = self.drop(torch.softmax(sc, dim=-1))
        out = torch.matmul(at, V).transpose(1, 2).contiguous().view(B, Tq, self.D)
        return self.Wo(out)


class _FF(nn.Module):
    def __init__(self, D: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(D * 4, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder block  (self-attn + heterogeneous cross-attn)
# ---------------------------------------------------------------------------


class _HACNEncoderBlock(nn.Module):
    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        self.sa = _MHA(D, H, dropout)
        self.het_l2b = _MHA(D, H, dropout)
        self.het_b2l = _MHA(D, H, dropout)
        self.ff = _FF(D, dropout)
        self.norm1 = _make_norm(use_in, D)
        self.norm2 = _make_norm(use_in, D)

    def forward(
        self,
        h: torch.Tensor,  # (B, N+1, D)
        h_l: torch.Tensor,  # (B, M,   D)
        h_b: torch.Tensor,  # (B, P,   D)
        l_idx: torch.Tensor,  # (M,)
        b_idx: torch.Tensor,  # (P,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_sa = self.sa(h, h, h)
        h_het = torch.zeros_like(h)
        if h_l.shape[1] > 0 and h_b.shape[1] > 0:
            h_het[:, l_idx] = self.het_l2b(h_l, h_b, h_b)
            h_het[:, b_idx] = self.het_b2l(h_b, h_l, h_l)
        h = self.norm1(h + h_sa + h_het)
        h = self.norm2(h + self.ff(h))
        h_l = h[:, l_idx] if h_l.shape[1] > 0 else h_l
        h_b = h[:, b_idx] if h_b.shape[1] > 0 else h_b
        return h, h_l, h_b


# ---------------------------------------------------------------------------
# Vehicle GNN
# ---------------------------------------------------------------------------


class _EdgeConvMessage(nn.Module):
    """
    One edge-conditioned message passing layer.

    For each edge (u→v) with edge feature e_uv:
        m_uv = MLP([h_u ; e_uv])
    Aggregation at v: h_v' = h_v + mean(m_uv for all u→v)
    Followed by residual LayerNorm.
    """

    def __init__(self, D: int, edge_feat_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(D + edge_feat_dim, D),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
        )
        self.norm = nn.LayerNorm(D)

    def forward(
        self,
        h: torch.Tensor,  # (V, D)  node embeddings
        edge_index: torch.Tensor,  # (2, E)  long
        edge_attr: torch.Tensor,  # (E, edge_feat_dim)
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return h  # empty graph: no messages

        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # (E, D)
        msg = self.msg_mlp(torch.cat([h_src, edge_attr], dim=-1))  # (E, D)

        # mean aggregation per destination node
        V = h.shape[0]
        agg = torch.zeros(V, h.shape[1], device=h.device, dtype=h.dtype)
        count = torch.zeros(V, 1, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        count.scatter_add_(
            0,
            dst.unsqueeze(1),
            torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype),
        )
        count = count.clamp(min=1.0)
        agg = agg / count

        return self.norm(h + agg)


class _VehicleGNN(nn.Module):
    """
    GNN over the partial-solution fleet graph.

    Input per visited node v:
        Z_node[v]          (D,)   cached encoder embedding
        visit_time[v]      (1,)   normalised
        visiting_fleet[v]  (K,)   one-hot fleet id
        visiting_vehicle[v](1,)   0=truck 1=drone
    Total node input dim: D + 1 + K + 1

    After L' layers of edge-conditioned message passing → Z_graph (V_t, D).

    Vehicle readout:
        base   = Z_graph[current_node]  (or zero_vec if no visited nodes yet)
        props  = MLP(vehicle_feature_row)
        Z_veh  = LayerNorm(base + props)
    """

    def __init__(
        self,
        D: int,
        K: int,
        edge_feat_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.D = D
        self.K = K
        node_in_dim = D + 1 + K + 1  # Z_node + visit_time + fleet_onehot + vtype

        self.node_proj = nn.Linear(node_in_dim, D)
        self.layers = nn.ModuleList(
            [_EdgeConvMessage(D, edge_feat_dim, dropout) for _ in range(n_layers)]
        )
        self.props_mlp = nn.Sequential(
            nn.Linear(VEH_FEAT_DIM, D), nn.ReLU(), nn.Linear(D, D)
        )
        self.readout_norm = nn.LayerNorm(D)

    def forward(
        self,
        Z_node_full: torch.Tensor,  # (N+1, D)   full encoder output
        veh_feat: torch.Tensor,  # (2K, VEH_FEAT_DIM)
        edge_index: torch.Tensor,  # (2, E)  long  or (2,0)
        edge_attr: torch.Tensor,  # (E, EDGE_FEAT_DIM)
        visited_nodes: List[int],  # sorted list of visited node indices
        visit_meta: torch.Tensor,  # (|V_t|, 1+K+1)  [time, fleet_oh, vtype]
        current_nodes: List[int],  # (2K,)  current node per vehicle
    ) -> torch.Tensor:  # (2K, D)
        D = self.D
        device = Z_node_full.device

        # ── Build per-visited-node features ──────────────────────────
        if len(visited_nodes) == 0:
            # empty graph: vehicle embeddings come from properties only
            props = self.props_mlp(veh_feat)  # (2K, D)
            zero_base = torch.zeros(len(current_nodes), D, device=device)
            return self.readout_norm(zero_base + props)

        v_idx = torch.tensor(visited_nodes, dtype=torch.long, device=device)
        Z_vis = Z_node_full[v_idx]  # (|V_t|, D)
        h_in = torch.cat([Z_vis, visit_meta], dim=-1)  # (|V_t|, D+1+K+1)
        h = self.node_proj(h_in)  # (|V_t|, D)

        # remap global node indices to local indices for message passing
        local_map = {g: l for l, g in enumerate(visited_nodes)}

        if edge_index.shape[1] > 0:
            src_g, dst_g = edge_index[0].tolist(), edge_index[1].tolist()
            # keep only edges where both endpoints are in visited_nodes
            valid = [
                (s, d)
                for s, d in zip(src_g, dst_g)
                if s in local_map and d in local_map
            ]
            if valid:
                src_l = torch.tensor(
                    [local_map[s] for s, _ in valid], dtype=torch.long, device=device
                )
                dst_l = torch.tensor(
                    [local_map[d] for _, d in valid], dtype=torch.long, device=device
                )
                e_idx_local = torch.stack([src_l, dst_l], dim=0)
                # reindex edge_attr to match valid edges
                valid_mask = torch.tensor(
                    [
                        i
                        for i, (s, d) in enumerate(zip(src_g, dst_g))
                        if s in local_map and d in local_map
                    ],
                    dtype=torch.long,
                    device=device,
                )
                e_attr_local = edge_attr[valid_mask]
            else:
                e_idx_local = torch.zeros(2, 0, dtype=torch.long, device=device)
                e_attr_local = torch.zeros(0, edge_attr.shape[-1], device=device)
        else:
            e_idx_local = torch.zeros(2, 0, dtype=torch.long, device=device)
            e_attr_local = torch.zeros(0, edge_attr.shape[-1], device=device)

        # ── Message passing ──────────────────────────────────────────
        for layer in self.layers:
            h = layer(h, e_idx_local, e_attr_local)  # (|V_t|, D)

        # ── Vehicle readout ──────────────────────────────────────────
        props = self.props_mlp(veh_feat)  # (2K, D)
        Z_veh = torch.zeros(len(current_nodes), D, device=device)
        for vi, cn in enumerate(current_nodes):
            if cn in local_map:
                Z_veh[vi] = h[local_map[cn]]
        return self.readout_norm(Z_veh + props)  # (2K, D)


# ---------------------------------------------------------------------------
# HACN Network
# ---------------------------------------------------------------------------


class PolicyNetwork(BaseNetwork):
    """
    Heterogeneous Attention Construction Network for VRPBTW.

    Parameters
    ----------
    obs_shape : (N+1, NODE_FEAT_DIM) from problem.observation_shape
    cfg       : NetworkConfig
    n_fleets  : K (number of fleets, i.e. truck-drone pairs)
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        cfg: NetworkConfig,
        n_fleets: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_shape = obs_shape
        self.K = n_fleets

        D = cfg.embed_dim
        H = cfg.n_heads
        L = cfg.n_encoder_layers
        drop = cfg.dropout
        use_in = getattr(cfg, "use_instance_norm", True)

        # ── Encoder ─────────────────────────────────────────────────
        self.node_embed = nn.Linear(NODE_FEAT_DIM, D)
        self.enc_blocks = nn.ModuleList(
            [_HACNEncoderBlock(D, H, drop, use_in) for _ in range(L)]
        )

        # ── Vehicle GNN ──────────────────────────────────────────────
        self.vehicle_gnn = _VehicleGNN(
            D=D,
            K=n_fleets,
            edge_feat_dim=EDGE_FEAT_DIM,
            n_layers=2,
            dropout=drop,
        )

        # ── Decoder: Upper policy (node selection) ───────────────────
        # context = MLP([graph_node ; graph_veh])
        self.ctx_upper = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU())
        self.Wq_upper = nn.Linear(D, D, bias=False)
        self.Wk_upper = nn.Linear(D, D, bias=False)

        # ── Decoder: Lower policy (vehicle selection) ────────────────
        # context = MLP([graph_veh ; Z_node[n*]])
        self.ctx_lower = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU())
        self.score_lower = nn.Sequential(
            nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, 1)
        )

        # ── Value head ───────────────────────────────────────────────
        self.value_head = nn.Sequential(nn.Linear(D * 2, D), nn.Tanh(), nn.Linear(D, 1))

        if cfg.ortho_init:
            self._ortho_init(self)

    # ------------------------------------------------------------------
    # Encode  (called once per instance; output cached by caller)
    # ------------------------------------------------------------------

    def encode(
        self,
        node_feat: torch.Tensor,  # (B, N+1, NODE_FEAT_DIM)
        l_idx: torch.Tensor,  # (M,)  linehaul indices
        b_idx: torch.Tensor,  # (P,)  backhaul indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns Z_node (B, N+1, D) and graph_node (B, D)."""
        h = self.node_embed(node_feat)
        h_l = h[:, l_idx] if l_idx.numel() > 0 else h[:, :0]
        h_b = h[:, b_idx] if b_idx.numel() > 0 else h[:, :0]
        for blk in self.enc_blocks:
            h, h_l, h_b = blk(h, h_l, h_b, l_idx, b_idx)
        return h, h.mean(dim=1)

    # ------------------------------------------------------------------
    # GNN vehicle embeddings  (called every step)
    # ------------------------------------------------------------------

    def embed_vehicles(
        self,
        Z_node: torch.Tensor,  # (N+1, D)   single-instance encoder output
        veh_feat: torch.Tensor,  # (2K, VEH_FEAT_DIM)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, EDGE_FEAT_DIM)
        visited_nodes: List[int],  # sorted list of visited global indices
        visit_meta: torch.Tensor,  # (|V_t|, 1+K+1)
        current_nodes: List[int],  # (2K,) current node per vehicle
    ) -> torch.Tensor:  # (2K, D)
        return self.vehicle_gnn(
            Z_node,
            veh_feat,
            edge_index,
            edge_attr,
            visited_nodes,
            visit_meta,
            current_nodes,
        )

    # ------------------------------------------------------------------
    # Decode Upper (node selection)
    # ------------------------------------------------------------------

    def _decode_upper(
        self,
        graph_node: torch.Tensor,  # (B, D)
        graph_veh: torch.Tensor,  # (B, D)
        Z_node: torch.Tensor,  # (B, N+1, D)
        node_mask: Optional[torch.Tensor],  # (B, N+1) True=feasible
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits_upper (B, N+1) and context_upper (B, D)."""
        ctx = self.ctx_upper(torch.cat([graph_node, graph_veh], dim=-1))  # (B, D)
        query = self.Wq_upper(ctx).unsqueeze(1)  # (B, 1, D)
        keys = self.Wk_upper(Z_node)  # (B, N+1, D)
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(
            self.cfg.embed_dim
        )  # (B, N+1)
        logits = self.cfg.clip_logits * torch.tanh(logits)
        logits = self._apply_mask(logits, node_mask)
        return logits, ctx

    # ------------------------------------------------------------------
    # Decode Lower (vehicle selection)
    # ------------------------------------------------------------------

    def _decode_lower(
        self,
        graph_veh: torch.Tensor,  # (B, D)
        Z_node_sel: torch.Tensor,  # (B, D)  Z_node[n*]
        Z_veh: torch.Tensor,  # (B, 2K, D)
        veh_mask: Optional[torch.Tensor],  # (B, 2K) True=feasible
    ) -> torch.Tensor:
        """Returns logits_lower (B, 2K)."""
        ctx = self.ctx_lower(torch.cat([graph_veh, Z_node_sel], dim=-1))  # (B, D)
        ctx_e = ctx.unsqueeze(1).expand_as(Z_veh)  # (B, 2K, D)
        logits = self.score_lower(torch.cat([Z_veh, ctx_e], dim=-1)).squeeze(
            -1
        )  # (B, 2K)
        logits = self._apply_mask(logits, veh_mask)
        return logits

    # ------------------------------------------------------------------
    # Helpers: obs → tensors
    # ------------------------------------------------------------------

    def _prep_batch(
        self, obs: Dict, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List, List]:
        """
        Returns:
            node_feat  (B, N+1, NODE_FEAT_DIM)
            veh_feat   (B, 2K,  VEH_FEAT_DIM)
            edge_index list[B]  each (2, E_b)  LongTensor
            edge_attr  list[B]  each (E_b, EDGE_FEAT_DIM) FloatTensor
            edge_fleet list[B]  each (E_b,)  LongTensor
        """
        nf = torch.FloatTensor(obs["node_features"]).to(device)
        vf = torch.FloatTensor(obs["vehicle_features"]).to(device)
        ei = obs["edge_index"]
        ea = obs["edge_attr"]
        ef = obs["edge_fleet"]
        # handle single-instance (non-batched) dicts
        if nf.dim() == 2:
            nf = nf.unsqueeze(0)
            vf = vf.unsqueeze(0)
            ei = [ei]
            ea = [ea]
            ef = [ef]
        ei_t = [torch.LongTensor(np.array(e, dtype=np.int64)).to(device) for e in ei]
        ea_t = [torch.FloatTensor(np.array(a, dtype=np.float32)).to(device) for a in ea]
        ef_t = [torch.LongTensor(np.array(f, dtype=np.int64)).to(device) for f in ef]
        return nf, vf, ei_t, ea_t, ef_t

    def _node_indices(self, node_feat: torch.Tensor):
        """Derive linehaul / backhaul indices from demand column (col 2)."""
        demand = node_feat[0, :, 2]
        l_idx = torch.where(demand > 0)[0]
        b_idx = torch.where(demand < 0)[0]
        return l_idx, b_idx

    @staticmethod
    def _visited_and_meta(
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, EDGE_FEAT_DIM) col0=vtype col4=arrive_time
        edge_fleet: torch.Tensor,  # (E,)  fleet id per edge
        K: int,
        device: str,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Derive visited node list and visit_meta tensor from edge data.

        visit_meta[i] shape: (1 + K + 1,)
            [0]      : latest arrive_time at node i  (edge_attr col 4)
            [1..K]   : one-hot fleet id of last edge arriving at i
            [K+1]    : vehicle type of that edge  (0=truck 1=drone, edge_attr col 0)
        """
        if edge_index.shape[1] == 0:
            return [], torch.zeros(0, 1 + K + 1, device=device)

        all_nodes_t = torch.cat([edge_index[0], edge_index[1]]).unique()
        all_nodes = sorted(all_nodes_t.tolist())
        V = len(all_nodes)
        meta = torch.zeros(V, 1 + K + 1, device=device)
        local_map = {int(g): l for l, g in enumerate(all_nodes)}

        # for each edge, update destination node meta (last edge wins)
        dst_list = edge_index[1].tolist()
        for ei_idx, dst_g in enumerate(dst_list):
            dst_g = int(dst_g)
            if dst_g not in local_map:
                continue
            li = local_map[dst_g]
            arr_t = float(edge_attr[ei_idx, 4].item())
            fleet_id = int(edge_fleet[ei_idx].item())
            vtype = float(edge_attr[ei_idx, 0].item())
            meta[li, 0] = arr_t
            meta[li, 1 : 1 + K] = 0.0
            if 0 <= fleet_id < K:
                meta[li, 1 + fleet_id] = 1.0
            meta[li, 1 + K] = vtype

        return all_nodes, meta

    @staticmethod
    def _upper_mask(action_mask: torch.Tensor, N1: int, V2K: int) -> torch.Tensor:
        """
        Node-level mask: node j is feasible if ANY vehicle can act on it.
        action_mask: (B, N1 * 2K)  flat = node * 2K + v_idx
        """
        B = action_mask.shape[0]
        m = action_mask.view(B, N1, V2K)
        return m.any(dim=-1)  # (B, N1)

    @staticmethod
    def _lower_mask(
        action_mask: torch.Tensor, node_id: torch.Tensor, N1: int, V2K: int
    ) -> torch.Tensor:
        """
        Vehicle-level mask for selected node n*.
        action_mask: (B, N1 * 2K)
        node_id:     (B,)
        Returns: (B, 2K)
        """
        B = action_mask.shape[0]
        m = action_mask.view(B, N1, V2K)
        idx = node_id.view(B, 1, 1).expand(B, 1, V2K)
        return m.gather(1, idx).squeeze(1)  # (B, 2K)

    # ------------------------------------------------------------------
    # forward  (for evaluate_actions in PPO update)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: Dict,
        action_mask: Optional[torch.Tensor] = None,  # (B, N1*2K)
        context: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device.type
        nf, vf, ei_t, ea_t, ef_t = self._prep_batch(obs, device)
        B, N1, _ = nf.shape
        V2K = vf.shape[1]
        l_idx, b_idx = self._node_indices(nf)

        # encoder (batch)
        Z_node, graph_node = self.encode(nf, l_idx, b_idx)  # (B, N+1, D), (B, D)

        # vehicle GNN — run per sample (variable graph size)
        Z_veh_list = []
        for b in range(B):
            visited, v_meta = self._visited_and_meta(
                ei_t[b], ea_t[b], ef_t[b], self.K, device
            )
            # current_node per vehicle: recover from veh_feat col 0 (node/N)
            # multiply back and round to nearest int
            cur_nodes = [
                int(round(float(vf[b, vi, 0].item()) * (N1 - 1))) for vi in range(V2K)
            ]
            z_v = self.vehicle_gnn(
                Z_node[b],
                vf[b],
                ei_t[b],
                ea_t[b],
                visited,
                v_meta,
                cur_nodes,
            )  # (2K, D)
            Z_veh_list.append(z_v)
        Z_veh = torch.stack(Z_veh_list, dim=0)  # (B, 2K, D)
        graph_veh = Z_veh.mean(dim=1)  # (B, D)

        # upper mask
        node_mask = (
            self._upper_mask(action_mask, N1, V2K) if action_mask is not None else None
        )

        logits_U, ctx_U = self._decode_upper(graph_node, graph_veh, Z_node, node_mask)

        # build flat logits by scoring all (node, vehicle) pairs
        flat_logits = self._build_flat_logits(
            logits_U, Z_node, graph_veh, Z_veh, action_mask, N1, V2K, device
        )

        value = self.value_head(
            torch.cat([ctx_U.detach(), graph_veh.detach()], dim=-1)
        ).squeeze(-1)

        return flat_logits, value

    # ------------------------------------------------------------------
    # get_action_and_log_prob  (two-level sampling)
    # ------------------------------------------------------------------

    def get_action_and_log_prob(
        self,
        obs: Dict,
        action_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device.type
        nf, vf, ei_t, ea_t, ef_t = self._prep_batch(obs, device)
        B, N1, _ = nf.shape
        V2K = vf.shape[1]
        l_idx, b_idx = self._node_indices(nf)

        Z_node, graph_node = self.encode(nf, l_idx, b_idx)

        Z_veh_list = []
        for b in range(B):
            visited, v_meta = self._visited_and_meta(
                ei_t[b], ea_t[b], ef_t[b], self.K, device
            )
            cur_nodes = [
                int(round(float(vf[b, vi, 0].item()) * (N1 - 1))) for vi in range(V2K)
            ]
            Z_veh_list.append(
                self.vehicle_gnn(
                    Z_node[b],
                    vf[b],
                    ei_t[b],
                    ea_t[b],
                    visited,
                    v_meta,
                    cur_nodes,
                )
            )
        Z_veh = torch.stack(Z_veh_list, dim=0)  # (B, 2K, D)
        graph_veh = Z_veh.mean(dim=1)  # (B, D)

        node_mask = (
            self._upper_mask(action_mask, N1, V2K) if action_mask is not None else None
        )

        logits_U, ctx_U = self._decode_upper(graph_node, graph_veh, Z_node, node_mask)

        dist_U = torch.distributions.Categorical(logits=logits_U)
        node_id = logits_U.argmax(dim=-1) if deterministic else dist_U.sample()
        lp_node = dist_U.log_prob(node_id)

        # lower policy
        Z_node_sel = Z_node[torch.arange(B, device=device), node_id]  # (B, D)
        veh_mask = (
            self._lower_mask(action_mask, node_id, N1, V2K)
            if action_mask is not None
            else None
        )
        logits_L = self._decode_lower(graph_veh, Z_node_sel, Z_veh, veh_mask)

        dist_L = torch.distributions.Categorical(logits=logits_L)
        veh_id = logits_L.argmax(dim=-1) if deterministic else dist_L.sample()
        lp_veh = dist_L.log_prob(veh_id)

        flat_action = node_id * V2K + veh_id
        log_prob = lp_node + lp_veh
        value = self.value_head(
            torch.cat([ctx_U.detach(), graph_veh.detach()], dim=-1)
        ).squeeze(-1)

        return flat_action, log_prob, value

    # ------------------------------------------------------------------
    # evaluate_actions  (PPO update)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: Dict,
        actions: torch.Tensor,  # (B,) flat
        action_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device.type
        nf, vf, ei_t, ea_t, ef_t = self._prep_batch(obs, device)
        B, N1, _ = nf.shape
        V2K = vf.shape[1]
        l_idx, b_idx = self._node_indices(nf)

        Z_node, graph_node = self.encode(nf, l_idx, b_idx)

        Z_veh_list = []
        for b in range(B):
            visited, v_meta = self._visited_and_meta(
                ei_t[b], ea_t[b], ef_t[b], self.K, device
            )
            cur_nodes = [
                int(round(float(vf[b, vi, 0].item()) * (N1 - 1))) for vi in range(V2K)
            ]
            Z_veh_list.append(
                self.vehicle_gnn(
                    Z_node[b],
                    vf[b],
                    ei_t[b],
                    ea_t[b],
                    visited,
                    v_meta,
                    cur_nodes,
                )
            )
        Z_veh = torch.stack(Z_veh_list, dim=0)
        graph_veh = Z_veh.mean(dim=1)

        node_id = actions // V2K
        veh_id = actions % V2K

        node_mask = (
            self._upper_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        logits_U, ctx_U = self._decode_upper(graph_node, graph_veh, Z_node, node_mask)
        dist_U = torch.distributions.Categorical(logits=logits_U)
        lp_node = dist_U.log_prob(node_id)
        ent_U = dist_U.entropy()

        Z_node_sel = Z_node[torch.arange(B, device=device), node_id]
        veh_mask = (
            self._lower_mask(action_mask, node_id, N1, V2K)
            if action_mask is not None
            else None
        )
        logits_L = self._decode_lower(graph_veh, Z_node_sel, Z_veh, veh_mask)
        dist_L = torch.distributions.Categorical(logits=logits_L)
        lp_veh = dist_L.log_prob(veh_id)
        ent_L = dist_L.entropy()

        log_probs = lp_node + lp_veh
        entropy = ent_U + ent_L
        value = self.value_head(
            torch.cat([ctx_U.detach(), graph_veh.detach()], dim=-1)
        ).squeeze(-1)

        return log_probs, value, entropy

    # ------------------------------------------------------------------
    # Flat logits builder (used by forward / beam search)
    # ------------------------------------------------------------------

    def _build_flat_logits(
        self,
        logits_U: torch.Tensor,  # (B, N+1)
        Z_node: torch.Tensor,  # (B, N+1, D)
        graph_veh: torch.Tensor,  # (B, D)
        Z_veh: torch.Tensor,  # (B, 2K, D)
        action_mask: Optional[torch.Tensor],
        N1: int,
        V2K: int,
        device: str,
    ) -> torch.Tensor:
        B = logits_U.shape[0]
        flat = torch.full((B, N1 * V2K), float("-inf"), device=logits_U.device)

        for n in range(N1):
            Z_n = Z_node[:, n, :]  # (B, D)
            v_mask = None
            if action_mask is not None:
                nid_t = torch.full((B,), n, dtype=torch.long, device=logits_U.device)
                v_mask = self._lower_mask(action_mask, nid_t, N1, V2K)
            l_L = self._decode_lower(graph_veh, Z_n, Z_veh, v_mask)  # (B, V2K)
            for v in range(V2K):
                flat[:, n * V2K + v] = logits_U[:, n] + l_L[:, v]

        if action_mask is not None:
            flat = flat.masked_fill(~action_mask, float("-inf"))
        return flat
