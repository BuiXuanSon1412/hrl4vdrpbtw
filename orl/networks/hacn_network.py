"""
networks/hacn_network.py
------------------------
Heterogeneous Attention Construction Network (HACN)
for VRPBTW with truck-drone fleets.

Architecture
------------
Encoder  — Two-stage attention (Wang et al. 2024)
  Stage 1: Multi-head self-attention over ALL nodes
           captures global spatial relationships
  Stage 2: Heterogeneous cross-attention
           linehaul nodes query backhaul embeddings  (and vice versa)
           explicitly models the linehaul/backhaul dependency
  Per block: skip + InstanceNorm + FF + InstanceNorm
  Runs ONCE per instance; static embeddings are cached.

Decoder  — Hierarchical, two-level  (simultaneous multi-route)
  Level 1 (Node Selector):
    Context = graph_mean + fleet_summary (all vehicles via attention)
    Pointer attention over node embeddings → node_logits (N+1,)
    Node-level mask: a node is feasible if ANY vehicle can serve it

  Level 2 (Vehicle Selector):
    Given selected node, score each vehicle
    Context per vehicle = [vehicle_emb_k ; target_node_emb ;
                           dist_to_node ; arrival_time ; tardiness_if_assigned]
    Vehicle-level mask: vehicle-specific feasibility

Joint log-prob (PPO compatible)
  log_prob = log_prob_L1(node) + log_prob_L2(vehicle | node)

Observation contract
--------------------
  obs["node_features"]    : (B, N+1, NODE_FEAT_DIM=9)  — encoder input
  obs["vehicle_features"] : (B, 2K,  VEH_FEAT_DIM=7)   — decoder context

Action encoding (matches environment)
  flat = fleet * 2*(N+1) + vehicle_type * (N+1) + node
  vehicle_index = fleet * 2 + vehicle_type  (0=truck, 1=drone per fleet)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork
from configs import NetworkConfig


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------


def _make_norm(use_instance_norm: bool, dim: int) -> nn.Module:
    """Instance norm (paper) or LayerNorm — toggled by config."""
    if use_instance_norm:
        # InstanceNorm1d expects (B, C, L); we wrap it to accept (B, L, C)
        return _InstanceNormWrapper(dim)
    return nn.LayerNorm(dim)


class _InstanceNormWrapper(nn.Module):
    """Wraps InstanceNorm1d to operate on (B, T, D) tensors."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → transpose → (B, D, T) → norm → (B, T, D)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# Multi-head attention  (shared by both encoder stages and decoder)
# ---------------------------------------------------------------------------


class _MHA(nn.Module):
    """
    Multi-head attention.
    query, key, value may come from different sequences (cross-attention).
    """

    def __init__(self, D: int, H: int, dropout: float = 0.0):
        super().__init__()
        assert D % H == 0
        self.H = H
        self.Dh = D // H
        self.D = D
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
        B, Tk, _ = k.shape
        H, Dh = self.H, self.Dh

        def reshape(t, T):
            return t.view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)

        Q = reshape(self.Wq(q), Tq)
        K = reshape(self.Wk(k), Tk)
        V = reshape(self.Wv(v), Tk)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,Tq,Tk)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)  # (B, H, Tq, Dh)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.D)
        return self.Wo(out)


# ---------------------------------------------------------------------------
# Feed-forward sub-layer
# ---------------------------------------------------------------------------


class _FF(nn.Module):
    def __init__(self, D: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D * 4, D),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Two-stage heterogeneous encoder block
# ---------------------------------------------------------------------------


class _HACNEncoderBlock(nn.Module):
    """
    One encoder block = Stage1 (self-attn) + Stage2 (het-attn) + FF.

    Following eq. 18 of Wang et al.:
      h = IN(h + h_sa + h_het)
      h = IN(h + FF(h))

    Stage 2 uses SEPARATE weight matrices for linehaul→backhaul
    and backhaul→linehaul attention, mirroring the paper's
    dual-structure heterogeneous attention (Fig. 5).
    """

    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        # Stage 1: self-attention over all nodes
        self.sa = _MHA(D, H, dropout)

        # Stage 2: separate weight matrices for each direction
        self.het_l2b = _MHA(D, H, dropout)  # linehaul queries → backhaul K/V
        self.het_b2l = _MHA(D, H, dropout)  # backhaul queries → linehaul K/V

        self.ff = _FF(D, dropout)
        self.norm1 = _make_norm(use_in, D)
        self.norm2 = _make_norm(use_in, D)

    def forward(
        self,
        h: torch.Tensor,  # (B, N+1, D)  all nodes
        h_l: torch.Tensor,  # (B, M,   D)  linehaul slice
        h_b: torch.Tensor,  # (B, P,   D)  backhaul slice
        l_idx: torch.Tensor,  # (M,)  linehaul node indices
        b_idx: torch.Tensor,  # (P,)  backhaul node indices
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage 1
        h_sa = self.sa(h, h, h)  # (B, N+1, D)

        # Stage 2 — only update linehaul and backhaul, depot unchanged
        h_het = torch.zeros_like(h)
        if h_l.shape[1] > 0 and h_b.shape[1] > 0:
            h_l_new = self.het_l2b(h_l, h_b, h_b)  # (B, M, D)
            h_b_new = self.het_b2l(h_b, h_l, h_l)  # (B, P, D)
            h_het[:, l_idx] = h_l_new
            h_het[:, b_idx] = h_b_new

        # Skip + norm (eq. 18)
        h = self.norm1(h + h_sa + h_het)
        h = self.norm2(h + self.ff(h))

        # Re-extract updated slices for next block
        h_l = h[:, l_idx] if h_l.shape[1] > 0 else h_l
        h_b = h[:, b_idx] if h_b.shape[1] > 0 else h_b
        return h, h_l, h_b


# ---------------------------------------------------------------------------
# Vehicle embedding MLP
# ---------------------------------------------------------------------------


class _VehicleEmbedder(nn.Module):
    """Projects raw vehicle features (2K, VEH_FEAT_DIM) → (2K, D)."""

    def __init__(self, veh_feat_dim: int, D: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(veh_feat_dim, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )

    def forward(self, veh: torch.Tensor) -> torch.Tensor:
        return self.proj(veh)  # (B, 2K, D)


# ---------------------------------------------------------------------------
# HACN Network
# ---------------------------------------------------------------------------


class HACNNetwork(BaseNetwork):
    """
    Heterogeneous Attention Construction Network.

    Parameters
    ----------
    obs_shape : (N+1, NODE_FEAT_DIM) from problem.observation_shape.
                The network infers all other sizes (N+1, 2K, feat_dim)
                from input tensors at forward time — no problem-size
                constants are stored on the network.
    cfg       : NetworkConfig
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        cfg: NetworkConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_shape = obs_shape
        # n_vehicles (2K) is inferred from vehicle_features.shape[1] at forward time

        N1 = obs_shape[0]  # N+1 nodes
        feat_dim = obs_shape[1]  # NODE_FEAT_DIM = 9
        D = cfg.embed_dim
        H = cfg.n_heads
        L = cfg.n_encoder_layers
        drop = cfg.dropout
        use_in = cfg.use_instance_norm

        # ── Encoder ────────────────────────────────────────────────────
        self.node_embed = nn.Linear(feat_dim, D)
        self.enc_blocks = nn.ModuleList(
            [_HACNEncoderBlock(D, H, drop, use_in) for _ in range(L)]
        )

        # ── Vehicle embedder ───────────────────────────────────────────
        from problems.vrpbtw import VEH_FEAT_DIM

        self.veh_embedder = _VehicleEmbedder(VEH_FEAT_DIM, D)

        # ── Decoder Level 1 (node selector) ───────────────────────────
        # Fleet context: graph_mean attends over vehicle embeddings
        self.fleet_attn = _MHA(D, H, drop)
        self.ctx_proj_L1 = nn.Linear(D * 2, D)  # [graph_mean ; fleet_ctx] → D

        # Pointer attention weights
        self.Wq_L1 = nn.Linear(D, D, bias=False)
        self.Wk_L1 = nn.Linear(D, D, bias=False)

        # ── Decoder Level 2 (vehicle selector) ────────────────────────
        # Per-vehicle context: [veh_emb ; target_node_emb ; dist ; arrival ; tardiness]
        # 2D + 3 scalars; we project scalars separately then add
        self.ctx_proj_L2 = nn.Linear(D * 2 + 3, D)
        self.score_L2 = nn.Linear(D, 1)

        # ── Value head (stop-gradient, for PPO critic) ─────────────────
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

        if cfg.ortho_init:
            self._ortho_init(self)

    # ------------------------------------------------------------------
    # Encode  (called once per instance in practice)
    # ------------------------------------------------------------------

    def encode(
        self,
        node_feat: torch.Tensor,  # (B, N+1, feat_dim)
        l_idx: torch.Tensor,  # (M,)  linehaul indices on device
        b_idx: torch.Tensor,  # (P,)  backhaul indices on device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        static_emb : (B, N+1, D)  node embeddings
        graph_emb  : (B, D)       mean pooling
        """
        h = self.node_embed(node_feat)  # (B, N+1, D)
        h_l = h[:, l_idx] if l_idx.numel() > 0 else h[:, :0]
        h_b = h[:, b_idx] if b_idx.numel() > 0 else h[:, :0]

        for block in self.enc_blocks:
            h, h_l, h_b = block(h, h_l, h_b, l_idx, b_idx)

        return h, h.mean(dim=1)

    # ------------------------------------------------------------------
    # Decode Level 1 — node selector
    # ------------------------------------------------------------------

    def _decode_L1(
        self,
        graph_emb: torch.Tensor,  # (B, D)
        static_emb: torch.Tensor,  # (B, N+1, D)
        veh_emb: torch.Tensor,  # (B, 2K, D)
        node_mask: Optional[
            torch.Tensor
        ],  # (B, N+1) bool True=feasible; None=all feasible
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits_L1   : (B, N+1) — masked node scores
        context_L1  : (B, D)   — context vector for value head
        """
        B = graph_emb.shape[0]

        # Fleet context: graph_mean queries all vehicle embeddings
        g = graph_emb.unsqueeze(1)  # (B, 1, D)
        fleet_ctx = self.fleet_attn(g, veh_emb, veh_emb)  # (B, 1, D)
        fleet_ctx = fleet_ctx.squeeze(1)  # (B, D)

        context_L1 = self.ctx_proj_L1(
            torch.cat([graph_emb, fleet_ctx], dim=-1)
        )  # (B, D)

        # Pointer attention
        query = self.Wq_L1(context_L1).unsqueeze(1)  # (B, 1, D)
        keys = self.Wk_L1(static_emb)  # (B, N+1, D)
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(
            self.cfg.embed_dim
        )
        logits = self.cfg.clip_logits * torch.tanh(logits)  # (B, N+1)
        logits = self._apply_mask(logits, node_mask)

        return logits, context_L1

    # ------------------------------------------------------------------
    # Decode Level 2 — vehicle selector
    # ------------------------------------------------------------------

    def _decode_L2(
        self,
        veh_emb: torch.Tensor,  # (B, 2K, D)
        target_emb: torch.Tensor,  # (B, D)   selected node embedding
        dist_to_node: torch.Tensor,  # (B, 2K)  travel distance
        arrival_time: torch.Tensor,  # (B, 2K)  estimated arrival
        tardiness: torch.Tensor,  # (B, 2K)  tardiness if assigned
        veh_mask: Optional[
            torch.Tensor
        ],  # (B, 2K)  bool True=feasible; None=all feasible
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits_L2 : (B, 2K) — masked vehicle scores
        """
        B, V, D = veh_emb.shape
        target = target_emb.unsqueeze(1).expand(B, V, D)  # (B, 2K, D)

        scalars = torch.stack(
            [dist_to_node, arrival_time, tardiness], dim=-1
        )  # (B, 2K, 3)

        ctx = torch.cat([veh_emb, target, scalars], dim=-1)  # (B, 2K, 2D+3)
        ctx = F.relu(self.ctx_proj_L2(ctx))  # (B, 2K, D)

        logits = self.score_L2(ctx).squeeze(-1)  # (B, 2K)
        logits = self._apply_mask(logits, veh_mask)
        return logits

    # ------------------------------------------------------------------
    # Forward  (full hierarchical decode)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict,
        action_mask: Optional[torch.Tensor] = None,  # (B, K*2*(N+1)) flat
        context: Optional[dict] = None,  # extra tensors from env
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        flat_logits : (B, K*2*(N+1))  joint logits over all (vehicle,node) pairs
        value       : (B,)

        This is the standard BaseNetwork forward() contract — used during
        PPO's evaluate_actions().  For action selection, use
        get_action_and_log_prob() which properly runs the two-level decode.
        """
        node_feat = obs["node_features"]  # (B, N+1, feat_dim)
        veh_feat = obs["vehicle_features"]  # (B, 2K, veh_feat_dim)

        l_idx, b_idx = self._node_indices(node_feat)
        static_emb, graph_emb = self.encode(node_feat, l_idx, b_idx)
        veh_emb = self.veh_embedder(veh_feat)  # (B, 2K, D)

        B, N1, D = static_emb.shape
        V = veh_emb.shape[1]
        K = V // 2

        # Build node-level mask: node i is feasible if ANY vehicle can serve it
        if action_mask is not None:
            node_mask = self._node_level_mask(action_mask, N1, V)
        else:
            node_mask = None

        logits_L1, ctx_L1 = self._decode_L1(graph_emb, static_emb, veh_emb, node_mask)

        # For flat logits (needed by evaluate_actions), we compute vehicle
        # scores for every node simultaneously
        flat_logits = self._build_flat_logits(
            logits_L1, static_emb, veh_emb, veh_feat, action_mask, N1, V
        )

        value = self.value_head(ctx_L1.detach()).squeeze(-1)
        return flat_logits, value

    # ------------------------------------------------------------------
    # get_action_and_log_prob  — proper two-level sampling
    # ------------------------------------------------------------------

    def get_action_and_log_prob(
        self,
        obs: dict,
        action_mask: Optional[torch.Tensor] = None,
        context: Optional[dict] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-level hierarchical action selection.

        Returns
        -------
        action   : (B,)  flat integer action
        log_prob : (B,)  log p(node) + log p(vehicle|node)
        value    : (B,)
        """
        node_feat = obs["node_features"]
        veh_feat = obs["vehicle_features"]

        l_idx, b_idx = self._node_indices(node_feat)
        static_emb, graph_emb = self.encode(node_feat, l_idx, b_idx)
        veh_emb = self.veh_embedder(veh_feat)

        B, N1, D = static_emb.shape
        V = veh_emb.shape[1]

        # ── Level 1: select node ──────────────────────────────────────
        if action_mask is not None:
            node_mask = self._node_level_mask(action_mask, N1, V)
        else:
            node_mask = torch.ones(B, N1, dtype=torch.bool, device=node_feat.device)

        logits_L1, ctx_L1 = self._decode_L1(graph_emb, static_emb, veh_emb, node_mask)

        dist_L1 = torch.distributions.Categorical(logits=logits_L1)
        node_id = logits_L1.argmax(dim=-1) if deterministic else dist_L1.sample()
        lp_node = dist_L1.log_prob(node_id)

        # ── Level 2: select vehicle for chosen node ────────────────────
        target_emb = static_emb[torch.arange(B), node_id]  # (B, D)
        dist_t, arr_t, tard_t = self._vehicle_scalars(veh_feat, node_feat, node_id)

        if action_mask is not None:
            veh_mask = self._vehicle_level_mask(action_mask, node_id, N1, V)
        else:
            veh_mask = torch.ones(B, V, dtype=torch.bool, device=node_feat.device)

        logits_L2 = self._decode_L2(
            veh_emb, target_emb, dist_t, arr_t, tard_t, veh_mask
        )

        dist_L2 = torch.distributions.Categorical(logits=logits_L2)
        vehicle_id = logits_L2.argmax(dim=-1) if deterministic else dist_L2.sample()
        lp_vehicle = dist_L2.log_prob(vehicle_id)

        # ── Encode back to flat action ─────────────────────────────────
        # vehicle_id = fleet*2 + vehicle_type
        fleet = vehicle_id // 2
        vehicle_type = vehicle_id % 2
        flat_action = fleet * (2 * N1) + vehicle_type * N1 + node_id

        log_prob = lp_node + lp_vehicle
        value = self.value_head(ctx_L1.detach()).squeeze(-1)

        return flat_action, log_prob, value

    # ------------------------------------------------------------------
    # evaluate_actions  (PPO update)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: dict,
        actions: torch.Tensor,  # (B,) flat actions
        action_mask: Optional[torch.Tensor] = None,
        context: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_feat = obs["node_features"]
        veh_feat = obs["vehicle_features"]

        l_idx, b_idx = self._node_indices(node_feat)
        static_emb, graph_emb = self.encode(node_feat, l_idx, b_idx)
        veh_emb = self.veh_embedder(veh_feat)

        B, N1, D = static_emb.shape
        V = veh_emb.shape[1]

        # Decode node_id and vehicle_id from flat actions
        fleet = actions // (2 * N1)
        vehicle_type = (actions // N1) % 2
        node_id = actions % N1
        vehicle_id = fleet * 2 + vehicle_type

        # Level 1 log-prob
        if action_mask is not None:
            node_mask = self._node_level_mask(action_mask, N1, V)
        else:
            node_mask = None

        logits_L1, ctx_L1 = self._decode_L1(graph_emb, static_emb, veh_emb, node_mask)
        dist_L1 = torch.distributions.Categorical(logits=logits_L1)
        lp_node = dist_L1.log_prob(node_id)
        ent_node = dist_L1.entropy()

        # Level 2 log-prob
        target_emb = static_emb[torch.arange(B), node_id]
        dist_t, arr_t, tard_t = self._vehicle_scalars(veh_feat, node_feat, node_id)

        if action_mask is not None:
            veh_mask = self._vehicle_level_mask(action_mask, node_id, N1, V)
        else:
            veh_mask = None

        logits_L2 = self._decode_L2(
            veh_emb, target_emb, dist_t, arr_t, tard_t, veh_mask
        )
        dist_L2 = torch.distributions.Categorical(logits=logits_L2)
        lp_vehicle = dist_L2.log_prob(vehicle_id)
        ent_vehicle = dist_L2.entropy()

        log_probs = lp_node + lp_vehicle
        entropy = ent_node + ent_vehicle
        value = self.value_head(ctx_L1.detach()).squeeze(-1)

        return log_probs, value, entropy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _node_indices(
        self,
        node_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Derive linehaul / backhaul node indices from demand column (col 2).
        Depot (index 0) is excluded from both sets.
        """
        demand = node_feat[0, :, 2]  # (N+1,)  use first batch item
        l_idx = torch.where(demand > 0)[0]
        b_idx = torch.where(demand < 0)[0]
        return l_idx, b_idx

    @staticmethod
    def _node_level_mask(
        flat_mask: torch.Tensor,  # (B, K*2*(N+1))
        N1: int,
        V: int,  # 2K
    ) -> torch.Tensor:
        """
        Node-level mask: node i is feasible if ANY vehicle can serve it.
        flat_mask is reshaped to (B, K, 2, N1) then collapsed over vehicle dims.
        """
        B = flat_mask.shape[0]
        K = V // 2
        m = flat_mask.view(B, K, 2, N1)  # (B, K, 2, N+1)
        return m.any(dim=1).any(dim=1)  # (B, N+1)

    @staticmethod
    def _vehicle_level_mask(
        flat_mask: torch.Tensor,  # (B, K*2*(N+1))
        node_id: torch.Tensor,  # (B,)
        N1: int,
        V: int,
    ) -> torch.Tensor:
        """
        Vehicle-level mask for selected node: (B, 2K).
        For vehicle v, check if flat_mask at (fleet=v//2, vehicle_type=v%2, node=node_id).
        """
        B = flat_mask.shape[0]
        K = V // 2
        m = flat_mask.view(B, K, 2, N1)  # (B, K, 2, N+1)
        # gather over node dimension
        nid = node_id.view(B, 1, 1, 1).expand(B, K, 2, 1)
        per_vehicle = m.gather(3, nid).squeeze(-1)  # (B, K, 2)
        return per_vehicle.view(B, V)  # (B, 2K)

    def _vehicle_scalars(
        self,
        veh_feat: torch.Tensor,  # (B, 2K, VEH_FEAT_DIM)
        node_feat: torch.Tensor,  # (B, N+1, NODE_FEAT_DIM)
        node_id: torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-vehicle scalars for Level 2 context:
          dist_to_node, arrival_time, tardiness_if_assigned
        All normalised to [0, 1].
        These are approximations derived from normalised features.
        """
        B, V, _ = veh_feat.shape

        # Vehicle current position encoded as node index (col 0) — proxy for distance
        veh_node_frac = veh_feat[:, :, 0]  # (B, 2K)  node/N
        veh_time_frac = veh_feat[:, :, 2]  # (B, 2K)  time/T

        # Target node time window (cols 3, 4 of node_feat)
        tw_open = node_feat[torch.arange(B), node_id, 3]  # (B,)
        tw_close = node_feat[torch.arange(B), node_id, 4]  # (B,)

        # Approximate distance as abs difference of normalised node positions
        # (coarse but gradient-friendly; actual dist matrix is in the env)
        dist_approx = (
            veh_node_frac
            - node_id.float().unsqueeze(1) / max(node_feat.shape[1] - 1, 1)
        ).abs()  # (B, 2K)

        # Approximate arrival: current time + distance (speed=1 for trucks, 2 for drones)
        # is_drone is col 4 of veh_feat
        speed = 1.0 + veh_feat[:, :, 4]  # (B, 2K) 1.0 truck, 2.0 drone
        arrival_approx = veh_time_frac + dist_approx / speed  # (B, 2K)

        tw_close_exp = tw_close.unsqueeze(1).expand(B, V)  # (B, 2K)
        tardiness = F.relu(arrival_approx - tw_close_exp)  # (B, 2K)

        return dist_approx, arrival_approx, tardiness

    def _build_flat_logits(
        self,
        logits_L1: torch.Tensor,  # (B, N+1)
        static_emb: torch.Tensor,  # (B, N+1, D)
        veh_emb: torch.Tensor,  # (B, 2K, D)
        veh_feat: torch.Tensor,  # (B, 2K, VEH_FEAT_DIM)
        action_mask: Optional[torch.Tensor],
        N1: int,
        V: int,
    ) -> torch.Tensor:
        """
        Build a flat (B, K*2*(N+1)) logit tensor from the two-level scores.
        Used by evaluate_actions and the standard forward() path.

        flat[b, fleet*(2*N1) + vtype*N1 + node] = logit_L1[node] + logit_L2[vehicle]
        """
        B = logits_L1.shape[0]
        device = logits_L1.device

        flat = torch.full((B, V // 2 * 2 * N1), float("-inf"), device=device)

        for n in range(N1):
            target_emb = static_emb[:, n, :]  # (B, D)
            node_id_t = torch.full((B,), n, dtype=torch.long, device=device)

            if action_mask is not None:
                veh_mask = self._vehicle_level_mask(action_mask, node_id_t, N1, V)
            else:
                veh_mask = None

            dist_t, arr_t, tard_t = self._vehicle_scalars(
                veh_feat, static_emb, node_id_t
            )
            l2 = self._decode_L2(
                veh_emb, target_emb, dist_t, arr_t, tard_t, veh_mask
            )  # (B,2K)

            # Distribute to flat layout
            for vid in range(V):
                fleet = vid // 2
                vtype = vid % 2
                flat_i = fleet * (2 * N1) + vtype * N1 + n
                # joint logit = L1(node) + L2(vehicle|node)
                valid_L1 = logits_L1[:, n]
                valid_L2 = l2[:, vid]
                flat[:, flat_i] = valid_L1 + valid_L2

        if action_mask is not None:
            flat = flat.masked_fill(~action_mask, float("-inf"))

        return flat
