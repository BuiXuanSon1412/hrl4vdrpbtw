"""
networks/hacn.py
----------------
Heterogeneous Attention Construction Network (HACN) for VRPBTW.

Architecture
============

ENCODING PHASE  (runs once per step, not once per instance)
------------------------------------------------------------

  NodeEncoder
    Projects raw node features and runs L heterogeneous self-attention
    blocks (linehaul ↔ backhaul cross-attention) to produce node
    embeddings Z_node (B, N+1, D) and a global node context g_node (B, D).

  VehicleEncoder
    Takes live vehicle state features and the global node context g_node,
    then produces vehicle embeddings Z_veh (B, 2K, D) through:
      1. Type-specific input projection (separate heads for truck / drone)
      2. Injection of learned vehicle-type embeddings
      3. Edge-conditioned message passing over the route graph
      4. Cross-attention from each vehicle to the full node set Z_node
         — this is where vehicles "read" the remaining problem structure
      5. Type-gated readout MLP

  HierarchicalEncoder
    Orchestrates one round of NodeEncoder → VehicleEncoder, then a
    second cross-attention pass where nodes attend back to vehicles
    (so nodes "read" the current fleet state before decoding).
    Produces the final Z_node, Z_veh, g_node, g_veh used by the decoder.

DECODING PHASE  (hierarchical two-pointer)
------------------------------------------

  NodeSelectionDecoder
    Upper pointer: selects which node n* to serve next.
    Query = f(g_node, g_veh).  Keys = Z_node.

  VehicleSelectionDecoder
    Lower pointer: selects which vehicle v* serves n*.
    Explicitly conditioned on vehicle type via type embeddings.
    Query = f(g_veh, Z_node[n*]).  Keys = Z_veh with type injection.

  Joint log-prob = log π_upper(n*) + log π_lower(v* | n*)
  Flat action index = n* × 2K + v*

Constants (from problems/vrpbtw.py)
------------------------------------
  NODE_FEAT_DIM = 5   [x, y, demand, tw_open, tw_close]
  VEH_FEAT_DIM  = 5   [x, y, load, time, deadline]
  EDGE_FEAT_DIM = 6   [vtype, travel_time, dist, depart, arrive, tardiness]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from core.module import BaseNetwork, _MHA, _FF, _make_norm
from problems.vrpbtw import NODE_FEAT_DIM, VEH_FEAT_DIM, EDGE_FEAT_DIM


# ---------------------------------------------------------------------------
# Shared sub-module: edge-conditioned message passing
# ---------------------------------------------------------------------------


class _EdgeConvLayer(nn.Module):
    """
    One layer of edge-conditioned message passing over a route graph.

    For each edge (src → dst), concatenates the source node embedding
    with the edge features, passes through an MLP to produce a message,
    then mean-aggregates messages at each destination.

    Used inside VehicleEncoder to propagate route-history information
    across visited nodes before vehicles attend to the node set.
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
        h: torch.Tensor,  # (V, D)  visited-node embeddings
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, edge_feat_dim)
    ) -> torch.Tensor:  # (V, D)
        if edge_index.shape[1] == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        msg = self.msg_mlp(torch.cat([h[src], edge_attr], dim=-1))
        V = h.shape[0]
        agg = torch.zeros(V, h.shape[1], device=h.device, dtype=h.dtype)
        count = torch.ones(V, 1, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        count.scatter_add_(
            0,
            dst.unsqueeze(1),
            torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype),
        )
        return self.norm(h + agg / count)


# ---------------------------------------------------------------------------
# NodeEncoder
# ---------------------------------------------------------------------------


class NodeEncoder(nn.Module):
    """
    Encodes the full set of nodes (depot + customers) into embeddings
    Z_node (B, N+1, D) using heterogeneous self-attention.

    Heterogeneous attention means linehaul and backhaul customer nodes
    exchange information through dedicated cross-attention heads
    (linehaul→backhaul and backhaul→linehaul), in addition to the
    standard full self-attention over all nodes.  This lets linehaul
    nodes "know about" nearby backhaul nodes before decoding, which is
    important because the phase constraint links the two groups.

    Also returns g_node = mean(Z_node) as a global node context vector
    used by VehicleEncoder and the decoders.
    """

    def __init__(
        self,
        D: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_instance_norm: bool,
    ):
        super().__init__()
        self.input_proj = nn.Linear(NODE_FEAT_DIM, D)
        self.layers = nn.ModuleList(
            [
                _HeterogeneousNodeEncoderLayer(D, n_heads, dropout, use_instance_norm)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        node_feat: torch.Tensor,  # (B, N+1, NODE_FEAT_DIM)
        l_idx: torch.Tensor,  # (M,)  linehaul node indices
        b_idx: torch.Tensor,  # (P,)  backhaul node indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        h_l = h[:, l_idx] if l_idx.numel() > 0 else h[:, :0]
        h_b = h[:, b_idx] if b_idx.numel() > 0 else h[:, :0]
        for layer in self.layers:
            h, h_l, h_b = layer(h, h_l, h_b, l_idx, b_idx)
        g_node = h.mean(dim=1)  # (B, D)
        return h, g_node


class _HeterogeneousNodeEncoderLayer(nn.Module):
    """
    One layer of the NodeEncoder.

    Applies:
      1. Full self-attention over all N+1 nodes
      2. Heterogeneous cross-attention: linehaul ↔ backhaul
      3. Residual + norm after attention
      4. Position-wise feed-forward + residual + norm
    """

    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        self.self_attn = _MHA(D, H, dropout)
        self.cross_l2b = _MHA(D, H, dropout)  # linehaul queries backhaul
        self.cross_b2l = _MHA(D, H, dropout)  # backhaul queries linehaul
        self.ff = _FF(D, dropout)
        self.norm_attn = _make_norm(use_in, D)
        self.norm_ff = _make_norm(use_in, D)

    def forward(
        self,
        h: torch.Tensor,  # (B, N+1, D)
        h_l: torch.Tensor,  # (B, M, D)
        h_b: torch.Tensor,  # (B, P, D)
        l_idx: torch.Tensor,
        b_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_sa = self.self_attn(h, h, h)
        h_het = torch.zeros_like(h)
        if h_l.shape[1] > 0 and h_b.shape[1] > 0:
            h_het[:, l_idx] = self.cross_l2b(h_l, h_b, h_b)
            h_het[:, b_idx] = self.cross_b2l(h_b, h_l, h_l)
        h = self.norm_attn(h + h_sa + h_het)
        h = torch.nan_to_num(h, nan=0.0)
        h = self.norm_ff(h + self.ff(h))
        h = torch.nan_to_num(h, nan=0.0)
        h_l = h[:, l_idx] if h_l.shape[1] > 0 else h_l
        h_b = h[:, b_idx] if h_b.shape[1] > 0 else h_b
        return h, h_l, h_b


# ---------------------------------------------------------------------------
# VehicleEncoder
# ---------------------------------------------------------------------------


class VehicleEncoder(nn.Module):
    """
    Encodes 2K vehicles (K trucks + K drones) into embeddings
    Z_veh (B, 2K, D), conditioning on both live vehicle state and the
    current node embeddings Z_node from NodeEncoder.

    Pipeline per batch item
    -----------------------
    1. Type-specific input projection
       Trucks and drones have separate Linear heads so each type learns
       its own feature importance (phase constraint vs trip budget,
       Manhattan vs Euclidean costs).

    2. Vehicle-type embedding injection
       A learned 2×D lookup (truck=0, drone=1) is added to every
       vehicle's projected embedding.  This gives a hard, persistent
       type signal through all subsequent layers.

    3. Route-graph message passing  (_EdgeConvLayer)
       Visited nodes are embedded and updated via edge-conditioned GNN
       over the incremental route graph.  Each vehicle's embedding is
       initialised from its current node in this graph.

    4. Vehicle→Node cross-attention
       Each vehicle attends over the full Z_node set to read the
       remaining problem structure (unserved nodes, their time windows,
       demands).  This is the key coupling that makes the encoding phase
       hierarchical: vehicles learn *what is left to serve* before the
       decoder runs.

    5. Type-gated readout MLP
       Final embedding = MLP([gnn_context | node_context | type_emb | props])
       The gate jointly processes all four signals so truck/drone
       differences in scoring are learned, not hard-coded.
    """

    TRUCK = 0
    DRONE = 1

    def __init__(
        self,
        D: int,
        K: int,
        edge_feat_dim: int,
        n_gnn_layers: int = 2,
        n_cross_attn_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.D = D
        self.K = K

        # ── 1 + 2: type embedding and separate input heads ────────────
        self.type_embed = nn.Embedding(2, D)  # 0=truck, 1=drone

        # visited-node input dim: Z_node(D) + visit_time(1) + fleet_onehot(K) + vtype(1)
        node_in_dim = D + 1 + K + 1
        self.truck_node_proj = nn.Sequential(
            nn.Linear(node_in_dim, D), nn.ReLU(), nn.Linear(D, D)
        )
        self.drone_node_proj = nn.Sequential(
            nn.Linear(node_in_dim, D), nn.ReLU(), nn.Linear(D, D)
        )

        # live vehicle-state heads (VEH_FEAT_DIM → D)
        self.truck_state_proj = nn.Sequential(
            nn.Linear(VEH_FEAT_DIM, D), nn.ReLU(), nn.Linear(D, D)
        )
        self.drone_state_proj = nn.Sequential(
            nn.Linear(VEH_FEAT_DIM, D), nn.ReLU(), nn.Linear(D, D)
        )

        # ── 3: route-graph GNN ────────────────────────────────────────
        self.gnn_layers = nn.ModuleList(
            [_EdgeConvLayer(D, edge_feat_dim, dropout) for _ in range(n_gnn_layers)]
        )

        # ── 4: vehicle → node cross-attention ────────────────────────
        # Vehicles (queries) attend over node embeddings (keys/values)
        self.veh_to_node_attn = _MHA(D, n_cross_attn_heads, dropout)
        self.cross_norm = nn.LayerNorm(D)

        # ── 5: type-gated readout ─────────────────────────────────────
        # inputs: gnn_context(D) + node_context(D) + type_emb(D) + state_props(D)
        self.readout_gate = nn.Sequential(
            nn.Linear(D * 4, D * 2),
            nn.ReLU(),
            nn.Linear(D * 2, D),
        )
        self.readout_norm = nn.LayerNorm(D)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def type_ids(self, device: torch.device) -> torch.Tensor:
        """(2K,) int tensor: trucks=0 for first K, drones=1 for last K."""
        ids = torch.zeros(2 * self.K, dtype=torch.long, device=device)
        ids[self.K :] = 1
        return ids

    def _project_visited_nodes(
        self,
        Z_node_full: torch.Tensor,  # (N+1, D)
        visited: List[int],
        meta: torch.Tensor,  # (|V|, 1+K+1)
    ) -> torch.Tensor:  # (|V|, D)
        """Project visited nodes through the appropriate type-specific head."""
        v_idx = torch.tensor(visited, dtype=torch.long, device=Z_node_full.device)
        h_in = torch.cat([Z_node_full[v_idx], meta], dim=-1)
        vtype = meta[:, 1 + self.K]  # 0=truck-routed, 1=drone-routed
        truck_m = vtype < 0.5
        drone_m = ~truck_m
        h = torch.zeros(len(visited), self.D, device=Z_node_full.device)
        if truck_m.any():
            h[truck_m] = self.truck_node_proj(h_in[truck_m])
        if drone_m.any():
            h[drone_m] = self.drone_node_proj(h_in[drone_m])
        # Inject type embedding into each visited node's hidden state
        h = h + self.type_embed(truck_m.logical_not().long())
        return h

    @staticmethod
    def _build_local_graph(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        visited: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remap global node indices to local indices over visited nodes."""
        E_dim = edge_attr.shape[-1] if edge_attr.numel() > 0 else EDGE_FEAT_DIM
        empty_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        empty_ea = torch.zeros(0, E_dim, device=device)
        if edge_index.shape[1] == 0 or not visited:
            return empty_ei, empty_ea
        local = {int(g): l for l, g in enumerate(visited)}
        src_g, dst_g = edge_index[0].tolist(), edge_index[1].tolist()
        keep = [
            i for i, (s, d) in enumerate(zip(src_g, dst_g)) if s in local and d in local
        ]
        if not keep:
            return empty_ei, empty_ea
        ki = torch.tensor(keep, dtype=torch.long, device=device)
        src_l = torch.tensor(
            [local[src_g[i]] for i in keep], dtype=torch.long, device=device
        )
        dst_l = torch.tensor(
            [local[dst_g[i]] for i in keep], dtype=torch.long, device=device
        )
        return torch.stack([src_l, dst_l]), edge_attr[ki]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        Z_node_full: torch.Tensor,  # (N+1, D)   from NodeEncoder
        Z_node_batch: torch.Tensor,  # (1, N+1, D) batched version for cross-attn
        veh_feat: torch.Tensor,  # (2K, VEH_FEAT_DIM)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, EDGE_FEAT_DIM)
        visited: List[int],
        meta: torch.Tensor,  # (|V|, 1+K+1)
        cur_nodes: List[int],  # (2K,)
    ) -> torch.Tensor:  # (2K, D)
        device = Z_node_full.device
        V2K = 2 * self.K
        t_ids = self.type_ids(device)  # (2K,)
        type_emb = self.type_embed(t_ids)  # (2K, D)

        # ── 1+2: type-specific state projections ─────────────────────
        t_state = self.truck_state_proj(veh_feat[: self.K])  # (K, D)
        d_state = self.drone_state_proj(veh_feat[self.K :])  # (K, D)
        state_props = torch.cat([t_state, d_state], dim=0)  # (2K, D)

        # ── 3: GNN over visited route graph ──────────────────────────
        if not visited:
            gnn_context = torch.zeros(V2K, self.D, device=device)
        else:
            h = self._project_visited_nodes(Z_node_full, visited, meta)
            ei_l, ea_l = self._build_local_graph(edge_index, edge_attr, visited, device)
            for layer in self.gnn_layers:
                h = layer(h, ei_l, ea_l)
            local_map = {int(g): l for l, g in enumerate(visited)}
            gnn_context = torch.zeros(V2K, self.D, device=device)
            for vi, cn in enumerate(cur_nodes):
                if cn in local_map:
                    gnn_context[vi] = h[local_map[cn]]

        # ── 4: vehicle → node cross-attention ────────────────────────
        # Each vehicle (query) attends over all nodes (keys/values) to
        # read remaining demand, time windows, and spatial structure.
        # Shape: queries (1, 2K, D), keys/values (1, N+1, D)
        veh_queries = (gnn_context + type_emb + state_props).unsqueeze(0)  # (1, 2K, D)
        node_context_raw = self.veh_to_node_attn(
            veh_queries,  # Q: vehicles
            Z_node_batch,  # K: nodes
            Z_node_batch,  # V: nodes
        )  # (1, 2K, D)
        node_context = self.cross_norm(veh_queries + node_context_raw).squeeze(
            0
        )  # (2K, D)

        # ── 5: type-gated readout ─────────────────────────────────────
        fused = torch.cat(
            [gnn_context, node_context, type_emb, state_props], dim=-1
        )  # (2K, 4D)
        out = self.readout_gate(fused)  # (2K, D)
        return self.readout_norm(out)  # (2K, D)


# ---------------------------------------------------------------------------
# HierarchicalEncoder
# ---------------------------------------------------------------------------


class HierarchicalEncoder(nn.Module):
    """
    Coordinates the bidirectional encoding pass between nodes and vehicles.

    Pass 1 — NodeEncoder
      Encodes nodes using heterogeneous self-attention.
      Produces Z_node (B, N+1, D) and g_node (B, D).

    Pass 2 — VehicleEncoder
      Encodes vehicles using Z_node as cross-attention context.
      Vehicles attend to nodes to read remaining problem structure.
      Produces Z_veh (B, 2K, D) and g_veh (B, D).

    Pass 3 — Node refinement cross-attention
      Nodes attend back to the vehicle embeddings so node embeddings
      are updated with information about the current fleet state
      (positions, loads, which nodes each vehicle has already visited).
      This makes Z_node "aware of" the fleet before the decoder runs,
      closing the bidirectional loop.

    The output Z_node from Pass 3, along with Z_veh from Pass 2, are
    what the NodeSelectionDecoder and VehicleSelectionDecoder use.
    """

    def __init__(
        self,
        D: int,
        K: int,
        n_heads: int,
        n_node_layers: int,
        n_gnn_layers: int,
        dropout: float,
        use_instance_norm: bool,
    ):
        super().__init__()
        self.node_encoder = NodeEncoder(
            D=D,
            n_heads=n_heads,
            n_layers=n_node_layers,
            dropout=dropout,
            use_instance_norm=use_instance_norm,
        )
        self.vehicle_encoder = VehicleEncoder(
            D=D,
            K=K,
            edge_feat_dim=EDGE_FEAT_DIM,
            n_gnn_layers=n_gnn_layers,
            n_cross_attn_heads=n_heads,
            dropout=dropout,
        )
        # Pass 3: nodes attend to vehicles
        # Nodes (queries) read vehicle embeddings (keys/values)
        self.node_to_veh_attn = _MHA(D, n_heads, dropout)
        self.node_refine_norm = _make_norm(use_instance_norm, D)

    def forward(
        self,
        node_feat: torch.Tensor,  # (B, N+1, NODE_FEAT_DIM)
        l_idx: torch.Tensor,  # (M,)
        b_idx: torch.Tensor,  # (P,)
        veh_feat: torch.Tensor,  # (B, 2K, VEH_FEAT_DIM)
        edge_indices: List,  # list of (2, E) per batch
        edge_attrs: List,  # list of (E, EDGE_FEAT_DIM) per batch
        edge_fleets: List,  # list of (E,) per batch
        visited_list: List[List[int]],  # per-batch visited node lists
        meta_list: List[torch.Tensor],  # per-batch visit meta tensors
        cur_nodes_list: List[List[int]],  # per-batch current node lists
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N1, _ = node_feat.shape

        # ── Pass 1: encode nodes ──────────────────────────────────────
        Z_node, g_node = self.node_encoder(node_feat, l_idx, b_idx)
        # Z_node: (B, N+1, D),  g_node: (B, D)

        # ── Pass 2: encode vehicles using Z_node as context ───────────
        Z_veh_list = []
        for b in range(B):
            Z_veh_b = self.vehicle_encoder(
                Z_node_full=Z_node[b],  # (N+1, D)
                Z_node_batch=Z_node[b].unsqueeze(0),  # (1, N+1, D)
                veh_feat=veh_feat[b],  # (2K, VEH_FEAT_DIM)
                edge_index=edge_indices[b],
                edge_attr=edge_attrs[b],
                visited=visited_list[b],
                meta=meta_list[b],
                cur_nodes=cur_nodes_list[b],
            )
            Z_veh_list.append(Z_veh_b)
        Z_veh = torch.stack(Z_veh_list, dim=0)  # (B, 2K, D)
        g_veh = Z_veh.mean(dim=1)  # (B, D)

        # ── Pass 3: refine nodes by attending to vehicles ─────────────
        # Z_node (B, N+1, D) queries Z_veh (B, 2K, D)
        node_veh_ctx = self.node_to_veh_attn(Z_node, Z_veh, Z_veh)  # (B, N+1, D)
        Z_node = self.node_refine_norm(Z_node + node_veh_ctx)
        Z_node = torch.nan_to_num(Z_node, nan=0.0)
        g_node = Z_node.mean(dim=1)  # (B, D)  recompute after refinement

        return Z_node, g_node, Z_veh, g_veh


# ---------------------------------------------------------------------------
# NodeSelectionDecoder  (upper pointer)
# ---------------------------------------------------------------------------


class NodeSelectionDecoder(nn.Module):
    """
    Selects which node n* to serve next.

    Query  = f(g_node, g_veh)  — a context vector summarising both
             the global node state and the current fleet.
    Keys   = W_k(Z_node)       — one key per node.
    Logits = clip_logits × tanh( Q·K^T / sqrt(D) )

    The tanh clip prevents logits from saturating early in training,
    which would produce near-zero gradients and slow learning.
    """

    def __init__(self, D: int, clip_logits: float):
        super().__init__()
        self.clip_logits = clip_logits
        self.context_proj = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU())
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self._scale = None  # set lazily from D

    def forward(
        self,
        g_node: torch.Tensor,  # (B, D)
        g_veh: torch.Tensor,  # (B, D)
        Z_node: torch.Tensor,  # (B, N+1, D)
        node_mask: Optional[torch.Tensor],  # (B, N+1) bool, True=feasible
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        D = Z_node.shape[-1]
        if self._scale is None:
            self._scale = math.sqrt(D)
        ctx = self.context_proj(torch.cat([g_node, g_veh], dim=-1))  # (B, D)
        query = self.Wq(ctx).unsqueeze(1)  # (B, 1, D)
        keys = self.Wk(Z_node)  # (B, N+1, D)
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / self._scale
        logits = self.clip_logits * torch.tanh(logits)
        if node_mask is not None:
            logits = logits.masked_fill(~node_mask, float("-inf"))
        return logits, ctx


# ---------------------------------------------------------------------------
# VehicleSelectionDecoder  (lower pointer)
# ---------------------------------------------------------------------------


class VehicleSelectionDecoder(nn.Module):
    """
    Selects which vehicle v* serves the chosen node n*.

    Explicitly conditioned on vehicle type: the scoring MLP receives
    [Z_veh_i | ctx | type_emb_i] so the network can learn distinct
    scoring preferences for trucks vs drones given the selected node.

    Context = f(g_veh, Z_node[n*])  — combines the fleet summary with
    the embedding of the node that was just selected.
    """

    def __init__(self, D: int):
        super().__init__()
        self.context_proj = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU())
        # Input: Z_veh(D) + context(D) + type_emb(D) → score(1)
        self.score_mlp = nn.Sequential(nn.Linear(D * 3, D), nn.ReLU(), nn.Linear(D, 1))

    def forward(
        self,
        g_veh: torch.Tensor,  # (B, D)
        Z_node_sel: torch.Tensor,  # (B, D)  embedding of selected node
        Z_veh: torch.Tensor,  # (B, 2K, D)
        veh_mask: Optional[torch.Tensor],  # (B, 2K) bool, True=feasible
        type_emb: torch.Tensor,  # (2K, D)  vehicle type embeddings
    ) -> torch.Tensor:  # (B, 2K)  logits
        B, V2K, D = Z_veh.shape
        ctx = self.context_proj(torch.cat([g_veh, Z_node_sel], dim=-1))  # (B, D)
        ctx_exp = ctx.unsqueeze(1).expand(B, V2K, D)  # (B, 2K, D)
        type_exp = type_emb.unsqueeze(0).expand(B, V2K, D)  # (B, 2K, D)
        logits = self.score_mlp(torch.cat([Z_veh, ctx_exp, type_exp], dim=-1)).squeeze(
            -1
        )  # (B, 2K)
        if veh_mask is not None:
            logits = logits.masked_fill(~veh_mask, float("-inf"))
        return logits


# ---------------------------------------------------------------------------
# PolicyNetwork  (top-level, implements BaseNetwork)
# ---------------------------------------------------------------------------


class PolicyNetwork(BaseNetwork):
    """
    HACN policy network for VRPBTW.

    Wires HierarchicalEncoder → NodeSelectionDecoder →
    VehicleSelectionDecoder into the BaseNetwork interface.

    The value head estimates V(s) from the concatenated global
    summaries [g_node | g_veh], giving it access to both the
    remaining node structure and the current fleet state.
    """

    def __init__(self, obs_shape: Tuple[int, ...], cfg, n_fleets: int = 2):
        super().__init__()
        self.cfg = cfg
        self.obs_shape = obs_shape
        self.K = n_fleets

        D = cfg.embed_dim
        H = cfg.n_heads
        L = cfg.n_encoder_layers
        drop = cfg.dropout
        use_in = getattr(cfg, "use_instance_norm", True)

        self.encoder = HierarchicalEncoder(
            D=D,
            K=n_fleets,
            n_heads=H,
            n_node_layers=L,
            n_gnn_layers=2,
            dropout=drop,
            use_instance_norm=use_in,
        )
        self.node_decoder = NodeSelectionDecoder(D=D, clip_logits=cfg.clip_logits)
        self.vehicle_decoder = VehicleSelectionDecoder(D=D)
        self.value_head = nn.Sequential(nn.Linear(D * 2, D), nn.Tanh(), nn.Linear(D, 1))

        if cfg.ortho_init:
            self._ortho_init(self)

    # ------------------------------------------------------------------
    # Batch preparation helpers
    # ------------------------------------------------------------------

    def _prep_obs(self, obs: Dict, device: str):
        """Unpack obs dict, add batch dim if needed, move to device."""
        nf = torch.FloatTensor(obs["node_features"]).to(device)
        vf = torch.FloatTensor(obs["vehicle_features"]).to(device)
        ei = obs["edge_index"]
        ea = obs["edge_attr"]
        ef = obs["edge_fleet"]
        if nf.dim() == 2:
            nf, vf = nf.unsqueeze(0), vf.unsqueeze(0)
            ei, ea, ef = [ei], [ea], [ef]
        ei_t = [torch.LongTensor(np.array(e, dtype=np.int64)).to(device) for e in ei]
        ea_t = [torch.FloatTensor(np.array(a, dtype=np.float32)).to(device) for a in ea]
        ef_t = [torch.LongTensor(np.array(f, dtype=np.int64)).to(device) for f in ef]
        return nf, vf, ei_t, ea_t, ef_t

    def _node_type_indices(self, node_feat: torch.Tensor):
        """Return linehaul and backhaul node index tensors from demand sign."""
        demand = node_feat[0, :, 2]
        return torch.where(demand > 0)[0], torch.where(demand < 0)[0]

    @staticmethod
    def _visited_and_meta(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_fleet: torch.Tensor,
        K: int,
        device: str,
    ) -> Tuple[List[int], torch.Tensor]:
        """Build visited-node list and metadata tensor from route graph."""
        if edge_index.shape[1] == 0:
            return [], torch.zeros(0, 1 + K + 1, device=device)
        all_nodes = sorted(torch.cat([edge_index[0], edge_index[1]]).unique().tolist())
        meta = torch.zeros(len(all_nodes), 1 + K + 1, device=device)
        local_map = {int(g): l for l, g in enumerate(all_nodes)}
        for i, dst_g in enumerate(edge_index[1].tolist()):
            dst_g = int(dst_g)
            if dst_g not in local_map:
                continue
            li = local_map[dst_g]
            meta[li, 0] = float(edge_attr[i, 4].item())
            fleet_id = int(edge_fleet[i].item())
            if 0 <= fleet_id < K:
                meta[li, 1 + fleet_id] = 1.0
            meta[li, 1 + K] = float(edge_attr[i, 0].item())
        return all_nodes, meta

    def _current_nodes(self, vf: torch.Tensor, b: int, N1: int, V2K: int) -> List[int]:
        """Decode current node indices from normalised vehicle x-coordinates."""
        return [int(round(float(vf[b, vi, 0].item()) * (N1 - 1))) for vi in range(V2K)]

    # ------------------------------------------------------------------
    # Action mask reshaping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_mask(action_mask: torch.Tensor, N1: int, V2K: int) -> torch.Tensor:
        return action_mask.view(action_mask.shape[0], N1, V2K).any(dim=-1)

    @staticmethod
    def _veh_mask_for_node(
        action_mask: torch.Tensor,
        node_id: torch.Tensor,
        N1: int,
        V2K: int,
    ) -> torch.Tensor:
        B = action_mask.shape[0]
        m = action_mask.view(B, N1, V2K)
        idx = node_id.view(B, 1, 1).expand(B, 1, V2K)
        return m.gather(1, idx).squeeze(1)

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def _encode(self, obs: Dict, device: str):
        """Run HierarchicalEncoder and return all embeddings + shape info."""
        nf, vf, ei_t, ea_t, ef_t = self._prep_obs(obs, device)
        B, N1, _ = nf.shape
        V2K = vf.shape[1]
        l_idx, b_idx = self._node_type_indices(nf)

        visited_list, meta_list, cur_nodes_list = [], [], []
        for b in range(B):
            vis, meta = self._visited_and_meta(
                ei_t[b], ea_t[b], ef_t[b], self.K, device
            )
            visited_list.append(vis)
            meta_list.append(meta)
            cur_nodes_list.append(self._current_nodes(vf, b, N1, V2K))

        Z_node, g_node, Z_veh, g_veh = self.encoder(
            node_feat=nf,
            l_idx=l_idx,
            b_idx=b_idx,
            veh_feat=vf,
            edge_indices=ei_t,
            edge_attrs=ea_t,
            edge_fleets=ef_t,
            visited_list=visited_list,
            meta_list=meta_list,
            cur_nodes_list=cur_nodes_list,
        )

        # Type embeddings — consistent with VehicleEncoder's type_embed
        t_ids = self.encoder.vehicle_encoder.type_ids(torch.device(device))
        type_emb = self.encoder.vehicle_encoder.type_embed(t_ids)  # (2K, D)

        return Z_node, g_node, Z_veh, g_veh, N1, V2K, type_emb

    # ------------------------------------------------------------------
    # BaseNetwork interface
    # ------------------------------------------------------------------

    def forward(self, obs: Dict, action_mask=None, context=None):
        device = next(self.parameters()).device.type
        Z_node, g_node, Z_veh, g_veh, N1, V2K, type_emb = self._encode(obs, device)

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        logits_U, _ = self.node_decoder(g_node, g_veh, Z_node, n_mask)
        flat = self._build_flat_logits(
            logits_U, Z_node, g_veh, Z_veh, action_mask, N1, V2K, type_emb
        )
        value = self.value_head(
            torch.cat([g_node.detach(), g_veh.detach()], dim=-1)
        ).squeeze(-1)
        return flat, value

    def get_action_and_log_prob(
        self, obs: Dict, action_mask=None, context=None, deterministic=False
    ):
        device = next(self.parameters()).device.type
        Z_node, g_node, Z_veh, g_veh, N1, V2K, type_emb = self._encode(obs, device)
        B = Z_node.shape[0]

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        logits_U, _ = self.node_decoder(g_node, g_veh, Z_node, n_mask)
        if not torch.isfinite(logits_U).any():
            logits_U = torch.zeros_like(logits_U)
        dist_U = torch.distributions.Categorical(logits=logits_U)
        node_id = logits_U.argmax(-1) if deterministic else dist_U.sample()
        lp_node = dist_U.log_prob(node_id)

        Z_node_sel = Z_node[torch.arange(B, device=device), node_id]
        v_mask = (
            self._veh_mask_for_node(action_mask, node_id, N1, V2K)
            if action_mask is not None
            else None
        )
        logits_L = self.vehicle_decoder(g_veh, Z_node_sel, Z_veh, v_mask, type_emb)
        if not torch.isfinite(logits_L).any():
            logits_L = torch.zeros_like(logits_L)
        dist_L = torch.distributions.Categorical(logits=logits_L)
        veh_id = logits_L.argmax(-1) if deterministic else dist_L.sample()
        lp_veh = dist_L.log_prob(veh_id)

        value = self.value_head(
            torch.cat([g_node.detach(), g_veh.detach()], dim=-1)
        ).squeeze(-1)
        return node_id * V2K + veh_id, lp_node + lp_veh, value

    def evaluate_actions(
        self, obs: Dict, actions: torch.Tensor, action_mask=None, context=None
    ):
        device = next(self.parameters()).device.type
        Z_node, g_node, Z_veh, g_veh, N1, V2K, type_emb = self._encode(obs, device)
        B = Z_node.shape[0]

        node_id = actions // V2K
        veh_id = actions % V2K

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        logits_U, _ = self.node_decoder(g_node, g_veh, Z_node, n_mask)
        if not torch.isfinite(logits_U).any():
            logits_U = torch.zeros_like(logits_U)
        dist_U = torch.distributions.Categorical(logits=logits_U)
        lp_node = dist_U.log_prob(node_id)
        ent_U = dist_U.entropy()

        Z_node_sel = Z_node[torch.arange(B, device=device), node_id]
        v_mask = (
            self._veh_mask_for_node(action_mask, node_id, N1, V2K)
            if action_mask is not None
            else None
        )
        logits_L = self.vehicle_decoder(g_veh, Z_node_sel, Z_veh, v_mask, type_emb)
        if not torch.isfinite(logits_L).any():
            logits_L = torch.zeros_like(logits_L)
        dist_L = torch.distributions.Categorical(logits=logits_L)
        lp_veh = dist_L.log_prob(veh_id)
        ent_L = dist_L.entropy()

        value = self.value_head(
            torch.cat([g_node.detach(), g_veh.detach()], dim=-1)
        ).squeeze(-1)
        return lp_node + lp_veh, value, ent_U + ent_L

    def _build_flat_logits(
        self, logits_U, Z_node, g_veh, Z_veh, action_mask, N1, V2K, type_emb
    ) -> torch.Tensor:
        B = logits_U.shape[0]
        flat = torch.full((B, N1 * V2K), float("-inf"), device=logits_U.device)
        for n in range(N1):
            Z_n = Z_node[:, n, :]
            v_mask = None
            if action_mask is not None:
                nid_t = torch.full((B,), n, dtype=torch.long, device=logits_U.device)
                v_mask = self._veh_mask_for_node(action_mask, nid_t, N1, V2K)
            l_L = self.vehicle_decoder(g_veh, Z_n, Z_veh, v_mask, type_emb)
            for v in range(V2K):
                flat[:, n * V2K + v] = logits_U[:, n] + l_L[:, v]
        if action_mask is not None:
            flat = flat.masked_fill(~action_mask, float("-inf"))
        return flat
