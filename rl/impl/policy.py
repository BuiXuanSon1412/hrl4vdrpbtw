"""
impl/network.py
-----------------------
Heterogeneous Graph Neural Network Policy (HGNN-Policy) for VRPBTW.

The network receives a fully pre-processed obs dict from
VRPBTWProblem.state_to_obs — all arrays are already normalised and the
static compatibility graph is already built.  No preprocessing happens
inside this file.

Obs dict keys (all numpy arrays, produced by state_to_obs):
  node_features     (B, N+1, 5)
  vehicle_features  (B, 2K,  5)
  truck_edge_index  (B, 2, E)   or  (2, E)  — shared across batch
  truck_edge_attr   (B, E, 2)   or  (E, 2)
  drone_edge_index  (B, 2, E)   or  (2, E)
  drone_edge_attr   (B, E, 2)   or  (E, 2)

Architecture
============
Encoder
  NodeEncoder      heterogeneous self-attention (linehaul ↔ backhaul)
                   → Z_node (B, N+1, D),  g_node (B, D)
  VehicleEncoder   self-attention + cross-attention to Z_node
                   → Z_veh  (B, 2K,  D),  g_veh  (B, D)
  GraphEncoder     3-layer multi-relational GNN (truck / drone relations)
                   → Z_graph (B, N+1, D),  g_graph (B, D)

Decoder  (bilevel, two independent heads)
  NodeDecoder      query = f(g_node, mean(Z_veh), g_graph)
                   dot-product attention + tanh clip over Z_node
  VehicleDecoder   query = f(g_veh, mean(Z_node), g_graph)
                   dot-product attention + tanh clip over Z_veh
  Sampling         rejection sampling (up to MAX_REJECTION_TRIES),
                   fallback to masked-flat argmax

Value head  MLP(g_node ‖ g_veh ‖ g_graph)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.policy import BasePolicy, _MHA, _FF, _make_norm
from impl.environment import NODE_FEAT_DIM, VEH_FEAT_DIM

GRAPH_EDGE_DIM = 2  # [cost, time]
GRAPH_NODE_DIM = 4  # [x, y, tw_open, tw_close] — demand excluded; GraphEncoder
# captures routing structure, not delivery semantics


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# NodeEncoder
# ---------------------------------------------------------------------------


class _HeteroNodeLayer(nn.Module):
    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        self.sa = _MHA(D, H, dropout)
        self.xl2b = _MHA(D, H, dropout)
        self.xb2l = _MHA(D, H, dropout)
        self.ff = _FF(D, dropout)
        self.norm1 = _make_norm(use_in, D)
        self.norm2 = _make_norm(use_in, D)

    def forward(
        self, h: torch.Tensor, l_idx: torch.Tensor, b_idx: torch.Tensor
    ) -> torch.Tensor:
        h_sa = self.sa(h, h, h)
        h_het = torch.zeros_like(h)
        if l_idx.numel() > 0 and b_idx.numel() > 0:
            h_l = h[:, l_idx]
            h_b = h[:, b_idx]
            h_het[:, l_idx] = self.xl2b(h_l, h_b, h_b)
            h_het[:, b_idx] = self.xb2l(h_b, h_l, h_l)
        h = self.norm1(h + h_sa + h_het)
        h = torch.nan_to_num(h, nan=0.0)
        h = self.norm2(h + self.ff(h))
        h = torch.nan_to_num(h, nan=0.0)
        return h


class NodeEncoder(nn.Module):
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
                _HeteroNodeLayer(D, n_heads, dropout, use_instance_norm)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, node_feat: torch.Tensor, l_idx: torch.Tensor, b_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        for layer in self.layers:
            h = layer(h, l_idx, b_idx)
        return h, h.mean(dim=1)


# ---------------------------------------------------------------------------
# VehicleEncoder
# ---------------------------------------------------------------------------


class _VehicleEncoderLayer(nn.Module):
    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        self.sa = _MHA(D, H, dropout)
        self.xa = _MHA(D, H, dropout)
        self.ff = _FF(D, dropout)
        self.norm1 = _make_norm(use_in, D)
        self.norm2 = _make_norm(use_in, D)
        self.norm3 = _make_norm(use_in, D)

    def forward(self, h: torch.Tensor, Z_node: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.sa(h, h, h))
        h = torch.nan_to_num(h, nan=0.0)
        h = self.norm2(h + self.xa(h, Z_node, Z_node))
        h = torch.nan_to_num(h, nan=0.0)
        h = self.norm3(h + self.ff(h))
        h = torch.nan_to_num(h, nan=0.0)
        return h


class VehicleEncoder(nn.Module):
    def __init__(
        self,
        D: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_instance_norm: bool,
    ):
        super().__init__()
        self.truck_proj = nn.Linear(VEH_FEAT_DIM, D)
        self.drone_proj = nn.Linear(VEH_FEAT_DIM, D)
        self.layers = nn.ModuleList(
            [
                _VehicleEncoderLayer(D, n_heads, dropout, use_instance_norm)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, veh_feat: torch.Tensor, Z_node: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V2K, _ = veh_feat.shape
        K = V2K // 2

        h = torch.cat(
            [
                self.truck_proj(veh_feat[:, :K]),
                self.drone_proj(veh_feat[:, K:]),
            ],
            dim=1,
        )

        for layer in self.layers:
            h = layer(h, Z_node)
        return h, h.mean(dim=1)


# ---------------------------------------------------------------------------
# GraphEncoder — multi-relational GNN
# ---------------------------------------------------------------------------


class _MRGNNLayer(nn.Module):
    """
    One multi-relational GNN layer (truck + drone relations).

    For each relation:
      msg(i→j) = MLP( h_i ‖ edge_feat_{ij} )
      agg(j)   = mean over incoming messages

    Update:
      h' = LayerNorm( h + proj( cat[agg_truck, agg_drone] ) )
    """

    def __init__(self, D: int, edge_feat_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_truck = _mlp(D + edge_feat_dim, D * 2, D, dropout)
        self.msg_drone = _mlp(D + edge_feat_dim, D * 2, D, dropout)
        self.update = nn.Linear(D * 2, D)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def _pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        """
        h          : (B, N, D)
        edge_index : (2, E)
        edge_attr  : (E, ef)   or (B, E, ef)
        returns      (B, N, D)  mean-aggregated messages
        """
        B, N, D = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        h_src = h[:, src, :]  # (B, E, D)
        if edge_attr.dim() == 2:
            ea = edge_attr.unsqueeze(0).expand(B, -1, -1)  # (B, E, ef)
        else:
            ea = edge_attr
        msg = mlp(torch.cat([h_src, ea], dim=-1))  # (B, E, D)

        agg = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        count = torch.zeros(N, device=h.device, dtype=h.dtype)
        dst_e = dst.unsqueeze(0).unsqueeze(-1).expand(B, E, D)
        agg.scatter_add_(1, dst_e, msg)
        count.scatter_add_(0, dst, torch.ones(E, device=h.device))
        count = count.clamp(min=1.0).view(1, N, 1)
        return agg / count

    def forward(
        self,
        h: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
    ) -> torch.Tensor:
        agg_t = self._pass(h, t_ei, t_ea, self.msg_truck)
        agg_d = self._pass(h, d_ei, d_ea, self.msg_drone)
        upd = self.drop(self.update(torch.cat([agg_t, agg_d], dim=-1)))
        return self.norm(h + upd)


class GraphEncoder(nn.Module):
    def __init__(self, D: int, n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(GRAPH_NODE_DIM, D)
        self.layers = nn.ModuleList(
            [_MRGNNLayer(D, GRAPH_EDGE_DIM, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(D)

    def forward(
        self,
        node_feat: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        for layer in self.layers:
            h = layer(h, t_ei, t_ea, d_ei, d_ea)
        h = self.out_norm(h)
        return h, h.mean(dim=1)


# ---------------------------------------------------------------------------
# NodeDecoder / VehicleDecoder
# ---------------------------------------------------------------------------


class NodeDecoder(nn.Module):
    def __init__(self, D: int, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.ctx_proj = nn.Sequential(nn.Linear(D * 3, D), nn.ReLU())
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self._scale: Optional[float] = None

    def forward(
        self,
        g_node: torch.Tensor,
        Z_veh: torch.Tensor,
        g_graph: torch.Tensor,
        Z_node: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._scale is None:
            self._scale = math.sqrt(Z_node.shape[-1])
        z_veh_mean = Z_veh.mean(1)
        ctx = self.ctx_proj(torch.cat([g_node, z_veh_mean, g_graph], dim=-1))
        Q = self.Wq(ctx).unsqueeze(1)
        logits = torch.bmm(Q, self.Wk(Z_node).transpose(1, 2)).squeeze(1) / self._scale
        logits = self.clip * torch.tanh(logits)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class VehicleDecoder(nn.Module):
    def __init__(self, D: int, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.ctx_proj = nn.Sequential(nn.Linear(D * 3, D), nn.ReLU())
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self._scale: Optional[float] = None

    def forward(
        self,
        g_veh: torch.Tensor,
        Z_node: torch.Tensor,
        g_graph: torch.Tensor,
        Z_veh: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._scale is None:
            self._scale = math.sqrt(Z_veh.shape[-1])
        z_node_mean = Z_node.mean(1)
        ctx = self.ctx_proj(torch.cat([g_veh, z_node_mean, g_graph], dim=-1))
        Q = self.Wq(ctx).unsqueeze(1)
        logits = torch.bmm(Q, self.Wk(Z_veh).transpose(1, 2)).squeeze(1) / self._scale
        logits = self.clip * torch.tanh(logits)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


# ---------------------------------------------------------------------------
# PolicyNetwork
# ---------------------------------------------------------------------------


class HGNNPolicy(BasePolicy):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Extract encoder and decoder configs
        encoder_cfg = cfg.get("encoder", {})
        decoder_cfg = cfg.get("decoder", {})
        node_enc_cfg = encoder_cfg.get("node_encoder", {})
        veh_enc_cfg = encoder_cfg.get("vehicle_encoder", {})
        graph_enc_cfg = encoder_cfg.get("graph_encoder", {})
        node_dec_cfg = decoder_cfg.get("node_decoder", {})

        # Read directly from current config structure
        D: int = int(node_enc_cfg.get("embedding_dim", 128))
        H: int = int(node_enc_cfg.get("n_heads", 4))
        drop: float = float(node_enc_cfg.get("dropout", 0.1))
        use_in: bool = bool(node_enc_cfg.get("use_instance_norm", True))
        clip: float = float(node_dec_cfg.get("clip", 10.0))

        # Encoder layers: read per-encoder n_layers
        n_node_layers: int = int(node_enc_cfg.get("n_layers", 3))
        n_veh_layers: int = int(veh_enc_cfg.get("n_layers", 3))
        n_graph_layers: int = int(graph_enc_cfg.get("n_layers", 3))

        self.node_encoder = NodeEncoder(D, H, n_node_layers, drop, use_in)
        self.vehicle_encoder = VehicleEncoder(D, H, n_veh_layers, drop, use_in)
        self.graph_encoder = GraphEncoder(D, n_graph_layers, drop)
        self.node_decoder = NodeDecoder(D, clip)
        self.vehicle_decoder = VehicleDecoder(D, clip)
        self.value_head = nn.Sequential(nn.Linear(D * 3, D), nn.Tanh(), nn.Linear(D, 1))

        ortho_init: bool = (
            cfg.get("regularization", {}).get("ortho_init", True)
            if isinstance(cfg.get("regularization"), dict)
            else cfg.get("ortho_init", True)
        )
        if ortho_init:
            self._ortho_init(self)

    @classmethod
    def from_config(cls, cfg: Dict) -> "HGNNPolicy":
        """Factory method: instantiate HGNNPolicy from config dict.

        Supports both old flat structure and new hierarchical structure with network key.
        """
        # Support both old (flat) and new (with network key) structure
        policy_cfg = cfg.get("network", cfg)
        return cls(cfg=policy_cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(x, device: str, dtype=torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def _node_mask(action_mask: torch.Tensor, N1: int, V2K: int) -> torch.Tensor:
        return action_mask.view(action_mask.shape[0], N1, V2K).any(dim=-1)

    @staticmethod
    def _veh_mask(action_mask: torch.Tensor, N1: int, V2K: int) -> torch.Tensor:
        return action_mask.view(action_mask.shape[0], N1, V2K).any(dim=1)

    # ------------------------------------------------------------------
    # Observation unpacking — arrays only, no preprocessing
    # ------------------------------------------------------------------

    def _unpack(self, obs: Dict, device: str):
        """
        Convert obs dict values to tensors on the correct device.
        All arrays are already normalised by state_to_obs.
        Adds batch dimension if missing.
        """
        t = lambda x, dt=torch.float32: self._to_tensor(x, device, dt)

        nf = t(obs["node_features"])  # (B, N+1, 5)  or (N+1, 5)
        vf = t(obs["vehicle_features"])  # (B, 2K,  5)  or (2K,  5)
        t_ei = t(obs["truck_edge_index"], torch.long)  # (2, E)
        t_ea = t(obs["truck_edge_attr"])  # (E, 2)
        d_ei = t(obs["drone_edge_index"], torch.long)  # (2, E)
        d_ea = t(obs["drone_edge_attr"])  # (E, 2)

        # Topology is shared across the batch; callers may still store it as
        # (1, 2, E) or (B, 2, E). Collapse that to the expected (2, E) form.
        if t_ei.dim() == 3:
            t_ei = t_ei[0]
        if d_ei.dim() == 3:
            d_ei = d_ei[0]

        if nf.dim() == 2:
            nf = nf.unsqueeze(0)
            vf = vf.unsqueeze(0)
        # edge tensors stay 2-D (shared across batch)

        return nf, vf, t_ei, t_ea, d_ei, d_ea

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, obs: Dict, device: str):
        nf, vf, t_ei, t_ea, d_ei, d_ea = self._unpack(obs, device)
        B, N1, _ = nf.shape
        V2K = vf.shape[1]

        demand = nf[0, :, 2]
        l_idx = torch.where(demand > 0)[0]
        b_idx = torch.where(demand < 0)[0]

        Z_node, g_node = self.node_encoder(nf, l_idx, b_idx)
        Z_veh, g_veh = self.vehicle_encoder(vf, Z_node)

        # GraphEncoder receives [x, y, tw_open, tw_close] only — demand is a
        # delivery semantic, not a routing-structure signal.  Dropping it keeps
        # the graph encoder focused on spatial-temporal topology.
        nf_gnn = torch.cat([nf[:, :, :2], nf[:, :, 3:]], dim=-1)  # (B, N+1, 4)
        Z_graph, g_graph = self.graph_encoder(nf_gnn, t_ei, t_ea, d_ei, d_ea)

        return Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K

    # ------------------------------------------------------------------
    # BaseNetwork interface
    # ------------------------------------------------------------------

    def forward(self, obs, action_mask=None, context=None):
        device = next(self.parameters()).device.type
        Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K = self._encode(
            obs, device
        )

        # Convert action_mask to torch tensor if it's numpy
        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            action_mask = torch.from_numpy(action_mask).to(device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        v_mask = (
            self._veh_mask(action_mask, N1, V2K) if action_mask is not None else None
        )

        ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)  # (B, N1)
        lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)  # (B, V2K)

        flat = (ln.unsqueeze(2) + lv.unsqueeze(1)).view(ln.shape[0], N1 * V2K)
        if action_mask is not None:
            flat = flat.masked_fill(~action_mask, float("-inf"))

        # Ensure all features are 2D (B, D) before concatenation
        g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)
        g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)
        g_graph_2d = g_graph if g_graph.dim() == 2 else g_graph.flatten(1)
        value = self.value_head(
            torch.cat([g_node_2d, g_veh_2d, g_graph_2d], dim=-1)
        ).squeeze(-1)
        return flat, value

    def get_action_and_log_prob(
        self, obs, action_mask=None, context=None, deterministic=False
    ):
        device = next(self.parameters()).device.type
        Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K = self._encode(
            obs, device
        )

        # Convert action_mask to torch tensor if it's numpy
        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            action_mask = torch.from_numpy(action_mask).to(device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        v_mask = (
            self._veh_mask(action_mask, N1, V2K) if action_mask is not None else None
        )

        ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)
        lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)

        # Joint logits over all (node, vehicle) pairs, masked to feasible pairs.
        # Sampling and log-prob use the single joint Categorical, so the policy
        # correctly represents p(node, vehicle | feasible) rather than the
        # factorized approximation p_node * p_vehicle which ignores infeasible pairs.
        B = ln.shape[0]
        flat_l = (ln.unsqueeze(2) + lv.unsqueeze(1)).view(B, N1 * V2K)
        if action_mask is not None:
            flat_l = flat_l.masked_fill(~action_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=flat_l)
        action = flat_l.argmax(-1) if deterministic else dist.sample()

        # Ensure all features are 2D (B, D) before concatenation
        g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)
        g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)
        g_graph_2d = g_graph if g_graph.dim() == 2 else g_graph.flatten(1)
        value = self.value_head(
            torch.cat([g_node_2d, g_veh_2d, g_graph_2d], dim=-1)
        ).squeeze(-1)
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions, action_mask=None, context=None):
        device = next(self.parameters()).device.type
        Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K = self._encode(
            obs, device
        )

        # Convert action_mask to torch tensor if it's numpy
        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            action_mask = torch.from_numpy(action_mask).to(device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        v_mask = (
            self._veh_mask(action_mask, N1, V2K) if action_mask is not None else None
        )

        ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)
        lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)

        # Same joint distribution as get_action_and_log_prob so that the
        # importance ratio in PPO is computed under the correct normalisation.
        flat_l = (ln.unsqueeze(2) + lv.unsqueeze(1)).view(ln.shape[0], N1 * V2K)
        if action_mask is not None:
            flat_l = flat_l.masked_fill(~action_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=flat_l)
        # Ensure all features are 2D (B, D) before concatenation
        g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)
        g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)
        g_graph_2d = g_graph if g_graph.dim() == 2 else g_graph.flatten(1)
        value = self.value_head(
            torch.cat([g_node_2d, g_veh_2d, g_graph_2d], dim=-1)
        ).squeeze(-1)
        return dist.log_prob(actions), value, dist.entropy()

    def compute_entropy(self, buffer, sample_size: int = 1):
        """
        Compute bilevel policy entropy on buffer data (sampled for efficiency).

        For bilevel policies (node + vehicle decoders):
          entropy = entropy_node + entropy_vehicle

        Lower entropy = more confident policy (ready for harder tasks).
        Higher entropy = more uncertain policy (needs more training).

        Args:
            buffer: RolloutBuffer with collected transitions
            sample_size: number of transitions to sample (default 32 for speed)

        Returns:
            scalar entropy value (sum of node and vehicle entropies)
        """
        if buffer._ptr == 0:
            return 0.0

        n_sample = min(sample_size, buffer._ptr)
        indices = np.random.choice(buffer._ptr, size=n_sample, replace=False)

        entropies_node = []
        entropies_vehicle = []

        with torch.no_grad():
            device = next(self.parameters()).device.type
            for i in indices:
                transition = buffer._data[i]
                obs = transition.obs
                mask = torch.tensor(transition.action_mask, dtype=torch.bool).unsqueeze(
                    0
                )

                Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K = self._encode(
                    obs, device
                )

                n_mask = self._node_mask(mask, N1, V2K) if mask is not None else None
                v_mask = self._veh_mask(mask, N1, V2K) if mask is not None else None

                ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)
                lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)

                dist_node = torch.distributions.Categorical(logits=ln)
                dist_vehicle = torch.distributions.Categorical(logits=lv)

                entropies_node.append(dist_node.entropy().mean().item())
                entropies_vehicle.append(dist_vehicle.entropy().mean().item())

        avg_entropy = (
            (float(np.mean(entropies_node)) + float(np.mean(entropies_vehicle)))
            if entropies_node
            else 0.0
        )
        return avg_entropy
