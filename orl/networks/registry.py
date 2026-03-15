"""
networks/registry.py
--------------------
Factory for building networks from config.

Adding a new network
--------------------
Register it here.  Nothing else needs to change.

    from networks.my_network import MyNetwork
    NETWORK_REGISTRY["mynet"] = MyNetwork
"""

from __future__ import annotations

from typing import Tuple

from configs import NetworkConfig
from .attention_network import AttentionNetwork
from .pointer_network import PointerNetwork
from .base_network import BaseNetwork


def build_network(
    obs_shape: Tuple[int, ...],
    action_space_size: int,
    cfg: NetworkConfig,
) -> BaseNetwork:
    """
    Instantiate a network from config.

    Parameters
    ----------
    obs_shape        : From problem.observation_shape — source of truth.
    action_space_size: From problem.action_space_size — source of truth.
    cfg              : NetworkConfig with network_type field.

    Returns
    -------
    BaseNetwork instance (not yet on any device).
    """
    t = cfg.network_type.lower()

    if t in ("attention", "transformer", "mlp"):
        return AttentionNetwork(obs_shape, action_space_size, cfg)

    if t == "pointer":
        if len(obs_shape) < 2:
            raise ValueError(
                "PointerNetwork requires 2D observations (N, feat_dim). "
                f"Got obs_shape={obs_shape}."
            )
        feat_dim = obs_shape[-1]
        max_nodes = obs_shape[0]
        return PointerNetwork(feat_dim, max_nodes, cfg)

    if t in ("hacn", "heterogeneous"):
        if len(obs_shape) < 2:
            raise ValueError(
                "HACNNetwork requires 2D observations (N, feat_dim). "
                f"Got obs_shape={obs_shape}."
            )
        from .hacn_network import HACNNetwork

        # n_vehicles must be passed; infer as action_space_size // obs_shape[0] // 2 * 2
        # Caller must supply n_vehicles kwarg via cfg or extra arg.
        # We derive it from action_space_size and N+1:
        N1 = obs_shape[0]
        n_vehicles = action_space_size // N1  # = 2K
        return HACNNetwork(obs_shape, action_space_size, n_vehicles, cfg)

    raise ValueError(
        f"Unknown network_type={t!r}. "
        f"Available: 'attention', 'transformer', 'mlp', 'pointer', 'hacn'."
    )
