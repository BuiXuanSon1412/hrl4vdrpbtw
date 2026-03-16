"""
networks/registry.py
--------------------
Factory for building networks from config.

Adding a new network
--------------------
Register it here. Nothing else needs to change.

    from networks.my_network import MyNetwork
    NETWORK_REGISTRY["mynet"] = MyNetwork

Note on AttentionNetwork
------------------------
AttentionNetwork has been removed from the registry. It required
action_space_size at construction time (for its fixed Linear decoder),
which couples it to a specific problem size. HACNNetwork is the correct
architecture for VRPBTW; for flat-obs problems like Knapsack the MLP
path inside HACNNetwork or a custom subclass should be used instead.
AttentionNetwork is kept in attention_network.py for reference only.
"""

from __future__ import annotations

from typing import Tuple

from configs import NetworkConfig
from .pointer_network import PointerNetwork
from .hacn_network import HACNNetwork
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
    obs_shape         : From problem.observation_shape — source of truth.
    action_space_size : From problem.action_space_size — used only here
                        in the registry to derive structural parameters
                        (e.g. n_vehicles for HACN). Never stored on the
                        network itself.
    cfg               : NetworkConfig with network_type field.

    Returns
    -------
    BaseNetwork instance (not yet on any device).
    """
    t = cfg.network_type.lower()

    if t in ("hacn", "heterogeneous", "attention", "transformer"):
        if len(obs_shape) < 2:
            raise ValueError(
                "HACNNetwork requires 2D observations (N+1, feat_dim). "
                f"Got obs_shape={obs_shape}."
            )
        return HACNNetwork(obs_shape, cfg)

    if t == "pointer":
        if len(obs_shape) < 2:
            raise ValueError(
                "PointerNetwork requires 2D observations (N, feat_dim). "
                f"Got obs_shape={obs_shape}."
            )
        feat_dim = obs_shape[-1]
        return PointerNetwork(feat_dim, cfg)

    if t == "mlp":
        # MLP path: flat obs, route through HACN with no encoder layers
        # (or raise if someone explicitly asks for it on a 2D problem)
        if len(obs_shape) == 1:
            raise ValueError(
                "MLP network requires a flat 1D observation. "
                "For graph problems use 'hacn'."
            )
        return HACNNetwork(obs_shape, cfg)

    raise ValueError(
        f"Unknown network_type={t!r}. "
        f"Available: 'hacn', 'heterogeneous', 'attention', 'transformer', 'pointer'."
    )
