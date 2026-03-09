# src/models/low_level.py
"""Low-level policy stub — replace body with actual architecture."""

import torch.nn as nn
import torch


class LowLevelPolicy(nn.Module):
    def __init__(
        self,
        customer_dim,
        vehicle_dim,
        num_customers,
        embedding_dim,
        num_layers,
        num_heads,
        qkv_dim,
        ff_hidden_dim,
        logit_clipping,
    ):
        super().__init__()
        self.num_customers = num_customers
        # TODO: implement pointer-network / attention architecture
        self._customer_proj = nn.Linear(customer_dim, embedding_dim)
        self._vehicle_proj = nn.Linear(vehicle_dim, embedding_dim)
        self._score = nn.Linear(embedding_dim, 1)
        self._drone = nn.Linear(embedding_dim, 2)
        self._value = nn.Linear(embedding_dim, 1)

    def forward(self, customer_features, vehicle_context, mask=None):
        c = self._customer_proj(customer_features)  # (B, N, E)
        v = self._vehicle_proj(vehicle_context)  # (B, E)
        scores = self._score(c + v.unsqueeze(1)).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        v_mean = v
        drone = self._drone(v_mean)
        value = self._value(v_mean)
        return scores, drone, value
