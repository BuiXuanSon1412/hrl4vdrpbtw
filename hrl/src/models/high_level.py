# src/models/high_level.py
"""High-level policy stub — replace body with actual architecture."""

import torch.nn as nn


class HighLevelPolicy(nn.Module):
    def __init__(
        self, state_dim, num_vehicles, embedding_dim, num_heads, qkv_dim, ff_hidden_dim
    ):
        super().__init__()
        # TODO: implement attention-based encoder + vehicle selector
        self._placeholder = nn.Linear(state_dim, num_vehicles)
        self._value = nn.Linear(state_dim, 1)

    def forward(self, state):
        return self._placeholder(state), self._value(state)
