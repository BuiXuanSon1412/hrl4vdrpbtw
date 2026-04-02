import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from encoder import TSPDEncoder
from decoder import TSPDDecoder


class CriticNetwork(nn.Module):
    def __init__(self, d_model: int = 128, n_nodes: int = 20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, mean_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(mean_embedding).squeeze(-1)


class TSPDActorCritic(nn.Module):
    def __init__(
        self,
        n_nodes: int = 20,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        n_encoder_layers: int = 3,
        n_mgu_layers: int = 2,
        dropout: float = 0.1,
        d_sparse: int = 16,
        depot_idx: int = 0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model

        # Encoder (shared between actor and critic input)
        self.encoder = TSPDEncoder(
            d_input=2,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_encoder_layers,
            dropout=dropout,
            d_sparse=d_sparse,
            depot_idx=depot_idx,
        )

        # Actor decoder
        self.decoder = TSPDDecoder(
            d_model=d_model,
            n_mgu_layers=n_mgu_layers,
            d_dynamic=4,  # [demand, truck_pos, drone_pos, is_depot]
            n_nodes=n_nodes,
        )

        # Critic value network (Eq. 26)
        self.critic = CriticNetwork(d_model, n_nodes)

    def encode(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(coords)

    def forward(self,coords: torch.Tensor,env,state,greedy: bool = False,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        E_static, mean_embed = self.encode(coords)

        # Critic baseline estimate (Eq. 26): b(s) = V_φ(s)
        baseline = self.critic(mean_embed)

        # Decode (actor)
        total_reward, log_probs, actions = self.decoder.forward(
            E_static, env, state, greedy=greedy
        )

        return total_reward, log_probs, baseline

    def get_value(self, coords: torch.Tensor) -> torch.Tensor:
        _, mean_embed = self.encode(coords)
        return self.critic(mean_embed)
