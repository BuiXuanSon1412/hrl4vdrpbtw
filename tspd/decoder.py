import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MGUCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:

        combined = torch.cat([h, x], dim=-1)  # [B, input_size + hidden_size]

        # Forget gate
        f = torch.sigmoid(self.W_f(combined))

        # Candidate using gated previous hidden
        combined_gated = torch.cat([f * h, x], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_gated))

        # Final hidden state
        h_new = (1 - f) * h + f * h_tilde
        return h_new


class MGULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList()
        for l in range(n_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(MGUCell(in_size, hidden_size))

    def forward(self,x: torch.Tensor,h: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        device = x.device

        if h is None:
            h = torch.zeros(B, self.n_layers, self.hidden_size, device=device)

        h_new = []
        curr_input = x

        for l, cell in enumerate(self.cells):
            h_l = h[:, l]  # [B, hidden_size]
            h_l_new = cell(curr_input, h_l)
            h_new.append(h_l_new)
            curr_input = h_l_new

        h_new = torch.stack(h_new, dim=1)  # [B, n_layers, hidden_size]
        output = curr_input  # Last layer output

        return output, h_new


class AttentionModule(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Linear projections for dynamic, static, and query
        self.proj_dynamic = nn.Linear(d_model, d_model)
        self.proj_static = nn.Linear(d_model, d_model)
        self.proj_query = nn.Linear(d_model, d_model)

        # Learnable scoring vector v ∈ R^{D_h}
        self.v = nn.Parameter(torch.randn(d_model))

        # Scaling factor C (learnable, initialized to 10 as common in TSP literature)
        self.C = nn.Parameter(torch.tensor(10.0))

    def forward(self,E_static: torch.Tensor,E_dynamic: torch.Tensor,r: torch.Tensor,mask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = E_static.shape

        # Project all to same space
        e = self.proj_dynamic(E_dynamic)          # [B, N, D]
        d = self.proj_static(E_static)            # [B, N, D]
        q = self.proj_query(r).unsqueeze(1)       # [B, 1, D] (broadcast)

        # Attention scores (Eq. 21)
        # u_i = v^T * tanh(e_i + d_i + q)
        combined = torch.tanh(e + d + q)          # [B, N, D]
        u = (combined * self.v).sum(dim=-1)        # [B, N]

        # Scaled tanh logits (Eq. 23)
        logits = self.C * torch.tanh(u)            # [B, N]

        # Apply action mask
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))

        # Softmax probabilities (Eq. 24)
        probs = F.softmax(logits, dim=-1)

        return logits, probs


class DynamicEmbedding(nn.Module):

    def __init__(self, d_input: int = 4, d_model: int = 128):
        super().__init__()
        self.embed = nn.Linear(d_input, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class TSPDDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_mgu_layers: int = 2,
        d_dynamic: int = 4,      # Dynamic feature dimension
        n_nodes: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = n_nodes

        # Dynamic feature embedding
        self.dynamic_embed = DynamicEmbedding(d_dynamic, d_model)

        # Decoder input projection: combine static context + dynamic info
        self.input_proj = nn.Linear(d_model * 2, d_model)

        # MGU recurrent layers (Eq. 20)
        self.mgu = MGULayer(d_model, d_model, n_mgu_layers)

        # Separate attention heads for truck and drone
        self.truck_attention = AttentionModule(d_model)
        self.drone_attention = AttentionModule(d_model)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_context(self, E_static: torch.Tensor, vehicle_locs: torch.Tensor) -> torch.Tensor:
        B, N, D = E_static.shape
        # Gather embeddings at vehicle locations
        idx = vehicle_locs.unsqueeze(-1).expand(-1, D)  # [B, D]
        context = E_static.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B, D]
        return context

    def forward_step(
        self,
        E_static: torch.Tensor,
        dynamic_features: torch.Tensor,
        truck_loc: torch.Tensor,
        drone_loc: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        truck_mask: torch.Tensor,
        drone_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            E_static: [B, N, D] - static encoder output
            dynamic_features: [B, N, d_dynamic] - current dynamic state
            truck_loc: [B] - current truck node
            drone_loc: [B] - current drone node
            h_prev: [B, L, D] - previous MGU hidden states
            truck_mask: [B, N] - invalid truck actions
            drone_mask: [B, N] - invalid drone actions
        Returns:
            truck_logits: [B, N]
            truck_probs: [B, N]
            drone_logits: [B, N]
            drone_probs: [B, N]
            h_new: [B, L, D]
        """
        B, N, D = E_static.shape

        # Embed dynamic features
        E_dynamic = self.dynamic_embed(dynamic_features)  # [B, N, D]

        # Build decoder input from truck and drone contexts
        truck_ctx = self._get_context(E_static, truck_loc)   # [B, D]
        drone_ctx = self._get_context(E_static, drone_loc)   # [B, D]

        # Combined context
        combined_ctx = (truck_ctx + drone_ctx) / 2           # [B, D]

        # Global dynamic context (mean of dynamic embeddings)
        global_dynamic = E_dynamic.mean(dim=1)               # [B, D]

        # Decoder input
        dec_input = self.input_proj(
            torch.cat([combined_ctx, global_dynamic], dim=-1)
        )  # [B, D]

        # MGU forward (Eq. 20)
        r, h_new = self.mgu(dec_input, h_prev)               # [B, D], [B, L, D]

        # Truck action selection (Eq. 21-24)
        truck_logits, truck_probs = self.truck_attention(
            E_static, E_dynamic, r, truck_mask
        )

        # Drone action selection
        drone_logits, drone_probs = self.drone_attention(
            E_static, E_dynamic, r, drone_mask
        )

        return truck_logits, truck_probs, drone_logits, drone_probs, h_new

    def forward(
        self,
        E_static: torch.Tensor,
        env,
        state,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full decoding loop - generate complete solution sequence.
        
        Args:
            E_static: [B, N, D] - encoder output
            env: TSPDEnvironment
            state: Initial TSPDState
            greedy: If True, use greedy selection; else sample
        Returns:
            total_reward: [B] - total negative cost (sum of rewards)
            log_probs: [B] - sum of log probabilities for REINFORCE
            actions: list of (truck_action, drone_action) pairs
        """
        B = E_static.shape[0]
        device = E_static.device

        h = None  # MGU hidden states
        total_reward = torch.zeros(B, device=device)
        log_probs = torch.zeros(B, device=device)
        actions_list = []

        max_steps = env.n_nodes * 3  # Upper bound on steps

        for step in range(max_steps):
            # Get dynamic features
            dynamic_feat = env.get_dynamic_features(state)  # [B, N, 4]

            # Get action masks
            truck_mask, drone_mask = env.get_action_mask(state)

            # Forward step
            truck_logits, truck_probs, drone_logits, drone_probs, h = self.forward_step(
                E_static,
                dynamic_feat,
                state.truck_loc,
                state.drone_loc,
                h,
                truck_mask,
                drone_mask,
            )

            # Action selection
            if greedy:
                truck_act = truck_probs.argmax(dim=-1)  # [B]
                drone_act = drone_probs.argmax(dim=-1)  # [B]
            else:
                # Temperature sampling
                if temperature != 1.0:
                    truck_probs_t = F.softmax(truck_logits / temperature, dim=-1)
                    drone_probs_t = F.softmax(drone_logits / temperature, dim=-1)
                else:
                    truck_probs_t = truck_probs
                    drone_probs_t = drone_probs

                # Handle NaN/Inf in probabilities
                truck_probs_t = torch.nan_to_num(truck_probs_t, nan=0.0)
                drone_probs_t = torch.nan_to_num(drone_probs_t, nan=0.0)

                # Normalize
                truck_sum = truck_probs_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                drone_sum = drone_probs_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                truck_probs_t = truck_probs_t / truck_sum
                drone_probs_t = drone_probs_t / drone_sum

                truck_dist = torch.distributions.Categorical(truck_probs_t)
                drone_dist = torch.distributions.Categorical(drone_probs_t)
                truck_act = truck_dist.sample()
                drone_act = drone_dist.sample()

            # Compute log probabilities
            truck_log_p = torch.log(truck_probs.gather(1, truck_act.unsqueeze(1)).squeeze(1) + 1e-8)
            drone_log_p = torch.log(drone_probs.gather(1, drone_act.unsqueeze(1)).squeeze(1) + 1e-8)
            log_probs = log_probs + truck_log_p + drone_log_p

            actions_list.append((truck_act, drone_act))

            # Step environment
            state, reward, done = env.step(state, truck_act, drone_act)
            total_reward = total_reward + reward

            # Check if all done
            if done.all():
                break

        return total_reward, log_probs, actions_list