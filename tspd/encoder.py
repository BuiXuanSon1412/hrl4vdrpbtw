import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RelativePositionalEncoding(nn.Module):
    def __init__(self, n_heads: int, d_sparse: int, d_k: int, n_buckets: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.d_sparse = d_sparse
        self.d_k = d_k
        self.n_buckets = n_buckets
        # Sparse positional embedding matrix R ∈ R^{H × D_sparse × D_k}
        self.embedding = nn.Embedding(n_buckets, n_heads * d_k)

    def forward(self, coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, E, _ = edge_index.shape
        H, Dk = self.n_heads, self.d_k

        # Compute pairwise distances and bucket them
        src = edge_index[:, :, 0]  # [B, E]
        dst = edge_index[:, :, 1]  # [B, E]

        # Gather coordinates
        src_coords = torch.gather(
            coords, 1, src.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, E, 2]
        dst_coords = torch.gather(
            coords, 1, dst.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, E, 2]

        dist = torch.norm(src_coords - dst_coords, dim=-1)  # [B, E]

        # Bucket distances into discrete bins
        dist_buckets = (dist * self.n_buckets).long().clamp(0, self.n_buckets - 1)

        # Look up embeddings
        r = self.embedding(dist_buckets)  # [B, E, H*Dk]
        r = r.view(B, E, H, Dk)
        return r


class ExpanderGraphBuilder:
    @staticmethod
    def build_knn_edges(h: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        B, N, D = h.shape
        if k is None:
            k = math.ceil(math.log2(N)) if N > 1 else 1
        k = min(k, N - 1)

        # Pairwise distances in latent space
        dists = torch.cdist(h, h)  # [B, N, N]
        dists[:, range(N), range(N)] = float('inf')  # No self-loops

        # Top-k nearest neighbors
        _, knn_idx = dists.topk(k, dim=-1, largest=False)  # [B, N, k]

        # Build edge list
        src = torch.arange(N, device=h.device).unsqueeze(0).unsqueeze(-1)
        src = src.expand(B, N, k)  # [B, N, k]

        edges = torch.stack([src, knn_idx], dim=-1)  # [B, N, k, 2]
        edges = edges.view(B, N * k, 2)
        return edges

    @staticmethod
    def build_hierarchical_edges(N: int, L: int = 8, device: str = "cpu") -> torch.Tensor:
        stride = max(1, int(math.sqrt(N / L)))
        edges = []
        layer_size = N // L if L <= N else 1

        for l in range(min(L, N)):
            start = l * layer_size
            end = min(start + layer_size, N)
            for i in range(start, end):
                for j in range(start, end):
                    if abs(i - j) <= stride and i != j:
                        edges.append([i, j])

        if not edges:
            return torch.zeros(0, 2, dtype=torch.long, device=device)
        return torch.tensor(edges, dtype=torch.long, device=device)

    @staticmethod
    def build_depot_edges(N: int, depot_idx: int = 0, device: str = "cpu") -> torch.Tensor:
        edges = [[depot_idx, i] for i in range(N) if i != depot_idx]
        edges += [[i, depot_idx] for i in range(N) if i != depot_idx]
        return torch.tensor(edges, dtype=torch.long, device=device)

    @classmethod
    def build_adjacency_mask(cls,h: torch.Tensor,depot_idx: int = 0,with_global_node: bool = True,) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = h.shape
        device = h.device
        N_total = N + 1 if with_global_node else N  # +1 for global node

        # Start with zeros (no connections)
        adj = torch.zeros(B, N_total, N_total, device=device)

        # 1. k-NN edges
        knn_edges = cls.build_knn_edges(h)  # [B, N*k, 2]
        for b in range(B):
            for e in range(knn_edges.shape[1]):
                i, j = knn_edges[b, e, 0].item(), knn_edges[b, e, 1].item()
                adj[b, i, j] = 1
                adj[b, j, i] = 1  # symmetric

        # 2. Hierarchical edges
        hier_edges = cls.build_hierarchical_edges(N, device=device)
        if hier_edges.shape[0] > 0:
            for e in range(hier_edges.shape[0]):
                i, j = hier_edges[e, 0].item(), hier_edges[e, 1].item()
                adj[:, i, j] = 1
                adj[:, j, i] = 1

        # 3. Depot connections
        depot_edges = cls.build_depot_edges(N, depot_idx, device=device)
        if depot_edges.shape[0] > 0:
            for e in range(depot_edges.shape[0]):
                i, j = depot_edges[e, 0].item(), depot_edges[e, 1].item()
                adj[:, i, j] = 1

        # 4. Self-connections
        adj[:, range(N), range(N)] = 1

        # 5. Global node connects to all
        if with_global_node:
            adj[:, N, :] = 1  # Global node attends to all
            adj[:, :, N] = 1  # All attend to global node

        # Build attention mask: 0 where connected, -inf where not
        mask = torch.zeros(B, N_total, N_total, device=device)
        mask[adj == 0] = float('-inf')

        return adj, mask


class EGALayer(nn.Module):
    def __init__(self,d_model: int = 128,n_heads: int = 8,d_ff: int = 512,dropout: float = 0.1,d_sparse: int = 16,n_buckets: int = 32,):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Linear projections W_Q, W_K, W_V (Eq. 6-8)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        # Output projection W_O (Eq. 12)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Relative positional encoding
        self.rel_pos_enc = RelativePositionalEncoding(n_heads, d_sparse, self.d_k, n_buckets)

        # Feed-Forward Network (Eq. 14)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        # Instance Normalization (Eq. 13, 15)
        self.norm1 = nn.InstanceNorm1d(d_model, affine=True)
        self.norm2 = nn.InstanceNorm1d(d_model, affine=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self,h: torch.Tensor,coords: torch.Tensor,mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        B, N, D = h.shape
        H, Dk, Dv = self.n_heads, self.d_k, self.d_v

        # Compute Q, K, V (Eq. 6-8)
        Q = self.W_Q(h).view(B, N, H, Dk).transpose(1, 2)  # [B, H, N, Dk]
        K = self.W_K(h).view(B, N, H, Dk).transpose(1, 2)  # [B, H, N, Dk]
        V = self.W_V(h).view(B, N, H, Dv).transpose(1, 2)  # [B, H, N, Dv]

        # Scaled dot-product attention scores: Q * K^T / sqrt(D_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dk)  # [B, H, N, N]

        i_idx = torch.arange(N, device=h.device).unsqueeze(1).expand(N, N)
        j_idx = torch.arange(N, device=h.device).unsqueeze(0).expand(N, N)
        edge_index = torch.stack([i_idx, j_idx], dim=-1).view(1, N * N, 2)
        edge_index = edge_index.expand(B, -1, -1)

        # Truncate coords for relative pos enc (exclude global node)
        N_orig = min(coords.shape[1], N)
        coords_ext = coords
        if N > N_orig:
            # Global node at origin
            global_coords = torch.zeros(B, 1, 2, device=h.device)
            coords_ext = torch.cat([coords, global_coords], dim=1)
            coords_ext = coords_ext[:, :N]

        r_ij = self.rel_pos_enc(coords_ext, edge_index)  # [B, N*N, H, Dk]
        r_ij = r_ij.view(B, N, N, H, Dk).permute(0, 3, 1, 2, 4)  # [B, H, N, N, Dk]

        # q_i^T r_{ij}: [B, H, N, 1, Dk] x [B, H, N, N, Dk] -> [B, H, N, N]
        Q_expanded = Q.unsqueeze(-2).expand(-1, -1, -1, N, -1)  # [B, H, N, N, Dk]
        rel_scores = (Q_expanded * r_ij).sum(dim=-1)  # [B, H, N, N]
        scores = scores + rel_scores

        # Apply attention mask (Eq. 10)
        if mask is not None:
            # mask: [B, N, N] -> [B, 1, N, N]
            scores = scores + mask.unsqueeze(1)

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, N, N]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values (Eq. 11)
        h_attn = torch.matmul(attn_weights, V)  # [B, H, N, Dv]
        h_attn = h_attn.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        # Output projection (Eq. 12)
        h_attn = self.W_O(h_attn)

        # Instance Norm + residual (Eq. 13)
        h_norm1 = self.norm1((h + h_attn).transpose(1, 2)).transpose(1, 2)

        # FFN (Eq. 14)
        h_ffn = self.ffn(h_norm1)

        # Instance Norm + residual (Eq. 15)
        h_out = self.norm2((h_norm1 + h_ffn).transpose(1, 2)).transpose(1, 2)

        return h_out


class TSPDEncoder(nn.Module):
    def __init__(
        self,
        d_input: int = 2,       # Input feature dim (x, y coordinates)
        d_model: int = 128,     # Hidden dimension D_h
        n_heads: int = 8,
        d_ff: int = 512,
        n_layers: int = 3,      # Number of EGA layers
        dropout: float = 0.1,
        d_sparse: int = 16,
        depot_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.depot_idx = depot_idx

        # Input embedding (Eq. 3): h^(0)_n = W_input * x_n + b_input
        self.input_embed = nn.Linear(d_input, d_model)

        # Global node embedding (initialized as learnable parameter)
        self.global_node_embed = nn.Parameter(torch.randn(1, 1, d_model))

        # EGA layers
        self.ega_layers = nn.ModuleList([
            EGALayer(d_model, n_heads, d_ff, dropout, d_sparse)
            for _ in range(n_layers)
        ])

        # Final projection for decoder input
        self.output_proj = nn.Linear(d_model, d_model)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = coords.shape
        device = coords.device

        # Initial embedding (Eq. 3)
        h = self.input_embed(coords)  # [B, N, D]

        # Add global node (Eq. 17 initialization)
        global_node = self.global_node_embed.expand(B, -1, -1)  # [B, 1, D]

        # Process through EGA layers
        multi_scale_pools = []

        for l, layer in enumerate(self.ega_layers):
            # Compute mean-pooled features from current node inputs (Eq. 16)
            p_l = h.mean(dim=1, keepdim=True)  # [B, 1, D]

            # Update global node (Eq. 17)
            global_node = global_node + p_l

            # Concatenate nodes with global node
            h_with_global = torch.cat([h, global_node], dim=1)  # [B, N+1, D]

            # Build attention mask dynamically based on current embeddings
            _, mask = ExpanderGraphBuilder.build_adjacency_mask(
                h, self.depot_idx, with_global_node=True
            )

            # Apply EGA layer
            h_with_global = layer(h_with_global, coords, mask)

            # Separate nodes and global
            h = h_with_global[:, :N]           # [B, N, D]
            global_node = h_with_global[:, N:] # [B, 1, D]

            multi_scale_pools.append(h.mean(dim=1))  # [B, D]

        # Multi-scale mean pooling (Eq. 18)
        p_multi = torch.stack(multi_scale_pools, dim=1).mean(dim=1)  # [B, D]

        # Final fusion (Eq. 19)
        h_final = h + p_multi.unsqueeze(1)  # [B, N, D]

        # Output projection (E^static)
        E_static = self.output_proj(h_final)  # [B, N, D]

        return E_static, p_multi