"""
RankMixer: Scaling Up Ranking Models in Industrial Recommenders
Paper: https://arxiv.org/abs/2507.15551

This implements the Dense RankMixer-100M variant:
  D=768, T=16, L=2, k=4  (~107M parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Multi-head Token Mixing
# ---------------------------------------------------------------------------

class MultiHeadTokenMixing(nn.Module):
    """
    Parameter-free token mixing module.

    Each token x_t ∈ R^D is split into H heads of size D/H.
    The h-th new token s^h is formed by concatenating the h-th head
    of ALL original tokens:
        s^h = Concat(x_1^h, x_2^h, ..., x_T^h)  ∈ R^D

    With H=T this is equivalent to a (T, H, D/H) → (H, T, D/H) transpose,
    i.e. swapping the token and head dimensions.

    No learnable parameters — purely a reshape/transpose operation.
    """

    def __init__(self, num_tokens: int, hidden_dim: int):
        super().__init__()
        assert hidden_dim % num_tokens == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_tokens ({num_tokens})"
        )
        self.T = num_tokens   # H = T in the paper
        self.D = hidden_dim
        self.head_dim = hidden_dim // num_tokens  # D/H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            s: (B, T, D)  — token-mixed output
        """
        B, T, D = x.shape
        # x: (B, T, D) → (B, T, H, D/H)
        x = x.view(B, T, self.T, self.head_dim)
        # transpose token & head dims: (B, T, H, D/H) → (B, H, T, D/H)
        # then flatten last two dims back to D: (B, H, T*D/H) = (B, T, D)
        s = x.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return s


# ---------------------------------------------------------------------------
# Per-token FFN
# ---------------------------------------------------------------------------

class PerTokenFFN(nn.Module):
    """
    Each token has its own independent 2-layer MLP:
        v_t = W_t^2 · GELU(W_t^1 · s_t + b_t^1) + b_t^2

    Implemented efficiently as a single batched matrix multiply over all tokens.
    W1: (T, D, k*D),  W2: (T, k*D, D)
    """

    def __init__(self, num_tokens: int, hidden_dim: int, ffn_ratio: int = 4):
        super().__init__()
        self.T = num_tokens
        self.D = hidden_dim
        self.inner_dim = ffn_ratio * hidden_dim

        # Separate weight matrices for each token position
        # Store as (T, D, inner_dim) for efficient batched matmul
        self.W1 = nn.Parameter(torch.empty(num_tokens, hidden_dim, self.inner_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, self.inner_dim))
        self.W2 = nn.Parameter(torch.empty(num_tokens, self.inner_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, hidden_dim))

        self._init_weights()

    def _init_weights(self):
        # Use small gain to prevent activation explosion at init
        gain = 0.1
        for t in range(self.T):
            nn.init.xavier_uniform_(self.W1[t], gain=gain)
            nn.init.xavier_uniform_(self.W2[t], gain=gain)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B, T, D)
        Returns:
            v: (B, T, D)
        """
        # s: (B, T, D) → unsqueeze → (B, T, 1, D)
        # W1: (T, D, inner_dim)
        # einsum: b t d, t d i -> b t i
        h = torch.einsum('btd,tdi->bti', s, self.W1) + self.b1  # (B, T, inner_dim)
        h = F.gelu(h)
        v = torch.einsum('bti,tio->bto', h, self.W2) + self.b2  # (B, T, D)
        return v


# ---------------------------------------------------------------------------
# RankMixer Block
# ---------------------------------------------------------------------------

class RankMixerBlock(nn.Module):
    """
    One RankMixer block:
        S_{n-1} = LN(TokenMixing(X_{n-1}) + X_{n-1})
        X_n     = LN(PFFN(S_{n-1})         + S_{n-1})
    """

    def __init__(self, num_tokens: int, hidden_dim: int, ffn_ratio: int = 4):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(num_tokens, hidden_dim)
        self.pffn = PerTokenFFN(num_tokens, hidden_dim, ffn_ratio)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-3)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T, D)
        """
        # Token Mixing with residual + LayerNorm
        s = self.norm1(self.token_mixing(x) + x)
        # Per-token FFN with residual + LayerNorm
        x = self.norm2(self.pffn(s) + s)
        return x


# ---------------------------------------------------------------------------
# Input Layer: Feature Tokenization
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """
    Converts raw input features into T aligned tokens of dimension D.

    For Criteo:
      - 13 numerical features (scalar → embed via linear projection)
      - 26 categorical features (int ID → nn.Embedding)

    Strategy: all embeddings are concatenated, then split+projected into T tokens.
    Each categorical feature gets embed_dim=16 dimensions.
    Each numerical feature gets embed_dim=16 dimensions via a linear layer.
    Total raw embedding = (13 + 26) * 16 = 624 dims.
    We pad/project to T * d_per_group, then Proj each chunk to D.
    """

    def __init__(
        self,
        num_numerical: int,
        cat_vocab_sizes: List[int],
        embed_dim: int,         # per-feature embedding dim (d in eq.2)
        num_tokens: int,        # T
        hidden_dim: int,        # D (output token dim)
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = len(cat_vocab_sizes)
        self.embed_dim = embed_dim
        self.T = num_tokens
        self.D = hidden_dim

        # Numerical: each scalar → embed_dim via linear
        self.num_proj = nn.Linear(num_numerical, num_numerical * embed_dim)

        # Categorical: each field → embed_dim
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in cat_vocab_sizes
        ])

        # Total raw dim after concat
        total_raw_dim = (num_numerical + self.num_categorical) * embed_dim

        # Pad raw dim up to T * (total_raw_dim // T) if not divisible
        # Then linearly project each chunk of size (total_raw_dim // T) → D
        chunk_size = (total_raw_dim + num_tokens - 1) // num_tokens
        padded_dim = chunk_size * num_tokens

        self.pad_proj = nn.Linear(total_raw_dim, padded_dim) if total_raw_dim != padded_dim else nn.Identity()
        self.padded_dim = padded_dim
        self.chunk_size = chunk_size

        # Token projection: chunk_size → D (one per token, shared across all T)
        self.token_proj = nn.Linear(chunk_size, hidden_dim)

        # LayerNorm after tokenization to stabilize input scale (eps=1e-3 for robustness)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-3)

        self._init_weights()

    def _init_weights(self):
        # Standard xavier init — do NOT zero biases; zero bias + small gain
        # causes all-zero output for PAD/missing samples → LayerNorm std=0 → NaN backward
        nn.init.xavier_uniform_(self.num_proj.weight)
        nn.init.xavier_uniform_(self.token_proj.weight)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: (B, num_numerical)  float32
            x_cat: (B, num_categorical) int64
        Returns:
            tokens: (B, T, D)
        """
        B = x_num.size(0)

        # Numerical embeddings: (B, num_numerical * embed_dim)
        num_emb = self.num_proj(x_num).view(B, self.num_numerical, self.embed_dim)

        # Categorical embeddings: (B, num_categorical, embed_dim)
        cat_emb = torch.stack([
            emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)
        ], dim=1)

        # Concat all: (B, (num_num + num_cat), embed_dim) → (B, total_raw_dim)
        e_input = torch.cat([num_emb, cat_emb], dim=1).view(B, -1)

        # Pad to padded_dim
        e_input = self.pad_proj(e_input)  # (B, padded_dim)

        # Split into T chunks: (B, T, chunk_size)
        e_input = e_input.view(B, self.T, self.chunk_size)

        # Project each chunk to D: (B, T, D)
        tokens = self.token_proj(e_input)

        # Normalize token scale before entering RankMixer blocks
        tokens = self.input_norm(tokens)

        return tokens


# ---------------------------------------------------------------------------
# RankMixer (Dense, 100M variant)
# ---------------------------------------------------------------------------

class RankMixer(nn.Module):
    """
    Full RankMixer model.

    Default config for 100M variant:
        D=768, T=16, L=2, k=4

    #Param ≈ 2 * k * L * T * D^2
            = 2 * 4 * 2 * 16 * 768^2
            ≈ 113M  (close to paper's 107M)
    """

    def __init__(
        self,
        num_numerical: int,
        cat_vocab_sizes: List[int],
        embed_dim: int = 16,
        num_tokens: int = 16,      # T
        hidden_dim: int = 768,     # D
        num_layers: int = 2,       # L
        ffn_ratio: int = 4,        # k
        num_tasks: int = 1,        # number of prediction heads
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        # Input layer
        self.tokenizer = FeatureTokenizer(
            num_numerical=num_numerical,
            cat_vocab_sizes=cat_vocab_sizes,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
        )

        # RankMixer blocks
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens, hidden_dim, ffn_ratio)
            for _ in range(num_layers)
        ])

        # Output head (one per task)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_tasks)
        ])

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[RankMixer] Total params: {total / 1e6:.1f}M  |  Trainable: {trainable / 1e6:.1f}M")

    def forward(
        self,
        x_num: torch.Tensor,   # (B, num_numerical)
        x_cat: torch.Tensor,   # (B, num_categorical)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, num_tasks) — raw logits (apply sigmoid for probability)
        """
        # Tokenize input features → (B, T, D)
        # nan_to_num guards against NaN in numericals (missing values in Criteo)
        x_num = torch.nan_to_num(x_num, nan=0.0)
        x = self.tokenizer(x_num, x_cat)

        # Stack of RankMixer blocks
        for block in self.blocks:
            x = block(x)

        # Mean pooling over tokens: (B, T, D) → (B, D)
        o = x.mean(dim=1)

        # Task heads
        outputs = [head(o) for head in self.output_heads]  # list of (B, 1)
        logits = torch.cat(outputs, dim=-1)                 # (B, num_tasks)

        return logits
