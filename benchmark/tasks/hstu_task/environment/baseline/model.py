"""Dense, portable HSTU following Meta's SequentialTransductionUnitJagged.

The upstream formulation is credited in ATTRIBUTION.md.  This module
keeps its SiLU attention, u/v/q/k projection, relative time-position bias,
attention-output normalization, multiplicative gate, and residual projection,
but operates on padded tensors so the Harbor baseline does not require FBGEMM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DenseHSTUBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.max_length = max_length
        output_dim = num_heads * (2 * linear_dim + 2 * attention_dim)
        self.uvqk = nn.Parameter(torch.empty(embedding_dim, output_dim))
        nn.init.normal_(self.uvqk, mean=0.0, std=0.02)
        self.output = nn.Linear(num_heads * linear_dim, embedding_dim)
        nn.init.xavier_uniform_(self.output.weight)
        self.position_bias = nn.Parameter(torch.empty(2 * max_length - 1))
        self.time_bias = nn.Parameter(torch.empty(129))
        nn.init.normal_(self.position_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.time_bias, mean=0.0, std=0.02)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        normalized = F.layer_norm(x, [self.embedding_dim], eps=1e-6)
        uvqk = F.silu(torch.matmul(normalized, self.uvqk))
        split_sizes = [
            self.num_heads * self.linear_dim,
            self.num_heads * self.linear_dim,
            self.num_heads * self.attention_dim,
            self.num_heads * self.attention_dim,
        ]
        u, v, q, k = torch.split(uvqk, split_sizes, dim=-1)
        u = u.view(batch, length, self.num_heads, self.linear_dim)
        v = v.view(batch, length, self.num_heads, self.linear_dim)
        q = q.view(batch, length, self.num_heads, self.attention_dim)
        k = k.view(batch, length, self.num_heads, self.attention_dim)

        attention = torch.einsum("bnhd,bmhd->bhnm", q, k)
        positions = torch.arange(length, device=x.device)
        relative = positions[None, :] - positions[:, None]
        relative = relative.clamp(-self.max_length + 1, self.max_length - 1)
        attention = attention + self.position_bias[
            relative + self.max_length - 1
        ][None, None, :, :]

        time_delta = torch.abs(timestamps[:, :, None] - timestamps[:, None, :])
        time_bucket = (torch.log(time_delta.clamp(min=1).float()) / 0.301).long()
        time_bucket = time_bucket.clamp(min=0, max=128)
        attention = attention + self.time_bias[time_bucket][:, None, :, :]
        attention = F.silu(attention) / max(length, 1)

        causal = positions[None, :] <= positions[:, None]
        mask = (
            causal[None, None, :, :]
            & valid[:, None, :, None]
            & valid[:, None, None, :]
        )
        attention = attention * mask.to(attention.dtype)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        attended = torch.einsum("bhnm,bmhd->bnhd", attention, v)
        attended = attended.reshape(batch, length, self.num_heads * self.linear_dim)
        attended = F.layer_norm(
            attended, [self.num_heads * self.linear_dim], eps=1e-6
        )
        gated = u.reshape_as(attended) * attended
        result = x + self.output(F.dropout(gated, p=self.dropout, training=self.training))
        return result * valid.unsqueeze(-1).to(result.dtype)


class HSTUModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_length: int = 50,
        embedding_dim: int = 64,
        num_blocks: int = 4,
        num_heads: int = 4,
        attention_dim: int = 16,
        linear_dim: int = 16,
        dropout: float = 0.5,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DenseHSTUBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dim=attention_dim,
                    linear_dim=linear_dim,
                    max_length=max_length,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, item_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        valid = item_ids.ne(0)
        positions = torch.arange(item_ids.size(1), device=item_ids.device)
        x = self.item_embedding(item_ids) + self.position_embedding(positions)[None]
        x = self.input_dropout(x) * valid.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, valid, timestamps)
        return F.normalize(x, dim=-1)

    def score_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        item_weights = F.normalize(self.item_embedding.weight[1:], dim=-1)
        return torch.matmul(hidden, item_weights.t()) / self.temperature

    def query(
        self, item_ids: torch.Tensor, timestamps: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encode(item_ids, timestamps)
        rows = torch.arange(len(lengths), device=lengths.device)
        return hidden[rows, lengths - 1]

    def config(self) -> dict:
        first = self.blocks[0]
        return {
            "num_items": self.num_items,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "num_blocks": len(self.blocks),
            "num_heads": first.num_heads,
            "attention_dim": first.attention_dim,
            "linear_dim": first.linear_dim,
            "dropout": self.input_dropout.p,
            "temperature": self.temperature,
        }
