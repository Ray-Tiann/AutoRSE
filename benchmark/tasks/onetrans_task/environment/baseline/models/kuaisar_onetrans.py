from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pytorch import MultiOneTransBlock
from models.taac_onetrans import linear_pyramid_schedule


def hashed_ids(values: torch.Tensor, buckets: int) -> torch.Tensor:
    values = values.to(torch.long)
    valid = values != 0
    hashed = torch.remainder(values.abs(), buckets - 1) + 1
    return torch.where(valid, hashed, torch.zeros_like(hashed))


class HashEmbedding(nn.Module):
    def __init__(self, buckets: int, dim: int, sparse: bool = False) -> None:
        super().__init__()
        self.buckets = buckets
        self.embedding = nn.Embedding(buckets, dim, padding_idx=0, sparse=sparse)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.embedding(hashed_ids(values, self.buckets))


class KuaiSARFeatureEncoder(nn.Module):
    def __init__(self, d_model: int, ns_len: int, sparse_embeddings: bool = False) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.user = HashEmbedding(32_768, 32, sparse=sparse_embeddings)
        self.item = HashEmbedding(1_048_576, 64, sparse=sparse_embeddings)
        self.author = HashEmbedding(262_144, 32, sparse=sparse_embeddings)
        self.category = HashEmbedding(65_536, 16, sparse=sparse_embeddings)
        self.item_type = HashEmbedding(65_536, 8, sparse=sparse_embeddings)
        self.upload_type = HashEmbedding(65_536, 8, sparse=sparse_embeddings)
        self.search_source = HashEmbedding(65_536, 8, sparse=sparse_embeddings)
        self.small = HashEmbedding(256, 8, sparse=sparse_embeddings)
        self.binary = nn.Embedding(2, 4)
        self.source = nn.Embedding(3, 4, padding_idx=0)
        self.time_gap = nn.Embedding(128, 8)

        event_dim = 64 + 32 + 16 + 8 + 4 + 5 * 4 + 8 + 8
        non_seq_dim = 32 + 64 + 32 + 16 + 8 + 8 + 7 * 8
        self.event_projection = nn.Sequential(
            nn.Linear(event_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.non_seq_projection = nn.Sequential(
            nn.Linear(non_seq_dim, ns_len * d_model),
            nn.SiLU(),
        )

    def encode_sequence(self, sequence: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        # sequence columns follow utils.kuaisar_data.EVENT_FIELDS plus time_gap.
        parts = [
            self.item(sequence[..., 0]),
            self.author(sequence[..., 1]),
            self.category(sequence[..., 2]),
            self.item_type(sequence[..., 3]),
            self.source((sequence[..., 4].clamp(0, 1) + 1) * seq_mask.to(torch.long)),
            self.binary(sequence[..., 5].clamp(0, 1)),
            self.binary(sequence[..., 6].clamp(0, 1)),
            self.binary(sequence[..., 7].clamp(0, 1)),
            self.binary(sequence[..., 8].clamp(0, 1)),
            self.binary(sequence[..., 9].clamp(0, 1)),
            self.search_source(sequence[..., 10]),
            self.time_gap(sequence[..., 11].clamp(0, 127)),
        ]
        encoded = self.event_projection(torch.cat(parts, dim=-1))
        return encoded * seq_mask.unsqueeze(-1).to(encoded.dtype)

    def encode_non_sequence(self, non_seq: torch.Tensor) -> torch.Tensor:
        # user, item, author, category, item type, upload type, then seven
        # compact numeric/categorical context fields.
        parts = [
            self.user(non_seq[:, 0]),
            self.item(non_seq[:, 1]),
            self.author(non_seq[:, 2]),
            self.category(non_seq[:, 3]),
            self.item_type(non_seq[:, 4]),
            self.upload_type(non_seq[:, 5]),
        ]
        parts.extend(self.small(non_seq[:, idx]) for idx in range(6, 13))
        projected = self.non_seq_projection(torch.cat(parts, dim=-1))
        return projected.view(non_seq.size(0), self.ns_len, -1)


class KuaiSAROneTrans(nn.Module):
    def __init__(
        self,
        seq_len: int = 128,
        ns_len: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_hidden: int = 512,
        num_layers: int = 6,
        pyramid_align: int = 32,
        mask_type: str = "paper_causal",
        use_checkpoint: bool = False,
        sparse_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ns_len = ns_len
        self.encoder = KuaiSARFeatureEncoder(
            d_model=d_model,
            ns_len=ns_len,
            sparse_embeddings=sparse_embeddings,
        )
        self.position = nn.Embedding(seq_len + ns_len, d_model)
        schedule = linear_pyramid_schedule(
            total_tokens=seq_len + ns_len,
            ns_len=ns_len,
            num_layers=num_layers,
            align_to=pyramid_align,
        )
        self.blocks = nn.ModuleList(
            [
                MultiOneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=(ffn_hidden, d_model),
                    n=1,
                    pyramid_stack_len=target,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for target in schedule
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.click_head = nn.Linear(d_model, 1)
        self.like_head = nn.Linear(d_model, 1)

    def forward(
        self,
        non_seq: torch.Tensor,
        sequence: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seq_tokens = self.encoder.encode_sequence(sequence, seq_mask)
        ns_tokens = self.encoder.encode_non_sequence(non_seq)
        x = torch.cat([seq_tokens, ns_tokens], dim=1)
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.position(positions).unsqueeze(0)
        valid_mask = torch.cat(
            [
                seq_mask.to(torch.bool),
                torch.ones(non_seq.size(0), self.ns_len, dtype=torch.bool, device=x.device),
            ],
            dim=1,
        )
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x, valid_mask = block(x, valid_mask)
        pooled = self.output_norm(x).mean(dim=1)
        return {
            "click": self.click_head(pooled).squeeze(-1),
            "like": self.like_head(pooled).squeeze(-1),
        }


class KuaiSARMLPBaseline(nn.Module):
    """A small smoke-test baseline that shares exactly the same encoders."""

    def __init__(
        self,
        d_model: int = 128,
        ns_len: int = 4,
        sparse_embeddings: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.encoder = KuaiSARFeatureEncoder(
            d_model=d_model,
            ns_len=ns_len,
            sparse_embeddings=sparse_embeddings,
        )
        self.backbone = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
        )
        self.click_head = nn.Linear(d_model, 1)
        self.like_head = nn.Linear(d_model, 1)

    def forward(
        self,
        non_seq: torch.Tensor,
        sequence: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seq_tokens = self.encoder.encode_sequence(sequence, seq_mask)
        denom = seq_mask.sum(dim=1, keepdim=True).clamp_min(1).to(seq_tokens.dtype)
        seq_pooled = seq_tokens.sum(dim=1) / denom
        ns_pooled = self.encoder.encode_non_sequence(non_seq).mean(dim=1)
        hidden = self.backbone(torch.cat([seq_pooled, ns_pooled], dim=-1))
        return {
            "click": self.click_head(hidden).squeeze(-1),
            "like": self.like_head(hidden).squeeze(-1),
        }
