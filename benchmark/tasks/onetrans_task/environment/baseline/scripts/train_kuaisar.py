from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.kuaisar_onetrans import HashEmbedding, KuaiSARMLPBaseline, KuaiSAROneTrans
from utils.kuaisar_data import KuaiSARHistoryDataset
from utils.kuaisar_metrics import (
    binary_auc,
    binary_logloss,
    macro_group_auc,
    macro_group_weighted_user_auc,
    weighted_user_auc,
)

import torch.distributed as dist


_STOP_REQUESTED = False
_STOP_REASON = ""
_STOP_MICRO = 0


def _handle_sigterm(signum: int, frame: Any) -> None:
    global _STOP_REQUESTED, _STOP_REASON
    _STOP_REQUESTED = True
    _STOP_REASON = "SIGTERM"
    print("[signal] SIGTERM received; requesting graceful stop", flush=True)


def _handle_sigint(signum: int, frame: Any) -> None:
    global _STOP_REQUESTED, _STOP_REASON
    _STOP_REQUESTED = True
    _STOP_REASON = "SIGINT"
    print("[signal] SIGINT received; requesting graceful stop", flush=True)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_rank0() -> bool:
    return rank() == 0


def log_rank0(message: str) -> None:
    if is_rank0():
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OneTrans on leakage-safe KuaiSAR histories (DDP-capable).")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("onetrans", "mlp"), default="onetrans")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ns-len", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--pyramid-align", type=int, default=32)
    parser.add_argument("--mask-type", default="paper_causal")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=("hybrid", "adamw"), default="hybrid")
    parser.add_argument("--sparse-lr", type=float, default=0.1)
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--like-weight", type=float, default=0.2)
    parser.add_argument("--like-pos-weight", type=float, default=1.0)
    parser.add_argument("--click-only", action="store_true",
                        help="ablate the like auxiliary task: skip like loss and freeze the like head")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="legacy single gradient clip threshold; overrides both dense/sparse when set")
    parser.add_argument("--grad-clip-dense", type=float, default=90.0)
    parser.add_argument("--grad-clip-sparse", type=float, default=120.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="do not load/evaluate test labels in training; prediction is produced separately",
    )
    parser.add_argument("--distributed", action="store_true",
                        help="launch with torchrun; rank/world are read from env")
    parser.add_argument("--dist-backend", default="nccl",
                        choices=("nccl", "gloo"),
                        help="distributed backend (gloo for CPU smoke tests)")
    parser.add_argument("--placeholder", action="store_true",
                        help="real-training placeholder: no checkpoints, runs until SIGTERM")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-global-steps", type=int, default=None,
                        help="stop training after this many optimizer steps")
    parser.add_argument("--validate-every", type=int, default=0,
                        help="run validation every N optimizer steps (0 = epoch-based)")
    parser.add_argument("--early-stop-after", type=int, default=1500,
                        help="step-driven: allow early stopping only after this many global optimizer steps")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="step-driven: stop after N consecutive validations without click-AUC improvement (0 disables)")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="step-driven: run only this many optimizer steps, save smoke.json + last.pt, then exit without test")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> nn.Module:
    common: dict[str, Any] = {
        "seq_len": args.seq_len,
        "ns_len": args.ns_len,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "ffn_hidden": args.ffn_hidden,
        "num_layers": args.num_layers,
        "pyramid_align": args.pyramid_align,
        "mask_type": args.mask_type,
        "use_checkpoint": args.activation_checkpoint,
        "sparse_embeddings": False,  # DDP/NCCL cannot all-reduce sparse grads
    }
    if args.model == "onetrans":
        return KuaiSAROneTrans(**common)
    return KuaiSARMLPBaseline(**common)


def amp_settings(args: argparse.Namespace) -> tuple[bool, torch.dtype]:
    enabled = args.device.startswith("cuda") and args.amp_dtype != "none"
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        log_rank0("[amp] BF16 is not supported; falling back to FP16")
        dtype = torch.float16
    return enabled, dtype


class OptimizerBundle:
    """Coordinate sparse Adagrad and dense RMSProp as one checkpointable unit."""

    def __init__(
        self,
        sparse: torch.optim.Optimizer,
        dense: torch.optim.Optimizer,
    ) -> None:
        self.sparse = sparse
        self.dense = dense

    @property
    def members(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        return self.sparse, self.dense

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.members:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.members:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "hybrid",
            "sparse": self.sparse.state_dict(),
            "dense": self.dense.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.sparse.load_state_dict(state["sparse"])
        self.dense.load_state_dict(state["dense"])


OptimizerLike = torch.optim.Optimizer | OptimizerBundle


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> OptimizerLike:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    sparse_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, HashEmbedding)
        for parameter in module.parameters()
    }
    sparse_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in sparse_ids and parameter.requires_grad
    ]
    dense_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in sparse_ids and parameter.requires_grad
    ]
    if not sparse_parameters or not dense_parameters:
        raise RuntimeError("hybrid optimizer requires both sparse and dense parameters")
    sparse_optimizer = torch.optim.Adagrad(
        sparse_parameters,
        lr=args.sparse_lr,
        initial_accumulator_value=0.1,
        weight_decay=0.0,
    )
    dense_optimizer = torch.optim.RMSprop(
        dense_parameters,
        lr=args.lr,
        alpha=args.rmsprop_alpha,
        momentum=0.0,
        weight_decay=args.weight_decay,
    )
    return OptimizerBundle(sparse=sparse_optimizer, dense=dense_optimizer)


def optimizer_members(optimizer: OptimizerLike) -> tuple[torch.optim.Optimizer, ...]:
    if isinstance(optimizer, OptimizerBundle):
        return optimizer.members
    return (optimizer,)


def clip_model_gradients(
    model: nn.Module,
    dense_norm: float,
    sparse_norm: float,
) -> tuple[float, float]:
    """Clip dense gradients by dense_norm and sparse gradients by sparse_norm.

    Returns the pre-clipping norms (dense, sparse). Sparse gradients use a
    coefficient-style clip on the coalesced values, matching the existing
    KuaiSAR implementation.
    """
    sparse_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, HashEmbedding)
        for parameter in module.parameters()
    }
    dense_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None and id(parameter) not in sparse_ids
    ]
    dense_before = 0.0
    if dense_parameters:
        dense_before = float(
            torch.nn.utils.clip_grad_norm_(dense_parameters, dense_norm).item()
        )

    sparse_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None and id(parameter) in sparse_ids
    ]
    sparse_before = 0.0
    if sparse_parameters:
        sparse_before = float(
            torch.nn.utils.clip_grad_norm_(sparse_parameters, sparse_norm).item()
        )
    return dense_before, sparse_before


def _finish_accumulation(
    model: nn.Module,
    optimizer: OptimizerLike,
    scaler: torch.amp.GradScaler | None,
    dense_clip: float,
    sparse_clip: float,
) -> tuple[float, float]:
    if scaler is not None and scaler.is_enabled():
        for member in optimizer_members(optimizer):
            scaler.unscale_(member)
        dense_before, sparse_before = clip_model_gradients(model, dense_clip, sparse_clip)
        for member in optimizer_members(optimizer):
            scaler.step(member)
        scaler.update()
    else:
        dense_before, sparse_before = clip_model_gradients(model, dense_clip, sparse_clip)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dense_before, sparse_before


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    like_weight: float,
    like_pos_weight: float,
    grad_clip_dense: float,
    grad_clip_sparse: float,
    click_only: bool = False,
    optimizer: OptimizerLike | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
    grad_accum: int = 1,
    log_every: int = 20,
    resume_micro_step: int = 0,
    epoch_index: int = 1,
    learning_rate: float | None = None,
    on_optimizer_step: Any | None = None,
) -> tuple[dict[str, float], int, str]:
    """Run one pass over loader.

    Returns (metrics, global_step_delta, stop_reason). stop_reason is non-empty
    when training was interrupted (SIGTERM/NaN) or when loss became invalid.
    """
    global _STOP_MICRO
    training = optimizer is not None
    grad_accum = max(int(grad_accum), 1)
    if not training and torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train(training)
    click_loss_fn = nn.BCEWithLogitsLoss()
    like_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(like_pos_weight, device=device)
    )
    total_loss = 0.0
    total_rows = 0
    click_logits: list[np.ndarray] = []
    click_labels: list[np.ndarray] = []
    like_logits: list[np.ndarray] = []
    like_labels: list[np.ndarray] = []
    users: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    started = time.perf_counter()
    step_timer = time.perf_counter()
    pending = 0
    global_step = 0
    stop_reason = ""
    last_unit_loss = float("nan")
    ema_rows_per_second = 0.0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if batch_index < resume_micro_step:
            continue
        if _STOP_REQUESTED:
            _STOP_MICRO = batch_index + 1
            stop_reason = _STOP_REASON or "stop_requested"
            break

        non_seq = batch["non_seq"].to(device, non_blocking=True)
        sequence = batch["sequence"].to(device, non_blocking=True)
        seq_mask = batch["seq_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if training and pending == 0:
            optimizer.zero_grad(set_to_none=True)

        autocast_ctx = torch.autocast(
            device_type=torch.device(device).type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        )
        grad_ctx = torch.no_grad() if not training else nullcontext()
        with autocast_ctx, grad_ctx:
            output = model(non_seq, sequence, seq_mask)
            click_loss = click_loss_fn(output["click"], labels[:, 0])
            if click_only:
                unit_loss = click_loss
            else:
                like_loss = like_loss_fn(output["like"], labels[:, 1])
                unit_loss = click_loss + like_weight * like_loss
            last_unit_loss = float(unit_loss.detach())

        if training:
            pending += 1
            loss = unit_loss / grad_accum
            is_update = pending % grad_accum == 0
            sync_context: Any = (
                nullcontext() if (not is_distributed() or is_update) else model.no_sync()
            )
            with sync_context:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if is_update:
                dense_before, sparse_before = _finish_accumulation(
                    model, optimizer, scaler, grad_clip_dense, grad_clip_sparse
                )
                pending = 0
                global_step += 1
                if on_optimizer_step is not None:
                    on_optimizer_step(global_step, batch_index + 1)

                if not (math.isfinite(last_unit_loss) and math.isfinite(dense_before)
                        and math.isfinite(sparse_before)):
                    _STOP_MICRO = batch_index + 1
                    stop_reason = f"nan_or_inf loss={last_unit_loss} dense_grad={dense_before} sparse_grad={sparse_before}"
                    log_rank0(f"[fatal] {stop_reason}")
                    break

                if global_step % log_every == 0 and is_rank0():
                    now = time.perf_counter()
                    step_time = now - step_timer
                    step_timer = now
                    rows_this_window = int(labels.size(0)) * log_every
                    ema_rows_per_second = (
                        rows_this_window / max(step_time, 1e-9)
                        if ema_rows_per_second <= 0
                        else 0.7 * ema_rows_per_second + 0.3 * (rows_this_window / max(step_time, 1e-9))
                    )
                    if device.startswith("cuda") and torch.cuda.is_available():
                        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                        mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    else:
                        mem_alloc = mem_reserved = 0.0
                    print(
                        json.dumps(
                            {
                                "event": "step",
                                "epoch": epoch_index,
                                "micro_step": batch_index + 1,
                                "global_step": global_step,
                                "loss": last_unit_loss,
                                "learning_rate": learning_rate,
                                "samples_per_second": round(ema_rows_per_second, 1),
                                "step_time": round(step_time, 3),
                                "gpu_mem_alloc_gb": round(mem_alloc, 2),
                                "gpu_mem_reserved_gb": round(mem_reserved, 2),
                                "dense_grad_norm": round(dense_before, 4),
                                "sparse_grad_norm": round(sparse_before, 4),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        else:
            loss = unit_loss

        rows = labels.size(0)
        total_loss += float(unit_loss.detach()) * rows
        total_rows += rows
        if not training:
            click_logits.append(output["click"].detach().float().cpu().numpy())
            click_labels.append(labels[:, 0].detach().cpu().numpy())
            like_logits.append(output["like"].detach().float().cpu().numpy())
            like_labels.append(labels[:, 1].detach().cpu().numpy())
            users.append(batch["user_id"].numpy())
            timestamps.append(batch["timestamp"].numpy())

    if training and pending > 0 and not stop_reason:
        _finish_accumulation(model, optimizer, scaler, grad_clip_dense, grad_clip_sparse)
        pending = 0
        global_step += 1

    elapsed = max(time.perf_counter() - started, 1e-9)
    result = {
        "loss": total_loss / max(total_rows, 1),
        "rows": float(total_rows),
        "rows_per_second": total_rows / elapsed,
    }

    if not training:
        if is_distributed():
            payload = {
                "click_logits": np.concatenate(click_logits) if click_logits else np.zeros(0, dtype=np.float32),
                "click_labels": np.concatenate(click_labels) if click_labels else np.zeros(0, dtype=np.int8),
                "like_logits": np.concatenate(like_logits) if like_logits else np.zeros(0, dtype=np.float32),
                "like_labels": np.concatenate(like_labels) if like_labels else np.zeros(0, dtype=np.int8),
                "users": np.concatenate(users) if users else np.zeros(0, dtype=np.int64),
                "timestamps": np.concatenate(timestamps) if timestamps else np.zeros(0, dtype=np.int64),
                "rows": float(total_rows),
                "loss_sum": float(total_loss),
            }
            gathered: list[dict[str, Any]] = [None] * world_size()
            dist.all_gather_object(gathered, payload)
            if is_rank0():
                click_logits = [g["click_logits"] for g in gathered]
                click_labels = [g["click_labels"] for g in gathered]
                like_logits = [g["like_logits"] for g in gathered]
                like_labels = [g["like_labels"] for g in gathered]
                users = [g["users"] for g in gathered]
                timestamps = [g["timestamps"] for g in gathered]
                total_rows = float(sum(g["rows"] for g in gathered))
                total_loss = float(sum(g["loss_sum"] for g in gathered))
        if is_rank0() and total_rows:
            click_score = np.concatenate(click_logits)
            click_label = np.concatenate(click_labels).astype(np.int8)
            like_score = np.concatenate(like_logits)
            like_label = np.concatenate(like_labels).astype(np.int8)
            user_ids = np.concatenate(users)
            timestamp_ms = np.concatenate(timestamps)
            # KuaiSAR documents its human-readable timestamps in Beijing time.
            day_ids = (timestamp_ms + 8 * 3_600_000) // 86_400_000
            click_prob = 1.0 / (1.0 + np.exp(-np.clip(click_score, -40, 40)))
            pos_mask = click_label == 1
            neg_mask = click_label == 0
            result.update(
                {
                    "loss": total_loss / max(total_rows, 1),
                    "rows": float(total_rows),
                    "pred_mean": float(click_prob.mean()),
                    "pred_std": float(click_prob.std()),
                    "pos_pred_mean": float(click_prob[pos_mask].mean()) if pos_mask.any() else float("nan"),
                    "neg_pred_mean": float(click_prob[neg_mask].mean()) if neg_mask.any() else float("nan"),
                    "click_auc": binary_auc(click_label, click_score),
                    "click_uauc": weighted_user_auc(click_label, click_score, user_ids),
                    "click_daily_auc": macro_group_auc(click_label, click_score, day_ids),
                    "click_daily_uauc": macro_group_weighted_user_auc(
                        click_label, click_score, user_ids, day_ids
                    ),
                    "click_logloss": binary_logloss(click_label, click_score),
                    "like_auc": binary_auc(like_label, like_score),
                    "like_uauc": weighted_user_auc(like_label, like_score, user_ids),
                    "like_daily_auc": macro_group_auc(like_label, like_score, day_ids),
                    "like_daily_uauc": macro_group_weighted_user_auc(
                        like_label, like_score, user_ids, day_ids
                    ),
                    "like_logloss": binary_logloss(like_label, like_score),
                }
            )
    return result, global_step, stop_reason


def make_loader(
    processed_dir: Path,
    split: str,
    args: argparse.Namespace,
    shuffle: bool,
    sampler: Any = None,
    dataset: KuaiSARHistoryDataset | None = None,
) -> DataLoader:
    if dataset is None:
        dataset = KuaiSARHistoryDataset(
            processed_dir=processed_dir,
            split=split,
            seq_len=args.seq_len,
            max_rows=204800 if split == "val" else args.max_rows,
        )
    generator = None
    if shuffle and is_distributed():
        generator = torch.Generator().manual_seed(args.seed * 1000 + rank())
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Report both deployable total and the paper's dense-only convention."""
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) in embedding_parameter_ids
    )
    return {"total": total, "embedding": embedding, "dense": total - embedding}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: OptimizerLike,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    micro_step: int,
    global_step: int,
    best_val_auc: float,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": epoch,
        "micro_step_in_epoch": micro_step,
        "global_step": global_step,
        "best_val_auc": best_val_auc,
        "world_size": world_size(),
        "per_gpu_batch": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_global_batch": world_size() * args.batch_size * args.grad_accum,
        "rng_state_cpu": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: OptimizerLike | None,
    scaler: torch.amp.GradScaler | None,
    device: str,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and scaler.is_enabled() and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if state.get("rng_state_cpu") is not None:
        torch.set_rng_state(state["rng_state_cpu"].to("cpu"))
    if state.get("rng_state_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["rng_state_cuda"])
        except Exception:
            pass
    return state


def run_placeholder(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
) -> None:
    """Real-training placeholder: full forward/backward/step, no checkpoints."""
    optimizer = build_optimizer(args, model)
    amp_enabled, amp_dtype = amp_settings(args)
    scaler = torch.amp.GradScaler(
        device=torch.device(device).type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    click_loss_fn = nn.BCEWithLogitsLoss()
    like_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.like_pos_weight, device=device)
    )
    global_step = 0
    epoch = 0
    log_rank0("[placeholder] real-training placeholder started (no checkpoints)")
    while not _STOP_REQUESTED:
        epoch += 1
        epoch_rows = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            if _STOP_REQUESTED:
                break
            non_seq = batch["non_seq"].to(device, non_blocking=True)
            sequence = batch["sequence"].to(device, non_blocking=True)
            seq_mask = batch["seq_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(non_seq, sequence, seq_mask)
                click_loss = click_loss_fn(output["click"], labels[:, 0])
                like_loss = like_loss_fn(output["like"], labels[:, 1])
                unit_loss = click_loss + args.like_weight * like_loss
            if scaler is not None and scaler.is_enabled():
                scaler.scale(unit_loss).backward()
                for member in optimizer_members(optimizer):
                    scaler.unscale_(member)
                clip_model_gradients(
                    model, args.grad_clip_dense, args.grad_clip_sparse
                )
                for member in optimizer_members(optimizer):
                    scaler.step(member)
                scaler.update()
            else:
                unit_loss.backward()
                clip_model_gradients(model, args.grad_clip_dense, args.grad_clip_sparse)
                optimizer.step()
            rows = labels.size(0)
            epoch_rows += rows
            global_step += 1
            if global_step % args.log_every == 0 and is_rank0():
                now = time.perf_counter()
                rps = epoch_rows / max(now - epoch_started, 1e-9)
                if device.startswith("cuda") and torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                else:
                    mem_alloc = 0.0
                print(
                    json.dumps(
                        {
                            "event": "placeholder_step",
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": float(unit_loss.detach()),
                            "samples_per_second": round(rps, 1),
                            "gpu_mem_alloc_gb": round(mem_alloc, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    log_rank0("[placeholder] stopped by stop request; exiting")


def main() -> None:
    args = parse_args()
    # Study selection always covers the fixed first 204800 validation rows.
    args.max_eval_batches = None
    args.skip_test_eval = True
    if args.grad_clip is not None:
        args.grad_clip_dense = args.grad_clip
        args.grad_clip_sparse = args.grad_clip
    set_seed(args.seed)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigint)

    distributed = args.distributed or ("LOCAL_RANK" in os.environ)
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend=args.dist_backend)
        if args.device.startswith("cuda") or args.dist_backend == "nccl":
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        else:
            args.device = "cpu"
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        log_rank0(f"[ddp] initialized world_size={args.world_size} rank={args.rank}")
    else:
        args.world_size = 1
        args.rank = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_sampler: Any = None
    train_loader = None
    if not args.eval_only:
        train_dataset = KuaiSARHistoryDataset(
            processed_dir=args.processed_dir,
            split="train",
            seq_len=args.seq_len,
            max_rows=args.max_rows,
        )
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size(),
                rank=rank(),
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
        train_loader = make_loader(
            args.processed_dir,
            "train",
            args,
            shuffle=True,
            sampler=train_sampler,
            dataset=train_dataset,
        )

    val_loader = test_loader = None
    val_sampler = test_sampler = None
    if distributed:
        val_dataset = KuaiSARHistoryDataset(
            processed_dir=args.processed_dir,
            split="val",
            seq_len=args.seq_len,
            max_rows=204800,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=False,
            drop_last=False,
        )
        val_loader = make_loader(
            args.processed_dir, "val", args, shuffle=False,
            sampler=val_sampler, dataset=val_dataset,
        )
        if not args.skip_test_eval:
            test_dataset = KuaiSARHistoryDataset(
                processed_dir=args.processed_dir,
                split="test",
                seq_len=args.seq_len,
                max_rows=args.max_rows,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size(),
                rank=rank(),
                shuffle=False,
                drop_last=False,
            )
            test_loader = make_loader(
                args.processed_dir, "test", args, shuffle=False,
                sampler=test_sampler, dataset=test_dataset,
            )
    else:
        val_loader = make_loader(args.processed_dir, "val", args, shuffle=False)
        if not args.skip_test_eval:
            test_loader = make_loader(args.processed_dir, "test", args, shuffle=False)

    model = build_model(args).to(args.device)
    if args.click_only:
        frozen_count = 0
        for module_name, module in model.named_modules():
            if module_name.endswith("like_head"):
                for parameter in module.parameters():
                    parameter.requires_grad = False
                    frozen_count += parameter.numel()
        log_rank0(
            f"[click_only] like auxiliary task disabled; "
            f"like head frozen ({frozen_count:,} parameters)"
        )
    if distributed:
        # The pyramid schedule prunes all S-tokens at the top layer (query
        # length == ns_len), so the shared S-token QKV/FFN parameters are
        # unused in that layer by design. DDP therefore requires
        # find_unused_parameters=True; this is architectural, not a bug.
        ddp_device_ids = [int(os.environ.get("LOCAL_RANK", "0"))] if args.device.startswith("cuda") else None
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=ddp_device_ids,
            output_device=ddp_device_ids[0] if ddp_device_ids else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    amp_enabled, amp_dtype = amp_settings(args)
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            device=torch.device(args.device).type,
            enabled=scaler_enabled,
        )
    else:
        # Harbor's shared CUDA image currently uses PyTorch 2.1, where the
        # unified torch.amp.GradScaler API is not available yet.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    if args.placeholder:
        if train_loader is None:
            raise RuntimeError("placeholder requires train data")
        run_placeholder(args, model, train_loader, args.device)
        if distributed:
            dist.destroy_process_group()
        return

    start_epoch = 1
    resume_micro_step = 0
    global_step = 0
    best_auc = float("-inf")
    if args.resume is not None:
        state = load_checkpoint(args.resume, model, optimizer if not args.eval_only else None, scaler, args.device)
        epoch_from_state = int(state.get("epoch", 0))
        resume_micro_step = int(state.get("micro_step_in_epoch", 0))
        start_epoch = epoch_from_state + 1 if resume_micro_step == 0 else epoch_from_state
        global_step = int(state.get("global_step", 0))
        best_auc = float(state.get("best_val_auc", float("-inf")))
        log_rank0(
            f"[resume] {args.resume} epoch={start_epoch} micro_step={resume_micro_step} "
            f"global_step={global_step} best_val_auc={best_auc:.6f}"
        )

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config["effective_global_batch"] = world_size() * args.batch_size * args.grad_accum
    parameters = parameter_counts(model)
    log_rank0(
        f"[model] {args.model} total={parameters['total']:,} "
        f"embedding={parameters['embedding']:,} dense={parameters['dense']:,} "
        f"device={args.device} world={world_size()} batch={args.batch_size} "
        f"accum={args.grad_accum} effective_global_batch={config['effective_global_batch']}"
    )
    history: list[dict[str, Any]] = []
    interrupted = False

    if not args.eval_only:
        assert train_loader is not None
        if args.max_global_steps is not None:
            # Step-driven training: run segments of `validate_every` optimizer
            # steps, validate, and track best validation AUC.
            val_every = args.validate_every or args.max_global_steps
            processed_micro = resume_micro_step
            step_epoch = 1
            early_stop_streak = 0
            final_stop_reason = "max_steps"
            while global_step < args.max_global_steps:
                seg_global = min(val_every, args.max_global_steps - global_step)
                seg_micro = seg_global * args.grad_accum

                def on_step(step_in_epoch: int, micro_step: int) -> None:
                    if is_rank0() and (global_step + step_in_epoch) % args.save_every == 0:
                        save_checkpoint(
                            args.output_dir / "last.pt", model, optimizer, scaler,
                            1, micro_step, global_step + step_in_epoch,
                            best_auc, args, config,
                        )

                train_metrics, step_delta, stop_reason = run_epoch(
                    model, train_loader, args.device, amp_enabled, amp_dtype,
                    args.like_weight, args.like_pos_weight,
                    click_only=args.click_only,
                    grad_clip_dense=args.grad_clip_dense,
                    grad_clip_sparse=args.grad_clip_sparse,
                    optimizer=optimizer, scaler=scaler,
                    max_batches=processed_micro + seg_micro,
                    grad_accum=args.grad_accum, log_every=args.log_every,
                    resume_micro_step=processed_micro, epoch_index=step_epoch,
                    learning_rate=args.lr, on_optimizer_step=on_step,
                )
                processed_micro += step_delta * args.grad_accum
                global_step += step_delta
                if step_delta < seg_global and not stop_reason:
                    if train_sampler is not None:
                        step_epoch += 1
                        train_sampler.set_epoch(step_epoch)
                    processed_micro = 0
                if stop_reason:
                    log_rank0(f"[stop] training interrupted: {stop_reason}")
                    if is_rank0():
                        save_checkpoint(
                            args.output_dir / "last.pt", model, optimizer, scaler,
                            1, _STOP_MICRO, global_step, best_auc, args, config,
                        )
                    interrupted = True
                    break

                val_metrics, _, _ = run_epoch(
                    model, val_loader, args.device, amp_enabled, amp_dtype,
                    args.like_weight, args.like_pos_weight,
                    click_only=args.click_only,
                    grad_clip_dense=args.grad_clip_dense,
                    grad_clip_sparse=args.grad_clip_sparse,
                    max_batches=args.max_eval_batches,
                )
                val_auc = val_metrics.get("click_auc", float("nan"))
                if not math.isnan(val_auc) and val_auc > best_auc:
                    best_auc = val_auc
                    if is_rank0():
                        save_checkpoint(
                            args.output_dir / "best.pt", model, optimizer, scaler,
                            1, processed_micro, global_step, best_auc, args, config,
                        )
                        log_rank0(f"[ckpt] saved best.pt val_auc={best_auc:.6f}")
                if is_rank0():
                    row = {"global_step": global_step, "train": train_metrics, "val": val_metrics}
                    history.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                if is_distributed():
                    dist.barrier()
                improved = (not math.isnan(val_auc)) and val_auc > best_auc - 1e-12
                if improved:
                    early_stop_streak = 0
                elif args.early_stop_patience > 0 and global_step >= args.early_stop_after:
                    early_stop_streak += 1
                    if early_stop_streak >= args.early_stop_patience:
                        log_rank0(
                            f"[stop] early stopping after {args.early_stop_patience} consecutive "
                            f"validations without click-AUC improvement at global_step={global_step} "
                            f"best_val_auc={best_auc:.6f}"
                        )
                        final_stop_reason = f"early_stop_streak_{early_stop_streak}"
                        stop_reason = "early_stop"
                        break
            if args.smoke_steps > 0:
                if is_distributed():
                    dist.barrier()
                if is_rank0():
                    mem = {}
                    if torch.cuda.is_available():
                        dev_count = torch.cuda.device_count()
                        mem["per_rank_allocated_mib"] = [
                            int(torch.cuda.memory_allocated(i) // (1024 * 1024))
                            for i in range(dev_count)
                        ]
                        mem["per_rank_reserved_mib"] = [
                            int(torch.cuda.memory_reserved(i) // (1024 * 1024))
                            for i in range(dev_count)
                        ]
                    smoke_report = {
                        "smoke_steps": args.smoke_steps,
                        "global_step": global_step,
                        "last_train_loss": (train_metrics or {}).get("loss"),
                        "train_rows_per_second": (train_metrics or {}).get("rows_per_second"),
                        "best_val_auc": best_auc,
                        "cuda": mem,
                        "config": config,
                    }
                    save_json(args.output_dir / "smoke.json", smoke_report)
                    print(json.dumps({"smoke": smoke_report}, ensure_ascii=False, indent=2), flush=True)
                if distributed:
                    dist.destroy_process_group()
                return
            if args.smoke_steps > 0:
                if is_distributed():
                    dist.barrier()
                if is_rank0():
                    mem = {}
                    if torch.cuda.is_available():
                        dev_count = torch.cuda.device_count()
                        mem["per_rank_allocated_mib"] = [
                            int(torch.cuda.memory_allocated(i) // (1024 * 1024))
                            for i in range(dev_count)
                        ]
                        mem["per_rank_reserved_mib"] = [
                            int(torch.cuda.memory_reserved(i) // (1024 * 1024))
                            for i in range(dev_count)
                        ]
                    smoke_report = {
                        "smoke_steps": args.smoke_steps,
                        "global_step": global_step,
                        "last_train_loss": (train_metrics or {}).get("loss"),
                        "train_rows_per_second": (train_metrics or {}).get("rows_per_second"),
                        "best_val_auc": best_auc,
                        "cuda": mem,
                        "config": config,
                    }
                    save_json(args.output_dir / "smoke.json", smoke_report)
                    print(json.dumps({"smoke": smoke_report}, ensure_ascii=False, indent=2), flush=True)
                if distributed:
                    dist.destroy_process_group()
                return
            # ---- step-driven branch finished: test with best checkpoint ----
            if interrupted:
                if is_distributed():
                    dist.barrier()
                if is_rank0():
                    print(json.dumps({"interrupted": True}, ensure_ascii=False), flush=True)
                if distributed:
                    dist.destroy_process_group()
                return
            if is_distributed():
                dist.barrier()
            best_path = args.output_dir / "best.pt"
            if best_path.exists():
                load_checkpoint(best_path, model, None, None, args.device)
            if is_distributed():
                dist.barrier()
            test_metrics = {}
            if test_loader is not None:
                test_metrics, _, _ = run_epoch(
                    model, test_loader, args.device, amp_enabled, amp_dtype,
                    args.like_weight, args.like_pos_weight,
                    click_only=args.click_only,
                    grad_clip_dense=args.grad_clip_dense,
                    grad_clip_sparse=args.grad_clip_sparse,
                    max_batches=args.max_eval_batches,
                )
            if is_rank0():
                report = {
                    "config": config, "parameters": parameters,
                    "best_val_auc": best_auc, "history": history,
                    "test": test_metrics, "stop_reason": final_stop_reason,
                }
                save_json(args.output_dir / "metrics.json", report)
                print(json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2), flush=True)
            if distributed:
                dist.destroy_process_group()
            return
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            def on_step(step_in_epoch: int, micro_step: int) -> None:
                if is_rank0() and step_in_epoch % args.save_every == 0:
                    save_checkpoint(
                        args.output_dir / "last.pt", model, optimizer, scaler,
                        epoch, micro_step, global_step + step_in_epoch,
                        best_auc, args, config,
                    )

            train_metrics, step_delta, stop_reason = run_epoch(
                model,
                train_loader,
                args.device,
                amp_enabled,
                amp_dtype,
                args.like_weight,
                args.like_pos_weight,
                click_only=args.click_only,
                grad_clip_dense=args.grad_clip_dense,
                grad_clip_sparse=args.grad_clip_sparse,
                optimizer=optimizer,
                scaler=scaler,
                max_batches=args.max_train_batches,
                grad_accum=args.grad_accum,
                log_every=args.log_every,
                resume_micro_step=resume_micro_step,
                epoch_index=epoch,
                learning_rate=args.lr,
                on_optimizer_step=on_step,
            )
            resume_micro_step = 0
            global_step += step_delta
            if stop_reason:
                log_rank0(f"[stop] training interrupted: {stop_reason}")
                if is_rank0():
                    save_checkpoint(
                        args.output_dir / "last.pt", model, optimizer, scaler,
                        epoch, _STOP_MICRO, global_step, best_auc, args, config,
                    )
                    log_rank0(f"[stop] last.pt saved on interrupt micro={_STOP_MICRO}")
                if "nan_or_inf" in stop_reason:
                    save_checkpoint(
                        args.output_dir / "emergency.pt", model, optimizer, scaler,
                        epoch, 0, global_step, best_auc, args, config,
                    )
                    log_rank0("[stop] emergency checkpoint saved: emergency.pt")
                interrupted = True
                break

            if is_distributed():
                dist.barrier()
            if is_rank0():
                epoch_ckpt = args.output_dir / f"epoch_{epoch}.pt"
                save_checkpoint(
                    epoch_ckpt, model, optimizer, scaler,
                    epoch, 0, global_step, best_auc, args, config,
                )
                log_rank0(f"[ckpt] saved {epoch_ckpt.name} global_step={global_step}")
            if is_distributed():
                dist.barrier()

            val_metrics, _, val_stop = run_epoch(
                model,
                val_loader,
                args.device,
                amp_enabled,
                amp_dtype,
                args.like_weight,
                args.like_pos_weight,
                click_only=args.click_only,
                grad_clip_dense=args.grad_clip_dense,
                grad_clip_sparse=args.grad_clip_sparse,
                max_batches=args.max_eval_batches,
            )
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            if is_rank0():
                history.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                val_auc = val_metrics.get("click_auc", float("nan"))
                if not math.isnan(val_auc) and val_auc > best_auc:
                    best_auc = val_auc
                    save_checkpoint(
                        args.output_dir / "best.pt", model, optimizer, scaler,
                        epoch, 0, global_step, best_auc, args, config,
                    )
                    log_rank0(f"[ckpt] saved best.pt val_auc={best_auc:.6f}")
            if is_distributed():
                dist.barrier()

            # Periodic last.pt is written inside run_epoch for training; here we
            # also refresh it at each epoch boundary.
            if is_rank0():
                save_checkpoint(
                    args.output_dir / "last.pt", model, optimizer, scaler,
                    epoch, 0, global_step, best_auc, args, config,
                )
            if is_distributed():
                dist.barrier()

        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, model, None, None, args.device)

    if interrupted:
        if is_distributed():
            dist.barrier()
        if is_rank0():
            print(json.dumps({"interrupted": True}, ensure_ascii=False), flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    if is_distributed():
        dist.barrier()
    test_metrics = {}
    if test_loader is not None:
        test_metrics, _, _ = run_epoch(
            model,
            test_loader,
            args.device,
            amp_enabled,
            amp_dtype,
            args.like_weight,
            args.like_pos_weight,
            click_only=args.click_only,
            grad_clip_dense=args.grad_clip_dense,
            grad_clip_sparse=args.grad_clip_sparse,
            max_batches=args.max_eval_batches,
        )
    if is_rank0():
        report = {
            "config": config,
            "parameters": parameters,
            "best_val_auc": best_auc,
            "history": history,
            "test": test_metrics,
        }
        save_json(args.output_dir / "metrics.json", report)
        print(json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2), flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
