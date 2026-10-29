"""
RankMixer Training Script — 8-GPU DDP

Usage:
    torchrun --nproc_per_node=8 train.py --config config.yaml --results_dir results/criteo

Or single GPU for debugging:
    python train.py --config config.yaml --no_ddp --results_dir results/criteo
"""

import os
import sys
import time
import json
import logging
import argparse
import random
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rankmixer import RankMixer
from data.dataset import build_dataloader, load_meta


# ---------------------------------------------------------------------------
# Logging setup — writes to both console and train.log
# ---------------------------------------------------------------------------

def setup_logger(log_path: str, rank: int) -> logging.Logger:
    logger = logging.getLogger("rankmixer")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File (main process only)
    if rank == 0:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ---------------------------------------------------------------------------
# AUC evaluation — returns (auc, all_preds, all_labels) on rank 0
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, rank, world_size, use_ddp):
    """
    Only rank-0 runs evaluation over the FULL val set (no DistributedSampler).
    This guarantees pred/label ordering is correct.
    """
    # Non-rank-0 processes skip evaluation entirely
    if rank != 0:
        return None, None, None

    raw_model = model.module if use_ddp else model
    raw_model.eval()

    all_preds  = []
    all_labels = []

    for x_num, x_cat, labels in tqdm(loader, desc="Eval"):
        x_num  = x_num.to(device)
        x_cat  = x_cat.to(device)

        logits = raw_model(x_num, x_cat)
        preds  = torch.sigmoid(logits.squeeze(-1))

        all_preds.append(preds.cpu())
        all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)

    if not np.isfinite(all_preds).all():
        raise ValueError("Nonfinite public-validation predictions")

    auc = roc_auc_score(all_labels, all_preds)
    return auc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Save prediction.csv
# ---------------------------------------------------------------------------

def save_prediction(preds: np.ndarray, out_path: str):
    """Save one score per line (no header) to out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, preds, fmt="%.8f")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, rank: int, local_rank: int, world_size: int,
          use_ddp: bool, results_dir: str):

    device = torch.device(f"cuda:{local_rank}")

    # Setup logger (file + console)
    log_path = os.path.join(results_dir, "train.log")
    logger   = setup_logger(log_path, rank)

    # Load metadata
    meta        = load_meta(cfg["data_dir"])
    vocab_sizes = meta["vocab_sizes"]

    # Build model
    model = RankMixer(
        num_numerical   = meta["num_numerical"],
        cat_vocab_sizes = vocab_sizes,
        embed_dim       = cfg["embed_dim"],
        num_tokens      = cfg["num_tokens"],
        hidden_dim      = cfg["hidden_dim"],
        num_layers      = cfg["num_layers"],
        ffn_ratio       = cfg["ffn_ratio"],
        num_tasks       = 1,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Separate dense params vs. embedding params
    def is_embedding(name):
        return "cat_embeds" in name

    embed_params = [p for n, p in model.named_parameters() if is_embedding(n)]
    dense_params = [p for n, p in model.named_parameters() if not is_embedding(n)]

    optimizer_dense = torch.optim.RMSprop(
        dense_params, lr=cfg["lr_dense"], alpha=0.99, eps=1e-8,
    )
    optimizer_embed = torch.optim.Adagrad(
        embed_params, lr=cfg["lr_embed"], eps=1e-8,
    )

    # LR scheduler: halve lr every epoch after epoch 1
    scheduler_dense = torch.optim.lr_scheduler.StepLR(
        optimizer_dense, step_size=1, gamma=cfg.get("lr_decay", 0.5)
    )
    scheduler_embed = torch.optim.lr_scheduler.StepLR(
        optimizer_embed, step_size=1, gamma=cfg.get("lr_decay", 0.5)
    )

    criterion = nn.BCEWithLogitsLoss()

    # DataLoaders
    train_loader = build_dataloader(
        data_dir    = cfg["data_dir"],
        split       = "train",
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        mmap        = cfg.get("mmap", True),
        distributed = use_ddp,
        seed        = cfg["seed"],
        rank        = rank,
        world_size  = world_size,
        pin_memory  = True,
    )
    # val_loader: only rank-0 uses it, no DistributedSampler to preserve order
    val_loader = build_dataloader(
        data_dir    = cfg["data_dir"],
        split       = "val",
        batch_size  = cfg["batch_size"] * 4,
        num_workers = cfg["num_workers"],
        mmap        = cfg.get("mmap", True),
        distributed = False,   # rank-0 only, full val set in order
        rank        = 0,
        world_size  = 1,
        pin_memory  = True,
    )

    # TensorBoard
    writer = None
    if is_main(rank):
        tb_dir = os.path.join(results_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"Starting training — {world_size} GPU(s)")
        logger.info(f"  batch_size per GPU : {cfg['batch_size']}")
        logger.info(f"  effective batch    : {cfg['batch_size'] * world_size}")
        logger.info(f"  steps per epoch    : {len(train_loader):,}")
        logger.info(f"  results_dir        : {results_dir}")

    os.makedirs(results_dir, exist_ok=True)
    best_auc       = 0.0
    best_preds     = None
    global_step    = 0
    train_start    = time.time()

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)

        epoch_loss   = 0.0
        t0           = time.time()
        total_steps  = len(train_loader)
        log_steps    = max(1, total_steps // 20)  # log every 5%

        for step, (x_num, x_cat, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        ):
            x_num  = x_num.to(device, non_blocking=True)
            x_cat  = x_cat.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer_dense.zero_grad()
            optimizer_embed.zero_grad()

            logits = model(x_num, x_cat).squeeze(-1)
            loss   = criterion(logits, labels)

            # All ranks must take the same failure path before backward.
            invalid = (~torch.isfinite(loss.detach())).to(dtype=torch.int32)
            if use_ddp:
                dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
            if invalid.item():
                raise FloatingPointError("Nonfinite training loss on a DDP rank")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer_dense.step()
            optimizer_embed.step()

            epoch_loss  += loss.item()
            global_step += 1

            # Log every 5% of epoch
            if is_main(rank) and (step + 1) % log_steps == 0:
                pct      = (step + 1) / total_steps * 100
                avg_loss = epoch_loss / (step + 1)
                logger.info(f"  Epoch {epoch} [{pct:5.1f}%] step={global_step}  loss={avg_loss:.4f}")
                if writer:
                    writer.add_scalar("train/loss", avg_loss, global_step)

        epoch_time = time.time() - t0
        avg_loss   = epoch_loss / len(train_loader)

        if is_main(rank):
            logger.info(f"Epoch {epoch}  loss={avg_loss:.4f}  time={epoch_time:.1f}s")

        # Keep all DDP workers aligned while rank 0 performs validation and
        # checkpoint I/O. Without these barriers, non-zero ranks can enter the
        # next epoch while rank 0 is still evaluating.
        if use_ddp:
            dist.barrier()

        # Evaluate
        auc, preds, _ = evaluate(model, val_loader, device, rank, world_size, use_ddp)

        if is_main(rank) and auc is not None:
            logger.info(f"Epoch {epoch}  Val AUC = {auc:.6f}")
            if writer:
                writer.add_scalar("val/auc", auc, epoch)

            # Save best checkpoint
            if auc > best_auc:
                best_auc   = auc
                best_preds = preds
                ckpt_path  = os.path.join(results_dir, "checkpoint.pt")
                raw_model  = model.module if use_ddp else model
                torch.save({
                    "epoch":       epoch,
                    "model_state": raw_model.state_dict(),
                    "auc":         float(auc),
                    "cfg":         cfg,
                }, ckpt_path)
                logger.info(f"  ✓ Saved best checkpoint  AUC={auc:.6f}  → {ckpt_path}")

        if use_ddp:
            dist.barrier()

        # Step LR schedulers after each epoch
        cur_lr_dense = optimizer_dense.param_groups[0]['lr']
        scheduler_dense.step()
        scheduler_embed.step()
        if is_main(rank):
            new_lr = optimizer_dense.param_groups[0]['lr']
            logger.info(f"  LR: {cur_lr_dense:.6f} → {new_lr:.6f}")

    total_time = time.time() - train_start

    if is_main(rank):
        logger.info(f"Training finished. Best Val AUC = {best_auc:.6f}  "
                    f"Total time = {total_time/60:.1f} min")

        # --- Save prediction.csv ---
        pred_path = os.path.join(results_dir, "prediction.csv")
        if best_preds is not None:
            save_prediction(best_preds, pred_path)
            logger.info(f"Saved predictions → {pred_path}  ({len(best_preds):,} rows)")

        # --- Save metrics.json ---
        metrics = {
            "AUC":              round(float(best_auc), 6),
            "train_time_min":   round(total_time / 60, 2),
            "num_epochs":       cfg["num_epochs"],
            "batch_size_per_gpu": cfg["batch_size"],
            "effective_batch":  cfg["batch_size"] * world_size,
        }
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics    → {metrics_path}")
        logger.info(f"  {metrics}")

        if writer:
            writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="config.yaml")
    parser.add_argument("--results_dir", type=str, default="results/criteo",
                        help="Directory to save checkpoint, prediction, metrics, log")
    parser.add_argument("--no_ddp",      action="store_true",
                        help="Disable DDP (single GPU / CPU debug mode)")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RESEARCH_SEED", "2024")))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_ddp = not args.no_ddp and torch.cuda.is_available()

    if use_ddp:
        rank, local_rank, world_size = setup_ddp()
    else:
        rank, local_rank, world_size = 0, 0, 1

    train(cfg, rank, local_rank, world_size, use_ddp, args.results_dir)

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
