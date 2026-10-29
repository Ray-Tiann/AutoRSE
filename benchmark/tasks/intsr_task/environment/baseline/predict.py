#!/usr/bin/env python3
"""Generate IntSR scores from label-free test candidates."""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import IntSR
from utils import const, utils
from utils.dataset import UnlabeledDataSet


def load_numpy_compatible_pickle(path):
    """Load NumPy 1.x/2.x pickles without upgrading the PyTorch runtime."""
    with open(path, "rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("numpy._core"):
                raise
            handle.seek(0)
            # Import pandas before installing the aliases so its compiled
            # extensions initialize against the runtime NumPy package.
            import pandas  # noqa: F401
            import numpy.core.multiarray as legacy_multiarray
            import numpy.core.numeric as legacy_numeric

            sys.modules.setdefault("numpy._core.multiarray", legacy_multiarray)
            sys.modules.setdefault("numpy._core.numeric", legacy_numeric)
            return pickle.load(handle)


def parser_with_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=("Amazon", "KuaiSAR"), required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--unlabeled", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--rec-checkpoint", default="")
    parser.add_argument("--src-checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random_seed", type=int, default=20230601)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--model", default="IntSR")
    parser = IntSR.parse_model_args(parser)
    preliminary, _ = parser.parse_known_args()
    utils.load_hyperparam(preliminary)
    args = parser.parse_args(namespace=preliminary)
    if not args.checkpoint and not (args.rec_checkpoint and
                                    args.src_checkpoint):
        parser.error("provide --checkpoint or both task-specific checkpoints")
    return args


def make_loader(dataset, args):
    kwargs = {
        "dataset": dataset,
        "batch_size": args.eval_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": dataset.collate_batch,
    }
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True,
                      prefetch_factor=max(2, args.eval_batch_size //
                                          args.num_workers + 1))
    return DataLoader(**kwargs)


@torch.no_grad()
def predict_task(model, loader, max_batches):
    model.eval()
    sample_ids = []
    candidate_items = []
    scores = []
    for step, batch in enumerate(loader):
        sample_ids.append(batch["sample_id"].numpy().astype(np.int64))
        candidate_items.append(
            batch["candidate_items"].numpy().astype(np.int32))
        batch = utils.batch_to_gpu(batch, model.device)
        scores.append(model.predict(batch).cpu().numpy().astype(np.float32))
        if step and step % 100 == 0:
            print(f"predicted {step * loader.batch_size:,} rows")
        if max_batches > 0 and step + 1 >= max_batches:
            break
    return (np.concatenate(sample_ids), np.concatenate(candidate_items),
            np.concatenate(scores))


def main():
    args = parser_with_model_args()
    if args.gpu == "cpu":
        args.device = torch.device("cpu")
    else:
        if args.gpu:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
    if args.data == "Amazon":
        const.init_setting_Amazon(args.train_data)
    else:
        const.init_setting_KuaiSAR(args.train_data)
    utils.setup_seed(args.random_seed)

    model = IntSR(args)
    checkpoint_by_task = {
        "rec": args.rec_checkpoint or args.checkpoint,
        "src": args.src_checkpoint or args.checkpoint,
    }
    user_vocab = utils.load_pickle(const.user_vocab)
    test_data = load_numpy_compatible_pickle(args.unlabeled)
    if set(test_data) != {"rec", "src"}:
        raise ValueError("unlabeled pickle must contain rec and src tasks")

    all_ids = []
    all_tasks = []
    all_candidates = []
    all_scores = []
    started = time.time()
    for task_id, task in enumerate(("rec", "src")):
        model.load_model(checkpoint_by_task[task])
        print(f"{task} checkpoint: {checkpoint_by_task[task]}")
        dataset = UnlabeledDataSet(test_data[task], task == "src", user_vocab)
        loader = make_loader(dataset, args)
        sample_ids, candidates, scores = predict_task(
            model, loader, args.max_eval_batches)
        all_ids.append(sample_ids)
        all_tasks.append(np.full(len(sample_ids), task_id, dtype=np.uint8))
        all_candidates.append(candidates)
        all_scores.append(scores)
        print(f"{task}: {len(sample_ids):,} rows")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        sample_ids=np.concatenate(all_ids),
        task_ids=np.concatenate(all_tasks),
        candidate_items=np.concatenate(all_candidates),
        scores=np.concatenate(all_scores),
        prediction_seconds=np.asarray([time.time() - started],
                                      dtype=np.float64),
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
