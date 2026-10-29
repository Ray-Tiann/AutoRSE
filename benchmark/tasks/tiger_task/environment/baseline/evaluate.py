#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import (
    UPSTREAM,
    input_dir,
    load_metadata,
    load_semantic_ids,
    load_sequences,
    make_inputs,
    working_dir,
)

import sys

sys.path.insert(0, str(UPSTREAM))
from genrec.models.tiger import Tiger


def load_model(checkpoint: Path, device: torch.device) -> tuple[Tiger, dict]:
    state = torch.load(checkpoint, map_location="cpu")
    model = Tiger(state["model_config"])
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), state["model_config"]


def semantic_lookup(semantic_ids: np.ndarray, popularity: np.ndarray) -> dict:
    lookup = defaultdict(list)
    for item_id in range(1, len(semantic_ids)):
        lookup[tuple(map(int, semantic_ids[item_id]))].append(item_id)
    for code, items in lookup.items():
        items.sort(key=lambda item: (-int(popularity[item]), item))
    return dict(lookup)


def decode_beams(tokens: np.ndarray, codebook_size: int) -> list[tuple[int, ...]]:
    output = []
    for row in tokens:
        codes = []
        valid = True
        for dimension, token in enumerate(row):
            code = int(token) - dimension * codebook_size - 1
            if code < 0 or code >= codebook_size:
                valid = False
                break
            codes.append(code)
        if valid:
            output.append(tuple(codes))
    return output


@torch.no_grad()
def recommend(
    model: Tiger,
    model_cfg: dict,
    histories: list[np.ndarray],
    batch_size: int,
    beam_size: int,
    device: torch.device,
) -> np.ndarray:
    semantic_ids = load_semantic_ids()
    metadata = load_metadata()
    popularity = np.load(input_dir() / "item_popularity.npy")
    lookup = semantic_lookup(semantic_ids, popularity)
    fallback = np.argsort(-popularity[1:], kind="stable") + 1
    output = np.empty((len(histories), 10), dtype=np.int32)
    codebook_size = int(metadata["semantic_codebook_size"])
    max_length = int(metadata["maximum_sequence_length"])
    semantic_dim = int(metadata["semantic_id_dimensions"])
    for start in range(0, len(histories), batch_size):
        stop = min(start + batch_size, len(histories))
        batch_histories = histories[start:stop]
        input_ids, attention = make_inputs(
            batch_histories, semantic_ids, max_length, codebook_size
        )
        generated = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
            num_beams=beam_size,
        )
        generated = generated[:, 1:].reshape(
            len(batch_histories), beam_size, semantic_dim
        ).cpu().numpy()
        for row, (history, beams) in enumerate(zip(batch_histories, generated)):
            seen = set(map(int, history))
            selected = []
            for code in decode_beams(beams, codebook_size):
                for item in lookup.get(code, []):
                    if item not in seen and item not in selected:
                        selected.append(item)
                        if len(selected) == 10:
                            break
                if len(selected) == 10:
                    break
            if len(selected) < 10:
                for item in fallback:
                    item = int(item)
                    if item not in seen and item not in selected:
                        selected.append(item)
                        if len(selected) == 10:
                            break
            output[start + row] = selected
    return output


def metrics(top_items: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    matches = top_items == targets[:, None]
    found = matches.any(axis=1)
    ranks = np.argmax(matches, axis=1)
    result = {}
    for cutoff in (5, 10):
        hits = found & (ranks < cutoff)
        result[f"recall_at_{cutoff}"] = float(np.mean(hits))
        result[f"ndcg_at_{cutoff}"] = float(
            np.mean(np.where(hits, 1.0 / np.log2(ranks + 2), 0.0))
        )
    result["mrr_at_10"] = float(
        np.mean(np.where(found, 1.0 / (ranks + 1), 0.0))
    )
    return result


def validation_histories(
    max_users: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    user_ids = np.load(input_dir() / "validation" / "user_ids.npy")
    targets = np.load(input_dir() / "validation" / "target_items.npy")
    if max_users is not None:
        user_ids = user_ids[:max_users]
        targets = targets[:max_users]
    store = load_sequences()
    rows = store.rows_for(user_ids)
    histories = [np.asarray(store.sequence(int(row)), dtype=np.int64) for row in rows]
    return user_ids, histories, targets


def evaluate_model(
    model: Tiger,
    config: dict,
    batch_size: int,
    beam_size: int,
    device: torch.device,
    max_users: int | None = None,
) -> dict[str, float]:
    _, histories, targets = validation_histories(max_users)
    return metrics(recommend(model, config, histories, batch_size, beam_size, device), targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=working_dir() / "tiger.pt")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    result = evaluate_model(
        model, config, args.batch_size, args.beam_size, device, args.max_users
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
