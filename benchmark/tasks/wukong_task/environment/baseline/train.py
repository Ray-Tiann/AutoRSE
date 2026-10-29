#!/usr/bin/env python3
"""Train the FuxiCTR WuKong implementation and predict without test labels."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
FUXICTR_ROOT = Path(__file__).resolve().parent / "fuxictr"
WUKONG_ROOT = FUXICTR_ROOT / "model_zoo" / "WuKong"
TEMP_ROOT = DATASETS_ROOT / ".tmp" / "wukong"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TEMP_ROOT))
os.environ.setdefault("TEMP", str(TEMP_ROOT))
os.environ.setdefault("TMP", str(TEMP_ROOT))
sys.path.insert(0, str(FUXICTR_ROOT))
sys.path.insert(0, str(WUKONG_ROOT))

from fuxictr.features import FeatureMap  # noqa: E402
from fuxictr.preprocess import FeatureProcessor, build_dataset, transform  # noqa: E402
from fuxictr.pytorch.dataloaders import RankDataLoader  # noqa: E402
from fuxictr.pytorch.torch_utils import seed_everything  # noqa: E402
from src import WuKong  # noqa: E402


def load_metadata(dataset: str) -> tuple[Path, dict]:
    dataset_dir = DATASETS_ROOT / dataset / "input"
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset is not prepared: {metadata_path}")
    return dataset_dir, json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--run-name",
        help="Write an isolated tuning run under results/tuning/DATASET/RUN_NAME",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        help="Reuse a prepared FuxiCTR data root instead of rebuilding features",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam")
    # Match the public FuxiCTR WuKong reference template.
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--compressed-features", type=int, default=40)
    parser.add_argument("--rank-k", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument(
        "--reduce-lr-on-plateau",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--embedding-regularizer")
    parser.add_argument("--net-regularizer")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RESEARCH_SEED", "2024")))
    args = parser.parse_args()

    dataset_dir, metadata = load_metadata(args.dataset)
    experiment_root = DATASETS_ROOT / args.dataset / "experiments"
    if args.run_name:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
            raise ValueError("--run-name may contain only letters, digits, dot, dash, and underscore")
        result_dir = experiment_root / args.run_name
    else:
        result_dir = experiment_root / "wukong_baseline"
    scratch_base = os.environ.get("WUKONG_SCRATCH")
    reuse_scratch = os.environ.get("WUKONG_REUSE_SCRATCH") == "1"
    scratch_run_dir = (
        Path(scratch_base) / args.dataset / (args.run_name or "main")
        if scratch_base
        else result_dir
    )
    scratch_run_dir.mkdir(parents=True, exist_ok=True)
    default_tuning_cache = experiment_root / "_data_cache" / "fuxictr_data"
    data_cache_root = args.data_cache_dir
    if data_cache_root is None and args.run_name and default_tuning_cache.exists():
        data_cache_root = default_tuning_cache
    generated_data_root = data_cache_root or (scratch_run_dir / "fuxictr_data")
    using_data_cache = data_cache_root is not None
    generated_model_root = scratch_run_dir / "fuxictr_model"
    generated_data_dir = generated_data_root / args.dataset
    generated_model_dir = generated_model_root / args.dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    if not using_data_cache and generated_data_dir.exists() and not reuse_scratch:
        shutil.rmtree(generated_data_dir)
    if generated_model_dir.exists():
        shutil.rmtree(generated_model_dir)
    generated_data_dir.mkdir(parents=True, exist_ok=True)
    generated_model_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    seed_everything(seed=args.seed)

    label_column = metadata["label_column"]
    train_path = dataset_dir / "train_data.csv"
    valid_path = dataset_dir / "validation_data.csv"
    unlabeled_path = dataset_dir / "test_data_unlabeled.csv"
    if not valid_path.exists():
        raise FileNotFoundError(
            f"A validation split is required for checkpoint selection: {valid_path}"
        )

    unlabeled = pd.read_csv(unlabeled_path)
    sample_ids = unlabeled.pop("sample_id").astype(str).tolist()
    inference_frame = unlabeled.copy()
    # Missing numeric values make pandas infer float columns and emit values
    # such as ``1.0``. FuxiCTR's Criteo schema expects integer CSV fields.
    for feature_spec in metadata["feature_config"]:
        if feature_spec.get("dtype") == "int" and feature_spec["name"] in inference_frame:
            name = feature_spec["name"]
            inference_frame[name] = (
                pd.to_numeric(inference_frame[name], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
    inference_frame[label_column] = 0.0
    inference_csv = scratch_run_dir / "test_features_with_dummy_label.csv"
    inference_frame.to_csv(inference_csv, index=False)

    # Metadata stores artifact names relative to the dataset directory so the
    # submission remains relocatable. FuxiCTR needs a concrete path while it
    # prepares its cache, so resolve those names only at runtime.
    feature_config = json.loads(json.dumps(metadata["feature_config"]))
    for feature_spec in feature_config:
        pretrained = feature_spec.get("pretrained_emb")
        if pretrained and not os.path.isabs(pretrained):
            feature_spec["pretrained_emb"] = str(dataset_dir / pretrained)

    params = {
        "dataset_id": args.dataset,
        "data_root": str(generated_data_root),
        "data_format": "csv",
        "train_data": str(train_path),
        "valid_data": str(valid_path),
        "test_data": str(inference_csv),
        "feature_cols": feature_config,
        "label_col": {"dtype": "float", "name": label_column},
        "min_categr_count": metadata.get("min_categr_count", 1),
        "rebuild_dataset": True,
    }
    if args.dataset == "criteodisplay":
        # This optional dataset adapter is not part of the MovieLens-only
        # Harbor task. Import it lazily so an absent Criteo plugin cannot stop
        # the supported MovieLens path before training begins.
        from fuxictr.datasets.criteo import CustomizedFeatureProcessor

        processor_cls = CustomizedFeatureProcessor
    else:
        processor_cls = FeatureProcessor
    feature_encoder = processor_cls(**params)
    cached_train = generated_data_dir / "train.parquet"
    cached_valid = generated_data_dir / "valid.parquet"
    cached_test = generated_data_dir / "test.parquet"
    if reuse_scratch and cached_train.exists() and cached_valid.exists():
        # The failed first run already produced train/valid parquet. Restore
        # its fitted processor and only build the missing test parquet.
        feature_encoder = feature_encoder.load_pickle(
            str(generated_data_dir / "feature_processor.pkl")
        )
        if not cached_test.exists():
            test_ddf = feature_encoder.preprocess(
                feature_encoder.read_data(str(inference_csv), data_format="csv")
            )
            transform(feature_encoder, test_ddf, "test")
        train_data, valid_data, test_data = map(str, (cached_train, cached_valid, cached_test))
    else:
        train_data, valid_data, test_data = build_dataset(feature_encoder, **params)

    feature_map = FeatureMap(args.dataset, str(generated_data_dir))
    feature_map.load(str(generated_data_dir / "feature_map.json"), params)
    # Prepared KuaiVideo features include frozen 64-d pretrained vectors.  Keep
    # their source dimension for loading, but project them to the requested
    # search dimension so every field can be stacked by WuKong.
    for feature_spec in feature_map.features.values():
        if feature_spec.get("type") == "meta":
            continue
        if "pretrained_emb" in feature_spec:
            feature_spec.setdefault(
                "pretrain_dim", feature_spec.get("embedding_dim", args.embedding_dim)
            )
        if "embedding_dim" in feature_spec:
            feature_spec["embedding_dim"] = args.embedding_dim
    model_params = {
        "model_id": f"WuKong_{args.dataset}_{args.run_name or 'main'}",
        "gpu": args.gpu,
        "task": "binary_classification",
        "model_root": str(generated_model_root),
        "optimizer": args.optimizer,
        "loss": "binary_crossentropy",
        "learning_rate": args.learning_rate,
        "metrics": ["logloss", "AUC"],
        "monitor": "AUC",
        "monitor_mode": "max",
        "early_stop_patience": args.early_stop_patience,
        "save_best_only": True,
        "eval_steps": None,
        "reduce_lr_on_plateau": args.reduce_lr_on_plateau,
        "verbose": 1,
        "embedding_regularizer": args.embedding_regularizer,
        "net_regularizer": args.net_regularizer,
        "embedding_dim": args.embedding_dim,
        "num_wukong_layers": args.num_layers,
        "lcb_features": args.compressed_features,
        "fmb_features": args.compressed_features,
        "fmb_mlp_units": [512, 256],
        "fmb_mlp_activations": "relu",
        "fmp_rank_k": args.rank_k,
        "mlp_hidden_units": [512, 256],
        "mlp_hidden_activations": "relu",
        "mlp_batch_norm": True,
        "layer_norm": True,
        "net_dropout": args.dropout,
    }
    model = WuKong(feature_map, **model_params)
    model.count_parameters()

    loader_params = {
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "data_format": "parquet",
    }
    train_gen, valid_gen, test_gen = RankDataLoader(
        feature_map, stage="both", **loader_params
    ).make_iterator()

    start = time.perf_counter()
    model.fit(train_gen, validation_data=valid_gen, epochs=args.epochs)
    training_seconds = time.perf_counter() - start
    validation_metrics = model.evaluate(valid_gen)
    scores = model.predict(test_gen)

    prediction = pd.DataFrame({"sample_id": sample_ids, "score": scores})
    prediction.to_csv(result_dir / "prediction.csv.gz", index=False, compression="gzip")
    shutil.copy2(model.checkpoint, result_dir / "checkpoint.pt")
    inference_csv.unlink(missing_ok=True)

    summary = {
        "dataset": args.dataset,
        "training_seconds": training_seconds,
        "validation_metrics": validation_metrics,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": vars(args),
        "implementation": "FuxiCTR WuKong commit 0245987fdc1d43c36164ea496a02324885a8557a",
    }
    (result_dir / "training_summary.json").write_text(
        # argparse returns ``Path`` for --data-cache-dir.  Keep the summary
        # portable without making callers special-case cached paired runs.
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    # Preserve small fitted preprocessing artifacts before deleting bulk parquet.
    preprocessing = result_dir / "preprocessing"
    preprocessing.mkdir(exist_ok=True)
    for fitted in ("feature_map.json", "feature_processor.pkl", "feature_vocab.json"):
        source = generated_data_dir / fitted
        if source.is_file():
            shutil.copy2(source, preprocessing / fitted)
    if not (preprocessing / "feature_map.json").is_file() or not (preprocessing / "feature_processor.pkl").is_file():
        raise RuntimeError("Fitted preprocessing artifacts are missing")
    if not args.keep_intermediates and not using_data_cache:
        shutil.rmtree(generated_data_root)
    if not args.keep_intermediates:
        shutil.rmtree(generated_model_root)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
