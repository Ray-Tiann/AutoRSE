import json
import logging
import os
import pickle
import platform
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


GLOBAL_SEED = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GLOBAL_WORKER_ID = None


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def load_pickle(path):
    with open(path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError as error:
            if not (error.name or '').startswith('numpy._core'):
                raise
            # Public artifacts can be produced with NumPy 2 while the pinned
            # PyTorch 2.1.2 container uses NumPy 1.26. Only trusted task data is
            # loaded here; preserve the pickle's data without upgrading NumPy.
            import numpy.core.multiarray as legacy_multiarray
            import numpy.core.numeric as legacy_numeric
            sys.modules.setdefault('numpy._core.multiarray', legacy_multiarray)
            sys.modules.setdefault('numpy._core.numeric', legacy_numeric)
            handle.seek(0)
            return pickle.load(handle)


def batch_to_gpu(batch: dict, device) -> dict:
    for c in batch:
        if isinstance(batch[c], torch.Tensor):
            batch[c] = batch[c].to(device)
        elif isinstance(batch[c], List):
            batch[c] = [[p.to(device)
                         for p in k] if isinstance(k, List) else k.to(device)
                        for k in batch[c]]
    return batch


def format_metric(result_dict: Dict[str, Any]) -> str:
    assert type(result_dict) == dict
    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys()])
    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = '{}@{}'.format(metric, topk)
            m = result_dict[name]
            if isinstance(m, (float, np.floating)):
                format_str.append('{}:{:<.4f}'.format(name, m))
            elif isinstance(m, (int, np.integer)):
                format_str.append('{}:{}'.format(name, m))
            # if type(m) is float or type(m) is np.float or type(
            #         m) is np.float32 or type(m) is np.float64:
            #     format_str.append('{}:{:<.4f}'.format(name, m))
            # elif type(m) is int or type(m) is np.int or type(
            #         m) is np.int32 or type(m) is np.int64:
            #     format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


def check_dir(file_name: str):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_hyperparam(args):
    load_path = "config/{}_{}.yaml".format(args.model, args.data)
    with open(load_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in configs.items():
        setattr(args, key, value)
    return args


def set_logging(args):
    log_path = getattr(args, 'log_path', '') or (
        "output/{}/{}/logs/{}.log".format(args.data, args.model, args.time))
    check_dir(log_path)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO,
                        filename=log_path,
                        filemode='w',
                        force=True)


def write_run_metadata(path, args, started_at, ended_at, training_summary):
    if not path:
        return
    device_name = None
    gpu_memory_bytes = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    metadata = {
        'status': 'trained_not_scored',
        'dataset': args.data,
        'model': args.model,
        'seed': args.random_seed,
        'started_at': started_at.isoformat(),
        'ended_at': ended_at.isoformat(),
        'training_seconds': (ended_at - started_at).total_seconds(),
        'validation': training_summary,
        'hyperparameters': {
            name: getattr(args, name) for name in (
                'batch_size', 'eval_batch_size', 'lr', 'l2', 'dropout',
                'intsr_dim', 'qdb_layers', 'dsf_num_scenarios', 'dsf_layers',
                'prediction_mode', 'historical_query_include_item',
                'candidate_scorer_layers', 'separate_task_embedding',
                'query_backend',
                'normalize_precomputed_queries', 'intsr_embedding_init_std',
                'query_augmentation_beta', 'train_schedule',
                'epoch', 'pretrain_epoch', 'finetune_epoch', 'patience',
                'early_stop') if hasattr(args, name)
        },
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'cpu_count': os.cpu_count(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu': device_name,
            'gpu_memory_bytes': gpu_memory_bytes,
        },
        'test_metrics': None,
    }
    check_dir(path)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
