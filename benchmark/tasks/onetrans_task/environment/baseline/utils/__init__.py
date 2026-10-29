"""Utility exports with optional training dependencies loaded lazily.

Data-only commands import modules below ``utils`` too.  Importing PyTorch and
Hugging Face datasets here would make those commands require the full training
environment, so keep the historical convenience exports but resolve them only
when requested.
"""

from importlib import import_module

__all__ = [
    "json_ready_args",
    "set_seed",
    "split_indices",
    "take_rows",
    "accuracy_from_logits",
    "binary_auc_from_scores",
    "multiclass_auc_from_logits",
]

_EXPORT_MODULES = {
    "json_ready_args": ".common",
    "set_seed": ".common",
    "split_indices": ".common",
    "take_rows": ".common",
    "accuracy_from_logits": ".metrics",
    "binary_auc_from_scores": ".metrics",
    "multiclass_auc_from_logits": ".metrics",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
