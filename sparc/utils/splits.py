"""Patient-wise cross-validation split loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch


def _to_str_list(x: Any) -> List[str]:
    """Coerce any iterable of IDs to ``list[str]``."""
    return [str(v) for v in x]


def load_splits(path: Path) -> Dict[str, List[str]]:
    """Load a patient-wise (train, val, test) split from disk.

    Two on-disk formats are accepted:

    1. **Dict** (preferred)::

           {"train": [...], "val": [...], "test": [...]}

    2. **Tuple / list** (legacy)::

           (train_ids, val_ids, test_ids, *ignored)

       Anything beyond the first three elements is dropped.

    Args:
        path: Path to a ``.pt`` / ``.pth`` file readable by :func:`torch.load`.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` with all IDs cast
        to strings.

    Raises:
        ValueError: If the file does not contain one of the supported formats
                    or is missing one of the three split keys.
    """
    obj = torch.load(path)

    if isinstance(obj, dict):
        for k in ("train", "val", "test"):
            if k not in obj:
                raise ValueError(f"Splits dict missing key: '{k}'")
        return {
            "train": _to_str_list(obj["train"]),
            "val":   _to_str_list(obj["val"]),
            "test":  _to_str_list(obj["test"]),
        }

    if isinstance(obj, (tuple, list)):
        if len(obj) < 3:
            raise ValueError(
                "Tuple/list splits must have at least 3 elements "
                f"(train, val, test), got {len(obj)}"
            )
        train_ids, val_ids, test_ids = obj[:3]
        return {
            "train": _to_str_list(train_ids),
            "val":   _to_str_list(val_ids),
            "test":  _to_str_list(test_ids),
        }

    raise ValueError(
        f"Splits file must contain a dict or tuple/list, got {type(obj)}"
    )
