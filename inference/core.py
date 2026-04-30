"""Shared inference primitives: model loading, per-slide forward, output writing.

These helpers are cohort-agnostic. Cohort-specific paths and metadata live in
``inference/cohorts.py``; the unified CLI is ``inference/run.py``.
"""

from __future__ import annotations

import os

# h5py file locking over NFS causes hangs; disable before importing h5py.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from sparc.models.factory import build_model


def load_checkpoints(checkpoint_dir: Path) -> List[dict]:
    """Load all 5 fold checkpoints from a directory.

    Looks for fold_{0..4}_best.pt; falls back to fold_{0..4}_final.pt if best
    isn't present.
    """
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = []
    for fold in range(5):
        p = checkpoint_dir / f"fold_{fold}_best.pt"
        if not p.exists():
            p = checkpoint_dir / f"fold_{fold}_final.pt"
        if p.exists():
            ckpts.append(torch.load(p, map_location="cpu", weights_only=False))
    if not ckpts:
        raise FileNotFoundError(
            f"No fold_*_best.pt or fold_*_final.pt found in {checkpoint_dir}"
        )
    return ckpts


def load_fold_models(checkpoints_data: List[dict], device: torch.device):
    """Build all fold models on a device, with hooks capturing pre-head embeddings.

    Returns a list of (model, captured_dict, hook_handle) tuples.
    """
    fold_models = []
    for ckpt in checkpoints_data:
        m = build_model(ckpt["config"])
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval().to(device)
        captured = {}

        def _make_hook(c):
            def _hook(_mod, inp, _out):
                c["emb"] = inp[0].detach().cpu()
            return _hook

        handle = m.head.register_forward_hook(_make_hook(captured))
        fold_models.append((m, captured, handle))
    return fold_models


def fusion_needs_gep(checkpoints_data: List[dict]) -> bool:
    """Inspect a checkpoint's config to decide whether GEP features must be loaded."""
    fusion = checkpoints_data[0]["config"]["model"]["fusion"]
    return not fusion.startswith("image_only")


def load_slide_features(
    emb_path: Path,
    gep_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load patch embeddings + (optional) gene-program scores + normalised coords.

    Coords are normalised to [0, 1] per slide (matches the training data pipeline).
    """
    with h5py.File(emb_path, "r") as f:
        emb = f["features"][:].astype(np.float32)
        coords = f["coords"][:].astype(np.float32)
    lo = coords.min(axis=0, keepdims=True)
    hi = coords.max(axis=0, keepdims=True)
    coords = (coords - lo) / np.maximum(hi - lo, 1e-6)
    gep = None
    if gep_path is not None and gep_path.exists():
        with h5py.File(gep_path, "r") as f:
            gep = f["features"][:].astype(np.float32)
    return emb, gep, coords


def run_model_on_slide(
    fold_models,
    emb_t: torch.Tensor,
    gep_t: Optional[torch.Tensor],
    coords_t: torch.Tensor,
    ct_idx: int,
    device: torch.device,
    per_fold: bool = False,
):
    """Forward one slide through all fold models.

    If ``per_fold`` is False (default), returns the (mean_risk, mean_embedding)
    averaged across the 5 folds; otherwise returns lists of per-fold values.
    """
    fold_risks: List[float] = []
    fold_embs: List[np.ndarray] = []
    for model, captured, _ in fold_models:
        batch = {
            "img_feats":       [emb_t],
            "gep_feats":       [gep_t] if gep_t is not None else [None],
            "coords":          [coords_t],
            "cancer_type_idx": torch.tensor([ct_idx], device=device),
        }
        with torch.no_grad():
            out = model(batch)
        fold_risks.append(out["risk"].item())
        fold_embs.append(captured["emb"].numpy().squeeze())
    if per_fold:
        return ([np.float32(r) for r in fold_risks],
                [e.astype(np.float32) for e in fold_embs])
    return (np.float32(np.mean(fold_risks)),
            np.mean(fold_embs, axis=0).astype(np.float32))


def save_slide_prediction(
    out_path: Path,
    risk,
    embedding,
    ct_idx: int,
    per_fold: bool = False,
) -> None:
    """Write a single slide's prediction to ``out_path`` as a compressed .npz."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if per_fold:
        np.savez_compressed(
            out_path,
            risk=np.array(risk, dtype=np.float32),          # (5,)
            embedding=np.array(embedding, dtype=np.float32),  # (5, 256)
            ct_idx=np.int32(ct_idx),
        )
    else:
        np.savez_compressed(
            out_path, risk=risk, embedding=embedding, ct_idx=np.int32(ct_idx)
        )
