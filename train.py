"""
Training script with 5-fold cross-validation and early stopping.

Usage:
    torchrun --nproc_per_node=1 train_cv.py --config configs/base_cv.yaml

Key features:
    - Stratified 5-fold CV where EVERY sample is in the test set exactly once
    - Each fold: 3 parts train, 1 part val (early stopping), 1 part test
    - Early stopping based on validation C-index
    - Aggregates test results across all folds for final performance estimate

Results are:
    1. Logged to wandb (if enabled) - per-fold and aggregated metrics
    2. Saved to {out_dir}/cv_results_{timestamp}.json - complete results
    3. Printed to console - summary at the end
"""
from __future__ import annotations

import os
# Fix HDF5 locking issues over NFS - must be set before importing h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import contextlib
import copy
import json
import math
import multiprocessing
from collections import defaultdict
from pathlib import Path

# Use spawn instead of fork to avoid deadlocks with CUDA + multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from typing import Dict, List, Iterable, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import yaml

from sparc.data.dataset import (
    SlideBagDataset,
    slide_collate_fn,
    load_clinical_table,
    get_slide_paths,
    CANCER_TYPE_TO_IDX,
)
from sparc.data.samplers import DistributedPatientSampler
from sparc.losses import cox_loss, nll_survival_loss
from sparc.models.factory import build_model
from sparc.utils.seed import set_seed
from sparc.utils.metrics import (
    aggregate_patient_level,
    c_index,
    td_auc_simple,
)

# Avoid HDF5 locking issues with multiple workers
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ------------------- Data classes for tracking results -------------------


@dataclass
class EarlyStoppingState:
    """Tracks early stopping state."""
    patience: int = 10
    min_delta: float = 0.001
    best_score: float = -float("inf")
    best_epoch: int = 0
    counter: int = 0
    should_stop: bool = False
    best_model_state: Optional[Dict] = None
    _patience_score: float = -float("inf")

    def step(self, score: float, epoch: int, model_state: Dict) -> bool:
        # Always track the absolute best model (regardless of min_delta)
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model_state)

        # Patience resets only on significant improvements (> min_delta)
        if score > self._patience_score + self.min_delta:
            self._patience_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


@dataclass
class FoldResult:
    """Results from a single fold."""
    fold: int
    best_epoch: int
    best_val_c_index: float  # Global C-index
    best_val_macro_c_index: float  # Macro-average C-index (all cancers)
    best_val_filtered_macro_c_index: float  # Filtered macro (14 key cancers, used for early stopping)
    test_c_index: float
    test_macro_c_index: float  # Macro-average C-index on test
    test_filtered_macro_c_index: float  # Filtered macro on test
    test_aucs: Dict[str, float] = field(default_factory=dict)
    test_per_organ: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_per_cancer: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_train_patients: int = 0
    n_val_patients: int = 0
    n_test_patients: int = 0
    # Store per-patient predictions for pooled analysis
    test_patient_results: List[Dict] = field(default_factory=list)


# ------------------- DDP setup / teardown -------------------


def setup_ddp():
    """Initialize process group and return (rank, world_size, local_rank, device)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, local_rank, device


def cleanup_ddp():
    dist.destroy_process_group()


# ------------------- Evaluation -------------------


def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patient_organs: Dict[str, str],
    patient_cancer_types: Dict[str, str],
    td_times: List[float],
    desc: str = "Evaluation",
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    cancer_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Run patient-level evaluation.

    If distributed=True, gathers results from all ranks before computing metrics.
    Returns dict with metrics and per-patient results for pooled analysis.
    """
    model.eval()

    slide_patient_ids: List[str] = []
    slide_times: List[float] = []
    slide_events: List[int] = []
    slide_risks: List[float] = []
    # Track pathway norms for SQ models
    _eval_h_img_norms: List[float] = []
    _eval_h_bio_norms: List[float] = []
    _eval_gate_values: List[float] = []

    if desc:
        eval_iter = tqdm(loader, total=len(loader), desc=desc, dynamic_ncols=True)
    else:
        eval_iter = loader  # No progress bar for non-rank-0

    with torch.no_grad():
        for batch in eval_iter:
            # Skip batch if all samples were corrupted
            # In distributed mode, all ranks must agree to skip
            batch_is_none = batch is None
            if distributed and world_size > 1:
                # Synchronize: if ANY rank has None, all ranks skip
                batch_valid = torch.tensor([0 if batch_is_none else 1], device=device)
                dist.all_reduce(batch_valid, op=dist.ReduceOp.MIN)
                batch_is_none = (batch_valid.item() == 0)

            if batch_is_none:
                continue

            time = batch["time"].to(device, non_blocking=True)
            event = batch["event"].to(device, non_blocking=True)

            for k in ["img_feats", "gep_feats", "coords"]:
                if k not in batch:
                    continue
                if len(batch[k]) == 0:
                    continue
                if batch[k][0] is None:
                    continue
                batch[k] = [x.to(device, non_blocking=True) for x in batch[k]]

            # Move cancer_type_idx to device (for cancer conditioning)
            if "cancer_type_idx" in batch:
                batch["cancer_type_idx"] = batch["cancer_type_idx"].to(device, non_blocking=True)

            output = model(batch)

            # Handle both Cox (returns tensor) and NLL (returns dict) heads
            if isinstance(output, dict):
                risk = output["risk"]
            else:
                risk = output

            slide_patient_ids.extend(batch["patient_id"])
            slide_times.extend(time.cpu().numpy().tolist())
            slide_events.extend(event.cpu().numpy().astype(int).tolist())
            slide_risks.extend(risk.detach().cpu().numpy().tolist())

            # Track pathway norms (for SQ models)
            if hasattr(model, 'fusion') and hasattr(model.fusion, '_last_h_img_norm'):
                _eval_h_img_norms.append(model.fusion._last_h_img_norm)
                _eval_h_bio_norms.append(model.fusion._last_h_bio_norm)
                _eval_gate_values.append(model.fusion._last_gate_value)

    # Gather results from all ranks if distributed
    if distributed and world_size > 1:
        # Gather counts first to know sizes
        local_count = torch.tensor([len(slide_patient_ids)], device=device)
        all_counts = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)

        # Gather patient IDs (as list of lists, then flatten)
        all_patient_ids = [None for _ in range(world_size)]
        dist.all_gather_object(all_patient_ids, slide_patient_ids)
        slide_patient_ids = [pid for sublist in all_patient_ids for pid in sublist]

        # Gather times
        all_times = [None for _ in range(world_size)]
        dist.all_gather_object(all_times, slide_times)
        slide_times = [t for sublist in all_times for t in sublist]

        # Gather events
        all_events = [None for _ in range(world_size)]
        dist.all_gather_object(all_events, slide_events)
        slide_events = [e for sublist in all_events for e in sublist]

        # Gather risks
        all_risks = [None for _ in range(world_size)]
        dist.all_gather_object(all_risks, slide_risks)
        slide_risks = [r for sublist in all_risks for r in sublist]

    # Patient-level aggregation
    records = aggregate_patient_level(
        slide_patient_ids=slide_patient_ids,
        slide_times=slide_times,
        slide_events=slide_events,
        slide_risks=slide_risks,
        patient_organs=patient_organs,
        patient_cancer_types=patient_cancer_types,
        agg="mean",
    )

    if len(records) == 0:
        return {"c_index": float("nan"), "per_organ": {}, "per_cancer": {}, "patient_results": []}

    times_np = np.array([r.time for r in records], dtype=float)
    events_np = np.array([r.event for r in records], dtype=int)
    risks_np = np.array([r.risk for r in records], dtype=float)

    # Overall metrics
    overall_c = c_index(times_np, events_np, risks_np)
    aucs = td_auc_simple(times_np, events_np, risks_np, td_times)

    # Per-organ metrics
    metrics_per_organ: Dict[str, Dict[str, float]] = {}
    organs = sorted(set(r.organ for r in records if r.organ is not None))
    for org in organs:
        mask = np.array([r.organ == org for r in records], dtype=bool)
        if mask.sum() < 5:
            continue
        t_o, e_o, s_o = times_np[mask], events_np[mask], risks_np[mask]
        org_c = c_index(t_o, e_o, s_o)
        org_aucs = td_auc_simple(t_o, e_o, s_o, td_times)
        metrics_per_organ[org] = {
            "c_index": org_c,
            **{f"auc_t{int(t)}": v for t, v in org_aucs.items()},
        }

    # Per-cancer metrics (e.g., TCGA-BRCA, TCGA-LUAD)
    metrics_per_cancer: Dict[str, Dict[str, float]] = {}
    cancer_types = sorted(set(r.cancer_type for r in records if r.cancer_type is not None))
    per_cancer_c_indices = []  # For macro-average calculation
    for ct in cancer_types:
        mask = np.array([r.cancer_type == ct for r in records], dtype=bool)
        if mask.sum() < 5:
            continue
        t_c, e_c, s_c = times_np[mask], events_np[mask], risks_np[mask]
        ct_c = c_index(t_c, e_c, s_c)
        ct_aucs = td_auc_simple(t_c, e_c, s_c, td_times)
        n_patients = int(mask.sum())
        n_events = int(e_c.sum())
        metrics_per_cancer[ct] = {
            "c_index": ct_c,
            "n_patients": n_patients,
            "n_events": n_events,
            "event_rate": n_events / n_patients if n_patients > 0 else 0.0,
            **{f"auc_t{int(t)}": v for t, v in ct_aucs.items()},
        }
        if not np.isnan(ct_c):
            per_cancer_c_indices.append(ct_c)

    # Macro-average C-index: mean of per-cancer C-indices (equal weight to each cancer)
    macro_avg_c_index = float(np.mean(per_cancer_c_indices)) if per_cancer_c_indices else float("nan")

    # Filtered macro-average: specific cancers for model selection / early stopping.
    # If cancer_groups is provided (dict: group_name → [TCGA codes]), each group
    # contributes one C-index (mean of its members) — matches collaborators who
    # combine e.g. LUAD+LUSC into "lung", KIRC+KIRP+KICH into "rcc".
    # Otherwise falls back to a default flat set of key cancers.
    if cancer_groups:
        group_c_indices = []
        for members in cancer_groups.values():
            member_c = [
                metrics_per_cancer[ct]["c_index"]
                for ct in members
                if ct in metrics_per_cancer and not np.isnan(metrics_per_cancer[ct]["c_index"])
            ]
            if member_c:
                group_c_indices.append(float(np.mean(member_c)))
        filtered_macro_avg_c_index = float(np.mean(group_c_indices)) if group_c_indices else float("nan")
    else:
        _DEFAULT_VALIDATION_CANCER_TYPES = {
            "TCGA-BLCA", "TCGA-BRCA", "TCGA-COAD", "TCGA-READ",  # COADREAD combined
            "TCGA-HNSC", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LGG",
            "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD",
            "TCGA-SKCM", "TCGA-STAD", "TCGA-UCEC",
        }
        filtered_c_indices = [
            metrics_per_cancer[ct]["c_index"]
            for ct in metrics_per_cancer
            if ct in _DEFAULT_VALIDATION_CANCER_TYPES and not np.isnan(metrics_per_cancer[ct]["c_index"])
        ]
        filtered_macro_avg_c_index = float(np.mean(filtered_c_indices)) if filtered_c_indices else float("nan")

    # Per-patient results for pooled CV analysis
    patient_results = [
        {
            "patient_id": r.patient_id,
            "time": r.time,
            "event": r.event,
            "risk": r.risk,
            "organ": r.organ,
            "cancer_type": r.cancer_type,
        }
        for r in records
    ]

    result = {
        "c_index": overall_c,
        "macro_avg_c_index": macro_avg_c_index,  # Mean of ALL per-cancer C-indices
        "filtered_macro_avg_c_index": filtered_macro_avg_c_index,  # Mean of 14 key cancers (for model selection)
        **{f"auc_t{int(t)}": v for t, v in aucs.items()},
        "per_organ": metrics_per_organ,
        "per_cancer": metrics_per_cancer,
        "patient_results": patient_results,
        "n_patients": len(records),
    }
    if _eval_h_img_norms:
        result["h_img_norm"] = float(np.mean(_eval_h_img_norms))
        result["h_bio_norm"] = float(np.mean(_eval_h_bio_norms))
        result["eval_gate_mean"] = float(np.mean(_eval_gate_values))
    return result


# ------------------- Single fold training -------------------


def train_single_fold(
    cfg: Dict,
    train_slide_paths: List[Path],
    val_slide_paths: List[Path],
    test_slide_paths: List[Path],
    clinical: Dict,
    gep_dir: Optional[Path],
    patient_organs: Dict[str, str],
    patient_cancer_types: Dict[str, str],
    fold: int,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
    use_wandb: bool,
    start_epoch: int = 0,
) -> Optional[FoldResult]:
    """Train a single fold and return results."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    
    early_stop_cfg = cfg.get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 10)
    min_delta = early_stop_cfg.get("min_delta", 0.001)
    cancer_groups = cfg.get("metrics", {}).get("cancer_groups", None)
    
    # --- Datasets ---
    max_patches = data_cfg.get("max_patches", None)  # Cap patches to avoid OOM
    gep_ablation = data_cfg.get("gep_ablation", None)  # "shuffle" or "random" for ablation studies
    gep_zscore = data_cfg.get("gep_zscore", False)  # Per-patch z-score across programs

    train_dataset = SlideBagDataset(
        slide_feature_paths=train_slide_paths,
        clinical=clinical,
        gep_feature_dir=gep_dir,
        include_gep=data_cfg.get("use_gep", True),
        max_patches=max_patches,
        gep_ablation=gep_ablation,
        gep_zscore=gep_zscore,
    )

    val_dataset = SlideBagDataset(
        slide_feature_paths=val_slide_paths,
        clinical=clinical,
        gep_feature_dir=gep_dir,
        include_gep=data_cfg.get("use_gep", True),
        max_patches=max_patches,
        gep_ablation=gep_ablation,
        gep_zscore=gep_zscore,
    )

    test_dataset = SlideBagDataset(
        slide_feature_paths=test_slide_paths,
        clinical=clinical,
        gep_feature_dir=gep_dir,
        include_gep=data_cfg.get("use_gep", True),
        max_patches=max_patches,
        gep_ablation=gep_ablation,
        gep_zscore=gep_zscore,
    )

    # Check if running without validation (for fixed-epoch training)
    no_validation = len(val_slide_paths) == 0

    # --- Loaders ---
    train_sampler = DistributedPatientSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )

    loader_kwargs = dict(
        num_workers=data_cfg["num_workers"],
        collate_fn=slide_collate_fn,
        pin_memory=True,
        persistent_workers=True if data_cfg["num_workers"] > 0 else False,
        prefetch_factor=2 if data_cfg["num_workers"] > 0 else None,
        timeout=600 if data_cfg["num_workers"] > 0 else 0,  # 10min timeout (NFS can be slow)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg["batch_size"], sampler=train_sampler,
        **loader_kwargs
    )
    # Use DistributedSampler for val/test to distribute work across GPUs
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        sampler=val_sampler, **loader_kwargs
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        sampler=test_sampler, **loader_kwargs
    )

    # --- Model ---
    model = build_model(cfg).to(device)

    # --- Two-stage training: load pretrained weights (e.g., from Image Only run) ---
    pretrained_run = train_cfg.get("pretrained_run", None)
    freeze_pretrained_epochs = train_cfg.get("freeze_pretrained_epochs", 0)
    _pretrained_param_names = set()  # Track which params were loaded (for freeze/unfreeze)

    if pretrained_run is not None:
        pretrained_dir = Path(pretrained_run) / "checkpoints"
        # Try fold-specific checkpoint (fold_N_final.pt or fold_N_best.pt)
        pretrained_path = pretrained_dir / f"fold_{fold}_final.pt"
        if not pretrained_path.exists():
            pretrained_path = pretrained_dir / f"fold_{fold}_best.pt"

        if pretrained_path.exists():
            pretrained_ckpt = torch.load(pretrained_path, map_location=device)
            pretrained_sd = pretrained_ckpt["model_state_dict"]
            current_sd = model.state_dict()

            # Build mapping: handle renamed keys (e.g., ImageOnly's fusion.img_proj → fusion.img_direct_proj)
            key_remap = {}
            for pk in pretrained_sd:
                if pk in current_sd:
                    key_remap[pk] = pk
                elif pk == "fusion.img_proj.weight" and "fusion.img_direct_proj.weight" in current_sd:
                    key_remap[pk] = "fusion.img_direct_proj.weight"
                elif pk == "fusion.img_proj.bias" and "fusion.img_direct_proj.bias" in current_sd:
                    key_remap[pk] = "fusion.img_direct_proj.bias"

            # Load matching keys with compatible shapes
            loaded, skipped = [], []
            for src_key, dst_key in key_remap.items():
                if pretrained_sd[src_key].shape == current_sd[dst_key].shape:
                    current_sd[dst_key] = pretrained_sd[src_key]
                    loaded.append(f"{src_key} -> {dst_key}" if src_key != dst_key else src_key)
                    _pretrained_param_names.add(dst_key)
                else:
                    skipped.append(f"{src_key} shape {pretrained_sd[src_key].shape} != {current_sd[dst_key].shape}")

            model.load_state_dict(current_sd)
            if rank == 0:
                print(f"[Fold {fold}] Loaded {len(loaded)} params from {pretrained_path.name}")
                if skipped:
                    print(f"[Fold {fold}]   Skipped (shape mismatch): {skipped}")
                new_params = set(current_sd.keys()) - _pretrained_param_names
                print(f"[Fold {fold}]   New params (training from scratch): {len(new_params)}")
                if freeze_pretrained_epochs > 0:
                    print(f"[Fold {fold}]   Freezing pretrained params for {freeze_pretrained_epochs} epochs")
        else:
            if rank == 0:
                print(f"[Fold {fold}] WARNING: pretrained checkpoint not found at {pretrained_path}, training from scratch")

    # Use find_unused_parameters=True when freezing pretrained params (some params won't get gradients)
    _find_unused = bool(_pretrained_param_names) and freeze_pretrained_epochs > 0
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=_find_unused)

    # --- Optimizer with per-group learning rates ---
    # Supports: gate_lr (for fusion gates), pretrained_lr (for pretrained params during fine-tuning)
    gate_lr = train_cfg.get("gate_lr", None)
    pretrained_lr = train_cfg.get("pretrained_lr", None)  # LR for pretrained params after unfreeze
    base_lr = train_cfg["lr"]

    # Categorize parameters into groups
    gate_params, pretrained_params, new_params = [], [], []
    for name, param in ddp_model.named_parameters():
        if gate_lr is not None and ("fusion_gate" in name or "cancer_alpha_table" in name or "slide_alpha_mlp" in name or "cancer_program_scale" in name):
            gate_params.append(param)
        elif pretrained_lr is not None and name in _pretrained_param_names:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    param_groups = [{"params": new_params, "lr": base_lr}]
    if gate_params:
        param_groups.append({"params": gate_params, "lr": gate_lr})
    if pretrained_params:
        param_groups.append({"params": pretrained_params, "lr": pretrained_lr})

    optim = torch.optim.Adam(param_groups, weight_decay=train_cfg["weight_decay"])

    if rank == 0:
        lr_info = [f"Base LR: {base_lr} ({len(new_params)} params)"]
        if gate_params:
            lr_info.append(f"Gate LR: {gate_lr} ({len(gate_params)} params)")
        if pretrained_params:
            lr_info.append(f"Pretrained LR: {pretrained_lr} ({len(pretrained_params)} params)")
        print(f"[Fold {fold}] {' | '.join(lr_info)}")

    max_epochs = train_cfg["max_epochs"]

    # --- Learning Rate Scheduler ---
    scheduler_name = train_cfg.get("scheduler", "none")
    scheduler = None
    if scheduler_name == "cosine":
        cosine_t_max = train_cfg.get("cosine_t_max", max_epochs)
        scheduler = CosineAnnealingLR(optim, T_max=cosine_t_max, eta_min=1e-6)
        if rank == 0:
            print(f"[Fold {fold}] Using CosineAnnealingLR scheduler (T_max={cosine_t_max})")
    elif scheduler_name == "cosine_warm_restarts":
        T_0 = train_cfg.get("scheduler_T_0", 10)
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=T_0, eta_min=1e-6)
        if rank == 0:
            print(f"[Fold {fold}] Using CosineAnnealingWarmRestarts scheduler (T_0={T_0})")
    elif scheduler_name != "none":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # --- Stochastic Weight Averaging (SWA) ---
    use_swa = train_cfg.get("use_swa", False)
    swa_model = None
    swa_start = train_cfg.get("swa_start_epoch", int(max_epochs * 0.75))  # Default: last 25% of training

    if use_swa:
        swa_model = AveragedModel(ddp_model.module)
        if rank == 0:
            print(f"[Fold {fold}] SWA enabled: start_epoch={swa_start} (cosine LR continues)")
    log_every = train_cfg["log_every"]
    accum_steps = train_cfg.get("accum_steps", 1)
    val_every = train_cfg.get("val_every", 1)
    td_times = cfg.get("metrics", {}).get("td_times", [365.0, 730.0, 1095.0])

    if no_validation and rank == 0:
        print(f"[Fold {fold}] No validation set — training for {max_epochs} epochs, using final model for test.")

    # --- Loss function setup ---
    head_type = cfg["model"].get("head", "cox")
    n_bins = cfg["model"].get("n_bins", 4)
    bin_edges = None

    # Orthogonality regularization weight (V3 identity_queries)
    ortho_reg_weight = cfg["model"].get("ortho_reg_weight", 0.0)

    # Auxiliary loss weights (for multimodal training)
    use_aux_losses = cfg["model"].get("use_aux_losses", False)
    aux_loss_weight_img = train_cfg.get("aux_loss_weight_img", 0.3)
    aux_loss_weight_gep = train_cfg.get("aux_loss_weight_gep", 0.3)
    if use_aux_losses and rank == 0:
        print(f"[Fold {fold}] Auxiliary losses enabled: img_weight={aux_loss_weight_img}, gep_weight={aux_loss_weight_gep}")

    # Per-cancer auxiliary loss: compute within-cancer survival loss for target cancers
    per_cancer_loss_cfg = train_cfg.get("per_cancer_loss", {})
    use_per_cancer_loss = per_cancer_loss_cfg.get("enabled", False)
    per_cancer_loss_weight = per_cancer_loss_cfg.get("weight", 0.2)
    per_cancer_min_samples = per_cancer_loss_cfg.get("min_samples", 3)
    # Build set of target cancer_type_idx values from cancer_groups in metrics config
    per_cancer_target_indices = set()
    if use_per_cancer_loss:
        cancer_groups = cfg.get("metrics", {}).get("cancer_groups", {})
        for group_name, cancer_list in cancer_groups.items():
            for ct in cancer_list:
                idx = CANCER_TYPE_TO_IDX.get(ct, -1)
                if idx >= 0:
                    per_cancer_target_indices.add(idx)
        if rank == 0:
            print(f"[Fold {fold}] Per-cancer loss enabled: weight={per_cancer_loss_weight}, "
                  f"min_samples={per_cancer_min_samples}, target cancers={len(per_cancer_target_indices)}")

    # Mixed loss: blend NLL + Cox when using nll_surv head (0 = pure NLL, 1 = pure Cox)
    cox_mix = train_cfg.get("cox_mix", 0.0)
    if cox_mix > 0 and head_type == "nll_surv" and rank == 0:
        print(f"[Fold {fold}] Mixed loss: NLL weight={1 - cox_mix:.2f}, Cox weight={cox_mix:.2f}")

    # Alpha entropy regularizer (for InterpretableSignatureQueryFusion)
    # "target" mode: loss += λ·(H(α) - H_target)²  (bidirectional, pulls toward target)
    # "clean"  mode: loss += λ·H(α)                 (unidirectional, pushes toward sparsity)
    alpha_entropy_weight = train_cfg.get("alpha_entropy_weight", 0.0)
    alpha_entropy_target = train_cfg.get("alpha_entropy_target", 2.3)  # ln(10) ≈ 2.30 → ~10 programs
    alpha_entropy_mode = train_cfg.get("alpha_entropy_mode", "target")  # "target" or "clean"
    alpha_diversity_weight = train_cfg.get("alpha_diversity_weight", 0.0)
    if alpha_entropy_weight > 0 and rank == 0:
        if alpha_entropy_mode == "clean":
            print(f"[Fold {fold}] Alpha entropy (clean): weight={alpha_entropy_weight}")
        else:
            print(f"[Fold {fold}] Alpha entropy (target): weight={alpha_entropy_weight}, H_target={alpha_entropy_target:.2f}")
    if alpha_diversity_weight > 0 and rank == 0:
        print(f"[Fold {fold}] Alpha diversity weight: {alpha_diversity_weight}")

    # Temperature annealing for α gate (InterpretableSignatureQueryFusion)
    # τ starts warm (near-uniform α) and anneals to cold (sharper selection).
    # No entropy penalty needed — sharpening happens naturally via temperature.
    tau_start = train_cfg.get("tau_start", 5.0)   # Warm: near-uniform
    tau_end = train_cfg.get("tau_end", 1.0)        # Cold: sharper selection
    tau_anneal = train_cfg.get("tau_anneal", True)  # Whether to anneal at all
    if tau_anneal and rank == 0:
        print(f"[Fold {fold}] Temperature annealing: τ {tau_start:.1f} → {tau_end:.1f} over {max_epochs} epochs")

    if head_type == "nll_surv":
        # Compute bin edges from training data survival times using quantiles
        train_times = []
        for sp in train_slide_paths:
            patient_id = sp.stem.split("_")[0][:12]  # TCGA patient ID
            if patient_id in clinical:
                train_times.append(clinical[patient_id].time)
        train_times = np.array(train_times)
        # Use quantiles to define bin edges (excluding 0 and 100 percentiles)
        quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # e.g., [25, 50, 75] for 4 bins
        bin_boundaries = np.percentile(train_times, quantiles)
        bin_edges = torch.tensor([0.0] + list(bin_boundaries) + [float("inf")], device=device)
        if rank == 0:
            print(f"[Fold {fold}] NLL survival bin edges: {bin_edges.tolist()}")

    early_stopping = EarlyStoppingState(patience=patience, min_delta=min_delta)

    # --- Two-stage: freeze pretrained params initially ---
    if _pretrained_param_names and freeze_pretrained_epochs > 0:
        for name, param in ddp_model.module.named_parameters():
            if name in _pretrained_param_names:
                param.requires_grad = False

    # --- Training loop ---
    for epoch in range(start_epoch, max_epochs):
        # Two-stage: unfreeze pretrained params after warmup
        if _pretrained_param_names and freeze_pretrained_epochs > 0 and epoch == freeze_pretrained_epochs:
            for name, param in ddp_model.module.named_parameters():
                if name in _pretrained_param_names:
                    param.requires_grad = True
            if rank == 0:
                print(f"[Fold {fold} | Epoch {epoch}] Unfreezing pretrained params — full fine-tuning begins")

        # Temperature annealing: update τ for α gate
        if tau_anneal and hasattr(ddp_model.module, 'fusion') and hasattr(ddp_model.module.fusion, 'gate_temperature'):
            progress = epoch / max(max_epochs - 1, 1)  # 0 → 1 over training
            tau = tau_start + (tau_end - tau_start) * progress
            ddp_model.module.fusion.gate_temperature = tau
            if rank == 0 and epoch % log_every == 0:
                print(f"[Fold {fold} | Epoch {epoch}] τ = {tau:.2f}")

        ddp_model.train()
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_aux_img_loss = 0.0
        running_aux_gep_loss = 0.0
        running_per_cancer_loss = 0.0
        running_alpha_entropy = 0.0
        n_updates = 0
        risk_buffer, time_buffer, event_buffer = [], [], []
        cancer_idx_buffer = []  # For per-cancer loss
        hazards_buffer, survival_buffer = [], []  # For NLL survival head
        alpha_buffer = []  # For interpretable α entropy penalty
        # Auxiliary loss buffers (separate time/event buffers since random aux selection)
        img_aux_risk_buffer, img_aux_hazards_buffer, img_aux_survival_buffer = [], [], []
        img_aux_time_buffer, img_aux_event_buffer = [], []
        gep_aux_risk_buffer, gep_aux_hazards_buffer, gep_aux_survival_buffer = [], [], []
        gep_aux_time_buffer, gep_aux_event_buffer = [], []

        optim.zero_grad()

        if rank == 0:
            train_iter = tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc=f"Fold {fold} Epoch {epoch}", dynamic_ncols=True
            )
        else:
            train_iter = enumerate(train_loader)

        for step, batch in train_iter:
            # DDP-synchronized batch skip (all ranks must agree to skip)
            batch_is_none = batch is None
            if world_size > 1:
                batch_valid = torch.tensor([0 if batch_is_none else 1], device=device)
                dist.all_reduce(batch_valid, op=dist.ReduceOp.MIN)
                batch_is_none = (batch_valid.item() == 0)

            if batch_is_none:
                continue

            time = batch["time"].to(device, non_blocking=True)
            event = batch["event"].to(device, non_blocking=True)

            for k in ["img_feats", "gep_feats", "coords"]:
                if len(batch[k]) == 0 or batch[k][0] is None:
                    continue
                batch[k] = [x.to(device, non_blocking=True) for x in batch[k]]

            # Move cancer_type_idx to device (for cancer conditioning)
            if "cancer_type_idx" in batch:
                batch["cancer_type_idx"] = batch["cancer_type_idx"].to(device, non_blocking=True)

            output = ddp_model(batch)

            # Handle auxiliary losses mode (returns dict with "fused", "img_aux", "gep_aux")
            if isinstance(output, dict) and "fused" in output:
                # Auxiliary losses mode
                fused_out = output["fused"]
                img_aux_out = output["img_aux"]
                gep_aux_out = output["gep_aux"]

                # Handle fused output (can be tensor for Cox or dict for NLL)
                if isinstance(fused_out, dict):
                    risk_buffer.append(fused_out["risk"])
                    hazards_buffer.append(fused_out["hazards"])
                    survival_buffer.append(fused_out["survival"])
                    if "alpha" in fused_out:
                        alpha_buffer.append(fused_out["alpha"])
                else:
                    risk_buffer.append(fused_out)

                # Handle img_aux output (may be None with random aux selection)
                if img_aux_out is not None:
                    if isinstance(img_aux_out, dict):
                        img_aux_risk_buffer.append(img_aux_out["risk"])
                        img_aux_hazards_buffer.append(img_aux_out["hazards"])
                        img_aux_survival_buffer.append(img_aux_out["survival"])
                    else:
                        img_aux_risk_buffer.append(img_aux_out)
                    # Track time/event for this aux path
                    img_aux_time_buffer.append(time)
                    img_aux_event_buffer.append(event)

                # Handle gep_aux output (may be None with random aux selection)
                if gep_aux_out is not None:
                    if isinstance(gep_aux_out, dict):
                        gep_aux_risk_buffer.append(gep_aux_out["risk"])
                        gep_aux_hazards_buffer.append(gep_aux_out["hazards"])
                        gep_aux_survival_buffer.append(gep_aux_out["survival"])
                    else:
                        gep_aux_risk_buffer.append(gep_aux_out)
                    # Track time/event for this aux path
                    gep_aux_time_buffer.append(time)
                    gep_aux_event_buffer.append(event)

            # Handle standard NLL head (returns dict with "risk", "hazards", "survival")
            elif isinstance(output, dict):
                risk_buffer.append(output["risk"])
                hazards_buffer.append(output["hazards"])
                survival_buffer.append(output["survival"])
                if "alpha" in output:
                    alpha_buffer.append(output["alpha"])
            # Handle Cox head (returns tensor)
            else:
                risk_buffer.append(output)

            time_buffer.append(time)
            event_buffer.append(event)
            if use_per_cancer_loss:
                cancer_idx_buffer.append(batch["cancer_type_idx"].to(device, non_blocking=True))

            if (step + 1) % accum_steps == 0:
                b_risk = torch.cat(risk_buffer, dim=0)
                b_time = torch.cat(time_buffer, dim=0)
                b_event = torch.cat(event_buffer, dim=0)

                # Compute main (fused) loss
                if head_type == "nll_surv":
                    b_hazards = torch.cat(hazards_buffer, dim=0)
                    b_survival = torch.cat(survival_buffer, dim=0)
                    loss_fused = nll_survival_loss(b_hazards, b_survival, b_time, b_event, bin_edges)
                    if cox_mix > 0:
                        loss_fused = (1 - cox_mix) * loss_fused + cox_mix * cox_loss(b_risk, b_time, b_event)
                    hazards_buffer.clear()
                    survival_buffer.clear()
                else:
                    loss_fused = cox_loss(b_risk, b_time, b_event)

                loss = loss_fused
                aux_img_loss_val = 0.0
                aux_gep_loss_val = 0.0

                # Compute auxiliary losses if enabled
                if use_aux_losses and len(img_aux_risk_buffer) > 0:
                    b_img_aux_risk = torch.cat(img_aux_risk_buffer, dim=0)
                    b_img_aux_time = torch.cat(img_aux_time_buffer, dim=0)
                    b_img_aux_event = torch.cat(img_aux_event_buffer, dim=0)
                    if head_type == "nll_surv" and len(img_aux_hazards_buffer) > 0:
                        b_img_aux_hazards = torch.cat(img_aux_hazards_buffer, dim=0)
                        b_img_aux_survival = torch.cat(img_aux_survival_buffer, dim=0)
                        loss_img_aux = nll_survival_loss(b_img_aux_hazards, b_img_aux_survival, b_img_aux_time, b_img_aux_event, bin_edges)
                        if cox_mix > 0:
                            loss_img_aux = (1 - cox_mix) * loss_img_aux + cox_mix * cox_loss(b_img_aux_risk, b_img_aux_time, b_img_aux_event)
                        img_aux_hazards_buffer.clear()
                        img_aux_survival_buffer.clear()
                    else:
                        loss_img_aux = cox_loss(b_img_aux_risk, b_img_aux_time, b_img_aux_event)
                    loss = loss + aux_loss_weight_img * loss_img_aux
                    aux_img_loss_val = loss_img_aux.item()
                    img_aux_risk_buffer.clear()
                    img_aux_time_buffer.clear()
                    img_aux_event_buffer.clear()

                if use_aux_losses and len(gep_aux_risk_buffer) > 0:
                    b_gep_aux_risk = torch.cat(gep_aux_risk_buffer, dim=0)
                    b_gep_aux_time = torch.cat(gep_aux_time_buffer, dim=0)
                    b_gep_aux_event = torch.cat(gep_aux_event_buffer, dim=0)
                    if head_type == "nll_surv" and len(gep_aux_hazards_buffer) > 0:
                        b_gep_aux_hazards = torch.cat(gep_aux_hazards_buffer, dim=0)
                        b_gep_aux_survival = torch.cat(gep_aux_survival_buffer, dim=0)
                        loss_gep_aux = nll_survival_loss(b_gep_aux_hazards, b_gep_aux_survival, b_gep_aux_time, b_gep_aux_event, bin_edges)
                        if cox_mix > 0:
                            loss_gep_aux = (1 - cox_mix) * loss_gep_aux + cox_mix * cox_loss(b_gep_aux_risk, b_gep_aux_time, b_gep_aux_event)
                        gep_aux_hazards_buffer.clear()
                        gep_aux_survival_buffer.clear()
                    else:
                        loss_gep_aux = cox_loss(b_gep_aux_risk, b_gep_aux_time, b_gep_aux_event)
                    loss = loss + aux_loss_weight_gep * loss_gep_aux
                    aux_gep_loss_val = loss_gep_aux.item()
                    gep_aux_risk_buffer.clear()
                    gep_aux_time_buffer.clear()
                    gep_aux_event_buffer.clear()

                # Per-cancer auxiliary loss
                per_cancer_loss_val = 0.0
                if use_per_cancer_loss and len(cancer_idx_buffer) > 0:
                    b_cancer_idx = torch.cat(cancer_idx_buffer, dim=0)  # [B]
                    cancer_losses = []
                    for ct_idx in per_cancer_target_indices:
                        mask = (b_cancer_idx == ct_idx)
                        n_ct = mask.sum().item()
                        if n_ct < per_cancer_min_samples:
                            continue
                        # Need at least 1 event for Cox/NLL to be meaningful
                        ct_event = b_event[mask]
                        if ct_event.sum().item() < 1:
                            continue
                        if head_type == "nll_surv":
                            ct_loss = nll_survival_loss(
                                b_hazards[mask], b_survival[mask],
                                b_time[mask], ct_event, bin_edges,
                            )
                        else:
                            ct_loss = cox_loss(b_risk[mask], b_time[mask], ct_event)
                        cancer_losses.append(ct_loss)
                    if cancer_losses:
                        per_cancer_loss_agg = torch.stack(cancer_losses).mean()
                        loss = loss + per_cancer_loss_weight * per_cancer_loss_agg
                        per_cancer_loss_val = per_cancer_loss_agg.item()
                    cancer_idx_buffer.clear()

                # Alpha entropy: always compute for logging, optionally regularize
                alpha_entropy_val = 0.0
                if len(alpha_buffer) > 0:
                    b_alpha = torch.cat(alpha_buffer, dim=0)  # [B_accum, K]
                    entropy = -(b_alpha * torch.log(b_alpha + 1e-8)).sum(dim=-1).mean()
                    alpha_entropy_val = entropy.item()
                    if alpha_entropy_weight > 0:
                        if alpha_entropy_mode == "clean":
                            loss = loss + alpha_entropy_weight * entropy
                        else:
                            loss = loss + alpha_entropy_weight * (entropy - alpha_entropy_target) ** 2
                    alpha_buffer.clear()

                # Cross-cancer diversity: penalize all cancers selecting the same programs
                if alpha_diversity_weight > 0 and hasattr(ddp_model.module, 'fusion') and hasattr(ddp_model.module.fusion, 'get_cancer_diversity_loss'):
                    div_loss = ddp_model.module.fusion.get_cancer_diversity_loss()
                    loss = loss + alpha_diversity_weight * div_loss

                # Orthogonality regularization (V3 identity_queries)
                if ortho_reg_weight > 0 and hasattr(ddp_model.module, 'fusion') and hasattr(ddp_model.module.fusion, 'get_ortho_loss'):
                    loss = loss + ortho_reg_weight * ddp_model.module.fusion.get_ortho_loss()

                optim.zero_grad()
                loss.backward()
                optim.step()

                running_loss += loss_fused.item()
                running_aux_img_loss += aux_img_loss_val
                running_aux_gep_loss += aux_gep_loss_val
                running_per_cancer_loss += per_cancer_loss_val
                running_alpha_entropy += alpha_entropy_val
                n_updates += 1
                risk_buffer.clear()
                time_buffer.clear()
                event_buffer.clear()

                if (step % log_every == 0) and (rank == 0):
                    # Get fusion gate value if available
                    _gate_str = ""
                    _norms_str = ""
                    _base = ddp_model.module
                    if hasattr(_base, 'fusion') and hasattr(_base.fusion, '_last_gate_value'):
                        _gate_str = f" | gate={_base.fusion._last_gate_value:.4f}"
                    if hasattr(_base, 'fusion') and hasattr(_base.fusion, '_last_h_img_norm'):
                        _norms_str = f" | img={_base.fusion._last_h_img_norm:.2f} bio={_base.fusion._last_h_bio_norm:.2f}"

                    _pc_str = f" | pc={per_cancer_loss_val:.4f}" if use_per_cancer_loss else ""
                    _ent_str = f" | H(α)={alpha_entropy_val:.3f}" if alpha_entropy_val > 0 else ""
                    if use_aux_losses:
                        tqdm.write(f"[Fold {fold} | Epoch {epoch:03d} | Step {step:04d}] loss={loss.item():.4f} (fused={loss_fused.item():.4f}, img={aux_img_loss_val:.4f}, gep={aux_gep_loss_val:.4f}){_gate_str}{_norms_str}{_pc_str}{_ent_str}")
                    else:
                        tqdm.write(f"[Fold {fold} | Epoch {epoch:03d} | Step {step:04d}] loss = {loss.item():.4f}{_gate_str}{_norms_str}{_pc_str}{_ent_str}")

        # Handle leftover
        if len(risk_buffer) > 0:
            b_risk = torch.cat(risk_buffer, dim=0)
            b_time = torch.cat(time_buffer, dim=0)
            b_event = torch.cat(event_buffer, dim=0)

            if head_type == "nll_surv" and len(hazards_buffer) > 0:
                b_hazards = torch.cat(hazards_buffer, dim=0)
                b_survival = torch.cat(survival_buffer, dim=0)
                loss = nll_survival_loss(b_hazards, b_survival, b_time, b_event, bin_edges)
                if cox_mix > 0:
                    loss = (1 - cox_mix) * loss + cox_mix * cox_loss(b_risk, b_time, b_event)
            else:
                loss = cox_loss(b_risk, b_time, b_event)

            # Handle auxiliary losses for leftover (use aux-specific time/event buffers)
            if use_aux_losses and len(img_aux_risk_buffer) > 0:
                b_img_aux_risk = torch.cat(img_aux_risk_buffer, dim=0)
                b_img_aux_time = torch.cat(img_aux_time_buffer, dim=0)
                b_img_aux_event = torch.cat(img_aux_event_buffer, dim=0)
                if head_type == "nll_surv" and len(img_aux_hazards_buffer) > 0:
                    b_img_aux_hazards = torch.cat(img_aux_hazards_buffer, dim=0)
                    b_img_aux_survival = torch.cat(img_aux_survival_buffer, dim=0)
                    loss_img_aux = nll_survival_loss(b_img_aux_hazards, b_img_aux_survival, b_img_aux_time, b_img_aux_event, bin_edges)
                    if cox_mix > 0:
                        loss_img_aux = (1 - cox_mix) * loss_img_aux + cox_mix * cox_loss(b_img_aux_risk, b_img_aux_time, b_img_aux_event)
                else:
                    loss_img_aux = cox_loss(b_img_aux_risk, b_img_aux_time, b_img_aux_event)
                loss = loss + aux_loss_weight_img * loss_img_aux

            if use_aux_losses and len(gep_aux_risk_buffer) > 0:
                b_gep_aux_risk = torch.cat(gep_aux_risk_buffer, dim=0)
                b_gep_aux_time = torch.cat(gep_aux_time_buffer, dim=0)
                b_gep_aux_event = torch.cat(gep_aux_event_buffer, dim=0)
                if head_type == "nll_surv" and len(gep_aux_hazards_buffer) > 0:
                    b_gep_aux_hazards = torch.cat(gep_aux_hazards_buffer, dim=0)
                    b_gep_aux_survival = torch.cat(gep_aux_survival_buffer, dim=0)
                    loss_gep_aux = nll_survival_loss(b_gep_aux_hazards, b_gep_aux_survival, b_gep_aux_time, b_gep_aux_event, bin_edges)
                    if cox_mix > 0:
                        loss_gep_aux = (1 - cox_mix) * loss_gep_aux + cox_mix * cox_loss(b_gep_aux_risk, b_gep_aux_time, b_gep_aux_event)
                else:
                    loss_gep_aux = cox_loss(b_gep_aux_risk, b_gep_aux_time, b_gep_aux_event)
                loss = loss + aux_loss_weight_gep * loss_gep_aux

            # Per-cancer loss for leftover
            if use_per_cancer_loss and len(cancer_idx_buffer) > 0:
                b_cancer_idx = torch.cat(cancer_idx_buffer, dim=0)
                cancer_losses = []
                for ct_idx in per_cancer_target_indices:
                    mask = (b_cancer_idx == ct_idx)
                    n_ct = mask.sum().item()
                    if n_ct < per_cancer_min_samples:
                        continue
                    ct_event = b_event[mask]
                    if ct_event.sum().item() < 1:
                        continue
                    if head_type == "nll_surv":
                        ct_loss = nll_survival_loss(
                            b_hazards[mask], b_survival[mask],
                            b_time[mask], ct_event, bin_edges,
                        )
                    else:
                        ct_loss = cox_loss(b_risk[mask], b_time[mask], ct_event)
                    cancer_losses.append(ct_loss)
                if cancer_losses:
                    per_cancer_loss_agg = torch.stack(cancer_losses).mean()
                    loss = loss + per_cancer_loss_weight * per_cancer_loss_agg

            # Alpha entropy: always compute for logging, optionally regularize
            if len(alpha_buffer) > 0:
                b_alpha = torch.cat(alpha_buffer, dim=0)
                entropy = -(b_alpha * torch.log(b_alpha + 1e-8)).sum(dim=-1).mean()
                alpha_entropy_val = entropy.item()
                running_alpha_entropy += alpha_entropy_val
                if alpha_entropy_weight > 0:
                    if alpha_entropy_mode == "clean":
                        loss = loss + alpha_entropy_weight * entropy
                    else:
                        loss = loss + alpha_entropy_weight * (entropy - alpha_entropy_target) ** 2
                alpha_buffer.clear()

            # Cross-cancer diversity loss (leftover block)
            if alpha_diversity_weight > 0 and hasattr(ddp_model.module, 'fusion') and hasattr(ddp_model.module.fusion, 'get_cancer_diversity_loss'):
                div_loss = ddp_model.module.fusion.get_cancer_diversity_loss()
                loss = loss + alpha_diversity_weight * div_loss

            # Orthogonality regularization (leftover block)
            if ortho_reg_weight > 0 and hasattr(ddp_model.module, 'fusion') and hasattr(ddp_model.module.fusion, 'get_ortho_loss'):
                loss = loss + ortho_reg_weight * ddp_model.module.fusion.get_ortho_loss()

            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            n_updates += 1

        epoch_loss = running_loss / max(n_updates, 1)
        epoch_aux_img_loss = running_aux_img_loss / max(n_updates, 1) if use_aux_losses else 0.0
        epoch_aux_gep_loss = running_aux_gep_loss / max(n_updates, 1) if use_aux_losses else 0.0
        epoch_per_cancer_loss = running_per_cancer_loss / max(n_updates, 1) if use_per_cancer_loss else 0.0
        epoch_alpha_entropy = running_alpha_entropy / max(n_updates, 1) if running_alpha_entropy > 0 else 0.0

        # --- Scheduler step ---
        if scheduler is not None:
            scheduler.step()
        if use_swa and epoch >= swa_start:
            # Update averaged model (no LR change — keep cosine schedule)
            swa_model.update_parameters(ddp_model.module)

        # --- Sync all ranks before validation (critical for DDP) ---
        dist.barrier()

        # --- Validation (all ranks run to avoid timeout, only rank 0 logs) ---
        should_stop = torch.tensor([0], device=device)

        # Check for fusion gate value (works for both SQ and interpretable SQ)
        fusion_gate_val = None
        if rank == 0:
            base_model = ddp_model.module
            if hasattr(base_model, 'fusion') and hasattr(base_model.fusion, 'get_fusion_gate_value'):
                fusion_gate_val = base_model.fusion.get_fusion_gate_value()

        if no_validation:
            # No validation — just log train loss
            if rank == 0:

                _gate_str = f" | gate={fusion_gate_val:.4f}" if fusion_gate_val is not None else ""
                _ent_str = f" | H(α)={epoch_alpha_entropy:.3f}" if epoch_alpha_entropy > 0 else ""
                if use_aux_losses:
                    print(f"[Fold {fold} | Epoch {epoch:03d}] loss={epoch_loss:.4f} (img_aux={epoch_aux_img_loss:.4f}, gep_aux={epoch_aux_gep_loss:.4f}){_gate_str}{_ent_str}")
                else:
                    print(f"[Fold {fold} | Epoch {epoch:03d}] loss={epoch_loss:.4f}{_gate_str}{_ent_str}")
                if use_wandb:
                    import wandb
                    log_dict = {f"fold_{fold}/train_loss": epoch_loss, f"fold_{fold}/epoch": epoch}
                    if use_aux_losses:
                        log_dict[f"fold_{fold}/train_aux_img_loss"] = epoch_aux_img_loss
                        log_dict[f"fold_{fold}/train_aux_gep_loss"] = epoch_aux_gep_loss
                    if use_per_cancer_loss:
                        log_dict[f"fold_{fold}/train_per_cancer_loss"] = epoch_per_cancer_loss
                    if epoch_alpha_entropy > 0:
                        log_dict[f"fold_{fold}/alpha_entropy"] = epoch_alpha_entropy
                    if fusion_gate_val is not None:
                        log_dict[f"fold_{fold}/fusion_gate_mean"] = fusion_gate_val
                        # Log per-cancer gate values
                        per_cancer_gates = base_model.fusion.get_fusion_gate_per_cancer()
                        if per_cancer_gates is not None:
                            from sparc.data.dataset import CANCER_TYPE_ORDER
                            for ct_name, gate_val in zip(CANCER_TYPE_ORDER, per_cancer_gates):
                                log_dict[f"fold_{fold}/gate/{ct_name}"] = gate_val.item()
                    # Log pathway norms (training: last batch)
                    if hasattr(base_model, 'fusion') and hasattr(base_model.fusion, '_last_h_img_norm'):
                        log_dict[f"fold_{fold}/train_h_img_norm"] = base_model.fusion._last_h_img_norm
                        log_dict[f"fold_{fold}/train_h_bio_norm"] = base_model.fusion._last_h_bio_norm
                        log_dict[f"fold_{fold}/train_effective_ratio"] = base_model.fusion._last_effective_ratio
                    wandb.log(log_dict)

        elif (epoch + 1) % val_every == 0:
            # Distributed validation - each rank processes 1/world_size of data, then gather
            val_metrics = run_evaluation(
                model=ddp_model.module, loader=val_loader, device=device,
                patient_organs=patient_organs, patient_cancer_types=patient_cancer_types,
                td_times=td_times,
                desc=f"Fold {fold} Val" if rank == 0 else None,
                distributed=True, world_size=world_size, rank=rank,
                cancer_groups=cancer_groups,
            )
            val_c = val_metrics["c_index"]
            val_macro_c = val_metrics["macro_avg_c_index"]
            val_filtered_macro_c = val_metrics["filtered_macro_avg_c_index"]

            if rank == 0:
                _gate_str = f" | gate={fusion_gate_val:.4f}" if fusion_gate_val is not None else ""
                _ent_str = f" | H(α)={epoch_alpha_entropy:.3f}" if epoch_alpha_entropy > 0 else ""
                if use_aux_losses:
                    print(f"[Fold {fold} | Epoch {epoch:03d}] loss={epoch_loss:.4f} (img={epoch_aux_img_loss:.4f}, gep={epoch_aux_gep_loss:.4f}) | val C={val_c:.4f} | val macro={val_macro_c:.4f} | val filtered={val_filtered_macro_c:.4f}{_gate_str}{_ent_str}")
                else:
                    print(f"[Fold {fold} | Epoch {epoch:03d}] loss={epoch_loss:.4f}{_gate_str}{_ent_str} | val C={val_c:.4f} | val macro={val_macro_c:.4f} | val filtered={val_filtered_macro_c:.4f}")

                if use_wandb:
                    import wandb
                    log_dict = {
                        f"fold_{fold}/train_loss": epoch_loss,
                        f"fold_{fold}/val_c_index": val_c,
                        f"fold_{fold}/val_macro_avg_c_index": val_macro_c,
                        f"fold_{fold}/val_filtered_macro_c_index": val_filtered_macro_c,  # 14 key cancers
                        f"fold_{fold}/epoch": epoch,
                        **{f"fold_{fold}/val_{k}": v for k, v in val_metrics.items() if k.startswith("auc_")},
                    }
                    if use_aux_losses:
                        log_dict[f"fold_{fold}/train_aux_img_loss"] = epoch_aux_img_loss
                        log_dict[f"fold_{fold}/train_aux_gep_loss"] = epoch_aux_gep_loss
                    if use_per_cancer_loss:
                        log_dict[f"fold_{fold}/train_per_cancer_loss"] = epoch_per_cancer_loss
                    if fusion_gate_val is not None:
                        log_dict[f"fold_{fold}/fusion_gate_mean"] = fusion_gate_val
                        per_cancer_gates = base_model.fusion.get_fusion_gate_per_cancer()
                        if per_cancer_gates is not None:
                            from sparc.data.dataset import CANCER_TYPE_ORDER
                            for ct_name, gate_val in zip(CANCER_TYPE_ORDER, per_cancer_gates):
                                log_dict[f"fold_{fold}/gate/{ct_name}"] = gate_val.item()
                    # Per-cancer validation metrics
                    for ct, ct_metrics in val_metrics.get("per_cancer", {}).items():
                        log_dict[f"fold_{fold}/val/{ct}/c_index"] = ct_metrics["c_index"]
                        log_dict[f"fold_{fold}/val/{ct}/n_patients"] = ct_metrics.get("n_patients", 0)
                    # Log pathway norms (training: last batch, val: averaged over eval)
                    if hasattr(base_model, 'fusion') and hasattr(base_model.fusion, '_last_h_img_norm'):
                        log_dict[f"fold_{fold}/train_h_img_norm"] = base_model.fusion._last_h_img_norm
                        log_dict[f"fold_{fold}/train_h_bio_norm"] = base_model.fusion._last_h_bio_norm
                        log_dict[f"fold_{fold}/train_effective_ratio"] = base_model.fusion._last_effective_ratio
                    if "h_img_norm" in val_metrics:
                        log_dict[f"fold_{fold}/val_h_img_norm"] = val_metrics["h_img_norm"]
                        log_dict[f"fold_{fold}/val_h_bio_norm"] = val_metrics["h_bio_norm"]
                        log_dict[f"fold_{fold}/val_gate_mean"] = val_metrics["eval_gate_mean"]
                    wandb.log(log_dict)

                # Early stopping check - USE FILTERED MACRO-AVERAGE (14 key cancers)
                # This focuses on clinically relevant cancers with good sample sizes
                early_stop_metric = val_filtered_macro_c if not np.isnan(val_filtered_macro_c) else val_c
                if early_stopping.step(early_stop_metric, epoch, ddp_model.module.state_dict()):
                    should_stop = torch.tensor([1], device=device)

            # Broadcast early stopping decision from rank 0 to all ranks
            dist.broadcast(should_stop, src=0)

            if should_stop.item() == 1:
                if rank == 0:
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}. Best: {early_stopping.best_epoch} (filtered macro C={early_stopping.best_score:.4f})")
                break

    # --- Test evaluation ---
    result = None

    if no_validation:
        # No validation — use final model (or SWA model if enabled)
        if use_swa and swa_model is not None:
            # Update BatchNorm statistics for SWA model
            if rank == 0:
                print(f"[Fold {fold}] Updating SWA BatchNorm statistics...")
            # Use a subset of training data for BN update (full pass can be slow)
            swa_model.to(device)
            update_bn(train_loader, swa_model, device=device)
            # Copy SWA weights to the main model for evaluation
            ddp_model.module.load_state_dict(swa_model.module.state_dict())
            if rank == 0:
                print(f"[Fold {fold}] Using SWA model for test.")

        if rank == 0:
            model_type = "SWA" if use_swa else "final"
            print(f"[Fold {fold}] Using {model_type} model (epoch {max_epochs - 1}) for test.")

            # Save final checkpoint
            run_dir = Path(cfg.get("_run_dir", "runs"))
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"fold_{fold}_final.pt"

            torch.save({
                "fold": fold,
                "epoch": max_epochs - 1,
                "model_state_dict": ddp_model.module.state_dict(),
                "config": cfg,
                "use_swa": use_swa,
            }, ckpt_path)
            print(f"[Fold {fold}] Saved checkpoint to {ckpt_path}")
    else:
        # Load best model (from early stopping) and save checkpoint (rank 0 only)
        if rank == 0 and early_stopping.best_model_state is not None:
            ddp_model.module.load_state_dict(early_stopping.best_model_state)
            print(f"[Fold {fold}] Loaded best model from epoch {early_stopping.best_epoch}")

            # Save checkpoint
            run_dir = Path(cfg.get("_run_dir", "runs"))
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"fold_{fold}_best.pt"

            torch.save({
                "fold": fold,
                "epoch": early_stopping.best_epoch,
                "model_state_dict": early_stopping.best_model_state,
                "val_c_index": early_stopping.best_score,
                "config": cfg,
            }, ckpt_path)
            print(f"[Fold {fold}] Saved checkpoint to {ckpt_path}")

    # Broadcast best model to all ranks for distributed test evaluation
    for param in ddp_model.module.parameters():
        dist.broadcast(param.data, src=0)
    dist.barrier()

    # Distributed test evaluation
    test_metrics = run_evaluation(
        model=ddp_model.module, loader=test_loader, device=device,
        patient_organs=patient_organs, patient_cancer_types=patient_cancer_types,
        td_times=td_times,
        desc=f"Fold {fold} Test" if rank == 0 else None,
        distributed=True, world_size=world_size, rank=rank,
        cancer_groups=cancer_groups,
    )

    if rank == 0:
        test_macro_c = test_metrics.get("macro_avg_c_index", float("nan"))
        test_filtered_macro_c = test_metrics.get("filtered_macro_avg_c_index", float("nan"))
        print(f"\n[Fold {fold}] TEST: C={test_metrics['c_index']:.4f} | macro={test_macro_c:.4f} | filtered={test_filtered_macro_c:.4f}")
        for t in td_times:
            key = f"auc_t{int(t)}"
            print(f"  {key}: {test_metrics.get(key, float('nan')):.4f}")

        # Print per-cancer metrics
        print(f"\n[Fold {fold}] Per-cancer test C-index:")
        for ct, ct_metrics in sorted(test_metrics.get("per_cancer", {}).items()):
            print(f"  {ct}: {ct_metrics['c_index']:.4f} (n={ct_metrics.get('n_patients', 0)})")

        if use_wandb:
            import wandb
            log_dict = {
                f"fold_{fold}/test_c_index": test_metrics["c_index"],
                f"fold_{fold}/test_macro_avg_c_index": test_metrics.get("macro_avg_c_index", float("nan")),
                f"fold_{fold}/test_filtered_macro_c_index": test_metrics.get("filtered_macro_avg_c_index", float("nan")),
                **{f"fold_{fold}/test_{k}": v for k, v in test_metrics.items() if k.startswith("auc_")},
            }
            # Per-cancer test metrics
            for ct, ct_metrics in test_metrics.get("per_cancer", {}).items():
                for metric_name, metric_val in ct_metrics.items():
                    log_dict[f"fold_{fold}/test/{ct}/{metric_name}"] = metric_val
            if "h_img_norm" in test_metrics:
                log_dict[f"fold_{fold}/final_test_h_img_norm"] = test_metrics["h_img_norm"]
                log_dict[f"fold_{fold}/final_test_h_bio_norm"] = test_metrics["h_bio_norm"]
                log_dict[f"fold_{fold}/final_test_gate_mean"] = test_metrics["eval_gate_mean"]

            # Bar chart: per-cancer summary (n_patients, n_events, c_index)
            per_cancer = test_metrics.get("per_cancer", {})
            if per_cancer:
                cancer_table = wandb.Table(
                    columns=["cancer_type", "n_patients", "n_events", "event_rate", "c_index"],
                    data=[
                        [ct, m["n_patients"], m["n_events"], m["event_rate"], m["c_index"]]
                        for ct, m in sorted(per_cancer.items())
                    ]
                )
                log_dict[f"fold_{fold}/per_cancer_summary"] = cancer_table
                # Bar charts
                log_dict[f"fold_{fold}/charts/n_patients_by_cancer"] = wandb.plot.bar(
                    cancer_table, "cancer_type", "n_patients", title="Patients per Cancer Type"
                )
                log_dict[f"fold_{fold}/charts/c_index_by_cancer"] = wandb.plot.bar(
                    cancer_table, "cancer_type", "c_index", title="C-index per Cancer Type"
                )

            wandb.log(log_dict)

        if 'val_metrics' not in locals():
            val_metrics = {}
            
        result = FoldResult(
            fold=fold,
            best_epoch=early_stopping.best_epoch,
            best_val_c_index=val_metrics.get("c_index", float("nan")),
            best_val_macro_c_index=val_metrics.get("macro_avg_c_index", float("nan")),
            best_val_filtered_macro_c_index=early_stopping.best_score,  # Filtered macro (early stopping metric)
            test_c_index=test_metrics["c_index"],
            test_macro_c_index=test_metrics.get("macro_avg_c_index", float("nan")),
            test_filtered_macro_c_index=test_metrics.get("filtered_macro_avg_c_index", float("nan")),
            test_aucs={k: v for k, v in test_metrics.items() if k.startswith("auc_")},
            test_per_organ=test_metrics.get("per_organ", {}),
            test_per_cancer=test_metrics.get("per_cancer", {}),
            n_train_patients=len(train_dataset),
            n_val_patients=len(val_dataset),
            n_test_patients=test_metrics.get("n_patients", 0),
            test_patient_results=test_metrics.get("patient_results", []),
        )

    # Synchronize ALL ranks before returning (prevents timeout on next fold)
    dist.barrier()
    return result


# ------------------- CV split creation -------------------


def load_external_splits(
    splits_csv: Path,
    img_feature_dir: Path,
    clinical: Dict,
    n_folds: int = 5,
    no_validation: bool = False,
    val_fraction: float = 0.15,
    cancer_group: Optional[str] = None,
    gep_feature_dir: Optional[Path] = None,
    dx_only: bool = False,
) -> List[Tuple[List[Path], List[Path], List[Path]]]:
    """
    Load pre-defined splits from an external CSV file.

    The CSV should have columns: slide_id, fold, split, cancer_group
    where split is 'train' or 'val' (val is used as test in their setup).

    Since external splits only have train/val (no separate validation for early stopping),
    we'll use val as test and split train into train/val ourselves.

    If no_validation=True, all train samples are used for training (no val split).
    This is useful when training for a fixed number of epochs without early stopping.

    If cancer_group is specified (e.g., "tcga-brca"), only slides from that cancer are used.

    If gep_feature_dir is provided, slides are filtered to those with both image AND GEP
    features, ensuring image-only and multimodal models train/test on identical patients.

    Slides are filtered to only those with both a feature file AND a clinical entry.

    Returns list of (train_paths, val_paths, test_paths) for each fold.
    """
    df = pd.read_csv(splits_csv)

    # Filter by cancer group if specified (for per-cancer training)
    if cancer_group is not None:
        if "cancer_group" not in df.columns:
            raise ValueError(f"External splits CSV doesn't have 'cancer_group' column for per-cancer filtering.")
        original_count = len(df)
        df = df[df["cancer_group"] == cancer_group]
        print(f"[Per-cancer] Filtered to {cancer_group}: {original_count} -> {len(df)} samples")
        if len(df) == 0:
            raise ValueError(f"No samples found for cancer_group '{cancer_group}'. Available: {df['cancer_group'].unique().tolist()}")

    # Build slide_id -> path mapping (DX-only if requested)
    slide_to_path = {}
    for ext in [".npz", ".h5"]:
        for p in img_feature_dir.glob(f"*{ext}"):
            if dx_only and "-DX" not in p.stem and "-dx" not in p.stem:
                continue
            slide_to_path[p.stem] = p

    # Filter to paired slides (both image AND GEP) for fair model comparison
    if gep_feature_dir is not None and gep_feature_dir.exists():
        original = len(slide_to_path)
        paired = {}
        for stem, path in slide_to_path.items():
            gep_path = gep_feature_dir / f"{stem}.npz"
            if not gep_path.exists():
                gep_path = gep_feature_dir / f"{stem}.h5"
            if gep_path.exists():
                paired[stem] = path
        slide_to_path = paired

    def _has_clinical(slide_id: str) -> bool:
        patient_id = "-".join(slide_id.split("-")[:3])
        return patient_id in clinical

    # Warn once about slides dropped for missing clinical data
    all_slides_in_csv = set(df["slide_id"].tolist())
    missing_clinical = {s for s in all_slides_in_csv if s in slide_to_path and not _has_clinical(s)}
    if missing_clinical:
        print(f"[WARNING] {len(missing_clinical)} slides in external splits have no clinical entry — skipping them.")

    fold_splits = []

    for fold in range(n_folds):
        fold_df = df[df["fold"] == fold]

        # Their "val" is our "test" (the held-out evaluation set)
        test_slides = fold_df[fold_df["split"] == "val"]["slide_id"].tolist()
        train_slides = fold_df[fold_df["split"] == "train"]["slide_id"].tolist()

        # Convert to paths (skip slides without features or clinical data)
        test_paths = [slide_to_path[s] for s in test_slides if s in slide_to_path and _has_clinical(s)]

        # Get all train paths (filtered)
        train_paths_all = [slide_to_path[s] for s in train_slides if s in slide_to_path and _has_clinical(s)]

        if no_validation:
            # Use all train data, no validation split (for fixed-epoch training)
            train_paths = train_paths_all
            val_paths = []
        else:
            # Stratified train/val split (90/10) by cancer type for early stopping.
            # Ensures every cancer type is represented in both train and val.
            rng = np.random.RandomState(42 + fold)

            # Build slide_id -> cancer_type lookup from the CSV
            train_df = fold_df[fold_df["split"] == "train"]
            slide_to_cancer = dict(zip(train_df["slide_id"], train_df["cancer_type"]))

            # Group train paths by cancer type
            cancer_to_indices = defaultdict(list)
            for i, p in enumerate(train_paths_all):
                ctype = slide_to_cancer.get(p.stem, "unknown")
                cancer_to_indices[ctype].append(i)

            val_indices = []
            train_indices = []
            for ctype, idxs in cancer_to_indices.items():
                idxs = np.array(idxs)
                rng.shuffle(idxs)
                n_val = max(1, round(len(idxs) * val_fraction))
                val_indices.extend(idxs[:n_val].tolist())
                train_indices.extend(idxs[n_val:].tolist())

            val_paths = [train_paths_all[i] for i in val_indices]
            train_paths = [train_paths_all[i] for i in train_indices]

        fold_splits.append((train_paths, val_paths, test_paths))

    return fold_splits


def create_cv_splits_full(
    slide_paths: List[Path],
    clinical: Dict,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Path], List[Path], List[Path]]]:
    """
    Create stratified k-fold splits at the PATIENT level.
    
    Returns list of (train_paths, val_paths, test_paths) for each fold.
    - test: 1/k of data (rotates each fold so every sample tested once)
    - val: 1/(k-1) of remaining (for early stopping)
    - train: rest
    """
    # Map slides to patients
    patient_to_slides: Dict[str, List[Path]] = {}
    patient_events: Dict[str, int] = {}
    
    for path in slide_paths:
        stem = path.stem
        patient_id = None
        
        # Try to match patient ID
        for pid in clinical.keys():
            if stem.startswith(pid) or stem == pid:
                patient_id = pid
                break
        
        if patient_id is None:
            parts = stem.split("-")
            if len(parts) >= 3:
                patient_id = "-".join(parts[:3])
        
        if patient_id and patient_id in clinical:
            if patient_id not in patient_to_slides:
                patient_to_slides[patient_id] = []
                patient_events[patient_id] = clinical[patient_id].event
            patient_to_slides[patient_id].append(path)
    
    patients = np.array(list(patient_to_slides.keys()))
    events = np.array([patient_events[p] for p in patients])

    # Check if we have enough patients
    if len(patients) == 0:
        raise ValueError(
            f"No patients found with matching slides and clinical data!\n"
            f"Total slides provided: {len(slide_paths)}\n"
            f"Patients in clinical table: {len(clinical)}\n"
            f"Check that slide naming matches TCGA format (TCGA-XX-XXXX-...)"
        )

    if len(patients) < n_folds:
        raise ValueError(
            f"Not enough patients ({len(patients)}) for {n_folds}-fold cross-validation. "
            f"Need at least {n_folds} patients."
        )

    # Outer CV for test folds
    outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_splits = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(outer_skf.split(patients, events)):
        test_patients = set(patients[test_idx])
        trainval_patients = patients[trainval_idx]
        trainval_events = events[trainval_idx]
        
        # Inner split: separate val from train (use 1/(n_folds-1) for val)
        inner_skf = StratifiedKFold(n_splits=n_folds-1, shuffle=True, random_state=seed + fold_idx)
        train_idx_inner, val_idx_inner = next(inner_skf.split(trainval_patients, trainval_events))
        
        train_patients = set(trainval_patients[train_idx_inner])
        val_patients = set(trainval_patients[val_idx_inner])
        
        # Convert to slide paths
        train_paths, val_paths, test_paths = [], [], []
        for pid, paths in patient_to_slides.items():
            if pid in train_patients:
                train_paths.extend(paths)
            elif pid in val_patients:
                val_paths.extend(paths)
            elif pid in test_patients:
                test_paths.extend(paths)
        
        fold_splits.append((train_paths, val_paths, test_paths))
    
    return fold_splits


# ------------------- Main -------------------


def main_worker(config_path: str, dx_only: bool = False, resume_dir: str = None, cancer_type: str = None, single_fold: int = None, start_epoch: int = 0):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank, device = setup_ddp()
    base_seed = cfg["training"]["seed"]
    set_seed(base_seed, rank=rank)

    n_folds = cfg.get("cross_validation", {}).get("n_folds", 5)

    if "early_stopping" not in cfg:
        cfg["early_stopping"] = {"patience": 10, "min_delta": 0.001}

    # Check for resume_dir in config if not provided via command line
    if resume_dir is None:
        resume_dir = cfg.get("logging", {}).get("resume_dir", None)

    # Handle resume vs new run
    start_fold = 0
    loaded_fold_results: List[FoldResult] = []

    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.exists():
            raise ValueError(f"Resume directory does not exist: {run_dir}")

        # Load config from resumed run (use original config)
        resumed_config_path = run_dir / "config.yaml"
        if resumed_config_path.exists():
            with open(resumed_config_path) as f:
                cfg = yaml.safe_load(f)
            # Update n_folds and seed from resumed config
            n_folds = cfg.get("cross_validation", {}).get("n_folds", 5)
            base_seed = cfg["training"]["seed"]
            if rank == 0:
                print(f"Loaded config from resumed run: {resumed_config_path}")

        # Detect completed folds by checking for checkpoint files
        ckpt_dir = run_dir / "checkpoints"
        for fold_idx in range(n_folds):
            ckpt_best = ckpt_dir / f"fold_{fold_idx}_best.pt"
            ckpt_final = ckpt_dir / f"fold_{fold_idx}_final.pt"
            fold_result_path = run_dir / f"fold_{fold_idx}_result.json"
            if (ckpt_best.exists() or ckpt_final.exists()) and fold_result_path.exists():
                # Load saved fold result
                with open(fold_result_path) as f:
                    fold_data = json.load(f)
                loaded_fold_results.append(FoldResult(**fold_data))
                start_fold = fold_idx + 1
            else:
                break

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"RESUMING from {run_dir}")
            print(f"Completed folds: {start_fold}/{n_folds}")
            print(f"Starting from fold {start_fold + 1}")
            print(f"{'='*60}\n")
    else:
        # Create unique run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = cfg["logging"].get("wandb_run_name", "run")
        # Include cancer type / fold in run directory name
        dir_name = run_name
        if cancer_type is not None:
            dir_name = f"{cancer_type}_{dir_name}"
        if single_fold is not None:
            dir_name = f"{dir_name}_fold{single_fold}"
        run_dir = Path(cfg.get("logging", {}).get("out_dir", "runs")) / f"{dir_name}_{timestamp}"

        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            # Save config to run directory
            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(cfg, f)
            # Also save cancer type if doing per-cancer training
            if cancer_type is not None:
                with open(run_dir / "cancer_type.txt", "w") as f:
                    f.write(cancer_type)

    # Store run_dir in cfg so train_single_fold can access it
    cfg["_run_dir"] = str(run_dir)

    use_wandb = cfg.get("logging", {}).get("use_wandb", True)
    if rank == 0 and use_wandb:
        import wandb
        wandb_run_name = cfg["logging"].get("wandb_run_name", None)
        if wandb_run_name:
            wandb_run_name = f"{wandb_run_name}_{n_folds}fold_cv"

        # For per-cancer training, use the cancer type as wandb group
        wandb_group = None
        if cancer_type is not None:
            wandb_group = cancer_type
            if wandb_run_name:
                wandb_run_name = f"{cancer_type}_{wandb_run_name}"
            else:
                wandb_run_name = f"{cancer_type}_{n_folds}fold_cv"

        # Append fold number to wandb run name for single-fold runs
        if single_fold is not None and wandb_run_name:
            wandb_run_name = f"{wandb_run_name}_fold{single_fold}"

        # Check if resuming an existing wandb run
        wandb_id_file = run_dir / "wandb_run_id.txt"
        if resume_dir and wandb_id_file.exists():
            # Resume existing wandb run
            wandb_run_id = wandb_id_file.read_text().strip()
            wandb.init(
                project=cfg["logging"].get("wandb_project", "survival_tcga"),
                id=wandb_run_id,
                resume="must",
                config=cfg,
            )
            print(f"Resumed wandb run: {wandb_run_id}")
        else:
            # Start new wandb run
            wandb.init(
                project=cfg["logging"].get("wandb_project", "survival_tcga"),
                name=wandb_run_name,
                group=wandb_group,  # Group by cancer type for per-cancer training
                config=cfg,
            )
            # Save run ID for potential resume later
            wandb_id_file.write_text(wandb.run.id)
            print(f"Started new wandb run: {wandb.run.id}")

    # --- Data ---
    data_cfg = cfg["data"]
    clinical_csv = Path(data_cfg["clinical_csv"])
    img_dir = Path(data_cfg["img_feature_dir"])
    # Always resolve gep_feature_dir for paired filtering (even if use_gep=False)
    gep_dir_path = Path(data_cfg["gep_feature_dir"]) if "gep_feature_dir" in data_cfg else None
    gep_dir = gep_dir_path if data_cfg.get("use_gep", True) else None

    # Load clinical table - check if we need to compute time from columns or if it's pre-computed
    compute_time_cols = data_cfg.get("compute_time_from_cols", None)
    if compute_time_cols is None and "time" not in pd.read_csv(clinical_csv, nrows=1).columns:
        # Old format - use default progression/recurrence columns
        compute_time_cols = {
            "event_flag": "progression_recurrence_event",
            "event_time": "days_to_progression_recurrence",
            "censor_time": "max_follow_up_days",
        }

    clinical = load_clinical_table(
        clinical_csv,
        compute_time_from_cols=compute_time_cols,
    )

    patient_organs: Dict[str, str] = {}
    patient_cancer_types: Dict[str, str] = {}
    for pid, rec in clinical.items():
        if rec.organ is not None:
            patient_organs[pid] = rec.organ
        if rec.cancer_type is not None:
            patient_cancer_types[pid] = rec.cancer_type

    all_slide_paths = get_slide_paths(img_dir)

    # Filter for DX slides only if requested
    if dx_only:
        original_count = len(all_slide_paths)
        all_slide_paths = [p for p in all_slide_paths if "-DX" in p.stem or "-dx" in p.stem]
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"FILTERING FOR DX (DIAGNOSTIC) SLIDES ONLY")
            print(f"Filtered {original_count} -> {len(all_slide_paths)} slides")
            print(f"{'='*60}")

    # Note: per-cancer filtering (--cancer) is applied AFTER CV split creation
    # so that fold assignments match the pan-cancer run.

    # Filter to paired slides (both image AND GEP features exist) for fair comparison
    # This ensures image-only and multimodal models are evaluated on the same patients.
    if gep_dir_path is not None and gep_dir_path.exists():
        original_count = len(all_slide_paths)
        paired_paths = []
        for p in all_slide_paths:
            gep_path = gep_dir_path / f"{p.stem}.npz"
            if not gep_path.exists():
                gep_path = gep_dir_path / f"{p.stem}.h5"
            if gep_path.exists():
                paired_paths.append(p)
        all_slide_paths = paired_paths

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"RUN DIRECTORY: {run_dir}")
        print(f"{'='*60}")
        print(f"Total slides: {len(all_slide_paths)}")
        if dx_only:
            print(f"Mode: DX slides only (diagnostic)")
        else:
            print(f"Mode: All slides (DX + TS/BS/MS)")
        if cancer_type:
            print(f"Cancer type: {cancer_type} (per-cancer training)")
        else:
            print(f"Cancer type: Pan-cancer (all types)")
        if single_fold is not None:
            print(f"Running SINGLE FOLD {single_fold} only (of {n_folds}-fold CV)")
        else:
            print(f"Running {n_folds}-fold cross-validation (full CV - every sample tested once)")

    # --- Create CV splits ---
    external_splits_path = cfg.get("cross_validation", {}).get("external_splits", None)

    if external_splits_path:
        # Use external pre-defined splits
        external_splits_path = Path(external_splits_path)
        if rank == 0:
            print(f"\nUsing EXTERNAL SPLITS from: {external_splits_path}")
        no_validation = cfg.get("cross_validation", {}).get("no_validation", False)
        val_fraction = cfg.get("early_stopping", {}).get("val_fraction", 0.15)
        # Map TCGA cancer type to external splits cancer_group (e.g., TCGA-BRCA -> tcga-brca)
        ext_cancer_group = None
        if cancer_type is not None:
            ext_cancer_group = cancer_type.lower().replace("_", "-")
            if not ext_cancer_group.startswith("tcga-"):
                ext_cancer_group = f"tcga-{ext_cancer_group}"
        cv_splits = load_external_splits(
            external_splits_path, img_dir, clinical,
            n_folds=n_folds, no_validation=no_validation, val_fraction=val_fraction,
            cancer_group=ext_cancer_group,
            gep_feature_dir=gep_dir_path, dx_only=dx_only,
        )
    else:
        # Create our own stratified splits
        cv_splits = create_cv_splits_full(all_slide_paths, clinical, n_folds=n_folds, seed=base_seed)
        no_validation = cfg.get("cross_validation", {}).get("no_validation", False)
        if no_validation:
            # Merge val into train — train on all non-test data
            cv_splits = [(train + val, [], test) for train, val, test in cv_splits]
            if rank == 0:
                print("no_validation=True: merging val into train (all non-test data used for training)")

    # Check if we have any patients after splitting
    if rank == 0:
        if len(cv_splits) == 0:
            raise ValueError("No valid cross-validation splits created! Check that slide paths match clinical data.")
        print(f"Successfully created {len(cv_splits)} folds")

    # Filter splits by cancer type AFTER creation (preserves fold assignments from pan-cancer)
    if cancer_type is not None and not external_splits_path:
        cancer_patients = {pid for pid, ct in patient_cancer_types.items() if ct == cancer_type}

        def _filter_paths(paths):
            return [p for p in paths if "-".join(p.stem.split("-")[:3]) in cancer_patients]

        filtered_splits = []
        for train, val, test in cv_splits:
            filtered_splits.append((_filter_paths(train), _filter_paths(val), _filter_paths(test)))
        cv_splits = filtered_splits

        # Report
        total_slides = sum(len(tr) + len(va) + len(te) for tr, va, te in cv_splits)
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"PER-CANCER FILTERING: {cancer_type}")
            print(f"Patients: {len(cancer_patients)} | Slides in splits: {total_slides}")
            print(f"(Fold assignments match pan-cancer run with same seed)")
            print(f"{'='*60}")

        if total_slides == 0:
            raise ValueError(f"No slides found for cancer type {cancer_type}! Check the cancer type name.")

    # --- Run folds ---
    fold_results: List[FoldResult] = list(loaded_fold_results)  # Start with any resumed results
    all_test_patient_results: List[Dict] = []  # Pooled across folds

    # Collect patient results from resumed folds
    for r in loaded_fold_results:
        all_test_patient_results.extend(r.test_patient_results)

    for fold, (train_paths, val_paths, test_paths) in enumerate(cv_splits):
        # Skip folds not matching --fold target
        if single_fold is not None and fold != single_fold:
            continue

        # Skip already completed folds when resuming
        if fold < start_fold:
            if rank == 0:
                print(f"\n[Fold {fold + 1}/{n_folds}] Already completed, skipping...")
            continue

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{n_folds}")
            print(f"Train: {len(train_paths)} slides | Val: {len(val_paths)} slides | Test: {len(test_paths)} slides")
            print(f"{'='*60}\n")

        set_seed(base_seed + fold, rank=rank)

        result = train_single_fold(
            cfg=cfg,
            train_slide_paths=train_paths,
            val_slide_paths=val_paths,
            test_slide_paths=test_paths,
            clinical=clinical,
            gep_dir=gep_dir,
            patient_organs=patient_organs,
            patient_cancer_types=patient_cancer_types,
            fold=fold,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            use_wandb=use_wandb,
            start_epoch=start_epoch,
        )

        if result is not None:
            fold_results.append(result)
            all_test_patient_results.extend(result.test_patient_results)

            # Save fold result incrementally (for resume capability)
            if rank == 0:
                fold_result_path = run_dir / f"fold_{fold}_result.json"
                # Convert to dict for JSON serialization (exclude non-serializable fields)
                fold_dict = {
                    "fold": result.fold,
                    "best_epoch": result.best_epoch,
                    "best_val_c_index": result.best_val_c_index,
                    "best_val_macro_c_index": result.best_val_macro_c_index,
                    "best_val_filtered_macro_c_index": result.best_val_filtered_macro_c_index,
                    "test_c_index": result.test_c_index,
                    "test_macro_c_index": result.test_macro_c_index,
                    "test_filtered_macro_c_index": result.test_filtered_macro_c_index,
                    "test_aucs": result.test_aucs,
                    "test_per_organ": result.test_per_organ,
                    "test_per_cancer": result.test_per_cancer,
                    "n_train_patients": result.n_train_patients,
                    "n_val_patients": result.n_val_patients,
                    "n_test_patients": result.n_test_patients,
                    "test_patient_results": result.test_patient_results,
                }
                with open(fold_result_path, "w") as f:
                    json.dump(fold_dict, f, indent=2)
                print(f"[Fold {fold}] Saved fold result to {fold_result_path}")

    # --- Report corrupted files ---
    if rank == 0:
        n_corrupted = SlideBagDataset.get_corrupted_count()
        if n_corrupted > 0:
            print(f"\n{'='*60}")
            print(f"WARNING: {n_corrupted} corrupted GEP files were skipped")
            print(f"{'='*60}")
            for path, error in list(SlideBagDataset.get_corrupted_files().items())[:10]:
                print(f"  {Path(path).name}: {error[:50]}...")
            if n_corrupted > 10:
                print(f"  ... and {n_corrupted - 10} more")
            print(f"{'='*60}\n")

    # --- Aggregate and save results ---
    if rank == 0 and len(fold_results) > 0:
        td_times = cfg.get("metrics", {}).get("td_times", [365.0, 730.0, 1095.0])
        
        # Per-fold metrics
        val_c_indices = [r.best_val_c_index for r in fold_results]
        val_macro_c_indices = [r.best_val_macro_c_index for r in fold_results]
        val_filtered_macro_c_indices = [r.best_val_filtered_macro_c_index for r in fold_results]
        test_c_indices = [r.test_c_index for r in fold_results]
        test_macro_c_indices = [r.test_macro_c_index for r in fold_results]
        test_filtered_macro_c_indices = [r.test_filtered_macro_c_index for r in fold_results]

        # Pooled test metrics (compute C-index on ALL test predictions together)
        pooled_times = np.array([p["time"] for p in all_test_patient_results])
        pooled_events = np.array([p["event"] for p in all_test_patient_results])
        pooled_risks = np.array([p["risk"] for p in all_test_patient_results])
        
        pooled_c_index = c_index(pooled_times, pooled_events, pooled_risks)
        pooled_aucs = td_auc_simple(pooled_times, pooled_events, pooled_risks, td_times)

        # Pooled per-organ metrics
        pooled_per_organ: Dict[str, Dict[str, float]] = {}
        organs = set(p["organ"] for p in all_test_patient_results if p["organ"] is not None)
        for org in sorted(organs):
            org_mask = [p["organ"] == org for p in all_test_patient_results]
            org_times = pooled_times[org_mask]
            org_events = pooled_events[org_mask]
            org_risks = pooled_risks[org_mask]

            if len(org_times) >= 5:
                org_c = c_index(org_times, org_events, org_risks)
                org_aucs = td_auc_simple(org_times, org_events, org_risks, td_times)
                pooled_per_organ[org] = {
                    "c_index": float(org_c),
                    "n_patients": int(len(org_times)),
                    **{f"auc_t{int(t)}": float(v) for t, v in org_aucs.items()},
                }

        # Pooled per-cancer metrics and macro-average
        pooled_per_cancer: Dict[str, Dict[str, float]] = {}
        pooled_per_cancer_c_indices = []
        cancer_types = set(p["cancer_type"] for p in all_test_patient_results if p.get("cancer_type") is not None)
        for ct in sorted(cancer_types):
            ct_mask = np.array([p.get("cancer_type") == ct for p in all_test_patient_results])
            ct_times = pooled_times[ct_mask]
            ct_events = pooled_events[ct_mask]
            ct_risks = pooled_risks[ct_mask]

            if len(ct_times) >= 5:
                ct_c = c_index(ct_times, ct_events, ct_risks)
                ct_aucs = td_auc_simple(ct_times, ct_events, ct_risks, td_times)
                pooled_per_cancer[ct] = {
                    "c_index": float(ct_c),
                    "n_patients": int(len(ct_times)),
                    "n_events": int(ct_events.sum()),
                    **{f"auc_t{int(t)}": float(v) for t, v in ct_aucs.items()},
                }
                if not np.isnan(ct_c):
                    pooled_per_cancer_c_indices.append(ct_c)

        pooled_macro_c_index = float(np.mean(pooled_per_cancer_c_indices)) if pooled_per_cancer_c_indices else float("nan")

        # Print summary
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*60}\n")

        print("Per-fold results:")
        for r in fold_results:
            print(f"  Fold {r.fold}: epoch={r.best_epoch}, val_C={r.best_val_c_index:.4f}, val_filtered={r.best_val_filtered_macro_c_index:.4f}, test_C={r.test_c_index:.4f}, test_filtered={r.test_filtered_macro_c_index:.4f}")

        print(f"\n--- Aggregated (mean +/- std across folds) ---")
        print(f"Validation C-index:           {np.mean(val_c_indices):.4f} +/- {np.std(val_c_indices):.4f}")
        print(f"Validation macro-avg C:       {np.mean(val_macro_c_indices):.4f} +/- {np.std(val_macro_c_indices):.4f}")
        print(f"Validation filtered macro C:  {np.mean(val_filtered_macro_c_indices):.4f} +/- {np.std(val_filtered_macro_c_indices):.4f}  <-- early stopping metric")
        print(f"Test C-index:                 {np.mean(test_c_indices):.4f} +/- {np.std(test_c_indices):.4f}")
        print(f"Test macro-avg C:             {np.mean(test_macro_c_indices):.4f} +/- {np.std(test_macro_c_indices):.4f}")
        print(f"Test filtered macro C:        {np.mean(test_filtered_macro_c_indices):.4f} +/- {np.std(test_filtered_macro_c_indices):.4f}")

        print(f"\n--- Pooled (all test predictions combined) ---")
        print(f"Pooled Test C-index:     {pooled_c_index:.4f}  (n={len(all_test_patient_results)} patients)")
        print(f"Pooled Test macro-avg C: {pooled_macro_c_index:.4f}  (across {len(pooled_per_cancer)} cancer types)")
        for t in td_times:
            key = f"auc_t{int(t)}"
            print(f"Pooled Test {key}: {pooled_aucs.get(t, float('nan')):.4f}")

        if pooled_per_organ:
            print(f"\n--- Pooled Per-Organ Results ---")
            for org, org_metrics in sorted(pooled_per_organ.items()):
                print(f"  {org} (n={org_metrics['n_patients']}): C-index={org_metrics['c_index']:.4f}")

        # Log to wandb
        if use_wandb:
            import wandb
            log_dict = {
                "cv/val_c_index_mean": np.mean(val_c_indices),
                "cv/val_c_index_std": np.std(val_c_indices),
                "cv/val_macro_avg_c_index_mean": np.mean(val_macro_c_indices),
                "cv/val_macro_avg_c_index_std": np.std(val_macro_c_indices),
                "cv/val_filtered_macro_c_index_mean": np.mean(val_filtered_macro_c_indices),
                "cv/val_filtered_macro_c_index_std": np.std(val_filtered_macro_c_indices),
                "cv/test_c_index_mean": np.mean(test_c_indices),
                "cv/test_c_index_std": np.std(test_c_indices),
                "cv/test_macro_avg_c_index_mean": np.mean(test_macro_c_indices),
                "cv/test_macro_avg_c_index_std": np.std(test_macro_c_indices),
                "cv/test_filtered_macro_c_index_mean": np.mean(test_filtered_macro_c_indices),
                "cv/test_filtered_macro_c_index_std": np.std(test_filtered_macro_c_indices),
                "cv/pooled_test_c_index": pooled_c_index,
                "cv/pooled_test_macro_avg_c_index": pooled_macro_c_index,
                **{f"cv/pooled_test_auc_t{int(t)}": v for t, v in pooled_aucs.items()},
            }
            # Per-organ pooled metrics
            for org, org_metrics in pooled_per_organ.items():
                for metric_name, metric_val in org_metrics.items():
                    log_dict[f"cv/pooled/{org}/{metric_name}"] = metric_val
            # Per-cancer pooled metrics
            for ct, ct_metrics in pooled_per_cancer.items():
                for metric_name, metric_val in ct_metrics.items():
                    log_dict[f"cv/pooled/{ct}/{metric_name}"] = metric_val
            wandb.log(log_dict)

        # Save to JSON
        run_dir = Path(cfg.get("_run_dir", "runs"))
        results_file = run_dir / "cv_results.json"
        
        results_dict = {
            "config": cfg,
            "n_folds": n_folds,
            "timestamp": timestamp,
            "aggregated": {
                "val_c_index_mean": float(np.mean(val_c_indices)),
                "val_c_index_std": float(np.std(val_c_indices)),
                "val_macro_avg_c_index_mean": float(np.mean(val_macro_c_indices)),
                "val_macro_avg_c_index_std": float(np.std(val_macro_c_indices)),
                "val_filtered_macro_c_index_mean": float(np.mean(val_filtered_macro_c_indices)),
                "val_filtered_macro_c_index_std": float(np.std(val_filtered_macro_c_indices)),
                "test_c_index_mean": float(np.mean(test_c_indices)),
                "test_c_index_std": float(np.std(test_c_indices)),
                "test_macro_avg_c_index_mean": float(np.mean(test_macro_c_indices)),
                "test_macro_avg_c_index_std": float(np.std(test_macro_c_indices)),
                "test_filtered_macro_c_index_mean": float(np.mean(test_filtered_macro_c_indices)),
                "test_filtered_macro_c_index_std": float(np.std(test_filtered_macro_c_indices)),
            },
            "pooled": {
                "test_c_index": float(pooled_c_index),
                "test_macro_avg_c_index": float(pooled_macro_c_index),
                "n_patients": len(all_test_patient_results),
                **{f"test_auc_t{int(t)}": float(v) for t, v in pooled_aucs.items()},
                "per_organ": pooled_per_organ,
                "per_cancer": pooled_per_cancer,
            },
            "per_fold": [
                {
                    "fold": r.fold,
                    "best_epoch": r.best_epoch,
                    "best_val_c_index": float(r.best_val_c_index),
                    "best_val_macro_c_index": float(r.best_val_macro_c_index),
                    "best_val_filtered_macro_c_index": float(r.best_val_filtered_macro_c_index),
                    "test_c_index": float(r.test_c_index),
                    "test_macro_c_index": float(r.test_macro_c_index),
                    "test_filtered_macro_c_index": float(r.test_filtered_macro_c_index),
                    "test_aucs": {k: float(v) for k, v in r.test_aucs.items()},
                    "test_per_organ": {
                        org: {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
                        for org, metrics in r.test_per_organ.items()
                    },
                    "test_per_cancer": {
                        ct: {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
                        for ct, metrics in r.test_per_cancer.items()
                    },
                    "n_train": r.n_train_patients,
                    "n_val": r.n_val_patients,
                    "n_test": r.n_test_patients,
                }
                for r in fold_results
            ],
        }
        
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Wandb logging: {'enabled' if use_wandb else 'disabled'}")

    if rank == 0:
        print("\nCross-validation complete.")

    if rank == 0 and use_wandb:
        import wandb
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_cv.yaml")
    parser.add_argument("--dx-only", action="store_true",
                        help="Only use diagnostic (DX) slides, exclude tissue slides (TS/BS/MS)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous run directory (e.g., runs/pan_tcga_dss_20260130_123456)")
    parser.add_argument("--cancer", type=str, default=None,
                        help="Train on a single cancer type (e.g., TCGA-BRCA). If not specified, trains pan-cancer.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train only a single fold (0-indexed, e.g., --fold 3). If not specified, trains all folds.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Start epoch counter at this value (useful for testing specific epoch shuffles)")
    args = parser.parse_args()
    main_worker(args.config, dx_only=args.dx_only, resume_dir=args.resume, cancer_type=args.cancer, single_fold=args.fold, start_epoch=args.start_epoch)
