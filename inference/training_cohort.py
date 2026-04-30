"""
Generate predictions CSV from trained model checkpoint.

Produces TWO outputs:
1. Slide-level predictions CSV with columns:
   - patient_id, slide_id, split, risk_score, time, event

2. Patient-level predictions CSV (aggregated across slides) with columns:
   - patient_id, split, risk_score (mean across slides), time, event, n_slides

Also computes and prints validation/test metrics (C-index, time-dependent AUCs).

Usage:
    python generate_predictions.py --checkpoint runs/localcrossattn_patchgcn_20251201_230624/checkpoints/fold_0_best.pt \
                                     --config configs/img_gep_patch_fancy.yaml \
                                     --output predictions.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import yaml
import os

from sparc.data.dataset import (
    SlideBagDataset,
    slide_collate_fn,
    load_clinical_table,
    get_slide_paths,
)
from sparc.models.factory import build_model
from sparc.utils.metrics import (
    aggregate_patient_level,
    c_index,
    td_auc_simple,
)

# Avoid HDF5 locking issues
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def create_cv_splits(clinical, slide_paths, n_folds=5, seed=1337):
    """Create cross-validation splits (same logic as new_train.py)."""
    from sklearn.model_selection import StratifiedKFold

    # Group slides by patient
    patient_to_slides = {}
    patient_events = {}

    for path in slide_paths:
        slide_id = Path(path).stem
        patient_id = "-".join(slide_id.split("-")[:3])

        if patient_id and patient_id in clinical:
            if patient_id not in patient_to_slides:
                patient_to_slides[patient_id] = []
                patient_events[patient_id] = clinical[patient_id].event
            patient_to_slides[patient_id].append(path)

    patients = np.array(list(patient_to_slides.keys()))
    events = np.array([patient_events[p] for p in patients])

    # Outer CV for test folds
    outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_splits = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(outer_skf.split(patients, events)):
        test_patients = set(patients[test_idx])
        trainval_patients = patients[trainval_idx]
        trainval_events = events[trainval_idx]

        # Inner split: separate val from train
        inner_skf = StratifiedKFold(n_splits=n_folds-1, shuffle=True, random_state=seed + fold_idx)
        train_idx_inner, val_idx_inner = next(inner_skf.split(trainval_patients, trainval_events))

        train_patients = set(trainval_patients[train_idx_inner])
        val_patients = set(trainval_patients[val_idx_inner])

        # Convert to slide paths and patient sets
        train_paths, val_paths, test_paths = [], [], []
        for pid, paths in patient_to_slides.items():
            if pid in train_patients:
                train_paths.extend(paths)
            elif pid in val_patients:
                val_paths.extend(paths)
            elif pid in test_patients:
                test_paths.extend(paths)

        fold_splits.append({
            'train_patients': train_patients,
            'val_patients': val_patients,
            'test_patients': test_patients,
            'train_paths': train_paths,
            'val_paths': val_paths,
            'test_paths': test_paths,
        })

    return fold_splits


def get_split_label(slide_id, fold_splits, fold_idx=0):
    """Determine which split a slide belongs to."""
    patient_id = "-".join(slide_id.split("-")[:3])
    fold = fold_splits[fold_idx]

    if patient_id in fold['train_patients']:
        return 'train'
    elif patient_id in fold['val_patients']:
        return 'val'
    elif patient_id in fold['test_patients']:
        return 'test'
    else:
        return 'unknown'


@torch.no_grad()
def generate_predictions(checkpoint_path, config_path, output_path, fold_idx=0, dx_only=False):
    """Generate predictions CSV from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_path: Output CSV path
        fold_idx: Fold index to use for split labels
        dx_only: If True, only use diagnostic (DX) slides
    """

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load clinical data
    print("Loading clinical data...")
    clinical_csv = cfg["data"]["clinical_csv"]

    # Check if we need to compute time from columns or if it's pre-computed
    compute_time_cols = cfg["data"].get("compute_time_from_cols", None)
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
    print(f"Loaded {len(clinical)} patients from clinical table")

    # Get slide paths
    print("Getting slide paths...")
    from pathlib import Path
    img_dir = Path(cfg["data"]["img_feature_dir"])
    all_slide_paths = get_slide_paths(img_dir, ext=".h5")

    # Filter for DX slides only if requested
    if dx_only:
        slide_paths = [p for p in all_slide_paths if "-DX" in p.stem or "-dx" in p.stem]
        print(f"Found {len(slide_paths)} DX (diagnostic) slides out of {len(all_slide_paths)} total")
    else:
        slide_paths = all_slide_paths
        print(f"Found {len(slide_paths)} slides")

    # Create CV splits (to determine train/val/test labels)
    print("Creating CV splits...")
    seed = cfg["training"]["seed"]
    n_folds = cfg.get("cross_validation", {}).get("n_folds", 5)
    fold_splits = create_cv_splits(clinical, slide_paths, n_folds=n_folds, seed=seed)
    print(f"Created {n_folds} folds")

    # Get only slides that have clinical data (same filtering as create_cv_splits)
    valid_slide_paths = []
    for path in slide_paths:
        stem = path.stem
        patient_id = "-".join(stem.split("-")[:3])  # TCGA-style patient ID
        if patient_id in clinical:
            valid_slide_paths.append(path)

    print(f"Filtered to {len(valid_slide_paths)} slides with clinical data (from {len(slide_paths)} total)")

    # Create dataset with valid slides only
    print("Creating dataset...")
    gep_dir = Path(cfg["data"]["gep_feature_dir"]) if cfg["data"].get("use_gep", False) else None
    dataset = SlideBagDataset(
        slide_feature_paths=valid_slide_paths,
        clinical=clinical,
        gep_feature_dir=gep_dir,
        include_gep=cfg["data"].get("use_gep", False),
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["data"].get("val_batch_size", 4),
        shuffle=False,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        collate_fn=slide_collate_fn,
    )

    # Build model
    print("Building model...")
    model = build_model(cfg)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint info: fold={checkpoint['fold']}, epoch={checkpoint['epoch']}, val_c_index={checkpoint.get('val_c_index', 'N/A')}")

    # Run inference
    print("Running inference...")
    results = []

    for batch in tqdm(dataloader, desc="Generating predictions"):
        # Move batch to device
        batch_device = {}
        for key, val in batch.items():
            if key in ["img_feats", "gep_feats", "coords"]:
                batch_device[key] = [x.to(device) if x is not None else None for x in val]
            elif isinstance(val, torch.Tensor):
                batch_device[key] = val.to(device)
            else:
                batch_device[key] = val

        # Get predictions
        risk_scores = model(batch_device).cpu().numpy()

        # Extract metadata
        slide_ids = batch["slide_id"]
        patient_ids = batch["patient_id"]
        times = batch["time"].cpu().numpy()
        events = batch["event"].cpu().numpy()

        # Determine splits
        for i in range(len(slide_ids)):
            slide_id = slide_ids[i]
            patient_id = patient_ids[i]
            split = get_split_label(slide_id, fold_splits, fold_idx)

            results.append({
                "patient_id": patient_id,
                "slide_id": slide_id,
                "split": split,
                "risk_score": float(risk_scores[i]),
                "time": float(times[i]),
                "event": int(events[i]),
            })

    # Create slide-level DataFrame
    print(f"Creating DataFrame with {len(results)} slide-level predictions...")
    df_slides = pd.DataFrame(results)

    # Print slide-level summary statistics
    print("\n" + "="*80)
    print("SLIDE-LEVEL PREDICTIONS SUMMARY")
    print("="*80)
    print(f"Total slides: {len(df_slides)}")
    print(f"Total patients: {df_slides['patient_id'].nunique()}")
    print(f"\nSplit distribution:")
    print(df_slides['split'].value_counts())
    print(f"\nEvent distribution:")
    print(f"  Events (event=1): {(df_slides['event']==1).sum()}")
    print(f"  Censored (event=0): {(df_slides['event']==0).sum()}")
    print(f"\nRisk score statistics:")
    print(df_slides['risk_score'].describe())
    print(f"\nTime to event statistics:")
    print(df_slides['time'].describe())

    # Aggregate to patient-level (mean across slides, same as training)
    print("\n" + "="*80)
    print("AGGREGATING TO PATIENT-LEVEL (mean across slides)...")
    print("="*80)

    # Get patient organs mapping
    patient_organs = {}
    for pid, clinical_record in clinical.items():
        if hasattr(clinical_record, 'organ') and clinical_record.organ:
            patient_organs[pid] = clinical_record.organ

    # Aggregate all predictions
    all_patient_records = aggregate_patient_level(
        slide_patient_ids=df_slides['patient_id'].tolist(),
        slide_times=df_slides['time'].tolist(),
        slide_events=df_slides['event'].tolist(),
        slide_risks=df_slides['risk_score'].tolist(),
        patient_organs=patient_organs,
        agg="mean",
    )

    # Create patient-level DataFrame
    patient_data = []
    for rec in all_patient_records:
        # Find which split this patient is in
        patient_slides = df_slides[df_slides['patient_id'] == rec.patient_id]
        split = patient_slides['split'].iloc[0]  # All slides from same patient have same split
        n_slides = len(patient_slides)

        patient_data.append({
            'patient_id': rec.patient_id,
            'split': split,
            'risk_score': rec.risk,
            'time': rec.time,
            'event': rec.event,
            'organ': rec.organ,
            'n_slides': n_slides,
        })

    df_patients = pd.DataFrame(patient_data)

    print(f"\nTotal patients: {len(df_patients)}")
    print(f"\nPatient-level split distribution:")
    print(df_patients['split'].value_counts())

    # Compute metrics for validation and test sets
    print("\n" + "="*80)
    print("EVALUATION METRICS (Patient-level)")
    print("="*80)

    # Get time-dependent AUC times from config
    td_times = cfg.get("metrics", {}).get("td_times", [30.0, 90.0, 180.0, 365.0, 1095.0])

    for split_name in ['val', 'test']:
        df_split = df_patients[df_patients['split'] == split_name]

        if len(df_split) == 0:
            print(f"\n{split_name.upper()} SET: No samples found")
            continue

        times = df_split['time'].values
        events = df_split['event'].values
        risks = df_split['risk_score'].values

        print(f"\n{split_name.upper()} SET:")
        print(f"  N patients: {len(df_split)}")
        print(f"  N events: {events.sum()}")
        print(f"  N censored: {(events==0).sum()}")

        # C-index
        c_idx = c_index(times, events, risks)
        print(f"  C-index: {c_idx:.4f}")

        # Time-dependent AUCs
        aucs = td_auc_simple(times, events, risks, td_times)
        print(f"  Time-dependent AUCs:")
        for t, auc_val in aucs.items():
            if not np.isnan(auc_val):
                print(f"    t={int(t):4d} days: {auc_val:.4f}")
            else:
                print(f"    t={int(t):4d} days: N/A")

        # Per-organ breakdown if available
        if 'organ' in df_split.columns and df_split['organ'].notna().any():
            print(f"\n  Per-organ C-index:")
            for organ in sorted(df_split['organ'].dropna().unique()):
                df_organ = df_split[df_split['organ'] == organ]
                if len(df_organ) >= 5:  # Need at least 5 samples
                    org_c = c_index(df_organ['time'].values,
                                   df_organ['event'].values,
                                   df_organ['risk_score'].values)
                    print(f"    {organ:10s}: {org_c:.4f} (n={len(df_organ)})")

    print("\n" + "="*80)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save slide-level predictions
    slide_output = output_path.parent / (output_path.stem + "_slides.csv")
    df_slides.to_csv(slide_output, index=False)
    print(f"\nSlide-level predictions saved to: {slide_output}")

    # Save patient-level predictions
    patient_output = output_path.parent / (output_path.stem + "_patients.csv")
    df_patients.to_csv(patient_output, index=False)
    print(f"Patient-level predictions saved to: {patient_output}")

    print("="*80)

    return df_slides, df_patients


def main():
    parser = argparse.ArgumentParser(description="Generate predictions CSV from trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (.yaml)")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold index to use for split labels (default: 0)")
    parser.add_argument("--dx-only", action="store_true",
                        help="Only use diagnostic (DX) slides, exclude tissue slides (TS/BS/MS)")

    args = parser.parse_args()

    generate_predictions(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        fold_idx=args.fold,
        dx_only=args.dx_only,
    )


if __name__ == "__main__":
    main()
