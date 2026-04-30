"""Slide-bag dataset for SPARC training and inference.

Each item yielded by :class:`SlideBagDataset` is a dict containing one slide's
worth of pre-computed patch features (image + optional gene-program scores),
their level-0 coordinates, and the patient's survival label. Slides without
matching clinical data are dropped at construction time.

The module also defines two index/order maps used throughout the codebase:

- ``CANCER_TYPE_ORDER`` and ``CANCER_TYPE_TO_IDX`` — stable alphabetical
  ordering of the 33 TCGA project codes, used by the ``cancer_conditioning``
  embedding in :class:`sparc.models.fusion.SignatureQueryFusion`.
- ``ORGAN_ORDER`` and ``ORGAN_TO_IDX`` — coarse organ buckets derived from
  ``ORGAN_MAP``, used by per-organ evaluation metrics.
"""

from __future__ import annotations

import os
# Disable HDF5 file locking — critical for NFS performance and avoiding hangs.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


Tensor = torch.Tensor

# ---- Organ mapping for all TCGA cancer types ----

ORGAN_MAP = {
    # Original 3
    "TCGA-LUAD": "Lung",
    "TCGA-LUSC": "Lung",
    "TCGA-READ": "Colorectal",
    "TCGA-COAD": "Colorectal",
    "TCGA-BRCA": "Breast",
    # Additional cancer types
    "TCGA-ACC": "Adrenal",
    "TCGA-BLCA": "Bladder",
    "TCGA-CESC": "Cervix",
    "TCGA-CHOL": "Bile Duct",
    "TCGA-DLBC": "Lymphoid",
    "TCGA-ESCA": "Esophagus",
    "TCGA-GBM": "Brain",
    "TCGA-HNSC": "Head and Neck",
    "TCGA-KICH": "Kidney",
    "TCGA-KIRC": "Kidney",
    "TCGA-KIRP": "Kidney",
    "TCGA-LAML": "Blood",
    "TCGA-LGG": "Brain",
    "TCGA-LIHC": "Liver",
    "TCGA-MESO": "Pleura",
    "TCGA-OV": "Ovary",
    "TCGA-PAAD": "Pancreas",
    "TCGA-PCPG": "Adrenal",
    "TCGA-PRAD": "Prostate",
    "TCGA-SARC": "Soft Tissue",
    "TCGA-SKCM": "Skin",
    "TCGA-STAD": "Stomach",
    "TCGA-TGCT": "Testis",
    "TCGA-THCA": "Thyroid",
    "TCGA-THYM": "Thymus",
    "TCGA-UCEC": "Uterus",
    "TCGA-UCS": "Uterus",
    "TCGA-UVM": "Eye",
}

ORGAN_ORDER = sorted(set(ORGAN_MAP.values()))
ORGAN_TO_IDX = {name: i for i, name in enumerate(ORGAN_ORDER)}

# Cancer type mapping (direct from project_id, e.g., TCGA-BRCA -> 0, TCGA-LUAD -> 1, etc.)
CANCER_TYPE_ORDER = sorted(ORGAN_MAP.keys())  # All TCGA cancer types
CANCER_TYPE_TO_IDX = {ct: i for i, ct in enumerate(CANCER_TYPE_ORDER)}
NUM_CANCER_TYPES = len(CANCER_TYPE_ORDER)


@dataclass
class ClinicalRecord:
    """Container for a single patient's clinical record."""
    patient_id: str
    time: float
    event: int
    organ: Optional[str]
    cancer_type: Optional[str]  # e.g., TCGA-BRCA, TCGA-LUAD
    extra: dict


def load_clinical_table(
    path: Path,
    patient_col: str = "patient_id",
    time_col: str = "time",
    event_col: str = "event",
    project_col: str = "project_id",
    compute_time_from_cols: Optional[Dict[str, str]] = None,
    organs_to_use: Optional[List[str]] = None,
) -> Dict[str, ClinicalRecord]:
    """
    Load clinical CSV into a dict mapping patient_id -> ClinicalRecord.

    Args:
        path: path to clinical CSV.
        patient_col: column with patient IDs.
        time_col:   column with survival time (used if compute_time_from_cols is None).
        event_col:  column with event indicator (0/1).
        project_col: column with TCGA project ID (for organ mapping).
        compute_time_from_cols: optional dict with keys:
            - "event_flag": e.g. "progression_recurrence_event"
            - "event_time": e.g. "days_to_progression_recurrence"
            - "censor_time": e.g. "max_follow_up_days"
          If provided, we'll compute "time" column like in your old code.
        organs_to_use: optional list of organ names to keep (e.g. ["Lung","Colon"]).

    Returns:
        dict[patient_id -> ClinicalRecord]
    """
    df = pd.read_csv(path)
    df[patient_col] = df[patient_col].astype(str).str.strip()

    # Map project_id -> organ
    if project_col in df.columns:
        df["organ"] = df[project_col].map(ORGAN_MAP)
    else:
        df["organ"] = None

    # Compute time/event if requested
    if compute_time_from_cols is not None:
        flag_col = compute_time_from_cols["event_flag"]
        event_time_col = compute_time_from_cols["event_time"]
        censor_time_col = compute_time_from_cols["censor_time"]

        def _choose_time(row):
            if row[flag_col] == 1:
                return row[event_time_col]
            return row[censor_time_col]

        df["time"] = df.apply(_choose_time, axis=1)
        df["event"] = df[flag_col]

        time_col = "time"
        event_col = "event"

    # Optional organ filtering
    if organs_to_use and organs_to_use != "all":
        df = df[df["organ"].isin(organs_to_use)]

    clinical: Dict[str, ClinicalRecord] = {}
    for _, row in df.iterrows():
        pid = str(row[patient_col])
        organ = row.get("organ", None)
        cancer_type = row.get(project_col, None) if project_col in df.columns else None
        clinical[pid] = ClinicalRecord(
            patient_id=pid,
            time=float(row[time_col]),
            event=int(row[event_col]),
            organ=str(organ) if organ is not None and organ == organ else None,
            cancer_type=str(cancer_type) if cancer_type is not None and cancer_type == cancer_type else None,
            extra=row.to_dict(),
        )

    return clinical


def get_slide_paths(img_feature_dir: Path, ext: str = ".h5") -> List[Path]:
    """
    Return all slide feature files in a directory (sorted).

    Supports both .h5 and .npz files. If ext=".h5" but .npz files exist,
    will prefer .npz (faster loading over NFS).

    Each file is expected to have:
      - 'features': [N_patches, D_img]
      - 'coords':   [N_patches, 2]
    """
    # Check if NPZ files exist (prefer these over H5 for speed)
    npz_files = sorted(p for p in img_feature_dir.glob("*.npz") if p.is_file())
    if npz_files:
        return npz_files

    # Fall back to original extension
    return sorted(p for p in img_feature_dir.glob(f"*{ext}") if p.is_file())


def _load_with_retry(load_fn, path: Path, max_retries: int = 3, delay: float = 1.0):
    """
    Wrapper to retry file loading with exponential backoff.

    Helps with transient NFS issues.
    """
    import time
    last_error = None

    for attempt in range(max_retries):
        try:
            return load_fn(path)
        except (OSError, IOError, Exception) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            continue

    # All retries failed
    raise last_error


def load_img_features(path: Path) -> Tuple[Tensor, Tensor]:
    """
    Load H-Optimus features + coords from .h5 or .npz.

    Expects datasets:
      - 'features': [N_patches, D_img]
      - 'coords':   [N_patches, 2]

    NPZ files are ~3-5x faster to load over NFS.
    Includes retry logic for transient NFS failures.
    """
    import numpy as np

    def _load(p):
        if p.suffix == ".npz":
            data = np.load(p)
            feats = torch.from_numpy(data["features"]).float()
            coords = torch.from_numpy(data["coords"]).float()
        else:
            # Use swmr=True for better NFS compatibility
            with h5py.File(p, "r", swmr=True, libver='latest') as f:
                feats = torch.from_numpy(f["features"][:]).float()
                coords = torch.from_numpy(f["coords"][:]).float()
        return feats, coords

    return _load_with_retry(_load, path)


def load_gep_features(
    path: Path,
    dataset_name: str = "features",
) -> Tensor:
    """
    Load patch-level gene expression program scores from .h5 or .npz file.

    By default expects dataset 'features', but this can be overridden
    if your GEP files use a different name.

    NPZ files are ~3-5x faster to load over NFS.
    Includes retry logic for transient NFS failures.
    """
    import numpy as np

    def _load(p):
        if p.suffix == ".npz":
            data = np.load(p)
            gep = torch.from_numpy(data[dataset_name]).float()
        else:
            # Use swmr=True for better NFS compatibility
            with h5py.File(p, "r", swmr=True, libver='latest') as f:
                gep = torch.from_numpy(f[dataset_name][:]).float()
        return gep

    return _load_with_retry(_load, path)


class SlideBagDataset(Dataset):
    """
    One sample = one slide (bag of patches).

    Returns a dict with:
        img_feats:   Tensor [N, D_img]
        gep_feats:   Tensor [N, K] or None
        coords:      Tensor [N, 2]
        time:        float
        event:       float
        patient_id:  str
        slide_id:    str
        organ:       str or None
        organ_idx:   int (or -1 if unknown)
    """

    # Class-level tracking of corrupted files (shared across all instances)
    _corrupted_files: Dict[str, str] = {}  # path -> error message
    _corrupted_warned: set = set()  # files we've already warned about

    def __init__(
        self,
        slide_feature_paths: List[Path],
        clinical: Dict[str, ClinicalRecord],
        gep_feature_dir: Optional[Path] = None,
        include_gep: bool = True,
        max_patches: Optional[int] = None,
        gep_ablation: Optional[str] = None,
        gep_zscore: bool = False,
    ) -> None:
        self.slide_feature_paths = slide_feature_paths
        self.clinical = clinical
        self.gep_feature_dir = gep_feature_dir
        self.include_gep = include_gep
        self.max_patches = max_patches  # Cap patches per slide to avoid OOM
        # Ablation modes for GEP features:
        #   None: use real program values (default)
        #   "shuffle": randomly permute program assignments per slide (breaks program identity)
        #   "random": replace with Gaussian noise matching per-slide mean/std (breaks molecular signal)
        self.gep_ablation = gep_ablation
        # Per-patch z-scoring across programs: normalize each patch's K program values
        # to mean=0, std=1. Captures relative activation profile rather than absolute levels.
        self.gep_zscore = gep_zscore

    @classmethod
    def get_corrupted_count(cls) -> int:
        """Return number of corrupted files encountered."""
        return len(cls._corrupted_files)

    @classmethod
    def get_corrupted_files(cls) -> Dict[str, str]:
        """Return dict of corrupted file paths and their error messages."""
        return cls._corrupted_files.copy()

    @classmethod
    def reset_corrupted_tracking(cls) -> None:
        """Reset corrupted file tracking."""
        cls._corrupted_files.clear()
        cls._corrupted_warned.clear()

    def __len__(self) -> int:
        return len(self.slide_feature_paths)

    @staticmethod
    def slide_id_to_patient_id(slide_id: str) -> str:
        """
        Map slide ID -> patient ID.

        Default: TCGA-style IDs where the first 3 chunks are the patient.
        Adapt this if your naming scheme is different.
        """
        return "-".join(slide_id.split("-")[:3])

    def __getitem__(self, idx: int) -> dict:
        slide_path = self.slide_feature_paths[idx]
        slide_id = slide_path.stem
        patient_id = self.slide_id_to_patient_id(slide_id)

        # Load image features with error handling for corrupted files
        img_corrupted = False
        try:
            img_feats, coords = load_img_features(slide_path)
        except (OSError, Exception) as e:
            # Track corrupted file
            path_str = str(slide_path)
            if path_str not in SlideBagDataset._corrupted_files:
                SlideBagDataset._corrupted_files[path_str] = str(e)
            # Warn once per file
            if path_str not in SlideBagDataset._corrupted_warned:
                SlideBagDataset._corrupted_warned.add(path_str)
                print(f"[WARNING] Corrupted image file skipped: {slide_path.name} ({e})", flush=True)
            img_corrupted = True
            # Return a minimal sample that will be filtered by collate_fn
            clin = self.clinical[patient_id]
            return {
                "img_feats": None,
                "gep_feats": None,
                "coords": None,
                "time": float(clin.time),
                "event": float(clin.event),
                "patient_id": patient_id,
                "slide_id": slide_id,
                "organ": clin.organ,
                "organ_idx": -1,
                "cancer_type": clin.cancer_type,
                "cancer_type_idx": -1,
                "img_corrupted": True,
                "gep_corrupted": False,
            }

        # Load GEP features (before subsampling so we can align patch counts)
        gep_feats = None
        if self.include_gep and self.gep_feature_dir is not None:
            # Prefer NPZ over H5 (faster loading)
            gep_path = self.gep_feature_dir / f"{slide_id}.npz"
            if not gep_path.exists():
                gep_path = self.gep_feature_dir / f"{slide_id}.h5"
            if gep_path.exists():
                try:
                    gep_feats = load_gep_features(gep_path)
                except (OSError, Exception) as e:
                    # Track corrupted file
                    path_str = str(gep_path)
                    if path_str not in SlideBagDataset._corrupted_files:
                        SlideBagDataset._corrupted_files[path_str] = str(e)
                    # Warn once per file
                    if path_str not in SlideBagDataset._corrupted_warned:
                        SlideBagDataset._corrupted_warned.add(path_str)
                        print(f"[WARNING] Corrupted GEP file skipped: {gep_path.name} ({e})", flush=True)
                    # Continue with gep_feats = None

        # Patch count mismatch = can't safely align (no shared coords in GEP files).
        # Skip the slide rather than risk silently misaligning patch pairs.
        if gep_feats is not None and gep_feats.shape[0] != img_feats.shape[0]:
            path_str = str(slide_path)
            if path_str not in SlideBagDataset._corrupted_files:
                SlideBagDataset._corrupted_files[path_str] = (
                    f"patch count mismatch: img={img_feats.shape[0]}, gep={gep_feats.shape[0]}"
                )
            if path_str not in SlideBagDataset._corrupted_warned:
                SlideBagDataset._corrupted_warned.add(path_str)
                print(f"[WARNING] Skipping {slide_id}: patch count mismatch img={img_feats.shape[0]} vs gep={gep_feats.shape[0]}.", flush=True)
            clin = self.clinical[patient_id]
            return {
                "img_feats": None, "gep_feats": None, "coords": None,
                "time": float(clin.time), "event": float(clin.event),
                "patient_id": patient_id, "slide_id": slide_id,
                "organ": clin.organ, "organ_idx": -1,
                "cancer_type": clin.cancer_type, "cancer_type_idx": -1,
                "img_corrupted": True, "gep_corrupted": True,
            }

        # Cap patches to avoid OOM on very large slides
        if self.max_patches is not None and img_feats.shape[0] > self.max_patches:
            subsample_indices = torch.randperm(img_feats.shape[0])[:self.max_patches]
            subsample_indices = subsample_indices.sort().values  # Keep spatial order
            img_feats = img_feats[subsample_indices]
            coords = coords[subsample_indices]
            if gep_feats is not None:
                gep_feats = gep_feats[subsample_indices]

        # Normalize coordinates to [0, 1]
        min_vals = coords.min(dim=0, keepdim=True).values
        max_vals = coords.max(dim=0, keepdim=True).values
        den = (max_vals - min_vals).clamp(min=1e-6)
        coords_norm = (coords - min_vals) / den

        clin = self.clinical[patient_id]
        organ = clin.organ
        organ_idx = ORGAN_TO_IDX.get(organ, -1) if organ is not None else -1
        cancer_type = clin.cancer_type
        cancer_type_idx = CANCER_TYPE_TO_IDX.get(cancer_type, -1) if cancer_type is not None else -1

        # Flag if GEP was expected but failed to load (corrupted file)
        gep_corrupted = self.include_gep and self.gep_feature_dir is not None and gep_feats is None

        # Per-patient z-scoring across programs: compute slide-level mean across patches
        # to get a single 40d profile, then z-score that profile (mean/std across the 40
        # programs). Apply the same shift/scale to all patches so spatial heterogeneity
        # is preserved. Matches the per-patient z-scoring used in treatment response notebooks.
        if gep_feats is not None and self.gep_zscore:
            slide_mean = gep_feats.mean(dim=0)  # [K] - mean program activation for this patient
            mu = slide_mean.mean()               # scalar - mean across programs
            sd = slide_mean.std().clamp(min=1e-8) # scalar - std across programs
            gep_feats = (gep_feats - mu) / sd

        # GEP ablation: modify program features for controlled experiments
        if gep_feats is not None and self.gep_ablation is not None:
            if self.gep_ablation == "shuffle":
                # Permute program columns: each patch keeps its values but programs are shuffled.
                # Breaks program-specific semantics (e.g., "program 3 = immune") while
                # preserving the value distribution. Permutation is per-slide, same across patches.
                perm = torch.randperm(gep_feats.shape[1])
                gep_feats = gep_feats[:, perm]
            elif self.gep_ablation == "random":
                # Replace with Gaussian noise matching per-slide statistics.
                # Breaks all molecular signal while preserving scale.
                gep_feats = torch.randn_like(gep_feats) * gep_feats.std() + gep_feats.mean()
            elif self.gep_ablation == "spatial_shuffle":
                # Permute patch order: each patch gets a different patch's GEP vector.
                # Breaks local spatial co-localization (morphology vs molecular).
                # Preserves slide's overall biological distribution.
                perm = torch.randperm(gep_feats.shape[0])
                gep_feats = gep_feats[perm]
            elif self.gep_ablation == "cross_patient_shuffle":
                # Replace with a random OTHER slide's GEP features.
                # Breaks patient-specific molecular signal while preserving
                # realistic GEP distributions. Resampled to match patch count.
                n_patches = gep_feats.shape[0]
                other_idx = idx
                while other_idx == idx:
                    other_idx = torch.randint(0, len(self), (1,)).item()
                other_path = self.slide_feature_paths[other_idx]
                other_gep_path = self.gep_feature_dir / f"{other_path.stem}.h5"
                if not other_gep_path.exists():
                    other_gep_path = self.gep_feature_dir / f"{other_path.stem}.npz"
                try:
                    other_gep = load_gep_features(other_gep_path)
                    if other_gep is not None:
                        if other_gep.shape[0] >= n_patches:
                            indices = torch.randperm(other_gep.shape[0])[:n_patches]
                        else:
                            indices = torch.randint(0, other_gep.shape[0], (n_patches,))
                        gep_feats = other_gep[indices]
                except Exception:
                    pass  # Keep original if loading fails

        return {
            "img_feats": img_feats,    # [N, D_img]
            "gep_feats": gep_feats,    # [N, K] or None
            "coords": coords_norm,          # [N, 2]
            "time": float(clin.time),
            "event": float(clin.event),
            "patient_id": patient_id,
            "slide_id": slide_id,
            "organ": organ,            # string or None
            "organ_idx": organ_idx,    # int (e.g. 0..2 or -1)
            "cancer_type": cancer_type,  # string or None (e.g., TCGA-BRCA)
            "cancer_type_idx": cancer_type_idx,  # int (0..32 or -1)
            "img_corrupted": False,    # Image loaded successfully
            "gep_corrupted": gep_corrupted,  # True if GEP file was corrupted
        }


def slide_collate_fn(batch: List[dict]) -> dict:
    """
    Collate a batch of slides (variable-length bags).

    We keep patch-level tensors as lists: one Tensor [N_i, *] per slide.
    Filters out samples with corrupted image or GEP files.
    """
    # Filter out corrupted samples (image corrupted OR GEP expected but failed to load)
    valid_batch = [
        s for s in batch
        if not s.get("img_corrupted", False) and not s.get("gep_corrupted", False)
    ]

    if len(valid_batch) == 0:
        # All samples corrupted - return empty batch marker
        return None

    if len(valid_batch) < len(batch):
        n_skipped = len(batch) - len(valid_batch)
        # Note: this will print frequently, but that's fine for debugging

    # Exclude corruption flags from output
    exclude_keys = {"gep_corrupted", "img_corrupted"}
    out: Dict[str, List] = {k: [] for k in valid_batch[0].keys() if k not in exclude_keys}

    for sample in valid_batch:
        for k, v in sample.items():
            if k not in exclude_keys:
                out[k].append(v)

    # Stack scalar fields into tensors
    out["time"] = torch.tensor(out["time"], dtype=torch.float32)
    out["event"] = torch.tensor(out["event"], dtype=torch.float32)
    out["organ_idx"] = torch.tensor(out["organ_idx"], dtype=torch.long)
    out["cancer_type_idx"] = torch.tensor(out["cancer_type_idx"], dtype=torch.long)

    return out