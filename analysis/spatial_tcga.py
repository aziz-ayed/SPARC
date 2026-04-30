#!/usr/bin/env python
"""SPARC Spatial TME Architecture Analysis.

Implements 6 analyses from sparc_spatial_spec_v5.1_FROZEN.md:
  A1: Intra-patch co-activation
  A2: Inter-patch spatial cross-correlation (+ null model)
  A3: Patch-level spatial niches
  A4: Prognostic value of spatial features
  A5: Patient-level TME archetypes
  A6: Boundary / interface (supplement)

Usage:
    python scripts/sparc_spatial_analysis.py --analyses 1 2 3 4 5 6
    python scripts/sparc_spatial_analysis.py --analyses 1 --n-workers 4  # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROGRAM_NAMES = [
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_COAGULATION",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_HYPOXIA",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION",
    "REACTOME_COLLAGEN_FORMATION",
    "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION",
    "REACTOME_INTEGRIN_CELL_SURFACE_INTERACTIONS",
    "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION",
    "REACTOME_MISMATCH_REPAIR",
    "REACTOME_NEUTROPHIL_DEGRANULATION",
    "REACTOME_TCR_SIGNALING",
    "REACTOME_TOLL_LIKE_RECEPTOR_CASCADES",
    "ANTIGEN_PRESENTATION_THOMPSON_2020_APM8",
    "B_CELL_CORE_BUDCZIES_2021",
    "CIN70_CHROMOSOMAL_INSTABILITY",
    "TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018",
    "TLS_CABRITA_9",
    "T_CELL_INFLAMED_GEP_18_AYERS_2017",
    "GOLDRATH_NAIVE_VS_MEMORY_CD8_TCELL_DN",
    "GSE13306_TREG_VS_TCONV_UP",
    "GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP",
    "GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_DN",
    "GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_UP",
    "GSE9650_EFFECTOR_VS_EXHAUSTED_CD8_TCELL_UP",
    "GSE9946_IMMATURE_VS_MATURE_STIMULATORY_DC_DN",
]
assert len(PROGRAM_NAMES) == 40

SHORT_NAMES = [
    "Angiogenesis", "Apoptosis", "Coagulation", "DNA Repair", "E2F Targets",
    "EMT", "G2M Checkpoint", "Glycolysis", "Hypoxia", "IL-6/JAK/STAT3",
    "Inflammatory Response", "IFN-α Response", "IFN-γ Response",
    "mTORC1 Signaling", "MYC Targets", "Oxidative Phosphorylation",
    "TGF-β Signaling", "TNF-α/NF-κB", "MHC-I Antigen Processing",
    "Collagen Formation", "ECM Organization", "Integrin Interactions",
    "MHC-II Presentation", "Mismatch Repair", "Neutrophil Degranulation",
    "TCR Signaling", "TLR Cascades", "Antigen Presentation (APM)",
    "B Cell Signature", "Chromosomal Instability",
    "TGF-β Stromal Exclusion", "Tertiary Lymphoid Structures",
    "T Cell-Inflamed GEP", "Naïve CD8⁺ T Cells", "Treg Signature",
    "PD-1⁺ CD8⁺ T Cells", "M1 Macrophages", "M2 Macrophages",
    "Exhausted CD8⁺ T Cells", "Immature Dendritic Cells",
]
assert len(SHORT_NAMES) == 40

# Program index lookup
_PI = {name: i for i, name in enumerate(PROGRAM_NAMES)}

# Tier 1 pairs (7 — BH-FDR corrected)
TIER1_PAIRS: List[Tuple[int, int, str]] = [
    (_PI["HALLMARK_HYPOXIA"], _PI["HALLMARK_ANGIOGENESIS"], "Hypoxia ↔ Angiogenesis"),
    (_PI["TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018"], _PI["GOLDRATH_NAIVE_VS_MEMORY_CD8_TCELL_DN"], "TGF-β Excl. ↔ Naïve CD8"),
    (_PI["TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018"], _PI["REACTOME_TCR_SIGNALING"], "TGF-β Excl. ↔ TCR"),
    (_PI["REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION"], _PI["HALLMARK_INFLAMMATORY_RESPONSE"], "ECM ↔ Inflammatory"),
    (_PI["REACTOME_COLLAGEN_FORMATION"], _PI["GSE9946_IMMATURE_VS_MATURE_STIMULATORY_DC_DN"], "Collagen ↔ Immature DCs"),
    (_PI["REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION"], _PI["GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP"], "MHC-I ↔ PD-1⁺ CD8"),
    (_PI["TLS_CABRITA_9"], _PI["REACTOME_TCR_SIGNALING"], "TLS ↔ TCR"),
]

# Tier 2 pairs (5 — exploratory)
TIER2_PAIRS: List[Tuple[int, int, str]] = [
    (_PI["GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_DN"], _PI["REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION"], "M2 ↔ MHC-II"),
    (_PI["HALLMARK_G2M_CHECKPOINT"], _PI["GSE9650_EFFECTOR_VS_EXHAUSTED_CD8_TCELL_UP"], "G2M ↔ Exhausted CD8"),
    (_PI["HALLMARK_OXIDATIVE_PHOSPHORYLATION"], _PI["HALLMARK_GLYCOLYSIS"], "OxPhos ↔ Glycolysis"),
    (_PI["HALLMARK_E2F_TARGETS"], _PI["HALLMARK_INTERFERON_GAMMA_RESPONSE"], "E2F ↔ IFN-γ"),
    (_PI["HALLMARK_MTORC1_SIGNALING"], _PI["GSE13306_TREG_VS_TCONV_UP"], "mTORC1 ↔ Treg"),
]

N_PROGRAMS = 40


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpatialConfig:
    """All paths and hyperparameters for the spatial analysis."""

    # Paths — overridable via env vars; see data/README.md
    gep_dir: Path = Path(os.environ.get("SPARC_TCGA_GEP", "features/tcga/predicted_programs_transformer"))
    coord_dir: Path = Path(os.environ.get("SPARC_TCGA_COORD", "features/tcga/hoptimus1"))
    splits_csv: Path = Path(os.environ.get("SPARC_SPLITS_CSV", "data/mmp_hybrid_splits_v2_20cancer.csv"))
    clinical_csv: Path = Path(os.environ.get("SPARC_CLINICAL_CSV", "data/clinical_dss.csv"))
    output_dir: Path = Path(os.environ.get("SPARC_RESULTS_ROOT", "results")) / "sparc_spatial"

    # Spatial graph
    # Note: coords are in level-0 pixels; 20x/224px patches → step=448 in coord space
    patch_step: int = 448
    radius: float = 448 * 1.4142135623730951 + 1  # 8-connected: d√2 + ε ≈ 634.6

    # Analysis 2: null model
    n_permutations: int = 200
    perm_slides_per_cancer: int = 50
    brca_validation_perms: int = 1000
    skip_null: bool = False
    skip_brca_validation: bool = False

    # Analysis 3: niches
    subsample_patches: int = 1_000_000
    max_patches_per_slide: int = 500
    pca_variance_threshold: float = 0.90
    niche_k_range: Tuple[int, ...] = (4, 5, 6, 7, 8)
    kmeans_n_init: int = 50
    niche_k_override: Optional[int] = None

    # Analysis 4: survival
    min_dss_events: int = 50

    # Analysis 5: archetypes
    archetype_k_range: Tuple[int, ...] = (2, 3, 4, 5)

    # Normalization
    patch_center: bool = True  # per-patch centering (remove global activation)

    # Execution
    n_workers: int = 16
    seed: int = 42
    skip_plots: bool = False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@dataclass
class SlideRecord:
    slide_id: str
    patient_id: str
    cancer_type: str
    gep_path: Path
    coord_path: Path


def _find_feature_file(directory: Path, stem: str) -> Optional[Path]:
    """Find a feature file by stem, preferring .npz over .h5."""
    npz = directory / f"{stem}.npz"
    if npz.exists():
        return npz
    h5 = directory / f"{stem}.h5"
    if h5.exists():
        return h5
    return None


def _load_array(path: Path, key: str) -> np.ndarray:
    """Load an array from .npz or .h5 file."""
    if path.suffix == ".npz":
        return np.load(path)[key]
    else:
        import h5py
        with h5py.File(path, "r") as f:
            return f[key][:]


def build_slide_registry(cfg: SpatialConfig) -> List[SlideRecord]:
    """Build list of slides with matched GEP + coord files."""
    splits = pd.read_csv(cfg.splits_csv)
    # Deduplicate across folds — keep one row per slide
    slides_df = splits.drop_duplicates(subset="slide_id")[["slide_id", "patient_id", "cancer_type"]]

    # Available files (support both .npz and .h5)
    gep_stems = {p.stem for p in cfg.gep_dir.iterdir() if p.suffix in (".npz", ".h5")}
    coord_stems = {p.stem for p in cfg.coord_dir.iterdir() if p.suffix in (".npz", ".h5")}
    available = gep_stems & coord_stems

    records = []
    for _, row in slides_df.iterrows():
        sid = row["slide_id"]
        if sid in available:
            records.append(SlideRecord(
                slide_id=sid,
                patient_id=row["patient_id"],
                cancer_type=row["cancer_type"],
                gep_path=_find_feature_file(cfg.gep_dir, sid),
                coord_path=_find_feature_file(cfg.coord_dir, sid),
            ))
    return records


def load_slide_data(rec: SlideRecord, patch_center: bool = True,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one slide, return normalized GEP, raw GEP, and coords.

    Steps:
        1. Within-slide z-score (per program, across patches)
        2. Per-patch centering (subtract patch mean across programs) — removes
           shared global activation per patch, isolating relative program profiles.

    Returns:
        Z: [N, 40] normalized (z-scored, optionally patch-centered)
        raw: [N, 40] raw GEP scores
        coords: [N, 2] pixel coordinates
    """
    raw = _load_array(rec.gep_path, "features").astype(np.float32)  # [N, 40]
    coords = _load_array(rec.coord_path, "coords")  # [N, 2]
    assert raw.shape[0] == coords.shape[0], f"Patch count mismatch for {rec.slide_id}"
    # Step 1: within-slide z-score (per program)
    mu = raw.mean(axis=0, keepdims=True)
    sd = raw.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Z = (raw - mu) / sd
    # Step 2: per-patch centering (remove global activation per patch)
    if patch_center:
        Z = Z - Z.mean(axis=1, keepdims=True)
    return Z, raw, coords


def build_radius_adj(coords: np.ndarray, radius: float) -> csr_matrix:
    """Build sparse adjacency matrix (no self-loops) via cKDTree."""
    N = coords.shape[0]
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius)
    rows, cols = [], []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if j != i:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))
    return adj


def row_normalize_sparse(adj: csr_matrix) -> csr_matrix:
    """Row-normalize a sparse adjacency matrix. Zero-degree rows stay zero."""
    deg = np.array(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv_deg = 1.0 / deg
    # Multiply each row by its inverse degree
    from scipy.sparse import diags
    return diags(inv_deg) @ adj


def fisher_z(rho: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(rho, -0.9999, 0.9999))


def fisher_z_inv(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def vectorized_spearman_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Spearman cross-correlation matrix between columns of A and B.

    Returns S[i,j] = Spearman(A[:,i], B[:,j])  shape [A.shape[1], B.shape[1]]
    """
    N = A.shape[0]
    # Rank each column
    rA = np.apply_along_axis(rankdata, 0, A)
    rB = np.apply_along_axis(rankdata, 0, B)
    # Standardize
    std_A = rA.std(0, ddof=0)
    std_B = rB.std(0, ddof=0)
    std_A[std_A < 1e-8] = 1.0
    std_B[std_B < 1e-8] = 1.0
    rA = (rA - rA.mean(0)) / std_A
    rB = (rB - rB.mean(0)) / std_B
    return (rA.T @ rB) / N


# ---------------------------------------------------------------------------
# Step 2: Shared Per-Slide Processing
# ---------------------------------------------------------------------------

def process_slide(rec: SlideRecord, cfg: SpatialConfig) -> Optional[Dict[str, Any]]:
    """Single-pass per-slide: load, z-score, build graph, compute all stats."""
    try:
        Z, raw, coords = load_slide_data(rec, patch_center=cfg.patch_center)
    except Exception as e:
        return None  # skip corrupted

    N = Z.shape[0]
    if N < 10:
        return None  # too few patches

    # Spatial graph
    adj = build_radius_adj(coords, cfg.radius)
    adj_norm = row_normalize_sparse(adj)

    # --- A1: Co-activation (Spearman within slide) ---
    coact_rho, _ = spearmanr(Z, axis=0)  # [40, 40]
    if coact_rho.ndim == 0:
        return None
    coact_fz = fisher_z(coact_rho)

    # --- A2: Neighbor means ---
    N_mean = adj_norm @ Z  # [N, 40] sparse @ dense

    # Cross-correlation S via vectorized Spearman
    S = vectorized_spearman_cross(Z, N_mean)  # [40, 40]
    S_fz = fisher_z(S)

    # D matrix: conditional neighbor mean difference (top 25% of g1)
    D = np.zeros((N_PROGRAMS, N_PROGRAMS), dtype=np.float32)
    global_mean = Z.mean(axis=0)
    for g1 in range(N_PROGRAMS):
        threshold = np.percentile(Z[:, g1], 75)
        mask = Z[:, g1] >= threshold
        if mask.sum() > 0:
            D[g1, :] = N_mean[mask].mean(axis=0) - global_mean

    # O matrix: binary adjacency fraction (z > 1)
    active = (Z > 1.0).astype(np.float32)
    O = np.zeros((N_PROGRAMS, N_PROGRAMS), dtype=np.float32)
    for g1 in range(N_PROGRAMS):
        active_mask = active[:, g1].astype(bool)
        n_active = active_mask.sum()
        if n_active == 0:
            continue
        # For each g1-active patch, check if any neighbor is g2-active
        adj_sub = adj[active_mask]  # [n_active, N] sparse
        for g2 in range(N_PROGRAMS):
            # Fraction of g1-active patches with ≥1 g2-active neighbor
            neigh_g2 = adj_sub @ active[:, g2:g2+1].astype(np.float32)
            if hasattr(neigh_g2, "toarray"):
                neigh_g2 = neigh_g2.toarray()
            has_g2_neigh = neigh_g2.ravel() > 0
            O[g1, g2] = has_g2_neigh.mean()

    # --- A3: Subsample patches for niche discovery ---
    n_sub = min(cfg.max_patches_per_slide, N)
    rng = np.random.RandomState(hash(rec.slide_id) % (2**31))
    sub_idx = rng.choice(N, size=n_sub, replace=False)
    Z_sub = Z[sub_idx]

    # Mean GEP (for abundance adjustment in A4)
    gep_means = raw.mean(axis=0)  # [40]

    return {
        "slide_id": rec.slide_id,
        "patient_id": rec.patient_id,
        "cancer_type": rec.cancer_type,
        "n_patches": N,
        "coact_fz": coact_fz,
        "S_fz": S_fz,
        "D": D,
        "O": O,
        "Z_sub": Z_sub,
        "gep_means": gep_means,
    }


def run_slide_processing(slides: List[SlideRecord], cfg: SpatialConfig) -> List[Dict]:
    """Process all slides in parallel. Returns list of per-slide dicts."""
    fn = partial(process_slide, cfg=cfg)
    results = []
    with Pool(cfg.n_workers) as pool:
        for res in tqdm(pool.imap_unordered(fn, slides), total=len(slides),
                        desc="Processing slides"):
            if res is not None:
                results.append(res)
    return results


def save_slide_cache(slide_stats: List[Dict], output_dir: Path):
    """Save per-slide results to cache directory."""
    cache_dir = output_dir / "slide_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save compact arrays
    slide_ids = [s["slide_id"] for s in slide_stats]
    patient_ids = [s["patient_id"] for s in slide_stats]
    cancer_types = [s["cancer_type"] for s in slide_stats]
    n_patches = np.array([s["n_patches"] for s in slide_stats])
    coact_fz = np.stack([s["coact_fz"] for s in slide_stats])  # [S, 40, 40]
    S_fz = np.stack([s["S_fz"] for s in slide_stats])
    D = np.stack([s["D"] for s in slide_stats])
    O = np.stack([s["O"] for s in slide_stats])
    gep_means = np.stack([s["gep_means"] for s in slide_stats])

    # Niche subsample patches
    Z_sub_all = np.concatenate([s["Z_sub"] for s in slide_stats], axis=0)

    np.savez_compressed(
        cache_dir / "matrices.npz",
        coact_fz=coact_fz, S_fz=S_fz, D=D, O=O, gep_means=gep_means,
        n_patches=n_patches, Z_sub=Z_sub_all,
    )
    pd.DataFrame({"slide_id": slide_ids, "patient_id": patient_ids,
                   "cancer_type": cancer_types, "n_patches": n_patches}).to_csv(
        cache_dir / "slide_meta.csv", index=False)

    # Summary
    ct_counts = pd.Series(cancer_types).value_counts().sort_index()
    with open(cache_dir / "summary.txt", "w") as f:
        f.write(f"Slides processed: {len(slide_stats)}\n")
        f.write(f"Total subsample patches: {Z_sub_all.shape[0]}\n")
        f.write(f"Mean patches/slide: {n_patches.mean():.0f}\n\n")
        f.write("Per-cancer slide counts:\n")
        for ct, cnt in ct_counts.items():
            f.write(f"  {ct}: {cnt}\n")

    print(f"✓ Slide cache saved: {len(slide_stats)} slides, "
          f"{Z_sub_all.shape[0]} subsample patches")


def load_slide_cache(output_dir: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Load cached slide results."""
    cache_dir = output_dir / "slide_cache"
    data = dict(np.load(cache_dir / "matrices.npz"))
    meta = pd.read_csv(cache_dir / "slide_meta.csv")
    return data, meta


# ---------------------------------------------------------------------------
# Analysis 1: Co-Activation
# ---------------------------------------------------------------------------

def run_analysis1(data: Dict, meta: pd.DataFrame, cfg: SpatialConfig):
    """Aggregate co-activation matrices, produce heatmaps."""
    out_dir = cfg.output_dir / "a1_coactivation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    coact_fz = data["coact_fz"]  # [S, 40, 40]
    cancer_types = meta["cancer_type"].values

    # Pan-cancer average
    pan_fz = coact_fz.mean(axis=0)
    pan_rho = fisher_z_inv(pan_fz)
    np.fill_diagonal(pan_rho, 1.0)

    np.savez(out_dir / "pan_cancer_matrix.npz", rho=pan_rho, fisher_z=pan_fz,
             program_names=PROGRAM_NAMES, short_names=SHORT_NAMES)

    # Per-cancer
    per_cancer_dir = out_dir / "per_cancer"
    per_cancer_dir.mkdir(exist_ok=True)
    unique_cts = sorted(set(cancer_types))
    for ct in unique_cts:
        mask = cancer_types == ct
        ct_fz = coact_fz[mask].mean(axis=0)
        ct_rho = fisher_z_inv(ct_fz)
        np.fill_diagonal(ct_rho, 1.0)
        np.savez(per_cancer_dir / f"{ct}_matrix.npz", rho=ct_rho, fisher_z=ct_fz)

    # Top pairs
    triu_idx = np.triu_indices(N_PROGRAMS, k=1)
    pair_vals = pan_rho[triu_idx]
    pair_names = [(SHORT_NAMES[i], SHORT_NAMES[j]) for i, j in zip(*triu_idx)]
    sort_idx = np.argsort(-np.abs(pair_vals))

    rows = []
    for k in sort_idx:
        rows.append({
            "program_1": pair_names[k][0],
            "program_2": pair_names[k][1],
            "rho": pair_vals[k],
        })
    top_df = pd.DataFrame(rows)
    top_df.to_csv(out_dir / "top_pairs.csv", index=False)

    # Print top 10
    print("\n" + "=" * 60)
    print("A1: CO-ACTIVATION — Top 10 pairs (pan-cancer)")
    print("=" * 60)
    for _, row in top_df.head(10).iterrows():
        print(f"  {row['program_1']:35s} ↔ {row['program_2']:35s}  ρ={row['rho']:+.3f}")
    print(f"\n  Anti-correlated:")
    for _, row in top_df.tail(10).iloc[::-1].iterrows():
        if row["rho"] < 0:
            print(f"  {row['program_1']:35s} ↔ {row['program_2']:35s}  ρ={row['rho']:+.3f}")

    # Heatmap
    if not cfg.skip_plots:
        Z_link = linkage(pan_rho, method="ward")
        order = leaves_list(Z_link)
        ordered = pan_rho[np.ix_(order, order)]
        labels = [SHORT_NAMES[i] for i in order]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(ordered, xticklabels=labels, yticklabels=labels,
                    cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                    square=True, linewidths=0.1, ax=ax,
                    cbar_kws={"shrink": 0.6, "label": "Spearman ρ"})
        ax.set_title("Pan-Cancer Co-Activation (Intra-Patch Spearman)", fontsize=14)
        plt.xticks(fontsize=7, rotation=90)
        plt.yticks(fontsize=7, rotation=0)
        plt.tight_layout()
        plt.savefig(fig_dir / "pan_cancer_heatmap.pdf", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {fig_dir / 'pan_cancer_heatmap.pdf'}")

    (out_dir / "done").touch()
    print("✓ Analysis 1 complete\n")


# ---------------------------------------------------------------------------
# Analysis 2: Cross-Correlation
# ---------------------------------------------------------------------------

def _permuted_S_for_slide(args):
    """Compute S matrix for one slide with permuted patch assignments."""
    rec, cfg, perm_seed = args
    try:
        Z, _, coords = load_slide_data(rec, patch_center=cfg.patch_center)
    except Exception:
        return None
    N = Z.shape[0]
    if N < 10:
        return None
    rng = np.random.RandomState(perm_seed)
    perm_idx = rng.permutation(N)
    Z_perm = Z[perm_idx]  # permute rows (break spatial-program coupling)
    adj = build_radius_adj(coords, cfg.radius)
    adj_norm = row_normalize_sparse(adj)
    N_mean = adj_norm @ Z_perm
    S = vectorized_spearman_cross(Z_perm, N_mean)
    return fisher_z(S)


def run_analysis2(data: Dict, meta: pd.DataFrame, slides: List[SlideRecord],
                  cfg: SpatialConfig):
    """Cross-correlation aggregation, null model, divergence."""
    out_dir = cfg.output_dir / "a2_cross_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    S_fz = data["S_fz"]  # [S, 40, 40]
    cancer_types = meta["cancer_type"].values

    # Pan-cancer average
    pan_S_fz = S_fz.mean(axis=0)
    pan_S = fisher_z_inv(pan_S_fz)
    pan_D = data["D"].mean(axis=0)
    pan_O = data["O"].mean(axis=0)

    np.savez(out_dir / "S_matrix.npz", S=pan_S, S_fz=pan_S_fz)
    np.savez(out_dir / "D_matrix.npz", D=pan_D)
    np.savez(out_dir / "O_matrix.npz", O=pan_O)

    # Per-cancer
    per_cancer_dir = out_dir / "per_cancer"
    per_cancer_dir.mkdir(exist_ok=True)
    unique_cts = sorted(set(cancer_types))
    for ct in unique_cts:
        mask = cancer_types == ct
        ct_S = fisher_z_inv(S_fz[mask].mean(axis=0))
        np.savez(per_cancer_dir / f"{ct}_S.npz", S=ct_S)

    # Tier 1 pairs summary
    rows = []
    for g1, g2, name in TIER1_PAIRS + TIER2_PAIRS:
        tier = "Tier1" if (g1, g2, name) in TIER1_PAIRS else "Tier2"
        rows.append({
            "pair": name, "tier": tier,
            "S": pan_S[g1, g2], "D": pan_D[g1, g2], "O": pan_O[g1, g2],
        })
    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(out_dir / "tier1_pairs_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("A2: CROSS-CORRELATION — Predefined pairs")
    print("=" * 60)
    for _, row in pairs_df.iterrows():
        print(f"  [{row['tier']}] {row['pair']:30s}  S={row['S']:+.3f}  D={row['D']:+.3f}  O={row['O']:.3f}")

    # --- Null model ---
    null_path = out_dir / "null_model.npz"
    if not cfg.skip_null and not null_path.exists():
        print("\n  Running null model (200 permutations)...")
        # Select subset: ~50 slides per cancer
        rng = np.random.RandomState(cfg.seed)
        slide_lookup = {s.slide_id: s for s in slides}
        subset = []
        for ct in unique_cts:
            ct_sids = meta[meta["cancer_type"] == ct]["slide_id"].values
            n_sel = min(cfg.perm_slides_per_cancer, len(ct_sids))
            sel = rng.choice(ct_sids, size=n_sel, replace=False)
            for sid in sel:
                if sid in slide_lookup:
                    subset.append(slide_lookup[sid])

        null_S_fz_all = []  # [n_perm, 40, 40]
        for perm_i in tqdm(range(cfg.n_permutations), desc="  Null permutations"):
            args_list = [(rec, cfg, cfg.seed + perm_i * 10000 + i)
                         for i, rec in enumerate(subset)]
            with Pool(cfg.n_workers) as pool:
                perm_results = list(pool.imap_unordered(_permuted_S_for_slide, args_list))
            valid = [r for r in perm_results if r is not None]
            if valid:
                null_S_fz_all.append(np.stack(valid).mean(axis=0))

        null_S_fz = np.stack(null_S_fz_all)  # [n_perm, 40, 40]
        null_mean = fisher_z_inv(null_S_fz.mean(axis=0))
        null_std = fisher_z_inv(null_S_fz.std(axis=0))

        np.savez(null_path, null_mean=null_mean, null_std=null_std,
                 null_S_fz=null_S_fz, n_perms=len(null_S_fz_all),
                 n_slides_per_perm=len(subset))
        print(f"  ✓ Null model: {len(null_S_fz_all)} perms × {len(subset)} slides")

    # --- Divergence panel ---
    if not cfg.skip_plots:
        coact_fz = data["coact_fz"].mean(axis=0)
        coact_rho = fisher_z_inv(coact_fz)
        divergence = np.abs(pan_S - coact_rho)

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        for ax, mat, title, cmap, center in [
            (axes[0], coact_rho, "Co-Activation (Intra-Patch)", "RdBu_r", 0),
            (axes[1], pan_S, "Cross-Correlation (Inter-Patch S)", "RdBu_r", 0),
            (axes[2], divergence, "Divergence |S − CoAct|", "YlOrRd", None),
        ]:
            sns.heatmap(mat, xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                        cmap=cmap, center=center, square=True, linewidths=0.05,
                        ax=ax, cbar_kws={"shrink": 0.6})
            ax.set_title(title, fontsize=12)
            ax.tick_params(labelsize=5)
        plt.tight_layout()
        plt.savefig(fig_dir / "divergence_panel.pdf", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {fig_dir / 'divergence_panel.pdf'}")

        # S heatmap (clustered)
        Z_link = linkage(pan_S, method="ward")
        order = leaves_list(Z_link)
        ordered = pan_S[np.ix_(order, order)]
        labels = [SHORT_NAMES[i] for i in order]
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(ordered, xticklabels=labels, yticklabels=labels,
                    cmap="RdBu_r", center=0, square=True, linewidths=0.1, ax=ax,
                    cbar_kws={"shrink": 0.6, "label": "Spearman S"})
        ax.set_title("Pan-Cancer Spatial Cross-Correlation (Inter-Patch)", fontsize=14)
        plt.xticks(fontsize=7, rotation=90)
        plt.yticks(fontsize=7, rotation=0)
        plt.tight_layout()
        plt.savefig(fig_dir / "S_heatmap.pdf", dpi=150, bbox_inches="tight")
        plt.close()

    (out_dir / "done").touch()
    print("✓ Analysis 2 complete\n")


# ---------------------------------------------------------------------------
# Analysis 3: Spatial Niches
# ---------------------------------------------------------------------------

_NICHE_CACHE = {}  # populated by pool initializer

def _assign_slide_init(pca_components, pca_mean, km_centroids, best_K, patch_center):
    """Pool initializer: store PCA + KMeans params in global (no pickle needed)."""
    global _NICHE_CACHE
    _NICHE_CACHE["pca_components"] = pca_components  # [n_pcs, 40]
    _NICHE_CACHE["pca_mean"] = pca_mean  # [40]
    _NICHE_CACHE["km_centroids"] = km_centroids  # [K, n_pcs]
    _NICHE_CACHE["best_K"] = best_K
    _NICHE_CACHE["patch_center"] = patch_center


def _assign_slide_worker(rec: SlideRecord):
    """Top-level function for multiprocessing niche assignment."""
    try:
        Z, _, _ = load_slide_data(rec, patch_center=_NICHE_CACHE["patch_center"])
        # Manual PCA transform: (Z - mean) @ components.T
        Z_pca = (Z - _NICHE_CACHE["pca_mean"]) @ _NICHE_CACHE["pca_components"].T
        # Manual KMeans predict: nearest centroid
        from scipy.spatial.distance import cdist
        dists = cdist(Z_pca, _NICHE_CACHE["km_centroids"])
        labels = dists.argmin(axis=1)
        counts = np.bincount(labels, minlength=_NICHE_CACHE["best_K"])
        return rec.slide_id, rec.patient_id, rec.cancer_type, counts
    except Exception:
        return None


def run_analysis3(data: Dict, meta: pd.DataFrame, slides: List[SlideRecord],
                  cfg: SpatialConfig):
    """Discover spatial niches, assign all patches, compute composition."""
    out_dir = cfg.output_dir / "a3_niches"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    Z_sub = data["Z_sub"]  # [M, 40] from subsample
    print(f"\n  Niche discovery: {Z_sub.shape[0]} subsampled patches")

    # PCA
    pca = PCA(n_components=cfg.pca_variance_threshold, svd_solver="full",
              random_state=cfg.seed)
    X_pca = pca.fit_transform(Z_sub)
    n_pcs = pca.n_components_
    print(f"  PCA: {n_pcs} components, {pca.explained_variance_ratio_.sum():.1%} variance")

    # K-Means for each K
    km_results = {}
    for K in tqdm(cfg.niche_k_range, desc="  K-Means"):
        km = KMeans(n_clusters=K, n_init=cfg.kmeans_n_init, random_state=cfg.seed,
                    max_iter=300)
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels, sample_size=min(50000, len(labels)),
                               random_state=cfg.seed)
        km_results[K] = {"model": km, "silhouette": sil, "labels": labels}

    # Select best K
    if cfg.niche_k_override:
        best_K = cfg.niche_k_override
    else:
        best_K = max(km_results, key=lambda k: km_results[k]["silhouette"])
    best_km = km_results[best_K]["model"]

    print(f"\n  Silhouette scores:")
    for K in sorted(km_results):
        marker = " ← best" if K == best_K else ""
        print(f"    K={K}: {km_results[K]['silhouette']:.4f}{marker}")

    # Save models
    with open(out_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(out_dir / "kmeans_models.pkl", "wb") as f:
        pickle.dump(km_results, f)
    with open(out_dir / "best_k.json", "w") as f:
        json.dump({"best_K": int(best_K), "silhouette": float(km_results[best_K]["silhouette"])}, f)

    # Niche centroids in original (z-scored) space
    centroids_pca = best_km.cluster_centers_  # [K, n_pcs]
    centroids_z = pca.inverse_transform(centroids_pca)  # [K, 40]
    np.savez(out_dir / "niche_centroids.npz", centroids_z=centroids_z,
             centroids_pca=centroids_pca, program_names=PROGRAM_NAMES,
             short_names=SHORT_NAMES, best_K=best_K)

    # Print niche profiles
    print(f"\n  Niche centroid profiles (K={best_K}):")
    for k in range(best_K):
        top3_idx = np.argsort(-centroids_z[k])[:3]
        top3 = ", ".join(f"{SHORT_NAMES[i]}({centroids_z[k, i]:+.2f})" for i in top3_idx)
        print(f"    Niche {k}: {top3}")

    # --- Assign ALL patches ---
    print("\n  Assigning all patches to niches...")
    slide_lookup = {s.slide_id: s for s in slides}
    slide_ids_ordered = meta["slide_id"].values
    patient_ids = meta["patient_id"].values
    cancer_types = meta["cancer_type"].values

    # Patient-level composition
    patient_data = {}  # patient_id -> {cancer_type, niche_counts[K], total_patches}

    assign_results = []
    with Pool(cfg.n_workers, initializer=_assign_slide_init,
              initargs=(pca.components_.astype(np.float32),
                        pca.mean_.astype(np.float32),
                        best_km.cluster_centers_.astype(np.float32),
                        best_K, cfg.patch_center)) as pool:
        valid_recs = [slide_lookup[sid] for sid in slide_ids_ordered if sid in slide_lookup]
        for res in tqdm(pool.imap_unordered(_assign_slide_worker, valid_recs),
                        total=len(valid_recs), desc="  Niche assignment"):
            if res is not None:
                assign_results.append(res)

    for sid, pid, ct, counts in assign_results:
        if pid not in patient_data:
            patient_data[pid] = {"cancer_type": ct, "counts": np.zeros(best_K), "total": 0}
        patient_data[pid]["counts"] += counts
        patient_data[pid]["total"] += counts.sum()

    # Build composition DataFrame
    rows = []
    for pid, d in patient_data.items():
        fracs = d["counts"] / max(d["total"], 1)
        entropy = -np.sum(fracs[fracs > 0] * np.log(fracs[fracs > 0]))
        row = {"patient_id": pid, "cancer_type": d["cancer_type"], "entropy": entropy}
        for k in range(best_K):
            row[f"p_{k}"] = fracs[k]
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "patient_composition.csv", index=False)
    print(f"  ✓ Patient composition: {len(comp_df)} patients, K={best_K}")

    # --- Per-cancer niche sensitivity check ---
    print("\n  Per-cancer niche sensitivity (BRCA, LUAD, COAD)...")
    sensitivity_cts = ["TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD"]
    for ct in sensitivity_cts:
        ct_mask = meta["cancer_type"].values == ct
        if ct_mask.sum() < 100:
            continue
        # Get subsample indices for this cancer
        ct_indices = np.where(ct_mask)[0]
        # Collect Z_sub patches for this cancer
        # We need to re-subsample from the full Z_sub — approximate by slide counts
        n_sub_per_slide = cfg.max_patches_per_slide
        start = 0
        ct_patches = []
        for i, n in enumerate(meta["n_patches"].values):
            n_sub_i = min(n_sub_per_slide, n)
            if ct_mask[i]:
                ct_patches.append(Z_sub[start:start + n_sub_i])
            start += n_sub_i
        if not ct_patches:
            continue
        ct_Z = np.concatenate(ct_patches, axis=0)
        if len(ct_Z) < 1000:
            continue
        ct_pca = pca.transform(ct_Z)
        ct_km = KMeans(n_clusters=best_K, n_init=20, random_state=cfg.seed)
        ct_km.fit(ct_pca)
        # Compare centroids via cosine similarity
        from scipy.spatial.distance import cdist
        ct_centroids_z = pca.inverse_transform(ct_km.cluster_centers_)
        sim = 1 - cdist(centroids_z, ct_centroids_z, metric="cosine")
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-sim)
        matched_sims = sim[row_ind, col_ind]
        print(f"    {ct}: matched centroid cosine similarities: "
              f"{', '.join(f'{s:.3f}' for s in matched_sims)} "
              f"(mean={matched_sims.mean():.3f})")

    # --- Figures ---
    if not cfg.skip_plots:
        # Centroid heatmap
        fig, ax = plt.subplots(figsize=(12, max(3, best_K * 0.8)))
        sns.heatmap(centroids_z, xticklabels=SHORT_NAMES, yticklabels=[f"Niche {k}" for k in range(best_K)],
                    cmap="RdBu_r", center=0, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.6, "label": "Mean z-score"})
        ax.set_title(f"Niche Centroid Profiles (K={best_K})", fontsize=14)
        plt.xticks(fontsize=7, rotation=90)
        plt.tight_layout()
        plt.savefig(fig_dir / "centroid_profiles.pdf", dpi=150, bbox_inches="tight")
        plt.close()

        # Composition by cancer type
        ct_comp = comp_df.groupby("cancer_type")[[f"p_{k}" for k in range(best_K)]].mean()
        ct_comp = ct_comp.loc[ct_comp.index.sort_values()]
        fig, ax = plt.subplots(figsize=(14, 6))
        ct_comp.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("Mean niche fraction")
        ax.set_title("Niche Composition by Cancer Type")
        ax.legend(title="Niche", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "niche_composition_by_cancer.pdf", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figures saved to {fig_dir}")

    (out_dir / "done").touch()
    print("✓ Analysis 3 complete\n")


# ---------------------------------------------------------------------------
# Analysis 4: Prognostic Value
# ---------------------------------------------------------------------------

def _fit_cox_univariate(X: np.ndarray, y: np.ndarray) -> Dict:
    """Fit univariate Cox model. X: [N, 1], y: structured array (event, time)."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    try:
        cox = CoxPHSurvivalAnalysis(alpha=1e-4)
        cox.fit(X, y)
        hr = np.exp(cox.coef_[0])
        # CI via Wald SE
        se = np.sqrt(np.diag(np.linalg.inv(cox._compute_baseline_model().hessian_ if hasattr(cox, '_compute_baseline_model') else np.eye(1))))[0] if False else np.nan
        ci = cox.score(X, y)
        return {"hr": hr, "coef": cox.coef_[0], "c_index": ci, "converged": True}
    except Exception as e:
        return {"hr": np.nan, "coef": np.nan, "c_index": np.nan, "converged": False}


def _fit_cox_multivariate(X: np.ndarray, y: np.ndarray, n_spatial: int = 1) -> Dict:
    """Fit multivariate Cox. First n_spatial columns are spatial features."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    try:
        cox = CoxPHSurvivalAnalysis(alpha=1e-3)
        cox.fit(X, y)
        hr_spatial = np.exp(cox.coef_[0])
        ci = cox.score(X, y)
        return {"hr": hr_spatial, "coef": cox.coef_[0], "c_index": ci,
                "all_coefs": cox.coef_.tolist(), "converged": True}
    except Exception:
        return {"hr": np.nan, "coef": np.nan, "c_index": np.nan, "converged": False}


def run_analysis4(data: Dict, meta: pd.DataFrame, cfg: SpatialConfig):
    """Prognostic value of spatial features via Cox models."""
    out_dir = cfg.output_dir / "a4_prognostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load clinical
    clin = pd.read_csv(cfg.clinical_csv)

    # Load niche composition
    comp_path = cfg.output_dir / "a3_niches" / "patient_composition.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
    else:
        comp_df = None
        print("  ⚠ Niche composition not found — skipping niche survival")

    # Build patient-level features from slide-level S matrices
    S_fz = data["S_fz"]  # [S, 40, 40]
    gep_means = data["gep_means"]  # [S, 40]
    slide_pids = meta["patient_id"].values
    slide_cts = meta["cancer_type"].values

    # Aggregate to patient level: mean S and mean GEP across slides
    patient_S = {}  # pid -> list of S_fz
    patient_gep = {}  # pid -> list of gep_means
    patient_ct = {}
    for i in range(len(slide_pids)):
        pid = slide_pids[i]
        if pid not in patient_S:
            patient_S[pid] = []
            patient_gep[pid] = []
            patient_ct[pid] = slide_cts[i]
        patient_S[pid].append(S_fz[i])
        patient_gep[pid].append(gep_means[i])

    # Mean across slides per patient
    patient_S_mean = {pid: fisher_z_inv(np.stack(v).mean(0)) for pid, v in patient_S.items()}
    patient_gep_mean = {pid: np.stack(v).mean(0) for pid, v in patient_gep.items()}

    # Merge with clinical
    clin_map = {}
    for _, row in clin.iterrows():
        clin_map[row["patient_id"]] = (row["time"], row["event"])

    all_results = []
    all_pairs = TIER1_PAIRS + TIER2_PAIRS

    # Per-cancer analysis
    unique_cts = sorted(set(patient_ct.values()))
    for ct in unique_cts:
        ct_pids = [pid for pid, c in patient_ct.items() if c == ct and pid in clin_map]
        times = np.array([clin_map[pid][0] for pid in ct_pids])
        events = np.array([clin_map[pid][1] for pid in ct_pids])
        n_events = events.sum()
        if n_events < cfg.min_dss_events:
            continue

        # sksurv structured array
        y = np.array([(bool(e), t) for e, t in zip(events, times)],
                     dtype=[("event", bool), ("time", float)])

        # --- Pairwise spatial features ---
        for g1, g2, pair_name in all_pairs:
            tier = "Tier1" if (g1, g2, pair_name) in TIER1_PAIRS else "Tier2"

            # Spatial feature: S(g1, g2) per patient
            x_spatial = np.array([patient_S_mean[pid][g1, g2] for pid in ct_pids]).reshape(-1, 1)
            # Abundance features
            x_abund_g1 = np.array([patient_gep_mean[pid][g1] for pid in ct_pids]).reshape(-1, 1)
            x_abund_g2 = np.array([patient_gep_mean[pid][g2] for pid in ct_pids]).reshape(-1, 1)

            # Standardize features
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            x_spatial_s = sc.fit_transform(x_spatial)

            # Model 1: univariate
            m1 = _fit_cox_univariate(x_spatial_s, y)

            # Model 2: abundance-adjusted
            X_m2 = np.hstack([x_spatial_s,
                              StandardScaler().fit_transform(x_abund_g1),
                              StandardScaler().fit_transform(x_abund_g2)])
            m2 = _fit_cox_multivariate(X_m2, y, n_spatial=1)

            all_results.append({
                "cancer_type": ct, "pair": pair_name, "tier": tier,
                "n_patients": len(ct_pids), "n_events": int(n_events),
                "m1_hr": m1["hr"], "m1_c": m1["c_index"],
                "m2_hr": m2["hr"], "m2_c": m2["c_index"],
            })

        # --- Niche features ---
        if comp_df is not None:
            niche_cols = [c for c in comp_df.columns if c.startswith("p_")]
            ct_comp = comp_df[comp_df["cancer_type"] == ct].set_index("patient_id")
            ct_pids_with_comp = [pid for pid in ct_pids if pid in ct_comp.index]
            if len(ct_pids_with_comp) < cfg.min_dss_events:
                continue
            y_niche = np.array(
                [(bool(clin_map[pid][1]), clin_map[pid][0]) for pid in ct_pids_with_comp],
                dtype=[("event", bool), ("time", float)])

            for col in niche_cols + ["entropy"]:
                x_niche = ct_comp.loc[ct_pids_with_comp, col].values.reshape(-1, 1)
                x_niche_s = StandardScaler().fit_transform(x_niche)
                m1 = _fit_cox_univariate(x_niche_s, y_niche)
                all_results.append({
                    "cancer_type": ct, "pair": col, "tier": "Niche",
                    "n_patients": len(ct_pids_with_comp),
                    "n_events": int(y_niche["event"].sum()),
                    "m1_hr": m1["hr"], "m1_c": m1["c_index"],
                    "m2_hr": np.nan, "m2_c": np.nan,
                })

    results_df = pd.DataFrame(all_results)
    if len(results_df) == 0:
        print("  ⚠ No cancers with enough events for survival analysis")
        (out_dir / "done").touch()
        return

    # BH-FDR correction (separate for Tier1 and Niche)
    # We approximate p-values from HRs using Wald test (coef / SE)
    # For now, report raw results — p-values need proper computation
    results_df.to_csv(out_dir / "all_results.csv", index=False)

    # Summary table
    print("\n" + "=" * 60)
    print("A4: PROGNOSTIC VALUE — Summary")
    print("=" * 60)

    for tier_name in ["Tier1", "Tier2", "Niche"]:
        tier_df = results_df[results_df["tier"] == tier_name]
        if len(tier_df) == 0:
            continue
        print(f"\n  {tier_name.upper()}")
        print(f"  {'Feature':<35s} {'Cancers':>8s} {'M1 HR range':>15s} {'M2 HR range':>15s}")
        print("  " + "-" * 75)
        for pair_name in tier_df["pair"].unique():
            pair_df = tier_df[tier_df["pair"] == pair_name]
            n_cancers = len(pair_df)
            m1_hrs = pair_df["m1_hr"].dropna()
            m2_hrs = pair_df["m2_hr"].dropna()
            m1_range = f"{m1_hrs.min():.2f}-{m1_hrs.max():.2f}" if len(m1_hrs) > 0 else "N/A"
            m2_range = f"{m2_hrs.min():.2f}-{m2_hrs.max():.2f}" if len(m2_hrs) > 0 else "N/A"
            print(f"  {pair_name:<35s} {n_cancers:>8d} {m1_range:>15s} {m2_range:>15s}")

    # Check for Tier 1 pairs with no significance
    for g1, g2, name in TIER1_PAIRS:
        pair_df = results_df[(results_df["pair"] == name) & (results_df["tier"] == "Tier1")]
        if len(pair_df) == 0:
            print(f"\n  ⚠ WARNING: {name} has no cancers with enough events")

    # Forest plot
    if not cfg.skip_plots and len(results_df) > 0:
        tier1_df = results_df[results_df["tier"] == "Tier1"].copy()
        if len(tier1_df) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(tier1_df) * 0.3)))
            y_pos = range(len(tier1_df))
            ax.scatter(tier1_df["m2_hr"], y_pos, c="steelblue", s=40, zorder=3)
            ax.axvline(1.0, color="gray", linewidth=0.5, linestyle="--")
            labels = [f"{row['pair']} ({row['cancer_type']})" for _, row in tier1_df.iterrows()]
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Hazard Ratio (abundance-adjusted)")
            ax.set_title("Tier 1 Spatial Pairs — Abundance-Adjusted Cox")
            plt.tight_layout()
            plt.savefig(fig_dir / "forest_plot_tier1.pdf", dpi=150, bbox_inches="tight")
            plt.close()

    (out_dir / "done").touch()
    print("✓ Analysis 4 complete\n")


# ---------------------------------------------------------------------------
# Analysis 5: Patient-Level TME Archetypes
# ---------------------------------------------------------------------------

def _kaplan_meier(time, event, groups):
    """Compute KM curves and log-rank test. Returns dict per group + p-value."""
    from collections import defaultdict
    unique_groups = sorted(set(groups))
    curves = {}
    for g in unique_groups:
        mask = groups == g
        t_g, e_g = time[mask], event[mask]
        sort_idx = np.argsort(t_g)
        t_sorted, e_sorted = t_g[sort_idx], e_g[sort_idx]
        unique_times = np.unique(t_sorted)
        surv = 1.0
        curve_t, curve_s = [0.0], [1.0]
        n_at_risk = len(t_sorted)
        for ut in unique_times:
            at_time = t_sorted == ut
            d = (e_sorted[at_time]).sum()
            n = n_at_risk
            if n > 0 and d > 0:
                surv *= (1 - d / n)
            curve_t.append(ut)
            curve_s.append(surv)
            n_at_risk -= at_time.sum()
        curves[g] = (np.array(curve_t), np.array(curve_s))

    # Log-rank test (simplified)
    p_val = _logrank_test(time, event, groups)
    return curves, p_val


def _logrank_test(time, event, groups):
    """Simplified log-rank test."""
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return 1.0
    all_times = np.unique(time[event == 1])
    O_E = np.zeros(len(unique_groups) - 1)
    V = np.zeros(len(unique_groups) - 1)
    for t in all_times:
        at_risk = {}
        deaths = {}
        for gi, g in enumerate(unique_groups):
            mask = groups == g
            at_risk[gi] = (time[mask] >= t).sum()
            deaths[gi] = ((time[mask] == t) & (event[mask] == 1)).sum()
        n_total = sum(at_risk.values())
        d_total = sum(deaths.values())
        if n_total == 0 or d_total == 0:
            continue
        for gi in range(len(unique_groups) - 1):
            E_gi = at_risk[gi] * d_total / n_total
            O_E[gi] += deaths[gi] - E_gi
            if n_total > 1:
                V[gi] += E_gi * (1 - at_risk[gi] / n_total) * (n_total - d_total) / (n_total - 1)

    chi2 = np.sum(O_E ** 2 / np.maximum(V, 1e-10))
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=len(unique_groups) - 1)
    return p


def run_analysis5(data: Dict, meta: pd.DataFrame, cfg: SpatialConfig):
    """Patient-level TME archetypes via consensus K-Means on niche composition."""
    out_dir = cfg.output_dir / "a5_archetypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    comp_path = cfg.output_dir / "a3_niches" / "patient_composition.csv"
    if not comp_path.exists():
        print("  ⚠ Niche composition not found — skipping archetypes")
        return

    comp_df = pd.read_csv(comp_path)
    clin = pd.read_csv(cfg.clinical_csv)
    clin_map = {row["patient_id"]: (row["time"], row["event"]) for _, row in clin.iterrows()}

    niche_cols = [c for c in comp_df.columns if c.startswith("p_")]
    X = comp_df[niche_cols + ["entropy"]].values

    # K-Means for each K
    best_K = None
    best_sil = -1
    km_results = {}
    for K in cfg.archetype_k_range:
        km = KMeans(n_clusters=K, n_init=50, random_state=cfg.seed)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, random_state=cfg.seed)
        km_results[K] = {"model": km, "labels": labels, "silhouette": sil}
        if sil > best_sil:
            best_sil = sil
            best_K = K

    comp_df["archetype"] = km_results[best_K]["labels"]
    comp_df.to_csv(out_dir / "archetype_assignments.csv", index=False)

    print(f"\n  Best archetype K={best_K} (silhouette={best_sil:.4f})")
    for k in range(best_K):
        n = (comp_df["archetype"] == k).sum()
        mean_comp = comp_df[comp_df["archetype"] == k][niche_cols].mean()
        top_niche = mean_comp.idxmax()
        print(f"    Archetype {k}: n={n}, dominant={top_niche} ({mean_comp[top_niche]:.2f})")

    # KM curves (pan-cancer)
    pids_with_surv = [pid for pid in comp_df["patient_id"] if pid in clin_map]
    surv_df = comp_df[comp_df["patient_id"].isin(pids_with_surv)].copy()
    times = np.array([clin_map[pid][0] for pid in surv_df["patient_id"]])
    events = np.array([clin_map[pid][1] for pid in surv_df["patient_id"]])
    groups = surv_df["archetype"].values

    curves, p_val = _kaplan_meier(times, events, groups)
    print(f"  Pan-cancer log-rank p={p_val:.4f}")

    if not cfg.skip_plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, best_K))
        for g, (t, s) in curves.items():
            n_g = (groups == g).sum()
            ax.step(t, s, where="post", color=colors[g], label=f"Archetype {g} (n={n_g})")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability")
        ax.set_title(f"TME Archetypes — Pan-Cancer KM (log-rank p={p_val:.4f})")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(fig_dir / "km_curves.pdf", dpi=150, bbox_inches="tight")
        plt.close()

        # Archetype composition
        arch_comp = comp_df.groupby("archetype")[niche_cols].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        arch_comp.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("Mean niche fraction")
        ax.set_xlabel("Archetype")
        ax.set_title("Niche Composition per Archetype")
        ax.legend(title="Niche", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "archetype_composition.pdf", dpi=150, bbox_inches="tight")
        plt.close()

    (out_dir / "done").touch()
    print("✓ Analysis 5 complete\n")


# ---------------------------------------------------------------------------
# Analysis 6: Boundary / Interface (Supplement)
# ---------------------------------------------------------------------------

def _process_slide_boundary(args):
    """Compute boundary fractions for predefined pairs."""
    rec, cfg, pairs = args
    try:
        Z, _, coords = load_slide_data(rec, patch_center=cfg.patch_center)
    except Exception:
        return None
    N = Z.shape[0]
    if N < 10:
        return None
    adj = build_radius_adj(coords, cfg.radius)
    active = (Z > 1.0)  # binary activation

    results = {}
    for g1, g2, name in pairs:
        # Boundary: g1-active AND g2-inactive AND has ≥1 g2-active neighbor
        g1_active = active[:, g1]
        g2_inactive = ~active[:, g2]
        candidate = g1_active & g2_inactive  # patches that are g1+, g2-
        if candidate.sum() == 0:
            results[name] = 0.0
            continue
        # Check neighbors
        adj_sub = adj[candidate]  # [n_cand, N]
        neigh_g2 = adj_sub @ active[:, g2:g2+1].astype(np.float32)
        if hasattr(neigh_g2, "toarray"):
            neigh_g2 = neigh_g2.toarray()
        has_g2_neigh = np.asarray(neigh_g2).ravel() > 0
        boundary_frac = has_g2_neigh.sum() / N  # fraction of ALL patches at boundary
        results[name] = boundary_frac

    return rec.slide_id, rec.patient_id, rec.cancer_type, results


def run_analysis6(slides: List[SlideRecord], cfg: SpatialConfig):
    """Boundary/interface detection for Tier 1 pairs (supplement)."""
    out_dir = cfg.output_dir / "a6_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Use top 5 Tier 1 pairs
    pairs = TIER1_PAIRS[:5]

    args_list = [(rec, cfg, pairs) for rec in slides]
    results = []
    with Pool(cfg.n_workers) as pool:
        for res in tqdm(pool.imap_unordered(_process_slide_boundary, args_list),
                        total=len(args_list), desc="  Boundary detection"):
            if res is not None:
                results.append(res)

    # Build DataFrame
    rows = []
    for sid, pid, ct, boundary_fracs in results:
        row = {"slide_id": sid, "patient_id": pid, "cancer_type": ct}
        row.update(boundary_fracs)
        rows.append(row)

    boundary_df = pd.DataFrame(rows)
    boundary_df.to_csv(out_dir / "boundary_stats.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("A6: BOUNDARY / INTERFACE — Summary")
    print("=" * 60)
    pair_names = [name for _, _, name in pairs]
    for name in pair_names:
        if name in boundary_df.columns:
            vals = boundary_df[name]
            print(f"  {name:<35s}  mean={vals.mean():.4f}  median={vals.median():.4f}")

    (out_dir / "done").touch()
    print("✓ Analysis 6 complete\n")


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def print_master_summary(cfg: SpatialConfig, total_time: float):
    """Print final results summary."""
    print("\n" + "═" * 60)
    print("SPARC SPATIAL TME ANALYSIS — RESULTS SUMMARY")
    print("═" * 60)

    checks = [
        ("A1 Co-activation", cfg.output_dir / "a1_coactivation" / "done"),
        ("A2 Cross-correlation", cfg.output_dir / "a2_cross_correlation" / "done"),
        ("A3 Niches", cfg.output_dir / "a3_niches" / "done"),
        ("A4 Prognostic", cfg.output_dir / "a4_prognostic" / "done"),
        ("A5 Archetypes", cfg.output_dir / "a5_archetypes" / "done"),
        ("A6 Boundary", cfg.output_dir / "a6_boundary" / "done"),
    ]
    for name, path in checks:
        status = "✓" if path.exists() else "✗ (not run)"
        print(f"  {name:<25s} {status}")

    mins = total_time / 60
    if mins > 60:
        print(f"\n  Total time: {mins / 60:.1f}h")
    else:
        print(f"\n  Total time: {mins:.1f}m")
    print(f"  All outputs: {cfg.output_dir}")
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SPARC Spatial TME Architecture Analysis")
    parser.add_argument("--analyses", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Which analyses to run (1-6)")
    parser.add_argument("--output-dir", type=str, default="results/sparc_spatial")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--gep-dir", type=str, default=None,
                        help="Override GEP feature directory")
    parser.add_argument("--coord-dir", type=str, default=None,
                        help="Override coordinate (image feature) directory")
    parser.add_argument("--no-patch-centering", action="store_true",
                        help="Disable per-patch centering (sensitivity analysis)")
    parser.add_argument("--skip-null", action="store_true",
                        help="Skip null model (Analysis 2)")
    parser.add_argument("--skip-brca-validation", action="store_true",
                        help="Skip BRCA validation (Analysis 2)")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--niche-k", type=int, default=None,
                        help="Force specific K for niches")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SpatialConfig(
        output_dir=Path(args.output_dir),
        n_workers=args.n_workers,
        skip_null=args.skip_null,
        skip_brca_validation=args.skip_brca_validation,
        skip_plots=args.skip_plots,
        niche_k_override=args.niche_k,
        seed=args.seed,
    )
    if args.gep_dir:
        cfg.gep_dir = Path(args.gep_dir)
    if args.coord_dir:
        cfg.coord_dir = Path(args.coord_dir)
    if args.no_patch_centering:
        cfg.patch_center = False
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    analyses = set(args.analyses)

    print("=" * 60)
    print("SPARC Spatial TME Architecture Analysis")
    print(f"  Analyses: {sorted(analyses)}")
    print(f"  Workers: {cfg.n_workers}")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 60 + "\n")

    # Build slide registry
    print("Building slide registry...")
    slides = build_slide_registry(cfg)
    print(f"  {len(slides)} slides with matched GEP + coords\n")

    # Phase 1: Shared slide processing (needed for A1, A2, A3)
    cache_path = cfg.output_dir / "slide_cache" / "matrices.npz"
    if any(a in analyses for a in [1, 2, 3, 4]) and not cache_path.exists():
        print("Phase 1: Processing all slides...")
        slide_stats = run_slide_processing(slides, cfg)
        save_slide_cache(slide_stats, cfg.output_dir)
    elif cache_path.exists():
        print("Phase 1: Loading cached slide data...")

    # Load cache
    data, meta = None, None
    if cache_path.exists():
        data, meta = load_slide_cache(cfg.output_dir)

    # Run analyses
    if 1 in analyses and data is not None:
        if not (cfg.output_dir / "a1_coactivation" / "done").exists():
            run_analysis1(data, meta, cfg)
        else:
            print("A1: Skipping (already done)\n")

    if 2 in analyses and data is not None:
        if not (cfg.output_dir / "a2_cross_correlation" / "done").exists():
            run_analysis2(data, meta, slides, cfg)
        else:
            print("A2: Skipping (already done)\n")

    if 3 in analyses and data is not None:
        if not (cfg.output_dir / "a3_niches" / "done").exists():
            run_analysis3(data, meta, slides, cfg)
        else:
            print("A3: Skipping (already done)\n")

    if 4 in analyses and data is not None:
        if not (cfg.output_dir / "a4_prognostic" / "done").exists():
            run_analysis4(data, meta, cfg)
        else:
            print("A4: Skipping (already done)\n")

    if 5 in analyses:
        if not (cfg.output_dir / "a5_archetypes" / "done").exists():
            run_analysis5(data, meta, cfg)
        else:
            print("A5: Skipping (already done)\n")

    if 6 in analyses:
        if not (cfg.output_dir / "a6_boundary" / "done").exists():
            run_analysis6(slides, cfg)
        else:
            print("A6: Skipping (already done)\n")

    print_master_summary(cfg, time.time() - t_start)


if __name__ == "__main__":
    main()
