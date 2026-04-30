#!/usr/bin/env python
"""SPARC Spatial TME Analysis — External Cohorts (SurGen CRC + NLST Lung).

Disease-specific spatial TME case studies complementing the TCGA pan-cancer analysis.
De novo niche discovery per cohort, predefined spatial pairs as anchors,
frozen TCGA niche projection as supplementary.

Usage:
    python scripts/sparc_spatial_external.py --cohort surgen --n-workers 16
    python scripts/sparc_spatial_external.py --cohort nlst --n-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (shared with TCGA script)
# ═══════════════════════════════════════════════════════════════════════════════

PROGRAM_NAMES = [
    "HALLMARK_ANGIOGENESIS", "HALLMARK_APOPTOSIS", "HALLMARK_COAGULATION",
    "HALLMARK_DNA_REPAIR", "HALLMARK_E2F_TARGETS",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_GLYCOLYSIS", "HALLMARK_HYPOXIA",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING", "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_MTORC1_SIGNALING", "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION",
    "REACTOME_COLLAGEN_FORMATION", "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION",
    "REACTOME_INTEGRIN_CELL_SURFACE_INTERACTIONS",
    "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION", "REACTOME_MISMATCH_REPAIR",
    "REACTOME_NEUTROPHIL_DEGRANULATION", "REACTOME_TCR_SIGNALING",
    "REACTOME_TOLL_LIKE_RECEPTOR_CASCADES",
    "ANTIGEN_PRESENTATION_THOMPSON_2020_APM8", "B_CELL_CORE_BUDCZIES_2021",
    "CIN70_CHROMOSOMAL_INSTABILITY",
    "TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018", "TLS_CABRITA_9",
    "T_CELL_INFLAMED_GEP_18_AYERS_2017",
    "GOLDRATH_NAIVE_VS_MEMORY_CD8_TCELL_DN", "GSE13306_TREG_VS_TCONV_UP",
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
    "Inflammatory", "IFN-α", "IFN-γ", "mTORC1", "MYC Targets", "OxPhos",
    "TGF-β Signaling", "TNF-α/NF-κB", "MHC-I Processing", "Collagen",
    "ECM Organization", "Integrin", "MHC-II", "Mismatch Repair", "Neutrophil",
    "TCR Signaling", "TLR Cascades", "Antigen Pres.", "B Cell", "CIN",
    "TGF-β Exclusion", "TLS", "T Cell GEP", "Naïve CD8", "Treg",
    "PD-1⁺ CD8", "M1 Macro", "M2 Macro", "Exhausted CD8", "Immature DCs",
]

_PI = {name: i for i, name in enumerate(PROGRAM_NAMES)}
N_PROGRAMS = 40

TIER1_PAIRS = [
    (_PI["HALLMARK_HYPOXIA"], _PI["HALLMARK_ANGIOGENESIS"], "Hypoxia ↔ Angiogenesis"),
    (_PI["TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018"], _PI["GOLDRATH_NAIVE_VS_MEMORY_CD8_TCELL_DN"], "TGF-β Excl. ↔ Naïve CD8"),
    (_PI["TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018"], _PI["REACTOME_TCR_SIGNALING"], "TGF-β Excl. ↔ TCR"),
    (_PI["REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION"], _PI["HALLMARK_INFLAMMATORY_RESPONSE"], "ECM ↔ Inflammatory"),
    (_PI["REACTOME_COLLAGEN_FORMATION"], _PI["GSE9946_IMMATURE_VS_MATURE_STIMULATORY_DC_DN"], "Collagen ↔ Immature DCs"),
    (_PI["REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION"], _PI["GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP"], "MHC-I ↔ PD-1⁺ CD8"),
    (_PI["TLS_CABRITA_9"], _PI["REACTOME_TCR_SIGNALING"], "TLS ↔ TCR"),
]

TIER2_PAIRS = [
    (_PI["GSE5099_CLASSICAL_M1_VS_ALTERNATIVE_M2_MACROPHAGE_DN"], _PI["REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION"], "M2 ↔ MHC-II"),
    (_PI["HALLMARK_G2M_CHECKPOINT"], _PI["GSE9650_EFFECTOR_VS_EXHAUSTED_CD8_TCELL_UP"], "G2M ↔ Exhausted CD8"),
    (_PI["HALLMARK_OXIDATIVE_PHOSPHORYLATION"], _PI["HALLMARK_GLYCOLYSIS"], "OxPhos ↔ Glycolysis"),
    (_PI["HALLMARK_E2F_TARGETS"], _PI["HALLMARK_INTERFERON_GAMMA_RESPONSE"], "E2F ↔ IFN-γ"),
    (_PI["HALLMARK_MTORC1_SIGNALING"], _PI["GSE13306_TREG_VS_TCONV_UP"], "mTORC1 ↔ Treg"),
]

LUAD_MORPHS = {8140, 8250, 8255, 8260, 8265, 8480, 8490, 8310, 8323}
LUSC_MORPHS = {8070, 8071, 8072, 8052, 8083}


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExternalSpatialConfig:
    cohort: str
    output_dir: Path
    gep_dir: Path
    coord_dir: Path
    clinical_csv: Path
    dicom_csv: Optional[Path] = None
    canc_csv: Optional[Path] = None
    tcga_pca_path: Path = Path("results/sparc_spatial/a3_niches/pca_model.pkl")
    tcga_kmeans_path: Path = Path("results/sparc_spatial/a3_niches/kmeans_models.pkl")
    patch_step: Optional[int] = None
    radius: Optional[float] = None
    subsample_patches: int = 500_000
    max_patches_per_slide: int = 500
    pca_variance_threshold: float = 0.90
    niche_k_range: Tuple[int, ...] = (4, 5, 6, 7, 8)
    kmeans_n_init: int = 50
    niche_k_override: Optional[int] = None
    niche_name_override_json: Optional[Path] = None
    min_events: int = 15
    n_boot: int = 100  # bootstrap resamples for Cox CIs (use 50 for exploratory, 200 for final)
    n_workers: int = 16
    seed: int = 42
    patch_center: bool = True
    skip_plots: bool = False
    skip_tcga_projection: bool = False
    subtype_only: Optional[str] = None


def make_config(cohort: str, output_base: str, **kw) -> ExternalSpatialConfig:
    """Build an ExternalSpatialConfig for one cohort. Paths are env-var driven.

    Override any default by setting the relevant SPARC_* environment variable
    or by passing keyword arguments (which take precedence).
    """
    defaults = {
        "surgen": dict(
            gep_dir=Path(os.environ.get("SPARC_SURGEN_GEP", "features/surgen/predicted_programs_transformer")),
            coord_dir=Path(os.environ.get("SPARC_SURGEN_COORD", "features/surgen/hoptimus1")),
            clinical_csv=Path(os.environ.get("SPARC_SURGEN_CLINICAL", "data/surgen_outcomes/SR386_labels.csv")),
        ),
        "nlst": dict(
            gep_dir=Path(os.environ.get("SPARC_NLST_GEP", "features/nlst/predicted_programs_transformer")),
            coord_dir=Path(os.environ.get("SPARC_NLST_COORD", "features/nlst/hoptimus1")),
            clinical_csv=Path(os.environ.get("SPARC_NLST_CLINICAL", "data/nlst_outcomes/nlst.csv")),
            dicom_csv=Path(os.environ.get("SPARC_NLST_DICOM_CSV", "data/nlst_outcomes/nlst_dicom_file_list.csv")),
            canc_csv=Path(os.environ.get("SPARC_NLST_CANC_CSV", "data/nlst_outcomes/nlst_780_canc_idc_20210527.csv")),
        ),
    }
    d = defaults[cohort]
    d.update(kw)
    return ExternalSpatialConfig(cohort=cohort, output_dir=Path(output_base) / cohort, **d)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Utilities (copied from sparc_spatial_analysis.py for self-containment)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SlideRecord:
    slide_id: str
    patient_id: str
    cancer_type: str
    gep_path: Path
    coord_path: Path


def _load_array(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npz":
        return np.load(path)[key]
    import h5py
    with h5py.File(path, "r") as f:
        return f[key][:]


def _find_feature_file(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".npz", ".h5"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_slide_data(rec: SlideRecord, patch_center: bool = True):
    raw = _load_array(rec.gep_path, "features").astype(np.float32)
    coords = _load_array(rec.coord_path, "coords")
    assert raw.shape[0] == coords.shape[0], f"Patch mismatch: {rec.slide_id}"
    mu = raw.mean(axis=0, keepdims=True)
    sd = raw.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Z = (raw - mu) / sd
    if patch_center:
        Z = Z - Z.mean(axis=1, keepdims=True)
    return Z, raw, coords


def build_radius_adj(coords, radius):
    N = coords.shape[0]
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius)
    rows, cols = [], []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if j != i:
                rows.append(i)
                cols.append(j)
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N))


def row_normalize_sparse(adj):
    from scipy.sparse import diags
    deg = np.array(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return diags(1.0 / deg) @ adj


def fisher_z(rho):
    return np.arctanh(np.clip(rho, -0.9999, 0.9999))


def fisher_z_inv(z):
    return np.tanh(z)


def vectorized_spearman_cross(A, B):
    N = A.shape[0]
    rA = np.apply_along_axis(rankdata, 0, A)
    rB = np.apply_along_axis(rankdata, 0, B)
    sA = rA.std(0, ddof=0); sA[sA < 1e-8] = 1.0
    sB = rB.std(0, ddof=0); sB[sB < 1e-8] = 1.0
    rA = (rA - rA.mean(0)) / sA
    rB = (rB - rB.mean(0)) / sB
    return (rA.T @ rB) / N


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Detect Patch Step
# ═══════════════════════════════════════════════════════════════════════════════

def auto_detect_patch_step(cfg: ExternalSpatialConfig):
    """Auto-detect from median of min nonzero step across 5 slides."""
    files = [f for f in cfg.coord_dir.iterdir() if f.suffix in (".h5", ".npz")][:10]
    steps = []
    for f in files[:5]:
        try:
            coords = _load_array(f, "coords")
            for axis in [0, 1]:
                u = np.sort(np.unique(coords[:, axis]))
                d = np.diff(u)
                d = d[d > 0]
                if len(d) > 0:
                    steps.append(d.min())
        except Exception:
            continue
    if not steps:
        raise RuntimeError(f"Cannot detect patch step from {cfg.coord_dir}")
    step = int(np.round(np.median(steps)))
    cfg.patch_step = step
    cfg.radius = step * 1.4142135623730951 + 1
    print(f"  Auto-detected: patch_step={step}, radius={cfg.radius:.1f} (from {len(steps)} measurements)")


# ═══════════════════════════════════════════════════════════════════════════════
# Cohort-Specific Registry Builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_surgen_registry(cfg: ExternalSpatialConfig):
    FIVE_YR = 5 * 365.25
    clin = pd.read_csv(cfg.clinical_csv)
    gep_stems = {p.stem for p in cfg.gep_dir.iterdir() if p.suffix in (".h5", ".npz")}
    coord_stems = {p.stem for p in cfg.coord_dir.iterdir() if p.suffix in (".h5", ".npz")}
    available = gep_stems & coord_stems

    slides, clin_rows = [], []
    for _, r in clin.iterrows():
        if pd.isna(r.get("died_within_5_years")):
            continue
        cid = int(r["case_id"])
        stem = f"SR386_40X_HE_T{cid:03d}_01"
        if stem not in available:
            continue
        died = int(r["died_within_5_years"])
        days = pd.to_numeric(r["days_till_death"], errors="coerce")
        T_os = days if (died and not np.isnan(days)) else FIVE_YR
        E_os = float(died)
        crc_cause = r.get("crc_primary_cause_of_death", 0) == 1
        E_dss = 1.0 if (died and crc_cause) else 0.0
        site = str(r.get("site_of_tumour_grouping", "")).lower()
        ct = "READ" if "rect" in site else "COAD"
        pid = f"SR386_{cid:03d}"

        slides.append(SlideRecord(
            slide_id=stem, patient_id=pid, cancer_type=ct,
            gep_path=_find_feature_file(cfg.gep_dir, stem),
            coord_path=_find_feature_file(cfg.coord_dir, stem),
        ))
        clin_rows.append({
            "patient_id": pid, "cancer_type": ct,
            "os_time": T_os, "os_event": E_os,
            "dss_time": T_os, "dss_event": E_dss,
            "stage": r.get("stage", np.nan),
            "mmr_status": r.get("mmr_loss_binary", np.nan),
            "braf_status": r.get("braf_mutant_status", np.nan),
        })

    clinical_df = pd.DataFrame(clin_rows)
    print(f"  SurGen: {len(slides)} slides, {len(clinical_df)} patients "
          f"(COAD={sum(clinical_df.cancer_type=='COAD')}, READ={sum(clinical_df.cancer_type=='READ')})")
    return slides, clinical_df


def build_nlst_registry(cfg: ExternalSpatialConfig):
    dicom = pd.read_csv(cfg.dicom_csv)
    dicom["uuid"] = dicom["file_name"].str.replace(".dcm", "", regex=False)
    dicom["pid"] = dicom["directory"].str.extract(r"/nlst/(\d+)/").astype(int)
    uuid_to_pid = dict(zip(dicom["uuid"], dicom["pid"]))

    nlst = pd.read_csv(cfg.clinical_csv, low_memory=False)
    canc = pd.read_csv(cfg.canc_csv)[["pid", "lc_morph"]].drop_duplicates("pid")

    gep_stems = {p.stem for p in cfg.gep_dir.iterdir() if p.suffix in (".h5", ".npz")}
    coord_stems = {p.stem for p in cfg.coord_dir.iterdir() if p.suffix in (".h5", ".npz")}
    available = gep_stems & coord_stems

    # Map uuid -> pid for available slides
    slide_pids = {}
    for stem in available:
        pid = uuid_to_pid.get(stem)
        if pid is not None:
            slide_pids[stem] = int(pid)

    # Clinical
    nl = nlst[nlst["pid"].isin(set(slide_pids.values()))].copy()
    nl["os_time"] = nl.apply(
        lambda r: r["death_days"] - r["candx_days"]
        if pd.notna(r["death_days"])
        else r["fup_days"] - r["candx_days"],
        axis=1,
    )
    nl["os_event"] = (nl["deathstat"] == 1).astype(int)
    nl["dss_time"] = nl["os_time"]
    nl["dss_event"] = nl["finaldeathlc"].fillna(0).astype(int)
    nl = nl[(nl["deathstat"] != 2) & (nl["progressed_ever"] != 9)]
    nl = nl[(nl["os_time"] > 0) & nl["os_time"].notna()]
    nl = nl.merge(canc, on="pid", how="left")

    def get_histo(m):
        if pd.isna(m): return "Other"
        m = int(m)
        return "LUAD" if m in LUAD_MORPHS else ("LUSC" if m in LUSC_MORPHS else "Other")
    nl["histo"] = nl["lc_morph"].apply(get_histo)

    valid_pids = set(nl["pid"].astype(int))

    slides, clin_rows = [], []
    pid_seen = set()
    for stem, pid in slide_pids.items():
        if pid not in valid_pids:
            continue
        row = nl[nl["pid"] == pid].iloc[0]
        ct = row["histo"]
        if cfg.subtype_only and ct != cfg.subtype_only:
            continue
        slides.append(SlideRecord(
            slide_id=stem, patient_id=str(pid), cancer_type=ct,
            gep_path=_find_feature_file(cfg.gep_dir, stem),
            coord_path=_find_feature_file(cfg.coord_dir, stem),
        ))
        if pid not in pid_seen:
            pid_seen.add(pid)
            clin_rows.append({
                "patient_id": str(pid), "cancer_type": ct,
                "os_time": row["os_time"], "os_event": float(row["os_event"]),
                "dss_time": row["dss_time"], "dss_event": float(row["dss_event"]),
            })

    clinical_df = pd.DataFrame(clin_rows)
    n_luad = sum(clinical_df.cancer_type == "LUAD")
    n_lusc = sum(clinical_df.cancer_type == "LUSC")
    n_other = len(clinical_df) - n_luad - n_lusc

    # Defensive checks
    n_available = len(available)
    n_mapped = len(slide_pids)
    n_survived = len(valid_pids)
    print(f"  NLST pipeline:")
    print(f"    Available slide files: {n_available}")
    print(f"    Mapped to patient IDs: {n_mapped}")
    print(f"    Survived clinical filters: {n_survived}")
    print(f"    Final: {len(slides)} slides, {len(clinical_df)} patients "
          f"(LUAD={n_luad}, LUSC={n_lusc}, Other={n_other})")
    if n_mapped < n_available * 0.8:
        print(f"    ⚠ WARNING: {n_available - n_mapped} slides could not be mapped to patient IDs")
    if n_survived < n_mapped * 0.5:
        print(f"    ⚠ WARNING: {n_mapped - n_survived} patients dropped by clinical filters")

    return slides, clinical_df


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Slide Processing (same as TCGA)
# ═══════════════════════════════════════════════════════════════════════════════

def process_slide(rec: SlideRecord, cfg: ExternalSpatialConfig) -> Optional[Dict]:
    try:
        Z, raw, coords = load_slide_data(rec, patch_center=cfg.patch_center)
    except Exception:
        return None
    N = Z.shape[0]
    if N < 10:
        return None

    adj = build_radius_adj(coords, cfg.radius)
    adj_norm = row_normalize_sparse(adj)

    coact_rho, _ = spearmanr(Z, axis=0)
    if coact_rho.ndim == 0:
        return None
    coact_fz = fisher_z(coact_rho)

    N_mean = adj_norm @ Z
    S = vectorized_spearman_cross(Z, N_mean)
    S_fz = fisher_z(S)

    D = np.zeros((N_PROGRAMS, N_PROGRAMS), dtype=np.float32)
    global_mean = Z.mean(axis=0)
    for g1 in range(N_PROGRAMS):
        threshold = np.percentile(Z[:, g1], 75)
        mask = Z[:, g1] >= threshold
        if mask.sum() > 0:
            D[g1, :] = N_mean[mask].mean(axis=0) - global_mean

    active = (Z > 1.0).astype(np.float32)
    O = np.zeros((N_PROGRAMS, N_PROGRAMS), dtype=np.float32)
    for g1 in range(N_PROGRAMS):
        active_mask = active[:, g1].astype(bool)
        if active_mask.sum() == 0:
            continue
        adj_sub = adj[active_mask]
        for g2 in range(N_PROGRAMS):
            neigh_g2 = adj_sub @ active[:, g2:g2+1].astype(np.float32)
            if hasattr(neigh_g2, "toarray"):
                neigh_g2 = neigh_g2.toarray()
            O[g1, g2] = (np.asarray(neigh_g2).ravel() > 0).mean()

    n_sub = min(cfg.max_patches_per_slide, N)
    rng = np.random.RandomState(hash(rec.slide_id) % (2**31))
    sub_idx = rng.choice(N, size=n_sub, replace=False)

    return {
        "slide_id": rec.slide_id, "patient_id": rec.patient_id,
        "cancer_type": rec.cancer_type, "n_patches": N,
        "coact_fz": coact_fz, "S_fz": S_fz, "D": D, "O": O,
        "Z_sub": Z[sub_idx], "gep_means": raw.mean(axis=0),
    }


def run_slide_processing(slides, cfg):
    fn = partial(process_slide, cfg=cfg)
    results = []
    with Pool(cfg.n_workers) as pool:
        for res in tqdm(pool.imap_unordered(fn, slides), total=len(slides),
                        desc="  Processing slides"):
            if res is not None:
                results.append(res)
    return results


def save_slide_cache(stats, output_dir):
    cache_dir = output_dir / "slide_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sids = [s["slide_id"] for s in stats]
    pids = [s["patient_id"] for s in stats]
    cts = [s["cancer_type"] for s in stats]
    n_patches = np.array([s["n_patches"] for s in stats])
    np.savez_compressed(
        cache_dir / "matrices.npz",
        coact_fz=np.stack([s["coact_fz"] for s in stats]),
        S_fz=np.stack([s["S_fz"] for s in stats]),
        D=np.stack([s["D"] for s in stats]),
        O=np.stack([s["O"] for s in stats]),
        gep_means=np.stack([s["gep_means"] for s in stats]),
        n_patches=n_patches,
        Z_sub=np.concatenate([s["Z_sub"] for s in stats], axis=0),
    )
    pd.DataFrame({"slide_id": sids, "patient_id": pids,
                   "cancer_type": cts, "n_patches": n_patches}).to_csv(
        cache_dir / "slide_meta.csv", index=False)
    print(f"  ✓ Cache: {len(stats)} slides, "
          f"{sum(s['Z_sub'].shape[0] for s in stats)} subsample patches")


def load_slide_cache(output_dir):
    cache_dir = output_dir / "slide_cache"
    data = dict(np.load(cache_dir / "matrices.npz"))
    meta = pd.read_csv(cache_dir / "slide_meta.csv")
    return data, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Niche Discovery + Auto-Naming
# ═══════════════════════════════════════════════════════════════════════════════

def auto_name_niches(centroids_z: np.ndarray) -> List[str]:
    """Rule-based niche naming from centroid profiles."""
    K = centroids_z.shape[0]
    names = []
    keywords = [
        ({"B Cell", "TLS"}, "Lymphoid"),
        ({"CIN", "E2F Targets", "G2M Checkpoint", "MYC Targets"}, "Proliferative"),
        ({"ECM Organization", "Collagen", "Integrin"}, "Stromal"),
        ({"T Cell GEP", "IFN-γ", "TCR Signaling"}, "Immune-Inflamed"),
        ({"Hypoxia", "Angiogenesis"}, "Hypoxic-Angiogenic"),
        ({"TGF-β Exclusion"}, "Immune-Excluded"),
        ({"OxPhos", "Glycolysis", "mTORC1"}, "Metabolic"),
        ({"EMT", "Angiogenesis"}, "Stromal-Vascular"),
        ({"Immature DCs", "Mismatch Repair"}, "Myeloid-Repair"),
    ]
    for k in range(K):
        top5_idx = np.argsort(-centroids_z[k])[:5]
        top5_names = {SHORT_NAMES[i] for i in top5_idx}
        assigned = False
        for kw_set, label in keywords:
            if kw_set & top5_names:
                names.append(label)
                assigned = True
                break
        if not assigned:
            names.append(f"Niche-{k} ({SHORT_NAMES[top5_idx[0]]})")

    # Deduplicate
    counts = {}
    for i, n in enumerate(names):
        if names.count(n) > 1:
            counts[n] = counts.get(n, 0) + 1
            top = SHORT_NAMES[np.argsort(-centroids_z[i])[0]]
            names[i] = f"{n} ({top})"
    return names


_NICHE_CACHE = {}

def _niche_assign_init(pca_components, pca_mean, km_centroids, best_K, patch_center):
    global _NICHE_CACHE
    _NICHE_CACHE.update({
        "pca_components": pca_components, "pca_mean": pca_mean,
        "km_centroids": km_centroids, "best_K": best_K,
        "patch_center": patch_center,
    })

def _niche_assign_worker(rec):
    try:
        Z, _, _ = load_slide_data(rec, patch_center=_NICHE_CACHE["patch_center"])
        Z_pca = (Z - _NICHE_CACHE["pca_mean"]) @ _NICHE_CACHE["pca_components"].T
        from scipy.spatial.distance import cdist
        labels = cdist(Z_pca, _NICHE_CACHE["km_centroids"]).argmin(axis=1)
        counts = np.bincount(labels, minlength=_NICHE_CACHE["best_K"])
        return rec.slide_id, rec.patient_id, rec.cancer_type, counts
    except Exception:
        return None


def discover_niches(data, meta, slides, cfg):
    out_dir = cfg.output_dir / "niches"
    out_dir.mkdir(parents=True, exist_ok=True)

    Z_sub = data["Z_sub"]
    # Global subsample cap (slide-level cap may still produce more than intended)
    if len(Z_sub) > cfg.subsample_patches:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(len(Z_sub), cfg.subsample_patches, replace=False)
        Z_sub = Z_sub[idx]
    print(f"\n  Niche discovery: {Z_sub.shape[0]} patches")

    pca = PCA(n_components=cfg.pca_variance_threshold, svd_solver="full",
              random_state=cfg.seed)
    X_pca = pca.fit_transform(Z_sub)
    print(f"  PCA: {pca.n_components_} components, {pca.explained_variance_ratio_.sum():.1%} variance")

    km_results = {}
    for K in tqdm(cfg.niche_k_range, desc="  K-Means"):
        km = KMeans(n_clusters=K, n_init=cfg.kmeans_n_init, random_state=cfg.seed, max_iter=300)
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels, sample_size=min(50000, len(labels)),
                               random_state=cfg.seed)
        km_results[K] = {"model": km, "silhouette": float(sil)}

    best_K = cfg.niche_k_override or max(km_results, key=lambda k: km_results[k]["silhouette"])
    best_km = km_results[best_K]["model"]
    centroids_z = pca.inverse_transform(best_km.cluster_centers_)

    print(f"\n  Silhouette scores:")
    for K in sorted(km_results):
        m = " ← selected" if K == best_K else ""
        print(f"    K={K}: {km_results[K]['silhouette']:.4f}{m}")

    # Auto-name
    niche_names = auto_name_niches(centroids_z)
    if cfg.niche_name_override_json and cfg.niche_name_override_json.exists():
        with open(cfg.niche_name_override_json) as f:
            niche_names = json.load(f)

    print(f"\n  Niche profiles (K={best_K}):")
    for k in range(best_K):
        top3 = np.argsort(-centroids_z[k])[:3]
        progs = ", ".join(f"{SHORT_NAMES[i]}({centroids_z[k,i]:+.2f})" for i in top3)
        print(f"    {niche_names[k]}: {progs}")

    # Save
    with open(out_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(out_dir / "kmeans_models.pkl", "wb") as f:
        pickle.dump(km_results, f)
    json.dump({"best_K": int(best_K), "silhouette": float(km_results[best_K]["silhouette"])},
              open(out_dir / "best_k.json", "w"))
    np.savez(out_dir / "niche_centroids.npz", centroids_z=centroids_z,
             short_names=SHORT_NAMES, best_K=best_K)
    json.dump(niche_names, open(out_dir / "niche_names.json", "w"))

    # Assign all patches
    print("\n  Assigning all patches...")
    slide_lookup = {s.slide_id: s for s in slides}
    valid_recs = [slide_lookup[sid] for sid in meta["slide_id"] if sid in slide_lookup]

    with Pool(cfg.n_workers, initializer=_niche_assign_init,
              initargs=(pca.components_.astype(np.float32), pca.mean_.astype(np.float32),
                        best_km.cluster_centers_.astype(np.float32), best_K, cfg.patch_center)) as pool:
        assign_results = []
        for res in tqdm(pool.imap_unordered(_niche_assign_worker, valid_recs),
                        total=len(valid_recs), desc="  Niche assignment"):
            if res is not None:
                assign_results.append(res)

    # Patient-level composition (aggregate multi-slide)
    patient_data = {}
    for sid, pid, ct, counts in assign_results:
        if pid not in patient_data:
            patient_data[pid] = {"ct": ct, "counts": np.zeros(best_K), "total": 0}
        patient_data[pid]["counts"] += counts
        patient_data[pid]["total"] += counts.sum()

    rows = []
    for pid, d in patient_data.items():
        fracs = d["counts"] / max(d["total"], 1)
        entropy = -np.sum(fracs[fracs > 0] * np.log(fracs[fracs > 0]))
        row = {"patient_id": pid, "cancer_type": d["ct"], "entropy": entropy}
        for k in range(best_K):
            row[niche_names[k]] = fracs[k]
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "patient_composition.csv", index=False)

    # KMeans stability check (fixed PCA, refit KMeans on random half)
    print("\n  KMeans stability check...")
    rng = np.random.RandomState(123)
    idx = rng.permutation(len(Z_sub))
    half = pca.transform(Z_sub[idx[:len(idx)//2]])
    km2 = KMeans(n_clusters=best_K, n_init=20, random_state=99).fit(half)
    c2 = pca.inverse_transform(km2.cluster_centers_)
    from scipy.spatial.distance import cdist as _cdist
    from scipy.optimize import linear_sum_assignment
    sim = 1 - _cdist(centroids_z, c2, metric="cosine")
    ri, ci = linear_sum_assignment(-sim)
    matched = sim[ri, ci]
    print(f"  Stability: cosine sim = [{', '.join(f'{s:.3f}' for s in sorted(matched, reverse=True))}] "
          f"mean={matched.mean():.3f}")

    print(f"  ✓ Niches: K={best_K}, {len(comp_df)} patients")
    return best_K, centroids_z, niche_names, comp_df, pca, best_km


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial Pair Features
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spatial_pair_features(data, meta, clinical_df, cfg):
    S_fz = data["S_fz"]
    gep_means = data["gep_means"]
    pids_slide = meta["patient_id"].astype(str).values

    patient_S = {}
    patient_gep = {}
    for i in range(len(pids_slide)):
        pid = pids_slide[i]
        patient_S.setdefault(pid, []).append(S_fz[i])
        patient_gep.setdefault(pid, []).append(gep_means[i])

    rows = []
    all_pairs = TIER1_PAIRS + TIER2_PAIRS
    for pid in clinical_df["patient_id"].astype(str):
        if pid not in patient_S:
            continue
        S_mean = fisher_z_inv(np.stack(patient_S[pid]).mean(0))
        gep_mean = np.stack(patient_gep[pid]).mean(0)
        row = {"patient_id": pid}
        for g1, g2, name in all_pairs:
            row[f"S_{name}"] = S_mean[g1, g2]
            row[f"abund_{name}_g1"] = gep_mean[g1]
            row[f"abund_{name}_g2"] = gep_mean[g2]
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_dir = cfg.output_dir / "spatial_features"
    feat_dir.mkdir(exist_ok=True)
    feat_df.to_csv(feat_dir / "patient_features.csv", index=False)
    print(f"  ✓ Spatial features: {len(feat_df)} patients, {len(all_pairs)} pairs")
    return feat_df


# ═══════════════════════════════════════════════════════════════════════════════
# Survival Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_cox(X, y, n_boot=100, coef_idx=0):
    """Fit Cox PH, report HR/CI/p for coef_idx-th covariate.

    For univariate models: coef_idx=0 (default).
    For multivariable: coef_idx=0 reports the first (spatial) feature,
    adjusted for all remaining covariates.

    Returns dict with hr, hr_lo, hr_hi, p_value, c_index, all_hrs (list),
    converged.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from scipy.stats import norm
    try:
        cox = CoxPHSurvivalAnalysis(alpha=1e-3, n_iter=200)
        cox.fit(X, y)
        ci = concordance_index_censored(y["event"], y["time"], cox.predict(X))[0]
        coef = cox.coef_[coef_idx]
        hr = np.exp(coef)
        all_hrs = np.exp(cox.coef_).tolist()

        # Bootstrap SE for the target coefficient
        rng = np.random.RandomState(42)
        boot_coefs = []
        for _ in range(n_boot):
            idx = rng.choice(len(X), len(X), replace=True)
            try:
                bm = CoxPHSurvivalAnalysis(alpha=1e-3, n_iter=200)
                bm.fit(X[idx], y[idx])
                boot_coefs.append(bm.coef_[coef_idx])
            except Exception:
                pass
        if len(boot_coefs) > 20:
            se = np.std(boot_coefs)
            z = coef / max(se, 1e-10)
            p = 2 * norm.sf(abs(z))
            hr_lo = np.exp(np.percentile(boot_coefs, 2.5))
            hr_hi = np.exp(np.percentile(boot_coefs, 97.5))
        else:
            p, hr_lo, hr_hi = np.nan, np.nan, np.nan

        return {"hr": hr, "hr_lo": hr_lo, "hr_hi": hr_hi, "p_value": p,
                "c_index": ci, "all_hrs": all_hrs, "converged": True}
    except Exception:
        return {"hr": np.nan, "hr_lo": np.nan, "hr_hi": np.nan, "p_value": np.nan,
                "c_index": np.nan, "all_hrs": [], "converged": False}


def _fit_cox_multivar_full(X, y, feature_names, n_boot=100):
    """Fit multivariable Cox, return HR/p for ALL covariates."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from scipy.stats import norm
    try:
        cox = CoxPHSurvivalAnalysis(alpha=1e-3, n_iter=200)
        cox.fit(X, y)
        ci = concordance_index_censored(y["event"], y["time"], cox.predict(X))[0]

        rng = np.random.RandomState(42)
        boot_coefs = []
        for _ in range(n_boot):
            idx = rng.choice(len(X), len(X), replace=True)
            try:
                bm = CoxPHSurvivalAnalysis(alpha=1e-3, n_iter=200)
                bm.fit(X[idx], y[idx])
                boot_coefs.append(bm.coef_)
            except Exception:
                pass

        results = {"c_index": ci, "converged": True, "features": {}}
        for j, name in enumerate(feature_names):
            hr = np.exp(cox.coef_[j])
            if len(boot_coefs) > 20:
                bc = [c[j] for c in boot_coefs]
                se = np.std(bc)
                z = cox.coef_[j] / max(se, 1e-10)
                p = 2 * norm.sf(abs(z))
                hr_lo = np.exp(np.percentile(bc, 2.5))
                hr_hi = np.exp(np.percentile(bc, 97.5))
            else:
                p, hr_lo, hr_hi = np.nan, np.nan, np.nan
            results["features"][name] = {"hr": hr, "hr_lo": hr_lo, "hr_hi": hr_hi, "p_value": p}
        return results
    except Exception:
        return {"c_index": np.nan, "converged": False, "features": {}}


def _logrank_p(time, event, groups):
    from scipy.stats import chi2
    unique_g = sorted(set(groups))
    if len(unique_g) < 2: return 1.0
    all_t = np.unique(time[event == 1])
    OE = np.zeros(len(unique_g) - 1)
    V = np.zeros(len(unique_g) - 1)
    for t in all_t:
        ar = {gi: (time[groups == g] >= t).sum() for gi, g in enumerate(unique_g)}
        de = {gi: ((time[groups == g] == t) & (event[groups == g] == 1)).sum() for gi, g in enumerate(unique_g)}
        n, d = sum(ar.values()), sum(de.values())
        if n < 2 or d == 0: continue
        for gi in range(len(unique_g) - 1):
            e_gi = ar[gi] * d / n
            OE[gi] += de[gi] - e_gi
            V[gi] += e_gi * (1 - ar[gi]/n) * (n-d) / (n-1)
    chi2_val = np.sum(OE**2 / np.maximum(V, 1e-10))
    return 1 - chi2.cdf(chi2_val, df=len(unique_g) - 1)


def run_survival(comp_df, feat_df, clinical_df, niche_names, cfg):
    from sklearn.preprocessing import StandardScaler
    surv_dir = cfg.output_dir / "survival"
    surv_dir.mkdir(exist_ok=True)

    # Ensure consistent patient_id types (string) across all dataframes
    clinical_df = clinical_df.copy()
    clinical_df["patient_id"] = clinical_df["patient_id"].astype(str)
    comp_df = comp_df.copy()
    comp_df["patient_id"] = comp_df["patient_id"].astype(str)
    feat_df = feat_df.copy()
    feat_df["patient_id"] = feat_df["patient_id"].astype(str)

    merged = clinical_df.merge(comp_df, on="patient_id", suffixes=("", "_comp"))
    merged = merged.merge(feat_df, on="patient_id", how="left")

    all_results = []
    subtypes = sorted(merged["cancer_type"].unique()) + ["All"]

    for ct in subtypes:
        sub = merged if ct == "All" else merged[merged["cancer_type"] == ct]
        for ep_name, t_col, e_col in [("DSS", "dss_time", "dss_event"), ("OS", "os_time", "os_event")]:
            T = sub[t_col].values.astype(float)
            E = sub[e_col].values.astype(float)
            valid = (T > 0) & ~np.isnan(T) & ~np.isnan(E)
            T, E = T[valid], E[valid]
            if E.sum() < cfg.min_events:
                continue
            y = np.array([(bool(e), t) for e, t in zip(E, T)],
                         dtype=[("event", bool), ("time", float)])
            sub_v = sub[valid].reset_index(drop=True)
            n_ev = int(E.sum())
            warn = " ⚠" if n_ev < 30 else ""

            # Niche fractions + entropy
            niche_cols = [n for n in niche_names if n in sub_v.columns]
            for col in niche_cols + ["entropy"]:
                if col not in sub_v.columns: continue
                x = sub_v[col].values.reshape(-1, 1)
                x = StandardScaler().fit_transform(x)
                res = _fit_cox(x, y, n_boot=cfg.n_boot)
                all_results.append({
                    "subtype": ct, "endpoint": ep_name, "feature": col, "feature_type": "niche",
                    "n": len(T), "events": n_ev, **res,
                })

            # Multivariate niche model (all fractions + entropy)
            niche_feat_cols = [c for c in niche_cols if c in sub_v.columns]
            if len(niche_feat_cols) >= 2 and "entropy" in sub_v.columns:
                mv_features = niche_feat_cols[:-1] + ["entropy"]  # drop last frac (collinearity)
                X_mv = StandardScaler().fit_transform(sub_v[mv_features].values)
                mv_res = _fit_cox_multivar_full(X_mv, y, mv_features, n_boot=cfg.n_boot)
                all_results.append({
                    "subtype": ct, "endpoint": ep_name, "feature": "niche_multivariate",
                    "feature_type": "niche_mv", "n": len(T), "events": n_ev,
                    "c_index": mv_res["c_index"], "converged": mv_res["converged"],
                    "hr": np.nan, "p_value": np.nan,  # individual HRs in mv_res["features"]
                })
                # Also log each niche's adjusted HR from the multivariate model
                for feat_name, feat_res in mv_res.get("features", {}).items():
                    all_results.append({
                        "subtype": ct, "endpoint": ep_name,
                        "feature": f"niche_mv:{feat_name}", "feature_type": "niche_mv_coef",
                        "n": len(T), "events": n_ev, **feat_res,
                    })

            # Spatial pairs
            for g1, g2, name in TIER1_PAIRS + TIER2_PAIRS:
                s_col = f"S_{name}"
                if s_col not in sub_v.columns: continue
                tier = "Tier1" if (g1, g2, name) in TIER1_PAIRS else "Tier2"
                x_s = StandardScaler().fit_transform(sub_v[s_col].values.reshape(-1, 1))
                # M1: univariate
                m1 = _fit_cox(x_s, y, n_boot=cfg.n_boot)
                # M2: abundance-adjusted
                a1 = sub_v[f"abund_{name}_g1"].values.reshape(-1, 1)
                a2 = sub_v[f"abund_{name}_g2"].values.reshape(-1, 1)
                X_m2 = np.hstack([x_s, StandardScaler().fit_transform(a1),
                                  StandardScaler().fit_transform(a2)])
                m2 = _fit_cox(X_m2, y, n_boot=cfg.n_boot)

                # M3: clinical-adjusted (stage + molecular where available)
                clin_covs = []
                if cfg.cohort == "surgen":
                    for cc in ["stage", "mmr_status", "braf_status"]:
                        if cc in sub_v.columns:
                            vals = pd.to_numeric(sub_v[cc], errors="coerce").values
                            if np.isfinite(vals).sum() > len(vals) * 0.5:
                                clin_covs.append(np.nan_to_num(vals, nan=np.nanmedian(vals[np.isfinite(vals)])))
                elif cfg.cohort == "nlst":
                    # NLST: try de_stag_7thed or similar stage column
                    for cc in ["de_stag_7thed", "stage"]:
                        if cc in sub_v.columns:
                            vals = pd.to_numeric(sub_v[cc], errors="coerce").values
                            if np.isfinite(vals).sum() > len(vals) * 0.3:
                                clin_covs.append(np.nan_to_num(vals, nan=np.nanmedian(vals[np.isfinite(vals)])))
                                break
                m3_hr = np.nan
                if clin_covs:
                    X_m3 = np.hstack([x_s] + [StandardScaler().fit_transform(c.reshape(-1,1)) for c in clin_covs])
                    m3 = _fit_cox(X_m3, y, n_boot=min(cfg.n_boot, 50))
                    m3_hr = m3["hr"]

                all_results.append({
                    "subtype": ct, "endpoint": ep_name, "feature": name,
                    "feature_type": tier, "n": len(T), "events": n_ev,
                    "hr": m1["hr"], "hr_lo": m1.get("hr_lo"), "hr_hi": m1.get("hr_hi"),
                    "p_value": m1.get("p_value"), "c_index": m1["c_index"],
                    "hr_adj": m2["hr"], "c_index_adj": m2["c_index"],
                    "hr_clin_adj": m3_hr,
                    "converged": m1["converged"],
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(surv_dir / "all_results.csv", index=False)

    # Print summary
    print(f"\n  {'='*70}")
    print(f"  SURVIVAL SUMMARY ({cfg.cohort.upper()})")
    print(f"  {'='*70}")
    for ct in subtypes:
        ct_df = results_df[results_df["subtype"] == ct]
        if len(ct_df) == 0: continue
        print(f"\n  --- {ct} ---")
        for ep in ["DSS", "OS"]:
            ep_df = ct_df[ct_df["endpoint"] == ep]
            if len(ep_df) == 0: continue
            print(f"  {ep} (n={ep_df.iloc[0]['n']}, events={ep_df.iloc[0]['events']}):")
            # Top features by |HR - 1|
            ep_df_valid = ep_df[ep_df["hr"].notna()].copy()
            ep_df_valid["_abs_hr"] = (ep_df_valid["hr"] - 1).abs()
            for _, r in ep_df_valid.nlargest(5, "_abs_hr").iterrows():
                p_str = f"p={r['p_value']:.3f}" if "p_value" in r and not np.isnan(r.get("p_value", np.nan)) else ""
                hr = r.get("hr_adj", r["hr"])
                if np.isnan(hr): hr = r["hr"]
                marker = "***" if (hr > 1.2 or hr < 0.8) else ""
                print(f"    {r['feature']:<35s}  HR={hr:.2f} {p_str} {marker}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# TCGA Projection (Supplementary)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tcga_projection(data, meta, slides, cfg, cohort_centroids, cohort_niche_names):
    if cfg.skip_tcga_projection:
        return
    if not cfg.tcga_pca_path.exists() or not cfg.tcga_kmeans_path.exists():
        print("  ⚠ TCGA models not found, skipping projection")
        return

    print("\n  TCGA frozen niche projection...")
    with open(cfg.tcga_pca_path, "rb") as f:
        tcga_pca = pickle.load(f)
    with open(cfg.tcga_kmeans_path, "rb") as f:
        tcga_km = pickle.load(f)[4]["model"]  # K=4

    tcga_centroids = tcga_pca.inverse_transform(tcga_km.cluster_centers_)
    tcga_names = ["Proliferative", "Stromal", "Lymphoid", "Immune-Inflamed"]

    # Compare centroids
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    sim = 1 - cdist(cohort_centroids, tcga_centroids, metric="cosine")
    ri, ci = linear_sum_assignment(-sim)
    matched = sim[ri, ci]
    print(f"  Cohort→TCGA centroid matching:")
    for i, j in zip(ri, ci):
        print(f"    {cohort_niche_names[i]:25s} → {tcga_names[j]:25s}  sim={sim[i,j]:.3f}")
    print(f"  Mean cosine similarity: {matched.mean():.3f}")

    out_dir = cfg.output_dir / "niches"
    np.savez(out_dir / "tcga_comparison.npz",
             cohort_centroids=cohort_centroids, tcga_centroids=tcga_centroids,
             similarity=sim, matched_pairs=list(zip(ri.tolist(), ci.tolist())),
             cohort_names=cohort_niche_names, tcga_names=tcga_names)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figures(centroids_z, niche_names, comp_df, survival_df, slides, cfg,
                     pca, best_km, best_K):
    if cfg.skip_plots:
        return
    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Niche centroids
    fig, ax = plt.subplots(figsize=(14, max(3, best_K * 0.8)))
    sns.heatmap(centroids_z, xticklabels=SHORT_NAMES, yticklabels=niche_names,
                cmap="RdBu_r", center=0, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.6, "label": "Mean z-score"})
    ax.set_title(f"{cfg.cohort.upper()} — TME Niche Profiles (K={best_K})", fontsize=14)
    plt.xticks(fontsize=6, rotation=90)
    plt.tight_layout()
    plt.savefig(fig_dir / "niche_centroids.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Niche composition by subtype
    niche_cols = [n for n in niche_names if n in comp_df.columns]
    ct_comp = comp_df.groupby("cancer_type")[niche_cols].mean()
    fig, ax = plt.subplots(figsize=(max(6, len(ct_comp) * 2), 5))
    ct_comp.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_ylabel("Mean niche fraction")
    ax.set_title(f"{cfg.cohort.upper()} — Niche Composition")
    ax.legend(title="Niche", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "niche_composition.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. WSI niche maps (3-4 representative slides)
    slide_lookup_by_pid = {}
    for s in slides:
        slide_lookup_by_pid.setdefault(s.patient_id, []).append(s)
    # Pick top patient per niche by fraction, then grab their first slide
    examples = {}
    for nc in niche_cols:
        if nc not in comp_df.columns:
            continue
        top_pid = comp_df.nlargest(1, nc).iloc[0]["patient_id"]
        if top_pid in slide_lookup_by_pid and nc not in examples:
            examples[nc] = slide_lookup_by_pid[top_pid][0]
        if len(examples) >= min(4, best_K):
            break

    if examples:
        niche_colors = plt.cm.tab10(np.linspace(0, 1, best_K))
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        cmap = ListedColormap(niche_colors[:best_K])

        n_ex = min(4, len(examples))
        fig, axes = plt.subplots(1, n_ex, figsize=(6 * n_ex, 6))
        if n_ex == 1: axes = [axes]

        for ax, (nc, rec) in zip(axes, list(examples.items())[:n_ex]):
            try:
                Z, _, coords = load_slide_data(rec, patch_center=cfg.patch_center)
                Z_pca = (Z - pca.mean_) @ pca.components_.T
                labels = best_km.predict(Z_pca)
                x, y_c = coords[:, 0], coords[:, 1]
                if len(x) > 40000:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(x), 40000, replace=False)
                    x, y_c, labels = x[idx], y_c[idx], labels[idx]
                ax.scatter(x, -y_c, c=labels, cmap=cmap, s=0.3, alpha=0.8,
                           vmin=0, vmax=best_K-1, rasterized=True)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{nc}-dominant", fontsize=10)
            except Exception:
                ax.set_title(f"{nc} (failed)")

        legend_elements = [Patch(facecolor=niche_colors[k], label=niche_names[k])
                           for k in range(best_K)]
        fig.legend(handles=legend_elements, loc="upper center", ncol=best_K, fontsize=9,
                   bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout()
        plt.savefig(fig_dir / "wsi_niche_maps.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. Spatial feature table (if survival results exist)
    if survival_df is not None and len(survival_df) > 0:
        tier_df = survival_df[survival_df["feature_type"].isin(["Tier1", "Tier2"])]
        if len(tier_df) > 0:
            # Print as a clean table
            print(f"\n  Spatial pair summary table saved to figures/")

    print(f"  ✓ Figures saved to {fig_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Results Markdown
# ═══════════════════════════════════════════════════════════════════════════════

def generate_results_markdown(cfg, comp_df, niche_names, centroids_z, survival_df,
                              clinical_df, feat_df):
    md_path = cfg.output_dir / f"{cfg.cohort.upper()}_RESULTS.md"
    cohort_upper = cfg.cohort.upper()

    lines = [f"# {cohort_upper} — Spatial TME Analysis Results\n"]
    lines.append(f"## Cohort: {len(clinical_df)} patients")
    for ct in sorted(clinical_df["cancer_type"].unique()):
        n = sum(clinical_df["cancer_type"] == ct)
        lines.append(f"- {ct}: n={n}")
    lines.append(f"\n## Niches (K={len(niche_names)})")
    for k, name in enumerate(niche_names):
        top3 = np.argsort(-centroids_z[k])[:3]
        progs = ", ".join(f"{SHORT_NAMES[i]} ({centroids_z[k,i]:+.2f})" for i in top3)
        lines.append(f"- **{name}**: {progs}")

    if survival_df is not None and len(survival_df) > 0:
        lines.append(f"\n## Top Survival Associations")
        for ep in ["DSS", "OS"]:
            for ct_label in sorted(survival_df["subtype"].unique()):
                ep_df = survival_df[(survival_df["endpoint"] == ep) & (survival_df["subtype"] == ct_label)]
                ep_df = ep_df[ep_df["feature_type"].isin(["Tier1", "Tier2", "niche"])]
                if len(ep_df) == 0: continue
                # Rank by p-value (smallest first)
                ep_ranked = ep_df.dropna(subset=["p_value"]).nsmallest(5, "p_value")
                if len(ep_ranked) == 0:
                    ep_ranked = ep_df.dropna(subset=["hr"]).copy()
                    ep_ranked["_abs"] = (ep_ranked["hr"] - 1).abs()
                    ep_ranked = ep_ranked.nlargest(5, "_abs")
                if len(ep_ranked) == 0: continue
                lines.append(f"\n### {ct_label} — {ep} (n={ep_ranked.iloc[0]['n']}, events={ep_ranked.iloc[0]['events']})")
                for _, r in ep_ranked.iterrows():
                    hr = r["hr"]
                    p = r.get("p_value", np.nan)
                    ci = ""
                    if not np.isnan(r.get("hr_lo", np.nan)):
                        ci = f" [{r['hr_lo']:.2f}–{r['hr_hi']:.2f}]"
                    p_str = f", p={p:.3f}" if not np.isnan(p) else ""
                    lines.append(f"- **{r['feature']}**: HR={hr:.2f}{ci}{p_str}")

        # Named clinical phenotypes
        lines.append(f"\n## Named Spatial Phenotypes")
        lines.append("Derived from de novo niches + predefined spatial pairs:\n")
        for k, name in enumerate(niche_names):
            top3 = np.argsort(-centroids_z[k])[:3]
            progs = ", ".join(SHORT_NAMES[i] for i in top3)
            lines.append(f"- **{name}**: {progs}")

        # Stability
        lines.append(f"\n## Niche Stability")
        lines.append("KMeans stability check (fixed PCA, refit on random half): reported in console output.")

    lines.append(f"\n## Figures")
    for fname in ["niche_centroids.png", "niche_composition.png", "wsi_niche_maps.png",
                   "km_curves.png", "spatial_feature_table.png"]:
        if (cfg.output_dir / "figures" / fname).exists():
            lines.append(f"- `figures/{fname}`")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Results: {md_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SPARC Spatial TME — External Cohorts")
    parser.add_argument("--cohort", required=True, choices=["surgen", "nlst"])
    parser.add_argument("--output-dir", default="results/sparc_spatial_external")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--niche-k", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-tcga-projection", action="store_true")
    parser.add_argument("--niche-names-json", type=str, default=None)
    parser.add_argument("--subtype-only", type=str, default=None, help="NLST only: LUAD or LUSC")
    parser.add_argument("--n-boot", type=int, default=100, help="Bootstrap resamples for Cox CIs (50=fast, 200=paper)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = make_config(args.cohort, args.output_dir,
                      n_workers=args.n_workers, seed=args.seed,
                      niche_k_override=args.niche_k, n_boot=args.n_boot,
                      skip_plots=args.skip_plots,
                      skip_tcga_projection=args.skip_tcga_projection,
                      subtype_only=args.subtype_only)
    if args.niche_names_json:
        cfg.niche_name_override_json = Path(args.niche_names_json)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 60)
    print(f"SPARC Spatial TME — {cfg.cohort.upper()}")
    print(f"  Workers: {cfg.n_workers}")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 60)

    # Step 1: Registry
    print("\nBuilding registry...")
    if cfg.cohort == "surgen":
        slides, clinical_df = build_surgen_registry(cfg)
    else:
        slides, clinical_df = build_nlst_registry(cfg)

    # Step 2: Auto-detect patch step
    auto_detect_patch_step(cfg)

    # Step 3: Slide processing
    cache_path = cfg.output_dir / "slide_cache" / "matrices.npz"
    if not cache_path.exists():
        print("\nProcessing slides...")
        stats = run_slide_processing(slides, cfg)
        save_slide_cache(stats, cfg.output_dir)

    data, meta = load_slide_cache(cfg.output_dir)

    # Step 4: Cross-correlation summary
    S_fz = data["S_fz"].mean(axis=0)
    S = fisher_z_inv(S_fz)
    print(f"\n  Spatial cross-correlation (Tier 1 pairs):")
    for g1, g2, name in TIER1_PAIRS:
        print(f"    {name:<35s}  S={S[g1,g2]:+.3f}")

    # Step 5: Niche discovery
    best_K, centroids_z, niche_names, comp_df, pca, best_km = discover_niches(
        data, meta, slides, cfg)

    # Step 6: Spatial features
    feat_df = compute_spatial_pair_features(data, meta, clinical_df, cfg)

    # Step 7: Survival
    survival_df = run_survival(comp_df, feat_df, clinical_df, niche_names, cfg)

    # Step 8: TCGA projection
    run_tcga_projection(data, meta, slides, cfg, centroids_z, niche_names)

    # Step 9: Figures
    generate_figures(centroids_z, niche_names, comp_df, survival_df, slides, cfg,
                     pca, best_km, best_K)

    # Step 10: Markdown
    generate_results_markdown(cfg, comp_df, niche_names, centroids_z, survival_df,
                              clinical_df, feat_df)

    # Dashboard
    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"{cfg.cohort.upper()} — COMPLETE")
    print(f"{'═'*60}")
    print(f"  Slides: {len(slides)}")
    print(f"  Patients: {len(clinical_df)}")
    print(f"  Niches: K={best_K} ({', '.join(niche_names)})")
    if survival_df is not None:
        n_large = sum((survival_df["hr"] > 1.2) | (survival_df["hr"] < 0.8))
        n_pval = sum(survival_df["p_value"].dropna() < 0.05) if "p_value" in survival_df.columns else 0
        print(f"  Features with large effect (|HR-1|>0.2): {n_large}")
        print(f"  Features with p<0.05: {n_pval}")
    print(f"  Time: {elapsed/60:.1f}m")
    print(f"  Output: {cfg.output_dir}")
    print(f"{'═'*60}\n")

    (cfg.output_dir / "done").touch()


if __name__ == "__main__":
    main()
