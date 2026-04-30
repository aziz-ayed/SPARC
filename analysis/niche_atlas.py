#!/usr/bin/env python
"""Compact niche-morphology atlas figures (Supp. Figs 1–2 of the SPARC paper).

For each cohort (SurGen CRC, NLST lung): one figure with
  rows = niches, columns = representative patches.

Patches are the highest-projection-score tiles drawn from a small set of
representative slides per niche.

Usage:
    python -m analysis.niche_atlas

Cohort-specific data paths are read from environment variables
(SPARC_SURGEN_*, SPARC_NLST_*); see data/README.md for details.
"""

import os

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import label as cc_label
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import warnings
warnings.filterwarnings('ignore')

import openslide

OUT = Path(os.environ.get('SPARC_FIGS_ROOT', 'figs')) / 'spatial'
OUT.mkdir(parents=True, exist_ok=True)
DPI = 600  # paper-grade

# ── Per-cohort layout ──
PATCHES_PER_NICHE = 6    # columns in the figure
PATCH_PX          = 448  # 2x224 (2x grid cell context) — matches pathologist packet
MIN_TF            = 0.50 # drop patches with <50% tissue


# ── Tissue helpers ──
def tissue_fraction(patch, bg=220, blk=30, edge_band=20, edge_black_thresh=0.5):
    """Tissue fraction with hard reject for CZI black-rectangle artifacts.

    Real CZI read failures show up as a solid black rectangle along one of the
    slide edges. We detect them by checking the four edge strips of the patch:
    if any one strip is mostly black, reject. Naturally-dark cellular tissue
    (dense lymphocytes etc.) is scattered, so no single strip is uniformly black.
    """
    if patch is None:
        return 0.0
    is_w = np.all(patch > bg, axis=2)
    is_b = np.all(patch < blk, axis=2)
    H, W = is_b.shape
    # 4 edge strips
    edges = [
        is_b[:edge_band, :],          # top
        is_b[-edge_band:, :],         # bottom
        is_b[:, :edge_band],          # left
        is_b[:, -edge_band:],         # right
    ]
    if any(e.mean() > edge_black_thresh for e in edges):
        return 0.0
    return float(1.0 - (is_w | is_b).mean())


def grid_coords_from_xy(coords, default_stride=224):
    x, y = coords[:, 0].astype(float), coords[:, 1].astype(float)
    ux = np.sort(np.unique(x)); dx = np.diff(ux); dx = dx[dx > 0]
    uy = np.sort(np.unique(y)); dy = np.diff(uy); dy = dy[dy > 0]
    sx = dx.min() if len(dx) > 0 else default_stride
    sy = dy.min() if len(dy) > 0 else sx
    xi = ((x - x.min()) / sx).astype(int)
    yi = ((y - y.min()) / sy).astype(int)
    return xi, yi, xi.max() + 1, yi.max() + 1


# ────────────────────────────────────────────────────────────────────────
# SurGen extraction
# ────────────────────────────────────────────────────────────────────────
SURGEN_GEP_DIR   = Path(os.environ.get('SPARC_SURGEN_GEP', 'features/surgen/predicted_programs_transformer'))
SURGEN_COORD_DIR = Path(os.environ.get('SPARC_SURGEN_COORD', 'features/surgen/hoptimus1'))
SURGEN_WSI_DIR   = Path(os.environ.get('SPARC_SURGEN_WSI', 'wsis/surgen'))
SURGEN_PCA       = Path(os.environ.get('SPARC_SURGEN_PCA', 'results/sparc_spatial_external/surgen/niches/pca_model.pkl'))
SURGEN_KM        = Path(os.environ.get('SPARC_SURGEN_KM', 'results/sparc_spatial_external/surgen/niches/kmeans_models.pkl'))
SURGEN_KM_KEY    = 5
SURGEN_NICHE_NAMES_JSON = Path(os.environ.get('SPARC_SURGEN_NICHE_NAMES', 'results/sparc_spatial_external/surgen/niches/niche_names.json'))

# SurGen K=5 niches → keep 4 (skip Niche 4 = Lymphoid; not visualisable on this cohort)
SURGEN_FORCED = {
    0: ['SR386_40X_HE_T019_01', 'SR386_40X_HE_T088_01', 'SR386_40X_HE_T419_01'],
    1: ['SR386_40X_HE_T394_01', 'SR386_40X_HE_T097_01', 'SR386_40X_HE_T318_01'],
    2: ['SR386_40X_HE_T105_01', 'SR386_40X_HE_T572_01', 'SR386_40X_HE_T195_01'],
    3: ['SR386_40X_HE_T126_01', 'SR386_40X_HE_T020_01', 'SR386_40X_HE_T567_01'],
}
SURGEN_RENAMES = {
    'Lymphoid (TGF-β Exclusion)': 'TGF-β / Mesenchymal',
    'Lymphoid (TGF-beta Exclusion)': 'TGF-β / Mesenchymal',
    'Stromal': 'Stromal (Desmoplastic)',
}


def extract_patches_czi(wsi_path, coords, patch_size=PATCH_PX):
    from pylibCZIrw import czi as pyczi
    with pyczi.open_czi(str(wsi_path)) as czidoc:
        bbox = czidoc.total_bounding_box
        x_off, y_off = bbox['X'][0], bbox['Y'][0]
        out = []
        for (cx, cy) in coords:
            try:
                region = czidoc.read(roi=(int(cx) + x_off, int(cy) + y_off,
                                          patch_size, patch_size))
                if region.ndim == 3 and region.shape[-1] >= 3:
                    if region.shape[-1] == 4:
                        region = region[:, :, [2, 1, 0]]
                    else:
                        region = region[:, :, :3]
                if region.dtype != np.uint8:
                    region = (region * 255).clip(0, 255).astype(np.uint8) \
                             if region.max() <= 1.5 else region.clip(0, 255).astype(np.uint8)
                if region.shape[0] < patch_size or region.shape[1] < patch_size:
                    padded = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
                    h, w = min(region.shape[0], patch_size), min(region.shape[1], patch_size)
                    padded[:h, :w] = region[:h, :w]
                    region = padded
                out.append(region)
            except Exception:
                out.append(None)
    return out


def gather_surgen():
    with open(SURGEN_PCA, 'rb') as f:
        pca = pickle.load(f)
    with open(SURGEN_KM, 'rb') as f:
        km_data = pickle.load(f)
        km = km_data[SURGEN_KM_KEY]['model']
    K = km.n_clusters
    with open(SURGEN_NICHE_NAMES_JSON) as f:
        names = json.load(f)
    names = [SURGEN_RENAMES.get(n, n) for n in names]
    centroids = pca.inverse_transform(km.cluster_centers_)
    cnorms = np.linalg.norm(centroids, axis=1) + 1e-8

    per_niche = {}
    for k, slide_ids in SURGEN_FORCED.items():
        # Pull projections + coords for all forced slides; pick top tiles globally
        bag = []  # list of (proj_score, slide_id, x, y)
        for sid in slide_ids:
            gep_f = SURGEN_GEP_DIR / f"{sid}.h5"
            coord_f = SURGEN_COORD_DIR / f"{sid}.h5"
            if not (gep_f.exists() and coord_f.exists()):
                continue
            with h5py.File(gep_f, 'r') as f: feats = f['features'][:].astype(np.float32)
            with h5py.File(coord_f, 'r') as f: coords = f['coords'][:].astype(np.float32)
            centered = feats - feats.mean(axis=1, keepdims=True)
            labels = km.predict(pca.transform(centered))
            projections = np.sum(centered * centroids[labels], axis=1) / cnorms[labels]
            niche_idx = np.where(labels == k)[0]
            for i in niche_idx:
                bag.append((float(projections[i]), sid, float(coords[i, 0]), float(coords[i, 1])))
        bag.sort(reverse=True)

        # Take top N candidates by projection score (heavy oversample for tissue filter).
        # No per-slide cap — pure best-of-best.
        chosen = bag[:PATCHES_PER_NICHE * 6]

        # Extract all candidates while preserving global projection-score order
        scored_patches = []  # list of (proj, patch)
        per_slide = {}
        for proj, sid, cx, cy in chosen:
            per_slide.setdefault(sid, []).append((proj, cx, cy))

        for sid, locs in per_slide.items():
            wsi_f = SURGEN_WSI_DIR / f"{sid}.czi"
            if not wsi_f.exists():
                continue
            coords_only = [(cx, cy) for (_, cx, cy) in locs]
            patches = extract_patches_czi(wsi_f, coords_only)
            for (proj, cx, cy), p in zip(locs, patches):
                if p is None: continue
                if tissue_fraction(p) >= MIN_TF:
                    scored_patches.append((proj, p))

        # Re-sort by projection (highest first), keep top N
        scored_patches.sort(key=lambda x: -x[0])
        per_niche[k] = [p for _, p in scored_patches[:PATCHES_PER_NICHE]]
        print(f"  SurGen niche {k} ({names[k]}): {len(per_niche[k])} patches")
    return names, per_niche


# ────────────────────────────────────────────────────────────────────────
# NLST extraction
# ────────────────────────────────────────────────────────────────────────
NLST_GEP_DIR    = Path(os.environ.get('SPARC_NLST_GEP', 'features/nlst/predicted_programs_transformer'))
NLST_COORD_DIR  = Path(os.environ.get('SPARC_NLST_COORD', 'features/nlst/hoptimus1'))
NLST_DICOM_CSV  = Path(os.environ.get('SPARC_NLST_DICOM_CSV', 'data/nlst_dicom_file_list.csv'))
NLST_CONTOURS   = Path(os.environ.get('SPARC_NLST_CONTOURS', 'features/nlst/contours_geojson'))
NLST_PCA        = Path(os.environ.get('SPARC_NLST_PCA', 'results/sparc_spatial_external/nlst/niches/pca_model.pkl'))
NLST_KM         = Path(os.environ.get('SPARC_NLST_KM', 'results/sparc_spatial_external/nlst/niches/kmeans_models.pkl'))
NLST_KM_KEY     = 4
NLST_NICHE_NAMES_JSON = Path(os.environ.get('SPARC_NLST_NICHE_NAMES', 'results/sparc_spatial_external/nlst/niches/niche_names.json'))

NLST_FORCED = {
    0: ['6bb9a739-2cd3-451e-ba3c-6bcc894126c8',
        '6c0e8d95-b15a-4868-a3b1-b5780fe72e5b',
        'd4e5e26a-962c-4e0c-b71c-c1f3da3e5e0f'],
    1: ['95316478-b801-47b8-bfa2-8cf6431afd01',
        '9f1a080a-6e8f-476c-bb77-f3770a98dcea',
        '695e7baf-fb69-444a-aa25-8bbaf0dbbef0'],
    2: ['b0f391cd-7108-49d3-93ef-7ece232f9412',
        '14ee43e1-52b6-4377-9eb4-ba0aa91cef2e',
        'eab2dcd0-3ceb-4476-a23a-89008e1fc6c4'],
    3: ['670675e5-d271-4eb2-8692-1db9e56abbd1',
        'a410e5ec-0665-4a35-adca-ef57ea937cc8',
        '66a198ac-7ce5-4a52-bea8-12060119516f'],
}
NLST_RENAMES = {}  # use original names


def deduce_nlst_grid(wsi_path, h5_n, tile=224):
    w = openslide.OpenSlide(str(wsi_path))
    W, H = w.dimensions
    c_base = W // tile; r_base = H // tile
    for dc in (0, 1, -1):
        for dr in (0, 1, -1):
            nc, nr = c_base + dc, r_base + dr
            if nc > 0 and nr > 0 and nr * nc == h5_n:
                return int(nr), int(nc), tile
    raise RuntimeError(f"grid: W={W} H={H} h5_n={h5_n}")


def nlst_tissue_mask(uuid, n_rows, n_cols, tile_px):
    p = NLST_CONTOURS / f"{uuid}.geojson"
    if not p.exists(): return None
    with open(p) as f: g = json.load(f)
    paths = []
    for feat in g.get('features', []):
        geom = feat.get('geometry') or {}
        if geom.get('type') == 'Polygon' and geom['coordinates']:
            paths.append(MPath(np.array(geom['coordinates'][0])))
        elif geom.get('type') == 'MultiPolygon':
            for poly in geom['coordinates']:
                if poly: paths.append(MPath(np.array(poly[0])))
    if not paths: return None
    rs, cs = np.indices((n_rows, n_cols))
    centers = np.stack([cs.ravel() * tile_px + tile_px / 2,
                        rs.ravel() * tile_px + tile_px / 2], axis=1)
    inside = np.zeros(centers.shape[0], dtype=bool)
    for p in paths:
        inside |= p.contains_points(centers)
    return inside


def extract_patches_openslide(wsi_path, coords, patch_size=PATCH_PX, target_mpp=0.5):
    w = openslide.OpenSlide(str(wsi_path))
    actual_mpp = float(w.properties.get('openslide.mpp-x', target_mpp))
    size_l0 = int(round(patch_size * target_mpp / actual_mpp))
    out = []
    for (cx, cy) in coords:
        try:
            region = w.read_region((int(cx), int(cy)), 0, (size_l0, size_l0))
            arr = np.array(region)[:, :, :3]
            if size_l0 != patch_size:
                arr = np.array(Image.fromarray(arr).resize((patch_size, patch_size), Image.LANCZOS))
            out.append(arr)
        except Exception:
            out.append(None)
    return out


def gather_nlst():
    with open(NLST_PCA, 'rb') as f: pca = pickle.load(f)
    with open(NLST_KM, 'rb') as f:
        km_data = pickle.load(f)
        km = km_data[NLST_KM_KEY]['model']
    K = km.n_clusters
    with open(NLST_NICHE_NAMES_JSON) as f:
        names = json.load(f)
    names = [NLST_RENAMES.get(n, n) for n in names]
    centroids = pca.inverse_transform(km.cluster_centers_)
    cnorms = np.linalg.norm(centroids, axis=1) + 1e-8

    dc = pd.read_csv(NLST_DICOM_CSV)
    dc_l0 = dc[dc['level'] == 0].copy()
    dc_l0['uuid'] = dc_l0['file_name'].str.replace('.dcm', '', regex=False)
    dicom_map = dict(zip(dc_l0['uuid'], dc_l0['file_path']))

    per_niche = {}
    for k, slide_ids in NLST_FORCED.items():
        bag = []
        for uuid in slide_ids:
            if uuid not in dicom_map: continue
            gep_f = NLST_GEP_DIR / f"{uuid}.h5"
            if not gep_f.exists(): continue
            with h5py.File(gep_f, 'r') as f: feats = f['features'][:].astype(np.float32)
            try:
                n_rows, n_cols, tile_px = deduce_nlst_grid(dicom_map[uuid], feats.shape[0])
            except RuntimeError:
                continue
            mask = nlst_tissue_mask(uuid, n_rows, n_cols, tile_px)
            if mask is None: continue
            tissue_idx = np.where(mask)[0]
            feats_t = feats[tissue_idx]
            idx = np.arange(feats.shape[0])
            xs = (idx % n_cols) * tile_px
            ys = (idx // n_cols) * tile_px
            coords_t = np.stack([xs, ys], axis=1)[tissue_idx]
            centered = feats_t - feats_t.mean(axis=1, keepdims=True)
            labels = km.predict(pca.transform(centered))
            projections = np.sum(centered * centroids[labels], axis=1) / cnorms[labels]
            niche_idx = np.where(labels == k)[0]
            for i in niche_idx:
                bag.append((float(projections[i]), uuid, float(coords_t[i, 0]), float(coords_t[i, 1])))
        bag.sort(reverse=True)

        # Heavy oversample for tissue filter; no per-slide cap.
        chosen = bag[:PATCHES_PER_NICHE * 6]

        scored_patches = []
        per_slide = {}
        for proj, uuid, cx, cy in chosen:
            per_slide.setdefault(uuid, []).append((proj, cx, cy))

        for uuid, locs in per_slide.items():
            coords_only = [(cx, cy) for (_, cx, cy) in locs]
            patches = extract_patches_openslide(dicom_map[uuid], coords_only)
            for (proj, cx, cy), p in zip(locs, patches):
                if p is None: continue
                if tissue_fraction(p) >= MIN_TF:
                    scored_patches.append((proj, p))

        scored_patches.sort(key=lambda x: -x[0])
        per_niche[k] = [p for _, p in scored_patches[:PATCHES_PER_NICHE]]
        print(f"  NLST niche {k} ({names[k]}): {len(per_niche[k])} patches")
    return names, per_niche


# ────────────────────────────────────────────────────────────────────────
# Compact figure renderer (publication style)
# ────────────────────────────────────────────────────────────────────────
# Match the niche palette used in fig3b (WSI niche maps).
NICHE_BAR_COLORS = {
    'Proliferative':           '#E64B35',  # red
    'Stromal':                 '#4DBBD5',  # cyan (NLST stromal)
    'Stromal (Desmoplastic)':  '#4DBBD5',  # cyan
    'Immune-Inflamed':         '#3C5488',  # navy
    'Lymphoid':                '#00A087',  # teal
    'TGF-β / Mesenchymal':     '#F39B7F',  # salmon (matches "TGF-β Excl." in fig3b)
}

# Pixel size at 20× = 0.5 μm/px. Patches are PATCH_PX wide → field of view in μm.
MPP_20X       = 0.5
SCALE_BAR_UM  = 50  # length of scale bar in μm


def _draw_scale_bar(ax, patch_px=PATCH_PX, mpp=MPP_20X, length_um=SCALE_BAR_UM):
    bar_px = length_um / mpp
    # Bottom-right corner of the patch with margin
    margin = patch_px * 0.06
    y = patch_px - margin
    x_end = patch_px - margin
    x_start = x_end - bar_px
    # White outline behind for contrast on dark patches
    ax.plot([x_start, x_end], [y, y], color='white', linewidth=4.5, solid_capstyle='butt', zorder=10)
    ax.plot([x_start, x_end], [y, y], color='black', linewidth=2.0, solid_capstyle='butt', zorder=11)
    ax.text((x_start + x_end) / 2, y - margin * 0.4, f'{length_um} μm',
            color='black', fontsize=7, ha='center', va='bottom', zorder=11,
            path_effects=[__import__('matplotlib.patheffects', fromlist=['withStroke']).withStroke(
                linewidth=2.5, foreground='white')])


def render_atlas(per_niche, niche_names, title, subtitle, out_stem,
                 niche_keys=None, fig_w=8.5, row_h=1.5, panel_letter=None):
    if niche_keys is None:
        niche_keys = sorted(per_niche.keys())
    n_rows = len(niche_keys)
    n_cols = PATCHES_PER_NICHE

    has_header = bool(title) or bool(subtitle) or bool(panel_letter)
    fig_h = row_h * n_rows + (0.38 if has_header else 0.15)
    top_margin = 0.93 if has_header else 0.99
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             facecolor='white',
                             gridspec_kw={'wspace': 0.025, 'hspace': 0.04,
                                          'left': 0.21, 'right': 0.99,
                                          'top': top_margin, 'bottom': 0.04})
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig_w_in_inches = fig_w
    label_x = 0.005   # niche label x in figure coords
    bar_x   = 0.197   # niche colour bar x

    for r, k in enumerate(niche_keys):
        patches = per_niche.get(k, [])
        nname = niche_names[k]
        bar_color = NICHE_BAR_COLORS.get(nname, '#777777')

        # patches
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(patches):
                ax.imshow(patches[c])
                # set extent so scale bar coordinates work in pixel space
                ax.set_xlim(0, PATCH_PX); ax.set_ylim(PATCH_PX, 0)
            else:
                ax.set_facecolor('#f5f5f5')
            for s in ax.spines.values():
                s.set_edgecolor('#bbb'); s.set_linewidth(0.4)

        # scale bar on the bottom-left patch only (last row, first column)
        if r == n_rows - 1 and patches:
            _draw_scale_bar(axes[r, 0])

        # row position in figure coords (axes[r, 0].get_position is the bbox)
        row_bbox = axes[r, 0].get_position()
        y_center = (row_bbox.y0 + row_bbox.y1) / 2

        # left vertical color bar
        bar_ax = fig.add_axes([bar_x - 0.012, row_bbox.y0, 0.005, row_bbox.height],
                              facecolor=bar_color, frameon=False)
        bar_ax.set_xticks([]); bar_ax.set_yticks([])
        bar_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=bar_color, transform=bar_ax.transAxes))
        for s in bar_ax.spines.values():
            s.set_visible(False)

        # niche name label (right-aligned to color bar)
        fig.text(bar_x - 0.018, y_center, nname, fontsize=10, fontweight='bold',
                 ha='right', va='center', color='#222222')

    # Compact title, left-aligned just above the patch grid
    if title:
        fig.text(0.21, 0.965, title, ha='left', va='top',
                 fontsize=11, fontweight='bold', color='#222222')
    if subtitle:
        fig.text(0.21, 0.94, subtitle, ha='left', va='top',
                 fontsize=8.5, color='#666666', style='italic')
    if panel_letter:
        fig.text(0.005, 0.985, panel_letter, ha='left', va='top',
                 fontsize=14, fontweight='bold')

    out_png = OUT / f"{out_stem}.png"
    out_pdf = OUT / f"{out_stem}.pdf"
    plt.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {out_png}")


# ── Main ──
print(f"{'='*60}\nSurGen CRC niche atlas\n{'='*60}")
sg_names, sg_per_niche = gather_surgen()
render_atlas(
    sg_per_niche, sg_names,
    title='SurGen — colorectal cancer',
    subtitle=None,
    out_stem='niche_atlas_surgen',
    niche_keys=[0, 1, 2, 3],   # skip niche 4 (Lymphoid) — not visualisable on this cohort
    panel_letter=None,
)

print(f"\n{'='*60}\nNLST Lung niche atlas\n{'='*60}")
nl_names, nl_per_niche = gather_nlst()
render_atlas(
    nl_per_niche, nl_names,
    title='NLST — non-small-cell lung cancer',
    subtitle=None,
    out_stem='niche_atlas_nlst',
    niche_keys=[0, 1, 2, 3],
    panel_letter=None,
)

print(f"\nDone. Atlases in {OUT}/")
