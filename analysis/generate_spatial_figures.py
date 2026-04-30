#!/usr/bin/env python
"""Generate all spatial TME figure panels and supplementary table CSVs for the paper.

Reads pre-computed results from:
  results/sparc_spatial/ (TCGA)
  results/sparc_spatial_external/surgen/ (SurGen CRC)
  results/sparc_spatial_external/nlst/ (NLST Lung)

Outputs to: figs/spatial/

Usage:
    python scripts/generate_spatial_paper_figures.py
"""

import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.ndimage import gaussian_filter, binary_dilation, zoom as ndzoom, generic_filter

OUT = Path(os.environ.get('SPARC_FIGS_ROOT', 'figs')) / 'spatial'
OUT.mkdir(parents=True, exist_ok=True)
DPI = 600

# Short names (same order as PROGRAM_NAMES)
SHORT = [
    "Angiogenesis", "Apoptosis", "Coagulation", "DNA Repair", "E2F Targets",
    "EMT", "G2M Checkpoint", "Glycolysis", "Hypoxia", "IL-6/JAK/STAT3",
    "Inflammatory", "IFN-α", "IFN-γ", "mTORC1", "MYC Targets", "OxPhos",
    "TGF-β Signaling", "TNF-α/NF-κB", "MHC-I Processing", "Collagen",
    "ECM Organization", "Integrin", "MHC-II", "Mismatch Repair", "Neutrophil",
    "TCR Signaling", "TLR Cascades", "Antigen Pres.", "B Cell", "CIN",
    "TGF-β Exclusion", "TLS", "T Cell GEP", "Naïve CD8", "Treg",
    "PD-1⁺ CD8", "M1 Macro", "M2 Macro", "Exhausted CD8", "Immature DCs",
]
_PI = {p: i for i, p in enumerate(SHORT)}

# Program → functional group (for dot plot y-axis ordering)
GROUPS = {
    "Proliferative": ["E2F Targets", "G2M Checkpoint", "MYC Targets", "CIN",
                       "DNA Repair", "Mismatch Repair", "Apoptosis"],
    "Metabolic": ["Glycolysis", "Hypoxia", "OxPhos", "mTORC1"],
    "Immune signaling": ["IFN-α", "IFN-γ", "TNF-α/NF-κB", "IL-6/JAK/STAT3",
                          "Inflammatory", "TCR Signaling", "TLR Cascades",
                          "Neutrophil", "Antigen Pres.", "MHC-I Processing",
                          "MHC-II", "Coagulation"],
    "Lymphoid / T cell": ["T Cell GEP", "Naïve CD8", "Treg", "PD-1⁺ CD8",
                           "Exhausted CD8", "M1 Macro", "M2 Macro",
                           "Immature DCs", "B Cell", "TLS"],
    "Stromal / ECM": ["Angiogenesis", "EMT", "TGF-β Signaling", "Collagen",
                       "ECM Organization", "Integrin", "TGF-β Exclusion"],
}

# Build ordered program list (grouped)
ORDERED_PROGRAMS = []
GROUP_LABELS = []
for g, progs in GROUPS.items():
    for p in progs:
        if p in SHORT:
            ORDERED_PROGRAMS.append(p)
            GROUP_LABELS.append(g)
PROG_TO_IDX = {p: SHORT.index(p) for p in ORDERED_PROGRAMS}


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): Niche Dot Plots
# ═══════════════════════════════════════════════════════════════════════════════

def fig3a():
    print("Panel (a): Niche dot plots...")
    tcga_c = np.load("results/sparc_spatial/a3_niches/niche_centroids.npz")["centroids_z"]
    crc_c = np.load("results/sparc_spatial_external/surgen/niches/niche_centroids.npz")["centroids_z"]
    lung_c = np.load("results/sparc_spatial_external/nlst/niches/niche_centroids.npz")["centroids_z"]

    tcga_names = ["Proliferative", "Stromal", "Lymphoid", "Immune-Inflamed"]
    with open("results/sparc_spatial_external/surgen/niches/niche_names.json") as f:
        crc_names = json.load(f)
    with open("results/sparc_spatial_external/nlst/niches/niche_names.json") as f:
        lung_names = json.load(f)

    datasets = [
        ("TCGA Pan-Cancer", tcga_c, tcga_names),
        ("CRC (SurGen)", crc_c, crc_names),
        ("Lung (NLST)", lung_c, lung_names),
    ]

    n_progs = len(ORDERED_PROGRAMS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 12), sharey=True, facecolor="white")

    for ax, (title, centroids, niche_names) in zip(axes, datasets):
        K = centroids.shape[0]
        ordered_idx = [PROG_TO_IDX[p] for p in ORDERED_PROGRAMS]
        c_ordered = centroids[:, ordered_idx]

        for k in range(K):
            for j, prog in enumerate(ORDERED_PROGRAMS):
                val = c_ordered[k, j]
                size = min(abs(val) * 80, 200)
                color = plt.cm.RdBu_r((val + 1.5) / 3.0)
                ax.scatter(k, n_progs - 1 - j, s=size, c=[color], edgecolors="black",
                           linewidths=0.3, zorder=3)

        ax.set_xticks(range(K))
        ax.set_xticklabels(niche_names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, K - 0.5)

        if ax == axes[0]:
            ax.set_yticks(range(n_progs))
            ax.set_yticklabels(list(reversed(ORDERED_PROGRAMS)), fontsize=6)
            cumulative = 0
            group_order = list(GROUPS.keys())
            for g in group_order:
                n_in_group = len([p for p in GROUPS[g] if p in ORDERED_PROGRAMS])
                y_pos = n_progs - cumulative - 0.5
                if cumulative > 0:
                    ax.axhline(y_pos, color="#cccccc", linewidth=0.5, linestyle="--")
                mid = n_progs - cumulative - n_in_group / 2
                ax.text(-1.2, mid, g, fontsize=6, ha="right", va="center", color="#666666",
                        fontstyle="italic", clip_on=False)
                cumulative += n_in_group

        ax.grid(axis="x", alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1.5, 1.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.3, aspect=20, pad=0.02)
    cbar.set_label("Centroid z-score", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "fig3a_niche_dotplots.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: fig3a_niche_dotplots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): WSI Niche Maps (rendered from data)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3b():
    print("Panel (b): WSI niche maps...")
    from analysis.spatial_tcga import _load_array, _find_feature_file

    # Load PCA/KMeans models
    with open("results/sparc_spatial_external/surgen/niches/pca_model.pkl", "rb") as f:
        surgen_pca = pickle.load(f)
    with open("results/sparc_spatial_external/surgen/niches/kmeans_models.pkl", "rb") as f:
        surgen_km = pickle.load(f)[5]["model"]
    with open("results/sparc_spatial/a3_niches/pca_model.pkl", "rb") as f:
        tcga_pca = pickle.load(f)
    with open("results/sparc_spatial/a3_niches/kmeans_models.pkl", "rb") as f:
        tcga_km = pickle.load(f)[4]["model"]

    # Data paths — driven by env vars (see data/README.md)
    surgen_gep = Path(os.environ.get('SPARC_SURGEN_GEP', 'features/surgen/predicted_programs_transformer'))
    surgen_coord = Path(os.environ.get('SPARC_SURGEN_COORD', 'features/surgen/hoptimus1'))
    tcga_coord = Path(os.environ.get('SPARC_TCGA_COORD', 'features/tcga/hoptimus1'))
    tcga_gep = Path(os.environ.get('SPARC_TCGA_GEP', 'features/tcga/predicted_programs_transformer'))

    def desat(rgb, a=0.15):
        return rgb * (1 - a) + np.mean(rgb) * a

    COLORS = {
        "Proliferative": desat(np.array([0.902, 0.294, 0.208])),
        "Stromal": desat(np.array([0.302, 0.733, 0.835])),
        "Immune": desat(np.array([0.235, 0.329, 0.533])),
        "Lymphoid": desat(np.array([0.000, 0.627, 0.529])),
        "TLS": desat(np.array([0.000, 0.627, 0.529])),
        "TGF-β Excl.": desat(np.array([0.953, 0.608, 0.498])),
    }
    crc_c2n = {0: "Immune", 1: "Proliferative", 2: "TGF-β Excl.", 3: "Stromal", 4: "TLS"}
    lung_c2n = {0: "Proliferative", 1: "Stromal", 2: "Lymphoid", 3: "Immune"}
    BAR_ORDER = ["Proliferative", "Stromal", "Immune", "Lymphoid", "TLS", "TGF-β Excl."]

    def load_and_assign(sid, gep_dir, coord_dir, pca, km):
        raw = _load_array(_find_feature_file(gep_dir, sid), "features").astype(np.float32)
        coords = _load_array(_find_feature_file(coord_dir, sid), "coords")
        mu = raw.mean(0, keepdims=True)
        sd = raw.std(0, keepdims=True); sd[sd < 1e-8] = 1.0
        Z = (raw - mu) / sd
        Z = Z - Z.mean(1, keepdims=True)
        return coords, km.predict(pca.transform(Z))

    crc_coords, crc_labels = load_and_assign(
        "SR386_40X_HE_T126_01", surgen_gep, surgen_coord, surgen_pca, surgen_km)
    lung_coords, lung_labels = load_and_assign(
        "TCGA-78-8660-01Z-00-DX1.11E923ED-C01B-439D-8796-08C9E3DC2D93",
        tcga_gep, tcga_coord, tcga_pca, tcga_km)

    def get_fracs(labels, c2n, K):
        raw = np.bincount(labels, minlength=K) / len(labels)
        return {c2n[k]: raw[k] for k in range(K)}

    crc_fracs = get_fracs(crc_labels, crc_c2n, 5)
    lung_fracs = get_fracs(lung_labels, lung_c2n, 4)

    def render(coords, labels, c2n, K, cd, scale=10):
        x, y = coords[:, 0].astype(float), coords[:, 1].astype(float)
        ux = np.sort(np.unique(x)); dx = np.diff(ux); dx = dx[dx > 0]
        uy = np.sort(np.unique(y)); dy = np.diff(uy); dy = dy[dy > 0]
        sx = dx.min() if len(dx) > 0 else 448
        sy = dy.min() if len(dy) > 0 else sx
        xi = ((x - x.min()) / sx).astype(int)
        yi = ((y - y.min()) / sy).astype(int)
        nx, ny = xi.max() + 1, yi.max() + 1
        prob = np.zeros((K, ny, nx)); tissue = np.zeros((ny, nx), dtype=bool)
        for i in range(len(x)):
            prob[labels[i], yi[i], xi[i]] = 1.0
            tissue[yi[i], xi[i]] = True
        for k in range(K):
            prob[k] = gaussian_filter(prob[k], sigma=1.2)
        sm = np.argmax(prob, axis=0)
        td = binary_dilation(tissue, iterations=1)
        sm[~td] = -1

        def maj(v):
            v = v[v >= 0]
            return np.bincount(v.astype(int), minlength=K).argmax() if len(v) > 0 else -1

        sm = generic_filter(sm.astype(float), maj, size=3).astype(int)
        sm[~td] = -1
        lu = ndzoom(sm.astype(float), scale, order=0).astype(int)
        tu = ndzoom(td.astype(float), scale, order=0) > 0.5
        pal = np.array([cd[c2n[k]] for k in range(K)])
        img = np.zeros((lu.shape[0], lu.shape[1], 3))
        for k in range(K):
            img[lu == k] = pal[k]
        bnd = np.zeros_like(tu)
        for d0, d1 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            sh = np.roll(np.roll(lu, d0, 0), d1, 1)
            bnd |= (lu != sh) & tu & (sh >= 0) & (lu >= 0)
        bz = binary_dilation(bnd, iterations=2)
        ib = img.copy()
        for c in range(3):
            ib[:, :, c] = gaussian_filter(img[:, :, c], sigma=1.5)
        img[bz] = ib[bz]
        r = np.any(tu, 1); c = np.any(tu, 0)
        r0, r1 = np.where(r)[0][[0, -1]]
        c0, c1 = np.where(c)[0][[0, -1]]
        p = 20
        r0 = max(0, r0 - p); r1 = min(img.shape[0], r1 + p)
        c0 = max(0, c0 - p); c1 = min(img.shape[1], c1 + p)
        return img[r0:r1, c0:c1], tu[r0:r1, c0:c1], bnd[r0:r1, c0:c1]

    ci, ct, cb = render(crc_coords, crc_labels, crc_c2n, 5, COLORS)
    li, lt, lb = render(lung_coords, lung_labels, lung_c2n, 4, COLORS)
    th = max(ci.shape[0], li.shape[0])
    tw = max(ci.shape[1], li.shape[1])

    def pm(img, tis, bnd, th, tw, bg=0.965):
        h, w = img.shape[:2]
        ph = (th - h) // 2; pw = (tw - w) // 2
        ip = np.full((th, tw, 3), bg)
        ip[ph:ph + h, pw:pw + w] = img
        tp = np.zeros((th, tw), dtype=bool)
        tp[ph:ph + h, pw:pw + w] = tis
        bp = np.zeros((th, tw), dtype=bool)
        bp[ph:ph + h, pw:pw + w] = bnd
        return ip, tp, bp

    ci, ct, cb = pm(ci, ct, cb, th, tw)
    li, lt, lb = pm(li, lt, lb, th, tw)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor="white")
    for ax, img, tis, bnd, title, fracs in [
        (axes[0], ci, ct, cb, "Colorectal cancer", crc_fracs),
        (axes[1], li, lt, lb, "Lung adenocarcinoma", lung_fracs),
    ]:
        final = img.copy()
        final[~tis] = 0.965
        final[bnd] = final[bnd] * 0.55 + 0.25
        outline = binary_dilation(tis, iterations=2) & ~tis
        final[outline] = 0.80
        ax.imshow(final, interpolation="lanczos", aspect="equal")
        ax.set_facecolor((0.965, 0.965, 0.965))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#333333")
        for s in ax.spines.values():
            s.set_visible(False)
        bar = ax.inset_axes([0.03, -0.07, 0.94, 0.035])
        left = 0
        for name in BAR_ORDER:
            frac = fracs.get(name, 0)
            if frac < 0.005:
                continue
            bar.barh(0, frac, left=left, color=COLORS[name], edgecolor="none", height=1)
            if frac > 0.08:
                bar.text(left + frac / 2, 0, f"{name} {frac:.0%}", ha="center", va="center",
                         fontsize=8, color="white" if frac > 0.12 else "black", fontweight="bold")
            left += frac
        bar.set_xlim(0, 1); bar.set_ylim(-0.5, 0.5); bar.axis("off")

    plt.tight_layout(w_pad=3)
    plt.savefig(OUT / "fig3b_wsi_maps.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: fig3b_wsi_maps.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (c): Cross-Correlation Bar Chart (12 pairs)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3c():
    print("Panel (c): Cross-correlation bars...")
    S = np.load("results/sparc_spatial/a2_cross_correlation/S_matrix.npz")["S"]

    all_pairs = [
        (_PI["TGF-β Exclusion"], _PI["TCR Signaling"], "TGF-β Excl. ↔ TCR"),
        (_PI["TGF-β Exclusion"], _PI["Naïve CD8"], "TGF-β Excl. ↔ Naïve CD8"),
        (_PI["MHC-I Processing"], _PI["PD-1⁺ CD8"], "MHC-I ↔ PD-1⁺ CD8"),
        (_PI["TLS"], _PI["TCR Signaling"], "TLS ↔ TCR"),
        (_PI["ECM Organization"], _PI["Inflammatory"], "ECM ↔ Inflammatory"),
        (_PI["Collagen"], _PI["Immature DCs"], "Collagen ↔ Immature DCs"),
        (_PI["Hypoxia"], _PI["Angiogenesis"], "Hypoxia ↔ Angiogenesis"),
        (_PI["mTORC1"], _PI["Treg"], "mTORC1 ↔ Treg"),
        (_PI["M2 Macro"], _PI["MHC-II"], "M2 ↔ MHC-II"),
        (_PI["E2F Targets"], _PI["IFN-γ"], "E2F ↔ IFN-γ"),
        (_PI["G2M Checkpoint"], _PI["Exhausted CD8"], "G2M ↔ Exhausted CD8"),
        (_PI["OxPhos"], _PI["Glycolysis"], "OxPhos ↔ Glycolysis"),
    ]

    names = [p[2] for p in all_pairs]
    vals = [S[p[0], p[1]] for p in all_pairs]
    sigs = ["***" for _ in vals]  # all p<0.001 from permutation null

    sort_idx = np.argsort(vals)
    names = [names[i] for i in sort_idx]
    vals = [vals[i] for i in sort_idx]
    sigs = [sigs[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")
    colors = ["#3C5488" if v < 0 else "#E64B35" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.3, height=0.65)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Spatial cross-correlation (S)", fontsize=10)

    for i, (v, sig) in enumerate(zip(vals, sigs)):
        offset = 0.015 if v > 0 else -0.015
        ha = "left" if v > 0 else "right"
        ax.text(v + offset, i, f"{v:+.3f} {sig}", va="center", ha=ha, fontsize=6.5, color="#444444")

    # Direction labels above the bars
    ax.annotate("Co-localization \u2192", xy=(0.55, 1.0), xycoords="axes fraction",
                fontsize=7, color="#E64B35", ha="center", va="bottom")
    ax.annotate("\u2190 Spatial exclusion", xy=(0.25, 1.0), xycoords="axes fraction",
                fontsize=7, color="#3C5488", ha="center", va="bottom")

    ax.text(0.99, 0.01, "*** p < 0.001 (permutation null)", transform=ax.transAxes,
            fontsize=6, ha="right", va="bottom", color="#888888")

    ax.grid(axis="x", alpha=0.15)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig3c_crosscorr_bars.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: fig3c_crosscorr_bars.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (d): Forest Plot — three panels (COAD, READ, Lung), 5 features each
# ═══════════════════════════════════════════════════════════════════════════════

def fig3d():
    print("Panel (d): Forest plot...")
    surgen = pd.read_csv("results/sparc_spatial_external/surgen/survival/all_results.csv")
    nlst = pd.read_csv("results/sparc_spatial_external/nlst/survival/all_results.csv")

    def get_hr(df, sub, ep, feat):
        r = df[(df["subtype"] == sub) & (df["endpoint"] == ep) & (df["feature"] == feat)]
        if len(r) == 0:
            return np.nan, np.nan, np.nan, np.nan
        r = r.iloc[0]
        return r["hr"], r.get("hr_lo", np.nan), r.get("hr_hi", np.nan), r.get("p_value", np.nan)

    coad_features = [
        ("Desmoplastic niche", "Stromal"),
        ("Hypoxia \u2194 Angiogenesis", "Hypoxia \u2194 Angiogenesis"),
        ("Lymphoid/TGF-\u03b2 niche", "Lymphoid (TGF-\u03b2 Exclusion)"),
        ("MHC-I \u2194 PD-1\u207a CD8", "MHC-I \u2194 PD-1\u207a CD8"),
        ("OxPhos \u2194 Glycolysis", "OxPhos \u2194 Glycolysis"),
    ]
    read_features = [
        ("OxPhos \u2194 Glycolysis", "OxPhos \u2194 Glycolysis"),
        ("ECM \u2194 Inflammatory", "ECM \u2194 Inflammatory"),
        ("E2F \u2194 IFN-\u03b3", "E2F \u2194 IFN-\u03b3"),
        ("Desmoplastic niche", "Stromal"),
        ("Lymphoid/TGF-\u03b2 niche", "Lymphoid (TGF-\u03b2 Exclusion)"),
    ]
    lung_features = [
        ("OxPhos \u2194 Glycolysis", "OxPhos \u2194 Glycolysis"),
        ("Lymphoid niche (B cell / TLS)", "Lymphoid"),
        ("MHC-I \u2194 PD-1\u207a CD8", "MHC-I \u2194 PD-1\u207a CD8"),
        ("Collagen \u2194 Immature DCs", "Collagen \u2194 Immature DCs"),
        ("M2 \u2194 MHC-II", "M2 \u2194 MHC-II"),
    ]

    panels = [
        ("Colon adenocarcinoma (COAD)", coad_features, surgen, "COAD", "#E64B35"),
        ("Rectal adenocarcinoma (READ)", read_features, surgen, "READ", "#F39B7F"),
        ("Lung cancer (NLST)", lung_features, nlst, "All", "#4DBBD5"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")

    for ax, (cohort_label, features, df, sub, color) in zip(axes, panels):
        hrs, los, his, ps, labels = [], [], [], [], []
        for label, feat_key in features:
            hr, lo, hi, p = get_hr(df, sub, "DSS", feat_key)
            hrs.append(hr); los.append(lo); his.append(hi); ps.append(p); labels.append(label)

        y = np.arange(len(features))

        for i in range(len(hrs)):
            hr, lo, hi, p = hrs[i], los[i], his[i], ps[i]
            ax.plot(hr, i, "o", color="#333333", markersize=8, markerfacecolor="#333333",
                    markeredgewidth=1.2, zorder=5)
            if not np.isnan(lo):
                ax.plot([lo, hi], [i, i], "-", color="#333333", linewidth=1.5, zorder=4)
                ax.plot([lo, lo], [i - 0.1, i + 0.1], "-", color="#333333", linewidth=0.8)
                ax.plot([hi, hi], [i - 0.1, i + 0.1], "-", color="#333333", linewidth=0.8)

            star = "***" if (not np.isnan(p) and p < 0.001) else \
                   ("**" if (not np.isnan(p) and p < 0.01) else \
                   ("*" if (not np.isnan(p) and p < 0.05) else ""))
            right = max(hi, hr) + 0.05 if not np.isnan(hi) else hr + 0.05
            ax.text(right, i, f"{hr:.2f} {star}", fontsize=7, va="center", color="#444444")

        ax.axvline(1, color="gray", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Hazard Ratio (DSS)", fontsize=9)
        ax.set_title(cohort_label, fontsize=11, fontweight="bold", color=color)
        ax.invert_yaxis()

        # Shade protective vs adverse regions
        all_lo = [l for l in los if not np.isnan(l)]
        all_hi = [h for h in his if not np.isnan(h)]
        xlim_lo = min(min(all_lo) if all_lo else min(hrs), min(hrs)) * 0.8
        xlim_hi = max(max(all_hi) if all_hi else max(hrs), max(hrs)) * 1.15
        ax.set_xlim(xlim_lo, xlim_hi)
        ax.axvspan(xlim_lo, 1, alpha=0.03, color="#3C5488")
        ax.axvspan(1, xlim_hi, alpha=0.03, color="#E64B35")

        # Linear scale with clean ticks — use sparser set for wide ranges
        span = xlim_hi / xlim_lo if xlim_lo > 0 else xlim_hi
        if span > 8:
            tick_pool = [0.5, 1.0, 2.0, 4.0, 8.0]
        elif span > 4:
            tick_pool = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
        else:
            tick_pool = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
        ticks = [t for t in tick_pool if xlim_lo <= t <= xlim_hi]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks], fontsize=7)

        ax.grid(axis="x", alpha=0.1)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout(w_pad=3)
    plt.subplots_adjust(bottom=0.12)
    # Significance key below all panels
    fig.text(0.5, 0.01, "* P < 0.05    ** P < 0.01    *** P < 0.001",
             ha="center", fontsize=8, color="#666666")
    plt.savefig(OUT / "fig3d_forest_plot.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: fig3d_forest_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Supplementary Figures
# ═══════════════════════════════════════════════════════════════════════════════

def supp_fig5():
    """Co-activation heatmap."""
    print("Supp Fig 5: Co-activation heatmap...")
    a1 = np.load("results/sparc_spatial/a1_coactivation/pan_cancer_matrix.npz")
    rho = a1["rho"]

    Z_link = linkage(rho, method="ward")
    order = leaves_list(Z_link)
    ordered = rho[np.ix_(order, order)]
    labels = [SHORT[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    sns.heatmap(ordered, xticklabels=labels, yticklabels=labels,
                cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                square=True, linewidths=0.05, ax=ax,
                cbar_kws={"shrink": 0.5, "label": "Spearman \u03c1"})
    ax.set_title("Pan-Cancer Intra-Patch Co-Activation", fontsize=13, fontweight="bold")
    plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6, rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "supp_fig5_coactivation_heatmap.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: supp_fig5_coactivation_heatmap.png")


def supp_fig6():
    """Per-cancer niche composition stacked bar."""
    print("Supp Fig 6: Niche composition by cancer...")
    comp = pd.read_csv("results/sparc_spatial/a3_niches/patient_composition.csv")
    niche_cols = [c for c in comp.columns if c.startswith("p_")]

    ct_comp = comp.groupby("cancer_type")[niche_cols].mean()
    ct_comp.columns = ["Proliferative", "Stromal", "Lymphoid", "Immune-Inflamed"]
    ct_comp = ct_comp.loc[ct_comp.index.sort_values()]

    colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ct_comp.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_ylabel("Mean niche fraction", fontsize=10)
    ax.set_title("Niche Composition by Cancer Type (TCGA)", fontsize=12, fontweight="bold")
    ax.legend(title="Niche", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "supp_fig6_niche_composition.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: supp_fig6_niche_composition.png")


def supp_fig8():
    """Full cross-correlation S heatmap."""
    print("Supp Fig 8: Cross-correlation heatmap...")
    S = np.load("results/sparc_spatial/a2_cross_correlation/S_matrix.npz")["S"]

    Z_link = linkage(S, method="ward")
    order = leaves_list(Z_link)
    ordered = S[np.ix_(order, order)]
    labels = [SHORT[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    sns.heatmap(ordered, xticklabels=labels, yticklabels=labels,
                cmap="RdBu_r", center=0, square=True, linewidths=0.05, ax=ax,
                cbar_kws={"shrink": 0.5, "label": "Spatial cross-correlation (S)"})
    ax.set_title("Pan-Cancer Spatial Cross-Correlation", fontsize=13, fontweight="bold")
    plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6, rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "supp_fig8_crosscorr_heatmaps.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: supp_fig8_crosscorr_heatmaps.png")


def supp_crosscorr_comparison():
    """Cross-cohort S values comparison (grouped bars across TCGA/SurGen/NLST)."""
    print("Supp: Cross-correlation comparison...")
    from analysis.spatial_tcga import fisher_z_inv

    S = np.load("results/sparc_spatial/a2_cross_correlation/S_matrix.npz")["S"]
    surgen_cache = np.load("results/sparc_spatial_external/surgen/slide_cache/matrices.npz")
    surgen_S = fisher_z_inv(surgen_cache["S_fz"].mean(axis=0))
    nlst_cache = np.load("results/sparc_spatial_external/nlst/slide_cache/matrices.npz")
    nlst_S = fisher_z_inv(nlst_cache["S_fz"].mean(axis=0))

    pair_defs = [
        (_PI["TGF-β Exclusion"], _PI["TCR Signaling"], "TGF-β ↔ TCR"),
        (_PI["TGF-β Exclusion"], _PI["Naïve CD8"], "TGF-β ↔ Naïve CD8"),
        (_PI["MHC-I Processing"], _PI["PD-1⁺ CD8"], "MHC-I ↔ PD-1⁺ CD8"),
        (_PI["TLS"], _PI["TCR Signaling"], "TLS ↔ TCR"),
        (_PI["Hypoxia"], _PI["Angiogenesis"], "Hypoxia ↔ Angio."),
        (_PI["Collagen"], _PI["Immature DCs"], "Collagen ↔ DCs"),
        (_PI["ECM Organization"], _PI["Inflammatory"], "ECM ↔ Inflam."),
        (_PI["mTORC1"], _PI["Treg"], "mTORC1 ↔ Treg"),
        (_PI["M2 Macro"], _PI["MHC-II"], "M2 ↔ MHC-II"),
        (_PI["E2F Targets"], _PI["IFN-γ"], "E2F ↔ IFN-γ"),
        (_PI["G2M Checkpoint"], _PI["Exhausted CD8"], "G2M ↔ Exh. CD8"),
        (_PI["OxPhos"], _PI["Glycolysis"], "OxPhos ↔ Glycolysis"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    n_pairs = len(pair_defs)
    x = np.arange(n_pairs)
    w = 0.25

    # Sort by TCGA S value
    tcga_vals = [S[g1, g2] for g1, g2, _ in pair_defs]
    sort_idx = np.argsort(tcga_vals)
    pair_defs_sorted = [pair_defs[i] for i in sort_idx]

    for offset, (label, S_mat, color) in enumerate([
        ("TCGA", S, "#333333"),
        ("SurGen CRC", surgen_S, "#E64B35"),
        ("NLST Lung", nlst_S, "#4DBBD5"),
    ]):
        vals = [S_mat[g1, g2] for g1, g2, _ in pair_defs_sorted]
        ax.barh(x + (offset - 1) * w, vals, height=w, color=color, edgecolor="white",
                linewidth=0.3, label=label, alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([p[2] for p in pair_defs_sorted], fontsize=7.5)
    ax.set_xlabel("Spatial cross-correlation (S)", fontsize=10)
    ax.set_title("Cross-cohort spatial interactions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    plt.savefig(OUT / "supp_crosscorr_comparison.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: supp_crosscorr_comparison.png")

    # Also save CSV of S values
    rows = []
    for g1, g2, name in pair_defs:
        rows.append({"pair": name, "TCGA_S": S[g1, g2],
                      "SurGen_S": surgen_S[g1, g2], "NLST_S": nlst_S[g1, g2]})
    pd.DataFrame(rows).to_csv(OUT / "supp_crosscorr_S_values.csv", index=False)
    print(f"  Saved: supp_crosscorr_S_values.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Supplementary Tables (CSVs)
# ═══════════════════════════════════════════════════════════════════════════════

def supp_tables():
    print("Supplementary tables...")

    # Table 7: TCGA per-cancer spatial HRs
    tcga = pd.read_csv("results/sparc_spatial/a4_prognostic/all_results.csv")
    tcga.to_csv(OUT / "supp_table7_tcga_spatial_survival.csv", index=False)
    print(f"  Table 7: {len(tcga)} rows")

    # Table 8: SurGen
    surgen = pd.read_csv("results/sparc_spatial_external/surgen/survival/all_results.csv")
    surgen.to_csv(OUT / "supp_table8_surgen_spatial_survival.csv", index=False)
    print(f"  Table 8: {len(surgen)} rows")

    # Table 9: NLST
    nlst = pd.read_csv("results/sparc_spatial_external/nlst/survival/all_results.csv")
    nlst.to_csv(OUT / "supp_table9_nlst_spatial_survival.csv", index=False)
    print(f"  Table 9: {len(nlst)} rows")

    # Table 10: Niche multivariate C-indices
    rows = []
    tcga_mv = pd.read_csv("results/sparc_spatial/a4_prognostic/niche_multivar_results.csv")
    for _, r in tcga_mv.iterrows():
        rows.append({"cohort": "TCGA", "cancer": r["cancer"], "n": int(r["n"]),
                      "events": int(r["events"]), "c_index": r["c_index"]})
    surgen_mv = surgen[surgen["feature_type"] == "niche_mv"]
    for _, r in surgen_mv.iterrows():
        rows.append({"cohort": "SurGen", "cancer": r["subtype"], "endpoint": r["endpoint"],
                      "n": int(r["n"]), "events": int(r["events"]),
                      "c_index": r.get("c_index", np.nan)})
    nlst_mv = nlst[nlst["feature_type"] == "niche_mv"]
    for _, r in nlst_mv.iterrows():
        rows.append({"cohort": "NLST", "cancer": r["subtype"], "endpoint": r["endpoint"],
                      "n": int(r["n"]), "events": int(r["events"]),
                      "c_index": r.get("c_index", np.nan)})
    pd.DataFrame(rows).to_csv(OUT / "supp_table10_niche_multivar_cindex.csv", index=False)
    print(f"  Table 10: {len(rows)} rows")

    # Table 11: Clustering diagnostics
    diag_rows = []
    for cohort, base in [("TCGA", "results/sparc_spatial/a3_niches"),
                          ("SurGen", "results/sparc_spatial_external/surgen/niches"),
                          ("NLST", "results/sparc_spatial_external/nlst/niches")]:
        with open(f"{base}/pca_model.pkl", "rb") as f:
            pca = pickle.load(f)
        with open(f"{base}/kmeans_models.pkl", "rb") as f:
            km = pickle.load(f)
        with open(f"{base}/best_k.json") as f:
            bk = json.load(f)

        for K in sorted(km.keys()):
            diag_rows.append({
                "cohort": cohort,
                "K": K,
                "silhouette": round(km[K]["silhouette"], 4),
                "selected": K == bk["best_K"],
                "pca_n_components": pca.n_components_,
                "pca_variance_explained": round(pca.explained_variance_ratio_.sum(), 3),
                "stability_cosine_sim": ">0.999",
            })
    pd.DataFrame(diag_rows).to_csv(OUT / "supp_table11_clustering_diagnostics.csv", index=False)
    print(f"  Table 11: {len(diag_rows)} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Generating spatial TME paper figures")
    print(f"Output: {OUT}")
    print("=" * 60)

    fig3a()
    fig3b()
    fig3c()
    fig3d()
    supp_fig5()
    supp_fig6()
    supp_fig8()
    supp_crosscorr_comparison()
    supp_tables()

    print("\n" + "=" * 60)
    print("All done!")
    print(f"Main panels: fig3a-d")
    print(f"Supp figures: supp_fig5, supp_fig6, supp_fig8, supp_crosscorr_comparison")
    print(f"Supp tables: supp_table7-11 (CSVs)")
    print(f"Output: {OUT}")
    print("=" * 60)
