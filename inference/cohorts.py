"""Per-cohort declarative configuration for external-cohort inference.

Each ``CohortSpec`` describes where a cohort's features and metadata live, plus
how to map a slide's identifier to the TCGA cancer-type index used by the model.

Add a new cohort by adding an entry to ``COHORTS`` below — no changes to
``inference/run.py`` are required.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from sparc.data.dataset import CANCER_TYPE_TO_IDX


@dataclass
class CohortSpec:
    """Everything ``inference/run.py`` needs to know about an external cohort."""

    name: str
    emb_dir: Path                                            # per-slide image-feature h5
    gep_dir: Path                                            # per-slide gene-program h5
    cancer_type_map: Callable[[Iterable[str]], Dict[str, int]]
    out_subdir: str                                          # results/<out_subdir>/<run>/<slide>.npz


# ─── Cohort-specific cancer-type-mapping helpers ─────────────────────────────

LUAD_IDX = CANCER_TYPE_TO_IDX["TCGA-LUAD"]
LUSC_IDX = CANCER_TYPE_TO_IDX["TCGA-LUSC"]
COAD_IDX = CANCER_TYPE_TO_IDX["TCGA-COAD"]
READ_IDX = CANCER_TYPE_TO_IDX["TCGA-READ"]

LUAD_MORPHS = {8140, 8250, 8255, 8260, 8265, 8480, 8490, 8310, 8323}
LUSC_MORPHS = {8070, 8071, 8072, 8052, 8083}


def _nlst_uuid_to_ct(uuids: Iterable[str]) -> Dict[str, int]:
    """Map NLST slide UUIDs to LUAD/LUSC indices using ICD-O-3 morphology codes."""
    dicom_csv = Path(os.environ.get(
        "SPARC_NLST_DICOM_CSV", "data/nlst_dicom_file_list.csv"))
    canc_csv = Path(os.environ.get(
        "SPARC_NLST_CANC_CSV", "data/nlst_outcomes/nlst_780_canc_idc_20210527.csv"))

    dicom = pd.read_csv(dicom_csv)
    dicom["uuid"] = dicom["file_name"].str.replace(".dcm", "", regex=False)
    dicom["pid"] = dicom["directory"].str.extract(r"/nlst/(\d+)/").astype(int)
    uuid_to_pid = dicom.set_index("uuid")["pid"].to_dict()

    canc = pd.read_csv(canc_csv)[["pid", "lc_morph"]].drop_duplicates("pid")
    pid_to_morph = canc.set_index("pid")["lc_morph"].to_dict()

    def morph_to_idx(m):
        if pd.isna(m):
            return LUAD_IDX
        try:
            m = int(m)
        except (ValueError, TypeError):
            return LUAD_IDX
        return LUSC_IDX if m in LUSC_MORPHS else LUAD_IDX

    return {
        u: morph_to_idx(pid_to_morph.get(uuid_to_pid.get(u), np.nan))
        for u in uuids
    }


def _surgen_stem_to_ct(stems: Iterable[str]) -> Dict[str, int]:
    """Map SurGen slide stems to COAD/READ indices using tumour-site labels."""
    clinical_csv = Path(os.environ.get(
        "SPARC_SURGEN_CLINICAL", "data/surgen_outcomes/SR386_labels.csv"))
    clin = pd.read_csv(clinical_csv)
    case_to_site = {
        row["case_id"]: str(row.get("site_of_tumour_grouping", "")).lower()
        for _, row in clin.iterrows()
    }
    out: Dict[str, int] = {}
    for stem in stems:
        m = re.search(r"T(\d+)", stem)
        if m:
            site = case_to_site.get(int(m.group(1)), "")
            out[stem] = READ_IDX if "rect" in site else COAD_IDX
        else:
            out[stem] = COAD_IDX
    return out


def _omer_stem_to_ct(stems: Iterable[str]) -> Dict[str, int]:
    """All Ömer cohort slides are CRC; map every stem to TCGA-COAD."""
    return {s: COAD_IDX for s in stems}


# ─── Cohort registry ─────────────────────────────────────────────────────────

COHORTS: Dict[str, CohortSpec] = {
    "nlst": CohortSpec(
        name="nlst",
        emb_dir=Path(os.environ.get(
            "SPARC_NLST_EMB", "features/nlst/hoptimus1")),
        gep_dir=Path(os.environ.get(
            "SPARC_NLST_GEP", "features/nlst/predicted_programs_transformer")),
        cancer_type_map=_nlst_uuid_to_ct,
        out_subdir="nlst_inference",
    ),
    "surgen": CohortSpec(
        name="surgen",
        emb_dir=Path(os.environ.get(
            "SPARC_SURGEN_EMB", "features/surgen/hoptimus1")),
        gep_dir=Path(os.environ.get(
            "SPARC_SURGEN_GEP", "features/surgen/predicted_programs_transformer")),
        cancer_type_map=_surgen_stem_to_ct,
        out_subdir="surgen_inference",
    ),
    "omer": CohortSpec(
        name="omer",
        emb_dir=Path(os.environ.get(
            "SPARC_OMER_EMB", "features/omer/hoptimus1")),
        gep_dir=Path(os.environ.get(
            "SPARC_OMER_GEP", "features/omer/predicted_programs_transformer")),
        cancer_type_map=_omer_stem_to_ct,
        out_subdir="omer_inference",
    ),
}
