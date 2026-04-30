# Data layout

This directory holds the cross-validation split file and the clinical-outcome
table that the configs reference. It does **not** ship with the patch-level
features (those are large and per-cohort) — see below for how to obtain them.

## Files in this directory

| File | Description |
|---|---|
| `mmp_hybrid_splits_v2_20cancer.csv` | 5-fold cross-validation splits over 20 TCGA cancer types. Each row is one slide; columns include `slide_id`, `patient_id`, `cancer_type`, `fold`, `split` (train / val / test). |
| `clinical_dss.csv` | Disease-specific survival labels per patient. Columns: `patient_id`, `dss_time` (days), `dss_event` (0/1), `cancer_type`, plus auxiliary covariates used by some downstream analyses. |

Both files were derived from TCGA clinical data and the published MMP splits
(Mahmood et al.). See `scripts/generate_mmp_hybrid_splits.py` in the source
research repo for the script that produced the splits CSV.

## Patch-level features

The configs reference two directories, neither of which is committed here:

| Path | Content |
|---|---|
| `features/hoptimus1` | Per-slide HDF5 files of patch features extracted with the H-optimus-1 foundation model at 20× / 224 px. One `.h5` per slide containing `features` (`N×1536`) and `coords` (`N×2`). |
| `predicted_programs_transformers` | Per-slide HDF5 files of the 40 SPARC-Map gene-program scores per patch. One `.h5` per slide containing `features` (`N×40`) and `coords` (`N×2`). |

To run training or inference, point either:

* the YAML config (`data.img_feature_dir`, `data.gep_feature_dir`), or
* the environment variables `SPARC_IMG_FEATURE_DIR` and `SPARC_GEP_FEATURE_DIR`,

at the directories holding these files on your machine.

### How to obtain the features

* **Image features** are produced by running the H-optimus-1 backbone on
  trident-detected tissue patches. See `inference/feature_extraction.md`.
* **Gene-program scores** are produced by SPARC-Map (this repo) — run
  inference with the SPARC-Map architecture on the same tissue patches
  to obtain a 40-dim score per patch.

## External cohorts

External cohorts (NLST, SurGen, Yale Breast, Ovarian) follow the same
two-directory convention. Cohort-specific layouts and DICOM/CZI handling
are described in `inference/cohorts.py`.