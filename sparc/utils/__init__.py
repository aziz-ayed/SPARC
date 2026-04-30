"""Cross-validation, metrics, external evaluation, and reproducibility.

- :mod:`sparc.utils.splits`        — load 5-fold cross-validation splits.
- :mod:`sparc.utils.metrics`       — Harrell C-index, time-dependent AUC,
                                     patient-level aggregation.
- :mod:`sparc.utils.external_eval` — utilities for external-cohort inference
                                     (per-fold loading, ridge-Cox calibration,
                                     ensemble scoring).
- :mod:`sparc.utils.seed`          — deterministic seeding for torch/numpy/python.
"""
