"""Survival metrics: concordance index, time-dependent AUC, patient-level aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class PatientRecord:
    """One patient's survival record together with a model risk prediction.

    Used by :func:`aggregate_patient_level` to mean-pool slide-level outputs.
    """

    patient_id: str
    time: float
    event: int
    risk: float
    organ: Optional[str] = None
    cancer_type: Optional[str] = None  # e.g., "TCGA-BRCA"


def c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index, computed pairwise.

    A pair ``(i, j)`` is *comparable* iff patient ``i`` had an event and
    patient ``j`` was still at risk at ``time[i]`` (i.e. ``time[j] > time[i]``).
    Of those comparable pairs, the C-index is the fraction in which the higher
    risk score corresponds to the shorter survival time, with ties counted as
    half a concordance.

    Args:
        time:  ``[N]`` survival or censoring times.
        event: ``[N]`` event indicators (``1`` = event observed, ``0`` = censored).
        risk:  ``[N]`` predicted risk scores (higher = worse prognosis).

    Returns:
        Concordance in ``[0, 1]``, or ``NaN`` if no comparable pairs exist
        (e.g. all-censored cohort).
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk)

    n = len(time)
    assert time.shape == event.shape == risk.shape

    num_pairs = 0
    num_concordant = 0
    num_ties = 0

    for i in range(n):
        if not event[i]:
            continue  # censored subject i does not define a pair
        for j in range(n):
            if time[j] <= time[i]:
                continue  # j not at-risk at i's event time
            # valid comparable pair
            num_pairs += 1
            if risk[i] > risk[j]:
                num_concordant += 1
            elif risk[i] == risk[j]:
                num_ties += 1

    if num_pairs == 0:
        return np.nan

    return (num_concordant + 0.5 * num_ties) / num_pairs


def td_auc_simple(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    times: Sequence[float],
) -> Dict[float, float]:
    """Naive (no IPCW) time-dependent AUC at a list of horizons.

    For each cutoff ``t`` the cases are patients with ``event == 1`` and
    ``time ≤ t``, the controls are patients with ``time > t`` (regardless of
    event), and the AUC is the ROC AUC of ``risk`` separating cases from
    controls.

    Useful for cheap training-time monitoring; for paper-grade reporting use
    an IPCW-corrected td-AUC (e.g. ``sksurv.metrics.cumulative_dynamic_auc``).

    Args:
        time:  ``[N]`` survival/censoring times.
        event: ``[N]`` event indicators.
        risk:  ``[N]`` predicted risk scores.
        times: Iterable of time horizons.

    Returns:
        Mapping from each cutoff in ``times`` to its AUC. A cutoff with zero
        cases or zero controls maps to ``NaN``.
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk)

    aucs: Dict[float, float] = {}
    for t in times:
        is_case = (event == 1) & (time <= t)
        is_control = time > t

        mask = is_case | is_control
        if mask.sum() < 10 or is_case.sum() == 0 or is_control.sum() == 0:
            aucs[t] = np.nan
            continue

        y_true = is_case[mask].astype(int)
        y_score = risk[mask]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan
        aucs[t] = float(auc)

    return aucs


def aggregate_patient_level(
    slide_patient_ids: Sequence[str],
    slide_times: Sequence[float],
    slide_events: Sequence[int],
    slide_risks: Sequence[float],
    patient_organs: Optional[Dict[str, str]] = None,
    patient_cancer_types: Optional[Dict[str, str]] = None,
    agg: str = "mean",
) -> List[PatientRecord]:
    """
    Aggregate slide-level predictions to patient-level.

    Args:
        slide_patient_ids: patient ID for each slide.
        slide_times:       survival time for each slide (same per patient).
        slide_events:      event indicator for each slide (same per patient).
        slide_risks:       predicted risk for each slide.
        patient_organs:    optional mapping patient_id -> organ string.
        patient_cancer_types: optional mapping patient_id -> cancer type (e.g., TCGA-BRCA).
        agg:               'mean' or 'max' or 'topk_mean:K'.

    Returns:
        List[PatientRecord] (one per patient).
    """
    slide_patient_ids = np.asarray(slide_patient_ids)
    slide_times = np.asarray(slide_times)
    slide_events = np.asarray(slide_events)
    slide_risks = np.asarray(slide_risks)

    patients = np.unique(slide_patient_ids)
    records: List[PatientRecord] = []

    topk = None
    if agg.startswith("topk_mean:"):
        try:
            topk = int(agg.split(":", 1)[1])
            agg_mode = "topk_mean"
        except Exception:
            raise ValueError(f"Could not parse agg={agg}")
    else:
        agg_mode = agg

    for pid in patients:
        mask = slide_patient_ids == pid
        risks = slide_risks[mask]
        times = slide_times[mask]
        events = slide_events[mask]

        # sanity: time/event should be identical within patient
        t = float(times[0])
        e = int(events[0])

        if agg_mode == "mean":
            patient_risk = float(risks.mean())
        elif agg_mode == "max":
            patient_risk = float(risks.max())
        elif agg_mode == "topk_mean":
            k = min(topk, len(risks))
            topk_vals = np.partition(risks, -k)[-k:]
            patient_risk = float(topk_vals.mean())
        else:
            raise ValueError(f"Unknown agg mode: {agg}")

        organ = patient_organs[pid] if patient_organs is not None and pid in patient_organs else None
        cancer_type = patient_cancer_types[pid] if patient_cancer_types is not None and pid in patient_cancer_types else None
        records.append(
            PatientRecord(
                patient_id=str(pid),
                time=t,
                event=e,
                risk=patient_risk,
                organ=organ,
                cancer_type=cancer_type,
            )
        )

    return records