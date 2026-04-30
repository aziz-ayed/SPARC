"""Helpers for evaluating SPARC predictions on external cohorts.

The external-cohort notebooks (``treatment_response_*.ipynb``,
``paper_external_figures.ipynb``) consume per-slide ``.npz`` files written by
:mod:`inference.run`, ensemble them across folds, and fit calibrated
ridge-Cox models on the resulting embeddings or risks. This module bundles
the loading, ensembling, and calibration primitives shared across those
notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

#: Default ridge-regularisation grid for Cox-PH calibration.
ALPHAS = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]


def load_perfold_slides(directory) -> Dict[str, Dict[str, np.ndarray]]:
    """Load every per-fold ``.npz`` written by :mod:`inference.run`.

    Args:
        directory: Path to a folder containing one ``<slide_stem>.npz`` per
            slide, each with ``risk`` of shape ``(5,)`` and ``embedding`` of
            shape ``(5, 256)``.

    Returns:
        Mapping ``stem -> {"risk": (5,), "embedding": (5, 256)}`` (both
        cast to ``float32``).
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for f in Path(directory).glob('*.npz'):
        x = np.load(f)
        out[f.stem] = {
            'risk': x['risk'].astype(np.float32),
            'embedding': x['embedding'].astype(np.float32),
        }
    return out


def ensemble_slide(slide_data: Dict[str, np.ndarray]) -> Tuple[float, np.ndarray]:
    """Average a single slide's per-fold predictions across the 5 CV folds.

    Args:
        slide_data: Output of :func:`load_perfold_slides` for one slide.

    Returns:
        ``(mean_risk, mean_embedding)`` — a scalar Python ``float`` and a
        ``(256,)`` NumPy array.
    """
    return float(slide_data['risk'].mean()), slide_data['embedding'].mean(axis=0)


def make_sksurv_y(T, E) -> np.ndarray:
    """Build the structured array expected by :mod:`sksurv` survival models.

    Args:
        T: Iterable of survival/censoring times.
        E: Iterable of event indicators (``1`` = event, ``0`` = censored).

    Returns:
        A structured array of dtype ``[('event', bool), ('time', float)]``.
    """
    return np.array(
        [(bool(e), float(t)) for e, t in zip(E, T)],
        dtype=[('event', bool), ('time', float)],
    )


def ridge_cox_cv(X, T, E, seed=42):
    """Ridge Cox with inner 3-fold CV for alpha selection and outer 5-fold CV
    for out-of-fold risk predictions.

    Args:
        X:    ``[N, D]`` feature matrix (typically the 256-d slide embeddings
              produced by SPARC-Risk or the image-only baseline).
        T, E: Survival times and event indicators.
        seed: Random seed for the outer / inner CV splits.

    Returns:
        ``[N]`` OOF risk predictions, pooled across the 5 outer folds. Higher
        risk = worse prognosis (matches the convention used by
        :func:`compute_cindex` and the rest of the pipeline).
    """
    oof = np.zeros(len(T))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_i, va_i in outer.split(X, E):
        imp = SimpleImputer(strategy='median').fit(X[tr_i])
        sc = StandardScaler().fit(imp.transform(X[tr_i]))
        Xtr = sc.transform(imp.transform(X[tr_i]))
        Xva = sc.transform(imp.transform(X[va_i]))
        best_a, best_c = ALPHAS[0], -1
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 1)
        for a in ALPHAS:
            cs = []
            for itr, iva in inner.split(Xtr, E[tr_i]):
                try:
                    m = CoxPHSurvivalAnalysis(alpha=a)
                    m.fit(Xtr[itr], make_sksurv_y(T[tr_i][itr], E[tr_i][itr]))
                    cs.append(concordance_index_censored(
                        E[tr_i][iva].astype(bool), T[tr_i][iva], m.predict(Xtr[iva]))[0])
                except Exception:
                    cs.append(0.5)
            if np.mean(cs) > best_c:
                best_c, best_a = np.mean(cs), a
        # Fit with best alpha (fallback to higher alpha if singular)
        for a in [best_a] + [x for x in ALPHAS if x > best_a]:
            try:
                m = CoxPHSurvivalAnalysis(alpha=a)
                m.fit(Xtr, make_sksurv_y(T[tr_i], E[tr_i]))
                oof[va_i] = m.predict(Xva)
                break
            except Exception:
                continue
    return oof


def compute_cindex(T, E, risk):
    """C-index in the model's fixed risk direction."""
    return concordance_index_censored(E.astype(bool), T, risk)[0]


def logrank_hr(T, E, risk):
    """Compute HR from median-split log-rank test.
    Returns (HR, p_value). HR > 1 means high-risk group has higher hazard."""
    med = np.median(risk)
    hi = risk >= med
    lo = ~hi
    if hi.sum() < 2 or lo.sum() < 2:
        return np.nan, np.nan

    t_hi, e_hi = T[hi], E[hi].astype(bool)
    t_lo, e_lo = T[lo], E[lo].astype(bool)
    event_times = np.unique(np.concatenate([t_hi[e_hi], t_lo[e_lo]]))

    O1, E1, V = 0.0, 0.0, 0.0
    hr_num, hr_den = 0.0, 0.0
    for et in event_times:
        n1, n2 = np.sum(t_hi >= et), np.sum(t_lo >= et)
        n = n1 + n2
        d1 = np.sum((t_hi == et) & e_hi)
        d2 = np.sum((t_lo == et) & e_lo)
        d = d1 + d2
        if n < 2:
            continue
        O1 += d1
        E1 += d * n1 / n
        V += d * n1 * n2 * (n - d) / (n**2 * (n - 1))
        hr_num += d1 * n2 / n
        hr_den += d2 * n1 / n

    if V < 1e-10:
        return np.nan, 1.0
    from scipy.stats import chi2
    p = float(chi2.sf((O1 - E1)**2 / V, df=1))
    hr = hr_num / hr_den if hr_den > 0 else np.inf
    return hr, p


def _ridge_cox_worker(args):
    """Worker for ``ThreadPoolExecutor``: fit ridge Cox + return OOF + C-index."""
    X, T, E, seed = args
    oof = ridge_cox_cv(X, T, E, seed=seed)
    c = concordance_index_censored(E.astype(bool), T, oof)[0]
    return oof, c


def _bootstrap_cindex(T, E, risk_sq, risk_img, n_boot=1000, seed=42):
    """Bootstrap CIs for paired C-index comparison. Returns dict with CIs and p-value."""
    rng = np.random.RandomState(seed)
    n = len(T)
    boot_sq = np.empty(n_boot)
    boot_img = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if E[idx].sum() < 1:
            boot_sq[b] = boot_img[b] = np.nan
            continue
        boot_sq[b] = compute_cindex(T[idx], E[idx], risk_sq[idx])
        boot_img[b] = compute_cindex(T[idx], E[idx], risk_img[idx])
    valid = ~(np.isnan(boot_sq) | np.isnan(boot_img))
    boot_delta = boot_sq[valid] - boot_img[valid]
    return {
        'sq_lo': float(np.percentile(boot_sq[valid], 2.5)),
        'sq_hi': float(np.percentile(boot_sq[valid], 97.5)),
        'img_lo': float(np.percentile(boot_img[valid], 2.5)),
        'img_hi': float(np.percentile(boot_img[valid], 97.5)),
        'p_value': float(np.mean(boot_delta <= 0)),
    }


def evaluate_cohort(patient_df, endpoints, label='All', n_boot=1000):
    """Evaluate ZS + FT C-index + HR for a patient DataFrame.

    patient_df must have columns:
        risk_sq, risk_img, emb_sq (256-d), emb_img (256-d),
        and for each endpoint: T_{ep}, E_{ep}

    Returns dict with results including bootstrap CIs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    res = {'n': len(patient_df), 'label': label}
    print(f"\n  Evaluating {label} (n={len(patient_df)})...", flush=True)

    # Collect all Ridge Cox tasks to run in parallel
    ft_tasks = {}
    ep_data = {}

    for ep_name, T_col, E_col in endpoints:
        T = patient_df[T_col].values.astype(float)
        E = patient_df[E_col].values.astype(float)
        valid = ~(np.isnan(T) | np.isnan(E) | (T <= 0))
        T, E = T[valid], E[valid]
        n_events = int(E.sum())
        res[f'ev_{ep_name.lower()}'] = n_events

        if len(T) < 10 or n_events < 3:
            print(f"    {ep_name}: skipped (n={len(T)}, events={n_events})", flush=True)
            continue

        risk_sq = patient_df['risk_sq'].values[valid]
        risk_img = patient_df['risk_img'].values[valid]
        emb_sq = np.stack(patient_df['emb_sq'].values)[valid]
        emb_img = np.stack(patient_df['emb_img'].values)[valid]

        ep_data[ep_name] = (T, E, risk_sq, risk_img, emb_sq, emb_img)

        # Zero-shot C-index + bootstrap CIs
        zs_sq = compute_cindex(T, E, risk_sq)
        zs_img = compute_cindex(T, E, risk_img)
        res[f'zs_{ep_name.lower()}_sq'] = round(zs_sq, 4)
        res[f'zs_{ep_name.lower()}_img'] = round(zs_img, 4)

        print(f"    {ep_name} ZS: bootstrapping...", end='', flush=True)
        boot = _bootstrap_cindex(T, E, risk_sq, risk_img, n_boot=n_boot)
        res[f'zs_{ep_name.lower()}_sq_lo'] = round(boot['sq_lo'], 4)
        res[f'zs_{ep_name.lower()}_sq_hi'] = round(boot['sq_hi'], 4)
        res[f'zs_{ep_name.lower()}_img_lo'] = round(boot['img_lo'], 4)
        res[f'zs_{ep_name.lower()}_img_hi'] = round(boot['img_hi'], 4)
        res[f'zs_{ep_name.lower()}_p'] = round(boot['p_value'], 4)

        # HR (instant)
        hr_sq, hr_p_sq = logrank_hr(T, E, risk_sq)
        hr_img, hr_p_img = logrank_hr(T, E, risk_img)
        res[f'hr_{ep_name.lower()}_sq'] = round(hr_sq, 3) if np.isfinite(hr_sq) else None
        res[f'hr_{ep_name.lower()}_img'] = round(hr_img, 3) if np.isfinite(hr_img) else None
        res[f'hr_p_{ep_name.lower()}_sq'] = round(hr_p_sq, 4) if np.isfinite(hr_p_sq) else None
        res[f'hr_p_{ep_name.lower()}_img'] = round(hr_p_img, 4) if np.isfinite(hr_p_img) else None

        zd = zs_sq - zs_img
        print(f" {zs_sq:.3f} [{boot['sq_lo']:.3f},{boot['sq_hi']:.3f}] vs "
              f"{zs_img:.3f} [{boot['img_lo']:.3f},{boot['img_hi']:.3f}] ({zd:+.3f}, p={boot['p_value']:.3f})  "
              f"HR: {hr_sq:.2f} vs {hr_img:.2f}", flush=True)

        # Queue Ridge Cox tasks
        ft_tasks[f'{ep_name}_sq'] = (emb_sq, T, E, 42)
        ft_tasks[f'{ep_name}_img'] = (emb_img, T, E, 42)

    # Run all Ridge Cox fits in parallel
    if ft_tasks:
        print(f"    Running {len(ft_tasks)} Ridge Cox fits in parallel...", end='', flush=True)
        ft_results = {}
        with ThreadPoolExecutor(max_workers=len(ft_tasks)) as pool:
            futures = {pool.submit(_ridge_cox_worker, args): key
                       for key, args in ft_tasks.items()}
            for fut in as_completed(futures):
                key = futures[fut]
                ft_results[key] = fut.result()
        print(" done", flush=True)

        for ep_name in ep_data:
            ep = ep_name.lower()
            T, E = ep_data[ep_name][0], ep_data[ep_name][1]
            oof_sq, ft_sq = ft_results[f'{ep_name}_sq']
            oof_img, ft_img = ft_results[f'{ep_name}_img']
            res[f'ft_{ep}_sq'] = round(ft_sq, 4)
            res[f'ft_{ep}_img'] = round(ft_img, 4)

            # Bootstrap CIs on the OOF predictions (fast — no re-fitting)
            boot_ft = _bootstrap_cindex(T, E, oof_sq, oof_img, n_boot=n_boot)
            res[f'ft_{ep}_sq_lo'] = round(boot_ft['sq_lo'], 4)
            res[f'ft_{ep}_sq_hi'] = round(boot_ft['sq_hi'], 4)
            res[f'ft_{ep}_img_lo'] = round(boot_ft['img_lo'], 4)
            res[f'ft_{ep}_img_hi'] = round(boot_ft['img_hi'], 4)
            res[f'ft_{ep}_p'] = round(boot_ft['p_value'], 4)

            fd = ft_sq - ft_img
            print(f"    {ep_name} FT: {ft_sq:.3f} [{boot_ft['sq_lo']:.3f},{boot_ft['sq_hi']:.3f}] vs "
                  f"{ft_img:.3f} [{boot_ft['img_lo']:.3f},{boot_ft['img_hi']:.3f}] ({fd:+.3f})", flush=True)

    return res


def print_results(res):
    """Pretty-print evaluation results."""
    label = res.get('label', 'All')
    n = res['n']
    print(f"\n  {label} (n={n}):")
    for ep in ['dss', 'os']:
        ev = res.get(f'ev_{ep}', '?')
        zs_sq = res.get(f'zs_{ep}_sq', '-')
        zs_img = res.get(f'zs_{ep}_img', '-')
        ft_sq = res.get(f'ft_{ep}_sq', '-')
        ft_img = res.get(f'ft_{ep}_img', '-')
        hr_sq = res.get(f'hr_{ep}_sq', '-')
        hr_img = res.get(f'hr_{ep}_img', '-')
        if zs_sq == '-':
            continue
        zd = zs_sq - zs_img if isinstance(zs_sq, float) else 0
        fd = ft_sq - ft_img if isinstance(ft_sq, float) else 0
        print(f"    {ep.upper():>3s} (ev={ev:>3d})  ZS: {zs_sq:.3f} vs {zs_img:.3f} ({zd:+.3f})"
              f"  FT: {ft_sq:.3f} vs {ft_img:.3f} ({fd:+.3f})"
              f"  HR: {hr_sq} vs {hr_img}")


def save_results(cohort_name, breakdowns, models, save_path):
    """Save results to JSON."""
    out = {
        'cohort': cohort_name,
        'models': models,
        'breakdowns': {r['label']: r for r in breakdowns},
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {save_path}")
