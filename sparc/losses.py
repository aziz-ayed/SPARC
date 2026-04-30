"""Survival losses used during training.

The paper's canonical configs use :func:`nll_survival_loss` exclusively;
:func:`cox_loss` is kept available because ``train.py`` supports an optional
``cox_mix`` term that adds a weighted Cox partial-likelihood contribution to
the NLL loss. With ``cox_mix = 0`` (the default), :func:`cox_loss` is dead at
runtime but the import must still resolve.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def cox_loss(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    """Negative Cox partial log-likelihood (Breslow tie handling).

    Args:
        risk:  ``[B]`` predicted log-risk scores (higher = riskier).
        time:  ``[B]`` survival/censoring times in days.
        event: ``[B]`` event indicators (``1`` = event observed, ``0`` = censored).

    Returns:
        Scalar loss tensor. Mean negative log-likelihood divided by the number
        of observed events (with a small epsilon to avoid division-by-zero
        when a batch contains only censored samples).
    """
    idx = torch.argsort(time, descending=True)
    time = time[idx]
    event = event[idx]
    risk = risk[idx]

    hazard_ratio = torch.exp(risk)
    log_cum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0))
    log_risk = risk - log_cum_hazard
    return -(log_risk * event).sum() / (event.sum() + 1e-8)


def nll_survival_loss(
    hazards: Tensor,
    survival: Tensor,
    time: Tensor,
    event: Tensor,
    bin_edges: Tensor,
    alpha: float = 0.0,
) -> Tensor:
    """Discrete-time negative log-likelihood for binned hazard models (MCAT-style).

    Patients are assigned to a discrete time bin based on their event/censoring
    time. The likelihood factorises as

    - **Uncensored** (``event = 1``): :math:`-\\log[S(t_{k-1})\\,h(t_k)]`
      where ``t_k`` is the patient's bin and ``S, h`` are predicted survival
      and hazard at that bin.
    - **Censored**   (``event = 0``): :math:`-\\log S(t_k)`.

    Args:
        hazards:   ``[B, n_bins]`` predicted per-bin hazards (each in ``[0, 1]``).
        survival:  ``[B, n_bins]`` survival probabilities, the running product
                   ``∏_{j ≤ k} (1 - hazard_j)``.
        time:      ``[B]`` survival/censoring times.
        event:     ``[B]`` event indicators.
        bin_edges: ``[n_bins + 1]`` boundaries used to discretise ``time``.
        alpha:     Optional up-weighting of uncensored patients in ``[0, ∞)``.
                   ``0`` (default) gives standard NLL.

    Returns:
        Scalar mean loss across the batch.
    """
    batch_size = time.shape[0]
    n_bins = hazards.shape[1]
    device = hazards.device

    # Patient i's event/censoring time falls in bin bin_idx[i]
    bin_idx = torch.searchsorted(bin_edges[1:], time.contiguous()).clamp(max=n_bins - 1)

    batch_indices = torch.arange(batch_size, device=device)
    S_bin = survival[batch_indices, bin_idx]
    h_bin = hazards[batch_indices, bin_idx]

    eps = 1e-8
    log_survival = torch.log(S_bin + eps)
    log_hazard = torch.log(h_bin + eps)
    log_1_minus_hazard = torch.log(1 - h_bin + eps)

    # Uncensored: -log(f(t)) = -log(S(t_{k-1}) * h(t_k))
    #                        = -[log S(t_k) - log(1 - h(t_k)) + log h(t_k)]
    uncensored_loss = -(log_survival - log_1_minus_hazard + log_hazard)
    censored_loss = -log_survival

    loss = event * uncensored_loss + (1 - event) * censored_loss
    if alpha > 0:
        weights = event * (1 + alpha) + (1 - event)
        loss = loss * weights
    return loss.mean()
