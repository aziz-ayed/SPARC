"""Discrete-time survival head used by all SPARC models."""

from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


class NLLSurvivalHead(nn.Module):
    """Discrete-time hazard head with negative-log-likelihood training (MCAT-style).

    Maps a slide-level embedding to ``n_bins`` independent hazard probabilities
    and derives the cumulative survival curve, plus a single scalar risk score
    used for C-index ranking.

    Args:
        in_dim: Width of the input slide embedding.
        n_bins: Number of discrete time bins. Bin edges live on the dataset /
                trainer side (``bin_edges`` in :func:`sparc.losses.nll_survival_loss`).

    Forward:
        ``slide_emb``: ``[B, in_dim]`` slide embeddings.

        Returns ``dict`` with:

        - ``hazards``  — ``[B, n_bins]`` per-bin hazard probabilities.
        - ``survival`` — ``[B, n_bins]`` cumulative survival
          ``∏_{j ≤ k} (1 - hazard_j)``.
        - ``risk``     — ``[B]`` scalar risk score, the negative sum of
          survival probabilities (higher = riskier; matches MCAT).
    """

    def __init__(self, in_dim: int, n_bins: int = 4) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(in_dim, n_bins),
        )

    def forward(self, slide_emb: Tensor) -> dict:
        logits = self.classifier(slide_emb)               # [B, n_bins]
        hazards = torch.sigmoid(logits)                   # [B, n_bins]
        survival = torch.cumprod(1 - hazards, dim=1)      # [B, n_bins]
        risk = -torch.sum(survival, dim=1)                # [B]
        return {"hazards": hazards, "survival": survival, "risk": risk}
