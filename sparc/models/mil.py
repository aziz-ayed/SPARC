"""Multiple-instance learning aggregators."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


class AttnMILAggregator(nn.Module):
    """Gated attention multiple-instance learning aggregator (Ilse et al., 2018).

    For each bag (slide) of patch embeddings ``h_i ∈ R^d`` the slide-level
    embedding is

    .. math::

        A_i \\propto w^\\top \\bigl(\\tanh(V h_i) \\odot \\sigma(U h_i)\\bigr),
        \\qquad
        m = \\sum_i A_i \\, h_i,

    where ``A`` is softmax-normalised across the slide's patches.

    Args:
        hidden_dim: Width of the patch-level embeddings (``d``).

    Forward:
        ``h_list``       — list of length ``B`` of ``[N_i, d]`` tensors.
        ``coords_list``  — unused; accepted for interface compatibility with
                           coordinate-aware aggregators.

        Returns ``[B, d]`` slide-level embeddings.

    Note:
        The most recent batch's per-slide attention weights are cached in
        ``_last_mil_attns`` (a list of ``[N_i, 1]`` tensors) for interpretability;
        retrieve them via :meth:`get_last_attention`.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_V = nn.Linear(hidden_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)

    def _single_slide(self, h: Tensor) -> Tensor:
        """Aggregate one slide's patches to a single embedding.

        Args:
            h: ``[N, d]`` patch embeddings for one slide.

        Returns:
            ``[d]`` slide embedding.
        """
        V = torch.tanh(self.attn_V(h))         # [N, d]
        U = torch.sigmoid(self.attn_U(h))      # [N, d]
        A = self.attn_w(V * U)                  # [N, 1]
        A = torch.softmax(A, dim=0)             # attention over patches
        self._last_mil_attn = A.detach()        # cached for interpretability
        return torch.sum(A * h, dim=0)          # [d]

    def forward(
        self,
        h_list: List[Tensor],
        coords_list: Optional[List[Tensor]] = None,
    ) -> Tensor:
        self._last_mil_attns: List[Tensor] = []
        slide_embs: List[Tensor] = []
        for h in h_list:
            slide_embs.append(self._single_slide(h))
            self._last_mil_attns.append(self._last_mil_attn)
        return torch.stack(slide_embs, dim=0)   # [B, d]

    def get_last_attention(self) -> Optional[List[Tensor]]:
        """Return per-slide attention weights from the most recent forward pass.

        Returns:
            A list of ``[N_i, 1]`` attention tensors, or ``None`` if no forward
            pass has run yet.
        """
        return getattr(self, "_last_mil_attns", None)
