"""Patch-level fusion modules used by the SPARC paper.

Two architectures live here:

- :class:`ImageOnlyFusion`        — projects ``hoptimus1`` features into the
                                    shared hidden space; used by the
                                    image-only baseline.
- :class:`SignatureQueryFusion`   — SPARC-Risk's main fusion: each gene
                                    program is a learned query that attends
                                    to local image patches.

Plus :class:`GatedAttentionPool`, a gated-attention pooler reused inside
``SignatureQueryFusion`` for the per-signature → slide reduction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

Tensor = torch.Tensor


class GatedAttentionPool(nn.Module):
    """Gated attention pooling (MCAT / CLAM).

    Uses two parallel branches:

    - **Tanh branch** learns the per-token attention features.
    - **Sigmoid branch** learns per-token gate values.

    The pre-softmax score for token ``i`` is ``w_c (tanh(W_a x_i) ⊙ σ(W_b x_i))``,
    which is more expressive than vanilla softmax attention while still being
    a single-layer module.

    Args:
        L:         Input feature width.
        D:         Hidden width inside the gated branches.
        dropout:   Dropout applied after each branch's first linear.
        n_classes: Number of independent attention scores per token (typically
                   ``1`` for slide-level pooling, more for multi-task variants).
    """

    def __init__(self, L: int, D: int, dropout: float = 0.25, n_classes: int = 1) -> None:
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute pre-softmax attention scores.

        Args:
            x: ``[B, N, L]`` input features.

        Returns:
            ``(A, x)`` where ``A`` has shape ``[B, N, n_classes]`` (pre-softmax
            scores) and ``x`` is the unchanged input (returned for caller
            convenience when chaining).
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x


class ImageOnlyFusion(nn.Module):
    """Image-only fusion: a single linear projection from image features to ``hidden_dim``.

    Used by the image-only baseline. The ``gep_feats`` and ``coords`` arguments
    of ``forward`` are accepted (and ignored) so this module can stand in for
    multimodal fusions wherever ``SlideSurvivalModel`` calls ``self.fusion(...)``.

    Args:
        img_dim:    Input image-feature width (e.g. ``1536`` for ``hoptimus1``).
        hidden_dim: Output width.

    Forward:
        ``img_feats`` — list of ``[N_i, img_dim]`` tensors.
        ``gep_feats`` — ignored.
        ``coords``    — ignored.

        Returns a list of ``[N_i, hidden_dim]`` tensors.
    """

    def __init__(self, img_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)

    def forward(
        self,
        img_feats: List[Tensor],
        gep_feats: Optional[List[Tensor]] = None,
        coords: Optional[List[Tensor]] = None,
    ) -> List[Tensor]:
        return [self.img_proj(x) for x in img_feats]


class SignatureQueryFusion(nn.Module):
    """
    Signature-as-Query Fusion with Residual Image Pathway.

    Each biological program (e.g., hypoxia, angiogenesis, immune infiltration)
    queries local image patches independently, producing a biological enrichment
    signal that is ADDED to the direct image representation via a learned gate.

    This guarantees the model is at least as good as image-only: the gate starts
    near zero (image-only) and opens only if biological programs improve prediction.

    Architecture:
        Image pathway (residual):
            img -> LayerNorm -> img_direct_proj -> h_img  [N, d]

        Biological enrichment pathway:
            1. Each signature value projected: [N, K, 1] -> [N, K, d]
            2. Signatures interact via transformer (captures co-occurrence)
            3. Image patches projected to K/V: [N, D_img] -> [N, d]
            4. Each signature attends to k-NN image neighbors: [N, K, d]
            5. Pool across signatures: [N, K, d] -> [N, d]
            6. Final projection -> h_bio  [N, d]

        Fusion:
            out = h_img + sigmoid(gate) * h_bio

    The learned gate (scalar) quantifies how much biological signal improves
    prediction beyond morphology alone. Attention weights [N, K, k] provide
    per-program spatial heatmaps for interpretability.
    """

    def __init__(
        self,
        img_dim: int,
        gep_dim: int,  # Number of signatures (e.g., 40)
        hidden_dim: int,
        num_heads: int = 4,
        k_nn: int = 16,
        modality_dropout: float = 0.0,
        signature_pool: str = "mean",  # "mean", "max", "attn", or "gated_attn"
        use_signature_transformer: bool = True,  # Can disable to save memory
        use_checkpoint: bool = True,  # Gradient checkpointing for memory efficiency
        chunk_size: int = 2048,  # Process patches in chunks to cap memory usage
        use_gep_residual: bool = False,  # Add gated residual to preserve GEP signal
        use_residual: bool = True,  # Residual image pathway with learned gate
        num_cancer_types: int = 33,  # Per-cancer gates when use_residual=True
        cancer_conditioning: bool = False,  # Kept for backward compat
        gate_init: float = -2.0,  # Initial logit for fusion gate (sigmoid(x): -2→0.12, 0→0.5)
        cancer_query_scaling: bool = False,  # Per-cancer, per-program scaling on GEP inputs
        deep_img_proj: bool = False,  # Use 2-layer MLP for image path (matches ImageOnlyV2 capacity)
        attn_dropout: float = 0.0,  # Dropout on attention weights in cross-attention
        n_cross_attn_layers: int = 1,  # Number of cross-attention layers (stacked decoder-style)
        orthogonal_queries: bool = False,  # Use per-program orthogonal embeddings instead of shared MLP
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gep_dim = gep_dim  # Number of signatures
        self.k_nn = k_nn
        self.modality_dropout = modality_dropout
        self.signature_pool = signature_pool
        self.use_signature_transformer = use_signature_transformer
        self.use_checkpoint = use_checkpoint
        self.chunk_size = chunk_size
        self.use_gep_residual = use_gep_residual
        self.use_residual = use_residual
        self.num_cancer_types = num_cancer_types
        self.cancer_query_scaling = cancer_query_scaling
        self.n_cross_attn_layers = n_cross_attn_layers
        self.orthogonal_queries = orthogonal_queries

        # === Per-program orthogonal embeddings (optional) ===
        # Instead of shared MLP (scalar→d), each program gets a unique direction.
        # GEP value scales the embedding: Q_k = gep_k * program_embed[k]
        if orthogonal_queries:
            self.program_embed = nn.Parameter(torch.empty(gep_dim, hidden_dim))
            nn.init.orthogonal_(self.program_embed)

        # === Direct image pathway (residual) ===
        if use_residual:
            # Preserves the image-only representation; GEP enrichment is added on top
            if deep_img_proj:
                # 2-layer MLP matching ImageOnlyFusionV2 capacity
                self.img_direct_proj = nn.Sequential(
                    nn.Linear(img_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                self.img_direct_proj = nn.Linear(img_dim, hidden_dim)
            # Per-cancer learnable gate: how much biological enrichment to add.
            # gate_init controls starting point: -2→sigmoid≈0.12 (near image-only), 0→0.5 (equal)
            # Each cancer type learns its own gate value.
            self.fusion_gate = nn.Parameter(torch.full((num_cancer_types,), gate_init))

        # LayerNorm on inputs
        self.img_norm = nn.LayerNorm(img_dim)

        # Project each signature value to hidden_dim
        # Input: [N, K, 1] -> Output: [N, K, d]
        if not orthogonal_queries:
            self.signature_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Per-cancer, per-program scaling: each cancer learns which programs to up/downweight
        if cancer_query_scaling:
            # [C, K] logits. 2*sigmoid(0) = 1.0 → neutral at init.
            self.cancer_program_scale = nn.Parameter(
                torch.zeros(num_cancer_types, gep_dim)
            )

        # Optional: let signatures interact before querying (captures co-occurrence)
        if use_signature_transformer:
            self.signature_transformer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
            )

        # Image projections for K/V
        self.k_proj = nn.Linear(img_dim, hidden_dim)
        self.v_proj = nn.Linear(img_dim, hidden_dim)

        # Cross-attention: signatures query image patches (stacked decoder-style)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=attn_dropout,
            )
            for _ in range(n_cross_attn_layers)
        ])
        if n_cross_attn_layers > 1:
            self.cross_attn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(n_cross_attn_layers)
            ])

        # Gated residual: preserve biological signal with per-signature learnable gates
        # Gate initialized to 1.0 so model starts with full GEP signal preservation
        if use_gep_residual:
            self.gep_gate = nn.Parameter(torch.ones(gep_dim))  # [K] one gate per signature

        # Pooling across signatures (if using attention pooling)
        if signature_pool == "attn":
            self.pool_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif signature_pool == "gated_attn":
            # Gated attention pooling (MCAT-style)
            self.gated_pool = GatedAttentionPool(
                L=hidden_dim,
                D=hidden_dim // 2,
                dropout=0.25,
                n_classes=1,
            )

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def build_knn_graph(coords: Tensor, k: int) -> Tensor:
        """Build k-NN graph from coordinates."""
        N = coords.size(0)
        if N <= 1:
            return coords.new_zeros((N, 1), dtype=torch.long)

        with torch.no_grad():
            dist = torch.cdist(coords, coords, p=2)
            inf = torch.tensor(float("inf"), device=coords.device, dtype=dist.dtype)
            dist.fill_diagonal_(inf)

            k_eff = min(k, N - 1)
            _, neigh_idx = torch.topk(dist, k=k_eff, dim=1, largest=False)

            if k_eff < k:
                pad_size = k - k_eff
                first_neigh = neigh_idx[:, :1].expand(N, pad_size)
                neigh_idx = torch.cat([neigh_idx, first_neigh], dim=1)

        return neigh_idx.long()

    def _process_chunk(
        self,
        gep_chunk: Tensor,
        K_img_full: Tensor,
        V_img_full: Tensor,
        idx_chunk: Tensor,
        program_scale: Tensor = None,
    ) -> Tensor:
        """
        Process a single chunk - designed to be wrapped in checkpoint.

        Args:
            gep_chunk: [chunk, K] GEP values for this chunk
            K_img_full: [N, d] pre-computed image keys for full slide
            V_img_full: [N, d] pre-computed image values for full slide
            idx_chunk: [chunk, k] neighbor indices for this chunk
            program_scale: [K] per-program scaling (from cancer_query_scaling)

        Returns: [chunk, hidden_dim] fused features for this chunk
        """
        # === Apply per-cancer program scaling ===
        if program_scale is not None:
            gep_chunk = gep_chunk * program_scale  # [chunk, K] * [K]

        # === Signature Queries for this chunk ===
        if self.orthogonal_queries:
            # Each program has a unique orthogonal direction, scaled by its GEP value
            # gep_chunk: [chunk, K] -> [chunk, K, 1] * [K, d] -> [chunk, K, d]
            Q = gep_chunk.unsqueeze(-1) * self.program_embed  # [chunk, K, d]
        else:
            gep_expanded = gep_chunk.unsqueeze(-1)  # [chunk, K, 1]
            Q = self.signature_proj(gep_expanded)  # [chunk, K, d]

        # Let signatures interact (captures co-occurrence patterns)
        if self.use_signature_transformer:
            Q = self.signature_transformer(Q)  # [chunk, K, d]

        # === Gather neighbors for this chunk ===
        K_neigh = K_img_full[idx_chunk]  # [chunk, k, d]
        V_neigh = V_img_full[idx_chunk]  # [chunk, k, d]

        # === Cross-Attention (stacked decoder-style) ===
        attended = Q
        for i, ca_layer in enumerate(self.cross_attn_layers):
            attn_out, _ = ca_layer(attended, K_neigh, V_neigh, need_weights=False)
            if self.n_cross_attn_layers > 1:
                attended = self.cross_attn_norms[i](attended + attn_out)  # residual + LN
            else:
                attended = attn_out

        # === Gated Residual: preserve biological signal ===
        # attended is image-refined, Q is the original biological query
        # The gate learns how much raw biology to preserve per signature
        if self.use_gep_residual:
            # gep_gate: [K] -> [1, K, 1] for broadcasting with [chunk, K, d]
            gate = self.gep_gate.view(1, -1, 1)
            attended = attended + gate * Q  # [chunk, K, d]

        # === Pool across signatures ===
        if self.signature_pool == "mean":
            pooled = attended.mean(dim=1)  # [chunk, d]
        elif self.signature_pool == "max":
            pooled = attended.max(dim=1)[0]  # [chunk, d]
        elif self.signature_pool == "attn":
            attn_scores = self.pool_attn(attended)  # [chunk, K, 1]
            attn_scores = torch.softmax(attn_scores, dim=1)
            pooled = (attended * attn_scores).sum(dim=1)  # [chunk, d]
        elif self.signature_pool == "gated_attn":
            # Gated attention pooling (MCAT-style)
            A, h = self.gated_pool(attended)  # A: [chunk, K, 1], h: [chunk, K, d]
            A = torch.softmax(A, dim=1)  # [chunk, K, 1]
            pooled = (h * A).sum(dim=1)  # [chunk, d]
        else:
            raise ValueError(f"Unknown signature_pool: {self.signature_pool}")

        # Final projection
        return self.out_proj(pooled)

    def _fuse_single(
        self,
        img: Tensor,
        gep: Tensor,
        coords: Tensor,
        training: bool,
        cancer_type_idx: Optional[int] = None,
    ) -> Tensor:
        """
        img:    [N, D_img]
        gep:    [N, K] where K = number of signatures
        coords: [N, 2]
        cancer_type_idx: int or None - cancer type index for this slide (per-cancer gating)

        returns: [N, hidden_dim] fused patch features

        Uses chunked processing with gradient checkpointing to cap memory usage.
        """
        device = img.device
        N = img.size(0)

        if N == 0:
            return img.new_zeros((1, self.hidden_dim))

        if coords is None:
            raise ValueError("SignatureQueryFusion requires coords but got None.")

        # === Pre-compute global structures (no gradients needed for these) ===
        with torch.no_grad():
            neigh_idx = self.build_knn_graph(coords, self.k_nn).to(device)  # [N, k]

        img_normed = self.img_norm(img)  # [N, D_img]

        # Pre-project K/V for full slide (avoids redundant computation in chunks)
        K_img_full = self.k_proj(img_normed)  # [N, d]
        V_img_full = self.v_proj(img_normed)  # [N, d]

        # === DDP-safe cancer program scaling ===
        if self.cancer_query_scaling:
            all_scales = 2.0 * torch.sigmoid(self.cancer_program_scale)  # [C, K]
            if cancer_type_idx is not None:
                program_scale = all_scales[cancer_type_idx]  # [K]
            else:
                program_scale = all_scales.mean(dim=0)
        else:
            program_scale = None

        # === Process in chunks with gradient checkpointing ===
        out_chunks = []
        attn_chunks = []  # For eval mode
        pool_attn_chunks = []  # For eval mode: program importance [chunk, K, 1]
        attended_norm_chunks = []  # For eval mode: feature norms [chunk, K]

        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)

            gep_chunk = gep[i:end]  # [chunk, K]
            idx_chunk = neigh_idx[i:end]  # [chunk, k]

            if training and self.use_checkpoint:
                # Checkpoint entire chunk processing - recompute during backward
                chunk_out = checkpoint(
                    self._process_chunk,
                    gep_chunk,
                    K_img_full,
                    V_img_full,
                    idx_chunk,
                    program_scale,
                    use_reentrant=False,
                )
            else:
                chunk_out = self._process_chunk(
                    gep_chunk, K_img_full, V_img_full, idx_chunk, program_scale
                )

                # Store attention weights for interpretability (eval mode only)
                if not training:
                    # Re-run to get attention weights (only during eval)
                    if self.orthogonal_queries:
                        Q = gep_chunk.unsqueeze(-1) * self.program_embed
                    else:
                        gep_expanded = gep_chunk.unsqueeze(-1)
                        Q = self.signature_proj(gep_expanded)
                    if self.use_signature_transformer:
                        Q = self.signature_transformer(Q)
                    K_neigh = K_img_full[idx_chunk]
                    V_neigh = V_img_full[idx_chunk]
                    # Use last cross-attention layer for attention weights
                    attended, attn_weights = self.cross_attn_layers[-1](Q, K_neigh, V_neigh, need_weights=True)
                    attn_chunks.append(attn_weights.detach())
                    # Pool attention: which programs matter per patch
                    if self.signature_pool == "attn":
                        pool_scores = torch.softmax(self.pool_attn(attended), dim=1)  # [chunk, K, 1]
                        pool_attn_chunks.append(pool_scores.detach())
                    # Attended feature norms: how much each program contributes
                    # ||attended[:, k, :]||_2 captures program importance via feature magnitude
                    attended_norm_chunks.append(attended.detach().norm(dim=-1))  # [chunk, K]

            out_chunks.append(chunk_out)

        # === Concatenate all chunks ===
        h_bio = torch.cat(out_chunks, dim=0)  # [N, d] - biological enrichment

        # === Residual fusion or direct output ===
        if self.use_residual:
            h_img_direct = self.img_direct_proj(img_normed)  # [N, d] - direct image
            # Per-cancer gate
            # IMPORTANT: compute sigmoid on the FULL gate vector before indexing.
            # Indexing first (fusion_gate[idx]) causes unused elements to have no gradient,
            # which breaks DDP (find_unused_parameters=False) when different ranks see
            # different cancer types in a gradient accumulation window.
            all_gates = torch.sigmoid(self.fusion_gate)  # [num_cancer_types]
            if cancer_type_idx is not None:
                gate = all_gates[cancer_type_idx]  # scalar
            else:
                gate = all_gates.mean()  # fallback
            # Modality dropout: randomly drop one pathway to force both to be
            # independently predictive. Applied at fusion step (not input) so
            # both pathways are computed normally and all params stay in the
            # computation graph for DDP.
            if training and self.modality_dropout > 0 and torch.rand(1).item() < self.modality_dropout:
                if torch.rand(1).item() < 0.5:
                    # Drop bio: force image pathway to be independently good
                    out = h_img_direct + (gate * h_bio) * 0
                else:
                    # Drop image: force bio pathway to carry survival signal
                    out = h_img_direct * 0 + gate * h_bio
            else:
                out = h_img_direct + gate * h_bio  # [N, d]
            self._last_gate_value = gate.item()

            # Store norms for interpretability (train + eval)
            self._last_h_img_norm = h_img_direct.detach().norm(dim=-1).mean().item()
            self._last_h_bio_norm = h_bio.detach().norm(dim=-1).mean().item()
            self._last_effective_ratio = (gate * self._last_h_bio_norm) / (self._last_h_img_norm + 1e-8)
            if isinstance(self._last_effective_ratio, torch.Tensor):
                self._last_effective_ratio = self._last_effective_ratio.item()
        else:
            out = h_bio  # Original behavior: no residual

        # Store attention weights for interpretability (eval mode only)
        if not training and attn_chunks:
            self._last_attn_weights = torch.cat(attn_chunks, dim=0)  # [N, K, k]
            self._last_neigh_idx = neigh_idx.detach()  # [N, k]
        if not training and pool_attn_chunks:
            self._last_pool_attn = torch.cat(pool_attn_chunks, dim=0)  # [N, K, 1]
        if not training and attended_norm_chunks:
            self._last_attended_norms = torch.cat(attended_norm_chunks, dim=0)  # [N, K]

        return out

    def forward(
        self,
        img_feats: List[Tensor],
        gep_feats: List[Tensor],
        coords: Optional[List[Tensor]] = None,
        cancer_type_idx: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Args:
            img_feats: list[Tensor [N_i, D_img]] - image features per slide
            gep_feats: list[Tensor [N_i, K]] - GEP features per slide
            coords: list[Tensor [N_i, 2]] - coordinates per slide
            cancer_type_idx: Tensor [B] - cancer type index per slide (for per-cancer gating)

        Returns:
            list[Tensor [N_i, hidden_dim]] - fused features per slide
        """
        if gep_feats is None or gep_feats[0] is None:
            raise ValueError("SignatureQueryFusion requires gep_feats but got None.")
        if coords is None or coords[0] is None:
            raise ValueError("SignatureQueryFusion requires coords but got None.")

        out: List[Tensor] = []
        for i, (f, g, c) in enumerate(zip(img_feats, gep_feats, coords)):
            ct_idx = cancer_type_idx[i].item() if cancer_type_idx is not None else None
            out.append(self._fuse_single(f, g, c, training=self.training, cancer_type_idx=ct_idx))
        return out

    def get_signature_attention_maps(self) -> Optional[tuple]:
        """
        Get the last computed attention weights for interpretability.

        Returns:
            attn_weights: [N, K, k] - attention from each signature to each neighbor
            neigh_idx: [N, k] - indices of neighbors for each patch

        Usage for heatmap generation:
            For signature s, the attention map shows which image patches
            that program focuses on. High attention = program found relevant features.
        """
        if hasattr(self, "_last_attn_weights"):
            return self._last_attn_weights, self._last_neigh_idx
        return None

    def get_pool_attention(self) -> Optional[Tensor]:
        """
        Get the last computed program-pooling attention weights.

        Returns:
            Tensor [N, K, 1] - softmax attention over K programs per patch.
            Higher value = program contributed more to h_bio at that patch.
        """
        if hasattr(self, "_last_pool_attn"):
            return self._last_pool_attn
        return None

    def get_attended_norms(self) -> Optional[Tensor]:
        """
        Get per-program attended feature norms (better importance signal than pool_attn).

        Returns:
            Tensor [N, K] - L2 norm of each program's attended feature per patch.
            Higher norm = program produces a larger feature vector = contributes more.
        """
        if hasattr(self, "_last_attended_norms"):
            return self._last_attended_norms
        return None

    def get_fusion_gate_value(self) -> float:
        """
        Get the mean fusion gate value across all cancer types and programs.

        Returns:
            float in [0, 1] - average biological enrichment contribution.
        """
        if not self.use_residual:
            return float('nan')
        return torch.sigmoid(self.fusion_gate).mean().item()

    def get_fusion_gate_per_cancer(self) -> Optional[Tensor]:
        """
        Get per-cancer-type gate values for interpretability.

        Returns:
            Tensor [num_cancer_types] with values in [0, 1].
            Higher = that cancer type benefits more from biological programs.
        """
        if not self.use_residual:
            return None
        return torch.sigmoid(self.fusion_gate).detach()

    def get_last_norms(self) -> Optional[dict]:
        """
        Get the norms from the last forward pass (eval mode only).

        Returns:
            dict with keys:
                h_img_norm: mean L2 norm of direct image pathway
                h_bio_norm: mean L2 norm of biological enrichment
                gate: sigmoid gate value used
                effective_ratio: gate * h_bio_norm / h_img_norm
        """
        if hasattr(self, "_last_h_img_norm"):
            return {
                "h_img_norm": self._last_h_img_norm,
                "h_bio_norm": self._last_h_bio_norm,
                "gate": self._last_gate_value,
                "effective_ratio": self._last_effective_ratio,
            }
        return None

    def get_gep_gate_values(self) -> Optional[Tensor]:
        """
        Get the learned GEP gate values for interpretability.

        Returns:
            gate_values: [K] - learned gate for each signature (e.g., 40 values)
            Higher values = model preserves more raw biological signal for that program
            Lower values = model relies more on image-refined representation

        Only available if use_gep_residual=True.
        """
        if hasattr(self, "gep_gate"):
            return self.gep_gate.detach().clone()
        return None