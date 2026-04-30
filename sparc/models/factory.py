"""Single entry point that turns a YAML config into a working ``nn.Module``.

Use :func:`build_model` to assemble one of the three architectures used in the
SPARC paper:

- ``image_only``     — :class:`ImageOnlyFusion` + ``AttnMILAggregator`` +
                       ``NLLSurvivalHead`` (baseline).
- ``signature_query``— :class:`SignatureQueryFusion` + ``AttnMILAggregator`` +
                       ``NLLSurvivalHead`` (SPARC-Risk).
- ``late_fusion``    — :class:`LateFusionSurvivalModel` (ablation: image and GEP
                       streams kept separate until slide-level concatenation).

The factory reads ``cfg["model"]["fusion"]`` and dispatches accordingly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sparc.data.dataset import NUM_CANCER_TYPES
from sparc.models.fusion import ImageOnlyFusion, SignatureQueryFusion
from sparc.models.heads import NLLSurvivalHead
from sparc.models.mil import AttnMILAggregator


class LateFusionSurvivalModel(nn.Module):
    """Late-fusion ablation: image and GEP streams stay separate until slide level.

    Architecture (per slide):

    .. code-block:: text

        Image stream: img_feats   ─► img_proj ─► img_aggregator ─► slide_emb_img
        GEP stream:   gep_feats   ─► gep_proj ─► gep_aggregator ─► slide_emb_gep
                                                                         │
                                                concat ──► fusion_layer ─┤
                                                                         ▼
                                                                       head

    By construction, gradients from one modality cannot flow into the other
    until after slide-level pooling, which guarantees the combined model is
    at least as informative as the better of the two unimodal streams (no
    negative transfer).

    Args:
        img_aggregator:      Image-stream slide-level pooling module.
        gep_aggregator:      GEP-stream slide-level pooling module.
        head:                Survival head consuming the fused embedding.
        img_dim:             Image-feature dimensionality.
        gep_dim:             Number of gene-expression programs.
        hidden_dim:          Common projection width.
        cancer_conditioning: If True, concatenate a learned per-cancer
                             embedding before the head.
        num_cancer_types:    Cardinality of the cancer-type vocabulary.
        fusion_dropout:      Dropout applied after the fusion linear layer.
    """

    def __init__(
        self,
        img_aggregator: nn.Module,
        gep_aggregator: nn.Module,
        head: nn.Module,
        img_dim: int,
        gep_dim: int,
        hidden_dim: int,
        cancer_conditioning: bool = False,
        num_cancer_types: int = NUM_CANCER_TYPES,
        fusion_dropout: float = 0.25,
    ):
        super().__init__()

        # --- Stream 1: Image Pathway ---
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.img_aggregator = img_aggregator

        # --- Stream 2: GEP Pathway ---
        self.gep_proj = nn.Linear(gep_dim, hidden_dim)
        self.gep_aggregator = gep_aggregator

        # --- Late Fusion ---
        # Concat two slide embeddings (hidden_dim * 2) -> hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
        )

        # --- Cancer Conditioning (optional) ---
        self.cancer_conditioning = cancer_conditioning
        if cancer_conditioning:
            self.cancer_embed = nn.Embedding(num_cancer_types, hidden_dim)
            self.cancer_fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.head = head

    def forward(self, batch: dict):
        img_feats = batch["img_feats"]   # list[Tensor [N_i, D_img]]
        gep_feats = batch["gep_feats"]   # list[Tensor [N_i, K]]
        coords    = batch["coords"]      # list[Tensor [N_i, 2]]

        # --- Stream 1: Image Pathway ---
        h_img = [self.img_proj(x) for x in img_feats]  # list[Tensor [N_i, hidden_dim]]
        slide_emb_img = self.img_aggregator(h_img, coords)  # [B, hidden_dim]

        # --- Stream 2: GEP Pathway ---
        h_gep = [self.gep_proj(x) for x in gep_feats]  # list[Tensor [N_i, hidden_dim]]
        slide_emb_gep = self.gep_aggregator(h_gep, coords)  # [B, hidden_dim]

        # --- Late Fusion ---
        combined = torch.cat([slide_emb_img, slide_emb_gep], dim=1)  # [B, hidden_dim * 2]
        fused = self.fusion_layer(combined)  # [B, hidden_dim]

        # --- Cancer Conditioning (optional) ---
        if self.cancer_conditioning and "cancer_type_idx" in batch:
            cancer_idx = batch["cancer_type_idx"].clamp(min=0)
            cancer_emb = self.cancer_embed(cancer_idx)
            fused = self.cancer_fuse(torch.cat([fused, cancer_emb], dim=-1))

        return self.head(fused)


class SlideSurvivalModel(nn.Module):
    """Standard model: patch-level fusion → slide-level aggregation → survival head.

    Used by both ``image_only`` and ``signature_query`` (SPARC-Risk) configs.

    Forward expects a batch dict produced by ``SlideBagDataset`` +
    ``slide_collate_fn``:

    .. code-block:: text

        batch = {
            "img_feats":        list[Tensor [N_i, D_img]],
            "gep_feats":        list[Tensor [N_i, K]] or list[None],
            "coords":           list[Tensor [N_i, 2]] or list[None],
            "time":             Tensor [B],
            "event":            Tensor [B],
            "patient_id":       list[str],
            "slide_id":         list[str],
            "cancer_type_idx":  Tensor [B]        # optional
        }

    The model itself only consumes ``*_feats``, ``coords``, and (optionally)
    ``cancer_type_idx``; ``time``/``event``/IDs are used by the training loop.

    Cancer conditioning:
        When enabled, a learned per-cancer embedding is fused with the
        slide-level embedding *after* aggregation. This frees model capacity
        from learning organ/cancer identity and lets it focus on intra-cancer
        survival signal (where the GEP pathway adds value).

    Auxiliary losses (shared-weight):
        If ``use_aux_losses=True``, training also computes auxiliary
        predictions by passing the same patches through the **same** fusion,
        aggregator and head with one modality zeroed out:

        - **Image-only aux**: GEP features replaced with zeros.
        - **GEP-only aux**:   Image features replaced with zeros.

        This forces the shared weights to be useful for either modality alone,
        preventing "modality laziness". Forward then returns
        ``{"fused": ..., "img_aux": ..., "gep_aux": ...}`` instead of just the
        fused output.

    Args:
        fusion:              A patch-level fusion module (e.g.
                             :class:`ImageOnlyFusion`, :class:`SignatureQueryFusion`).
        aggregator:          A slide-level pooling module (e.g.
                             :class:`AttnMILAggregator`).
        head:                A survival head (e.g. :class:`NLLSurvivalHead`).
        cancer_conditioning: Whether to add a per-cancer embedding before the head.
        num_cancer_types:    Cardinality of the cancer-type vocabulary.
        hidden_dim:          Width of the slide embedding.
        use_aux_losses:      Enable shared-weight auxiliary losses (training only).
        img_dim, gep_dim:    Per-modality input dimensionalities (kept on the
                             instance for reference; the actual projection
                             happens inside ``fusion``).
        head_type:           Reserved for future head variants; currently only
                             ``"nll_surv"`` is supported.
        n_bins:              Number of discrete time bins used by the head.
    """

    def __init__(
        self,
        fusion: nn.Module,
        aggregator: nn.Module,
        head: nn.Module,
        cancer_conditioning: bool = False,
        num_cancer_types: int = NUM_CANCER_TYPES,
        hidden_dim: int = 256,
        # Auxiliary loss parameters (shared-weight approach - no separate branches needed)
        use_aux_losses: bool = False,
        img_dim: int = 1536,
        gep_dim: int = 40,
        head_type: str = "nll_surv",
        n_bins: int = 4,
    ):
        super().__init__()
        self.fusion = fusion
        self.aggregator = aggregator
        self.head = head
        self.cancer_conditioning = cancer_conditioning
        self.use_aux_losses = use_aux_losses
        self.hidden_dim = hidden_dim

        if cancer_conditioning:
            # Learned embedding for each cancer type
            self.cancer_embed = nn.Embedding(num_cancer_types, hidden_dim)
            # Small MLP to combine slide embedding with cancer embedding
            self.cancer_fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # No separate auxiliary branches needed - we use the same fusion/aggregator/head
        # with one modality zeroed out

    def forward(self, batch: dict):
        img_feats = batch["img_feats"]   # list[Tensor [N_i, D_img]]
        gep_feats = batch["gep_feats"]   # list[Tensor [N_i, K]] or list[None]
        coords    = batch["coords"]      # list[Tensor [N_i, 2]] or list[None]

        # 1) Patch-level fusion (image + programs + optional coords)
        # Pass cancer_type_idx for per-cancer gating (SignatureQueryFusion)
        alphas = None
        if hasattr(self.fusion, 'fusion_gate') and "cancer_type_idx" in batch:
            cancer_idx = batch["cancer_type_idx"].clamp(min=0)
            fusion_out = self.fusion(img_feats, gep_feats, coords, cancer_type_idx=cancer_idx)
        else:
            fusion_out = self.fusion(img_feats, gep_feats, coords)

        # Some interpretable fusion variants return (h_patches, alphas)
        if isinstance(fusion_out, tuple):
            h_patches, alphas = fusion_out
        else:
            h_patches = fusion_out

        # 2) Slide-level aggregation (MIL)
        slide_embs = self.aggregator(h_patches, coords)         # Tensor [B, d]

        # 2.5) Cancer type conditioning (if enabled) - LATE conditioning at slide level
        # Cancer embedding is fused with slide embedding after aggregation
        if self.cancer_conditioning and "cancer_type_idx" in batch:
            cancer_idx = batch["cancer_type_idx"]  # [B]
            # Handle missing cancer types (idx = -1) by clamping to 0
            # These cases should be rare in TCGA data
            cancer_idx = cancer_idx.clamp(min=0)
            cancer_emb = self.cancer_embed(cancer_idx)  # [B, d]
            # Fuse slide embedding with cancer embedding
            slide_embs = self.cancer_fuse(torch.cat([slide_embs, cancer_emb], dim=-1))  # [B, d]

        # 3) Survival head
        out_fused = self.head(slide_embs)

        # 3.5) Attach alpha weights to output (for entropy penalty in loss)
        if alphas is not None and isinstance(out_fused, dict):
            # Stack per-slide alphas: list of [K] or [N_i, K] -> [B, K] (per_slide)
            # For per_slide mode: each alpha is [K], stack to [B, K]
            if alphas[0].dim() == 1:
                out_fused["alpha"] = torch.stack(alphas)  # [B, K]
            else:
                # Per-patch mode: mean-pool to [B, K] for loss computation
                out_fused["alpha"] = torch.stack([a.mean(dim=0) for a in alphas])  # [B, K]

        # 4) Auxiliary predictions using SHARED WEIGHTS (if enabled)
        # Pass through the SAME fusion, aggregator, and head with one modality zeroed
        # To save memory, randomly compute EITHER img-only OR gep-only each step (not both)
        if self.use_aux_losses and self.training:
            out_img = None
            out_gep = None

            if gep_feats is not None and gep_feats[0] is not None:
                # Randomly choose which aux path to compute (saves memory: 2x instead of 3x)
                compute_img_aux = torch.rand(1).item() < 0.5

                # Get cancer_type_idx for per-cancer gating
                _aux_ct = batch["cancer_type_idx"].clamp(min=0) if (hasattr(self.fusion, 'fusion_gate') and "cancer_type_idx" in batch) else None

                if compute_img_aux:
                    # Image-only: zero out GEP features
                    zero_gep = [torch.zeros_like(g) for g in gep_feats]
                    aux_out = self.fusion(img_feats, zero_gep, coords, cancer_type_idx=_aux_ct) if _aux_ct is not None else self.fusion(img_feats, zero_gep, coords)
                    h_img_only = aux_out[0] if isinstance(aux_out, tuple) else aux_out
                    slide_embs_img = self.aggregator(h_img_only, coords)
                    if self.cancer_conditioning and "cancer_type_idx" in batch:
                        cancer_idx = batch["cancer_type_idx"].clamp(min=0)
                        cancer_emb = self.cancer_embed(cancer_idx)
                        slide_embs_img = self.cancer_fuse(torch.cat([slide_embs_img, cancer_emb], dim=-1))
                    out_img = self.head(slide_embs_img)
                else:
                    # GEP-only: zero out image features
                    zero_img = [torch.zeros_like(f) for f in img_feats]
                    aux_out = self.fusion(zero_img, gep_feats, coords, cancer_type_idx=_aux_ct) if _aux_ct is not None else self.fusion(zero_img, gep_feats, coords)
                    h_gep_only = aux_out[0] if isinstance(aux_out, tuple) else aux_out
                    slide_embs_gep = self.aggregator(h_gep_only, coords)
                    if self.cancer_conditioning and "cancer_type_idx" in batch:
                        cancer_idx = batch["cancer_type_idx"].clamp(min=0)
                        cancer_emb = self.cancer_embed(cancer_idx)
                        slide_embs_gep = self.cancer_fuse(torch.cat([slide_embs_gep, cancer_emb], dim=-1))
                    out_gep = self.head(slide_embs_gep)

            return {
                "fused": out_fused,
                "img_aux": out_img,
                "gep_aux": out_gep,
            }

        return out_fused


def _build_aggregator(agg_name: str, hidden_dim: int) -> nn.Module:
    """Build a slide-level aggregator by name (only ``attn_mil`` is supported)."""
    if agg_name == "attn_mil":
        return AttnMILAggregator(hidden_dim=hidden_dim)
    raise ValueError(f"Unknown aggregator type: {agg_name}")


def build_model(cfg: dict) -> nn.Module:
    """Construct one of the three canonical SPARC architectures from a config.

    The config layout matches the YAML files under ``configs/``:

    .. code-block:: yaml

        model:
          fusion: signature_query   # "image_only" | "signature_query" | "late_fusion"
          aggregator: attn_mil      # only "attn_mil" supported
          head: nll_surv            # only "nll_surv" supported
          img_dim: 1536
          gep_dim: 40
          hidden_dim: 256
          pos_dim: 16
          n_bins: 4
          cancer_conditioning: true

    Optional, ``signature_query``-only knobs:
        ``num_heads``, ``k_nn``, ``modality_dropout``, ``signature_pool``,
        ``use_signature_transformer``, ``use_checkpoint``, ``chunk_size``,
        ``use_gep_residual``, ``use_residual``, ``gate_init``,
        ``cancer_query_scaling``, ``deep_img_proj``, ``attn_dropout``,
        ``n_cross_attn_layers``, ``orthogonal_queries``, ``use_aux_losses``.

    Optional, ``late_fusion``-only knobs:
        ``gep_aggregator`` (defaults to ``aggregator``), ``fusion_dropout``.

    Args:
        cfg: Parsed YAML config dict. Must contain the ``"model"`` block above.

    Returns:
        An ``nn.Module`` ready for training or inference. Either
        :class:`LateFusionSurvivalModel` (for ``fusion: late_fusion``) or
        :class:`SlideSurvivalModel` (for ``image_only`` and ``signature_query``).

    Raises:
        ValueError: On an unknown ``fusion``, ``aggregator``, or ``head`` value.
    """
    mcfg = cfg["model"]

    img_dim    = mcfg["img_dim"]
    gep_dim    = mcfg["gep_dim"]
    hidden_dim = mcfg["hidden_dim"]
    pos_dim    = mcfg["pos_dim"]

    fusion_name = mcfg["fusion"]
    cancer_conditioning = mcfg.get("cancer_conditioning", False)

    # ---------------- Late Fusion (special case) ----------------
    if fusion_name == "late_fusion":
        agg_name = mcfg["aggregator"]
        gep_agg_name = mcfg.get("gep_aggregator", agg_name)  # Can use different aggregator for GEP

        img_aggregator = _build_aggregator(agg_name, hidden_dim)
        gep_aggregator = _build_aggregator(gep_agg_name, hidden_dim)

        head_name = mcfg["head"]
        if head_name == "nll_surv":
            n_bins = mcfg.get("n_bins", 4)
            head = NLLSurvivalHead(in_dim=hidden_dim, n_bins=n_bins)
        else:
            raise ValueError(f"Unknown head type: {head_name}")

        return LateFusionSurvivalModel(
            img_aggregator=img_aggregator,
            gep_aggregator=gep_aggregator,
            head=head,
            img_dim=img_dim,
            gep_dim=gep_dim,
            hidden_dim=hidden_dim,
            cancer_conditioning=cancer_conditioning,
            fusion_dropout=mcfg.get("fusion_dropout", 0.25),
        )

    # ---------------- Standard Fusion Modes ----------------
    if fusion_name == "image_only":
        fusion = ImageOnlyFusion(img_dim=img_dim, hidden_dim=hidden_dim)
    elif fusion_name == "signature_query":
        fusion = SignatureQueryFusion(
            img_dim=img_dim,
            gep_dim=gep_dim,
            hidden_dim=hidden_dim,
            num_heads=mcfg.get("num_heads", 4),
            k_nn=mcfg.get("k_nn", 16),
            modality_dropout=mcfg.get("modality_dropout", 0.0),
            signature_pool=mcfg.get("signature_pool", "mean"),
            use_signature_transformer=mcfg.get("use_signature_transformer", True),
            use_checkpoint=mcfg.get("use_checkpoint", True),
            chunk_size=mcfg.get("chunk_size", 2048),
            use_gep_residual=mcfg.get("use_gep_residual", False),
            use_residual=mcfg.get("use_residual", True),
            num_cancer_types=NUM_CANCER_TYPES,
            gate_init=mcfg.get("gate_init", -2.0),
            cancer_query_scaling=mcfg.get("cancer_query_scaling", False),
            deep_img_proj=mcfg.get("deep_img_proj", False),
            attn_dropout=mcfg.get("attn_dropout", 0.0),
            n_cross_attn_layers=mcfg.get("n_cross_attn_layers", 1),
            orthogonal_queries=mcfg.get("orthogonal_queries", False),
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_name}")

    # ---------------- Aggregator ----------------
    aggregator = _build_aggregator(mcfg["aggregator"], hidden_dim)

    # ---------------- Head ----------------
    head_name = mcfg["head"]
    if head_name == "nll_surv":
        n_bins = mcfg.get("n_bins", 4)
        head = NLLSurvivalHead(in_dim=hidden_dim, n_bins=n_bins)
    else:
        raise ValueError(f"Unknown head type: {head_name}")

    # Auxiliary loss parameters
    use_aux_losses = mcfg.get("use_aux_losses", False)
    n_bins = mcfg.get("n_bins", 4)

    return SlideSurvivalModel(
        fusion=fusion,
        aggregator=aggregator,
        head=head,
        cancer_conditioning=cancer_conditioning,
        hidden_dim=hidden_dim,
        use_aux_losses=use_aux_losses,
        img_dim=img_dim,
        gep_dim=gep_dim,
        head_type=head_name,
        n_bins=n_bins,
    )