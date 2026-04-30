"""Smoke tests: imports + model construction from each canonical config.

These tests run without GPUs or data. They verify that the package layout is
intact and that each of the three canonical configs (sparc_risk, image_only,
late_fusion) can be turned into a working ``nn.Module``.
"""

from pathlib import Path

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["sparc_risk", "image_only", "late_fusion"]


def test_package_imports():
    """Every public symbol referenced by the paper analyses imports cleanly."""
    from sparc.data.dataset import (
        SlideBagDataset, slide_collate_fn, load_clinical_table,
        get_slide_paths, CANCER_TYPE_TO_IDX,
    )
    from sparc.data.samplers import DistributedPatientSampler
    from sparc.losses import nll_survival_loss, cox_loss
    from sparc.models.factory import build_model
    from sparc.models.fusion import (
        ImageOnlyFusion, SignatureQueryFusion, GatedAttentionPool,
    )
    from sparc.models.heads import NLLSurvivalHead
    from sparc.models.mil import AttnMILAggregator
    from sparc.utils.metrics import (
        c_index, td_auc_simple, aggregate_patient_level,
    )
    from sparc.utils.seed import set_seed
    from sparc.utils.splits import load_splits


def test_inference_imports():
    from inference.cohorts import COHORTS, CohortSpec
    from inference.core import (
        load_checkpoints, load_fold_models, load_slide_features,
        run_model_on_slide, save_slide_prediction,
    )
    assert set(COHORTS) == {"nlst", "surgen", "omer"}


@pytest.mark.parametrize("name", CONFIGS)
def test_build_model_from_config(name):
    """Each canonical config should build an `nn.Module` with > 0 parameters."""
    from sparc.models.factory import build_model
    cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text())
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    assert isinstance(model, torch.nn.Module)


def test_loss_shapes():
    """nll_survival_loss returns a scalar on dummy inputs."""
    from sparc.losses import nll_survival_loss
    B, K = 4, 4
    hazards = torch.rand(B, K).softmax(-1)
    survival = torch.cumprod(1 - hazards, dim=-1)
    time = torch.tensor([100.0, 365.0, 700.0, 1500.0])
    event = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bin_edges = torch.tensor([0.0, 365.0, 730.0, 1095.0, 1825.0])
    loss = nll_survival_loss(hazards, survival, time, event, bin_edges)
    assert loss.dim() == 0
