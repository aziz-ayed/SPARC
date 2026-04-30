"""Unified CLI for external-cohort inference.

Runs the 5-fold checkpoints of one or more trained models on every slide of a
named cohort, writing one ``.npz`` per slide. Slides whose output already
exists are skipped, so the script is resumable.

Examples:
    # SPARC-Risk on NLST, all visible GPUs in parallel
    python -m inference.run --cohort nlst \\
        --checkpoint_dir checkpoints/sparc_risk --gpus 0,1,2,3

    # Image-only baseline on SurGen, single GPU, per-fold predictions
    python -m inference.run --cohort surgen \\
        --checkpoint_dir checkpoints/image_only --per_fold

    # Ömer cohort, SPARC-Risk only, custom output directory
    python -m inference.run --cohort omer \\
        --checkpoint_dir checkpoints/sparc_risk \\
        --out_dir results/omer_custom
"""

from __future__ import annotations

import argparse
import os

# Disable HDF5 file locking before importing h5py (set inside core.py too).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
from typing import List, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from inference.cohorts import COHORTS, CohortSpec
from inference.core import (
    fusion_needs_gep,
    load_checkpoints,
    load_fold_models,
    load_slide_features,
    run_model_on_slide,
    save_slide_prediction,
)


# ─── Worker (one process per GPU) ────────────────────────────────────────────

def _worker(rank: int, world_size: int, slides: List[str],
            slide_to_ct: dict, model_specs: List[Tuple[str, list, bool, Path]],
            cohort: CohortSpec, per_fold: bool):
    device = torch.device(f"cuda:{rank}")
    my_slides = slides[rank::world_size]

    # Per-model pending sets (skip slides whose output already exists)
    pending = {}
    for label, _, _, out_dir in model_specs:
        out_dir.mkdir(parents=True, exist_ok=True)
        pending[label] = {s for s in my_slides
                          if not (out_dir / f"{s}.npz").exists()}
    all_pending = sorted(set().union(*pending.values()))
    print(f"[GPU {rank}] {len(all_pending)}/{len(my_slides)} slides to process",
          flush=True)
    if not all_pending:
        return

    # Build models on this GPU
    loaded = {}
    needs_gep = {}
    for label, ckpts, ng, _ in model_specs:
        if pending[label]:
            loaded[label] = load_fold_models(ckpts, device)
            needs_gep[label] = ng
            print(f"[GPU {rank}] Loaded {len(ckpts)} folds for {label}",
                  flush=True)

    for slide in tqdm(all_pending, desc=f"GPU {rank}", position=rank, leave=True):
        load_gep = any(needs_gep[label] for label in loaded
                       if slide in pending[label])
        emb_path = cohort.emb_dir / f"{slide}.h5"
        gep_path = (cohort.gep_dir / f"{slide}.h5") if load_gep else None
        try:
            emb, gep, coords = load_slide_features(emb_path, gep_path)
        except Exception as e:                # noqa: BLE001
            print(f"[GPU {rank}] SKIP {slide}: {e}", flush=True)
            continue

        ct_idx = slide_to_ct.get(slide, 0)
        emb_t = torch.from_numpy(emb).to(device)
        gep_t = torch.from_numpy(gep).to(device) if gep is not None else None
        coords_t = torch.from_numpy(coords).to(device)

        for label, _, ng, out_dir in model_specs:
            if slide not in pending[label] or label not in loaded:
                continue
            risk, embedding = run_model_on_slide(
                loaded[label], emb_t, gep_t if ng else None, coords_t,
                ct_idx, device, per_fold=per_fold,
            )
            save_slide_prediction(out_dir / f"{slide}.npz", risk, embedding,
                                  ct_idx, per_fold=per_fold)

    # Cleanup hook handles
    for fold_models in loaded.values():
        for _, _, handle in fold_models:
            handle.remove()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=list(COHORTS),
                        help="External cohort to run inference on.")
    parser.add_argument("--checkpoint_dir", required=True, action="append",
                        help="Path to a 5-fold checkpoint directory. May be "
                             "passed multiple times to run several models in "
                             "one pass (re-using slide features).")
    parser.add_argument("--out_dir", default=None,
                        help="Output base directory. Defaults to "
                             "results/<cohort>_inference/<run_name>/.")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated CUDA device indices.")
    parser.add_argument("--per_fold", action="store_true",
                        help="Save per-fold risks/embeddings instead of averaging.")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    world_size = torch.cuda.device_count()
    assert world_size > 0, "No GPUs visible"

    cohort = COHORTS[args.cohort]

    # Build model specs
    model_specs: List[Tuple[str, list, bool, Path]] = []
    for ckpt_dir_str in args.checkpoint_dir:
        ckpt_dir = Path(ckpt_dir_str).resolve()
        ckpts = load_checkpoints(ckpt_dir)
        run_name = ckpt_dir.parent.name
        ng = fusion_needs_gep(ckpts)
        if args.out_dir:
            base_out = Path(args.out_dir) / run_name
        else:
            base_out = Path("results") / cohort.out_subdir / run_name
        out_dir = (Path(str(base_out) + "_perfold")
                   if args.per_fold else base_out)
        model_specs.append((run_name, ckpts, ng, out_dir))
        print(f"{run_name}: fusion={ckpts[0]['config']['model']['fusion']} "
              f"needs_gep={ng} → {out_dir}")

    any_needs_gep = any(ng for _, _, ng, _ in model_specs)
    if any_needs_gep:
        slides = sorted(
            f.stem for f in cohort.emb_dir.glob("*.h5")
            if (cohort.gep_dir / f"{f.stem}.h5").exists()
        )
    else:
        slides = sorted(f.stem for f in cohort.emb_dir.glob("*.h5"))
    print(f"Total slides: {len(slides)}")

    print("Building slide → cancer-type-index map ...")
    slide_to_ct = cohort.cancer_type_map(slides)

    mp.spawn(
        _worker,
        args=(world_size, slides, slide_to_ct, model_specs, cohort, args.per_fold),
        nprocs=world_size,
        join=True,
    )

    for label, _, _, out_dir in model_specs:
        done = list(out_dir.glob("*.npz"))
        print(f"{label}: {len(done)}/{len(slides)} slides in {out_dir}/")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
