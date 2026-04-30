# Paper notebooks

These notebooks reproduce the analyses and figures shown in the SPARC paper.
They consume per-fold prediction `.npz` files produced by the inference
pipeline (`inference/run.py`) plus the patch-level features described in
`data/README.md`.

| Notebook | Produces |
|---|---|
| `compare_with_baselines.ipynb` | `figs/mmp_comparison.{pdf,png}` (main): per-cancer C-index for SPARC-Risk vs. MMP and PANTHER baselines |
| `compare_es_models.ipynb` | `figs/tcga_cindex_bootstrap.png`, `figs/tcga_hr_bootstrap.png`, `figs/km_signature_query_groups.png` (main); `figs/gate_a_logodds.png`, `figs/gate_b_delta_cindex.png`, `figs/gate_c_loo.png`, `figs/gate_d_delta_loghr.png` (supp) |
| `multivariable_cox.ipynb` | `figs/multivariable_cox_progression.png` (supp): incremental C-index when adding SPARC-Risk on top of clinical + molecular covariates |
| `paper_external_figures.ipynb` | `figs/external_zeroshot.png`, `figs/external_finetuned.png` (main): plots external-cohort zero-shot + ridge-Cox-calibrated ("fine-tuned") C-index. Reads JSONs produced by the two notebooks below. |
| `external_eval_nlst.ipynb` | `results/external_results/nlst.json`: per-patient aggregation of `inference/run.py --per_fold` outputs → zero-shot + ridge-Cox C-index / HR via `evaluate_cohort` |
| `external_eval_surgen.ipynb` | `results/external_results/surgen.json`: same as above for SurGen CRC |
| `treatment_response_breast.ipynb` | `figs/breast_or_forest.png`, `figs/breast_program_heatmap.png` (main); `figs/breast_raw_activation_supp.png` (supp): trastuzumab response analysis on Yale Breast |
| `treatment_response_ovarian.ipynb` | `figs/ovarian_or_forest.png`, `figs/ovarian_program_heatmap.png` (main): bevacizumab response analysis on the ovarian cohort |

## Running

The notebooks expect the public-repo data layout (see `data/README.md`) and
the `sparc/` package importable. Activate your environment, then:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

The notebooks have been pruned to keep only the cells required for the paper
figures and their immediate dependencies. Outputs and execution counters were
stripped — re-execute end-to-end to populate them.