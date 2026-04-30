"""SPARC: gene-program-aware survival modelling from H&E whole-slide images.

The :mod:`sparc` package provides three subpackages:

- :mod:`sparc.data`   — slide-bag dataset and patient-balanced samplers
- :mod:`sparc.models` — fusion architectures, MIL aggregators, survival heads
                        and a single :func:`sparc.models.factory.build_model`
                        entry point driven by YAML configs
- :mod:`sparc.utils`  — cross-validation splits, metrics, and reproducibility helpers

Plus :mod:`sparc.losses` (NLL discrete-time survival and Cox partial likelihood).

The canonical training entry point is the top-level ``train.py`` script;
external-cohort inference lives in the top-level :mod:`inference` package.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
