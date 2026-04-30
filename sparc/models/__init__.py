"""Model architectures used in the SPARC paper.

The package exposes three composable layers and a builder:

- **fusion**     — patch-level transformations from image + gene-program inputs
                   to a unified hidden-dim embedding. Three options:
                   :class:`~sparc.models.fusion.ImageOnlyFusion`,
                   :class:`~sparc.models.fusion.SignatureQueryFusion`, and
                   the late-fusion split implemented in
                   :func:`~sparc.models.factory.build_model`.
- **aggregator** — slide-level pooling over patch embeddings; only
                   :class:`~sparc.models.mil.AttnMILAggregator` is used by the
                   paper.
- **head**       — survival head; only :class:`~sparc.models.heads.NLLSurvivalHead`
                   (discrete-time NLL) is used by the paper.

Build any of the three canonical models from a YAML config with
:func:`sparc.models.factory.build_model`.
"""
