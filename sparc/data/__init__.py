"""Slide-level dataset and patient-aware distributed samplers.

The slide-bag dataset (:class:`sparc.data.dataset.SlideBagDataset`) loads
per-slide patch features (image + optional gene-expression-program scores)
with their level-0 coordinates, and pairs them with patient-level survival
labels.

:class:`sparc.data.samplers.DistributedPatientSampler` ensures that every
patient appears exactly once per epoch when training with DDP across multiple
GPUs.
"""
