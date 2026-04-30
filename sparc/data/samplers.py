"""DDP-aware patient-balanced sampler for multi-GPU survival training."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List

import numpy as np
from torch.utils.data import Sampler


class DistributedPatientSampler(Sampler[int]):
    """Patient-disjoint partitioning of a slide-level dataset across DDP ranks.

    The sampler:

    1. Groups slide indices by patient id.
    2. Round-robins patients across ``num_replicas`` ranks at construction
       time, so each patient is owned by exactly one rank for the entire run.
    3. At each epoch (after :meth:`set_epoch`), the owning rank shuffles its
       patients and the slides within each patient.
    4. Shorter ranks are padded by recycling the first slides of their own
       index list, so ``__len__`` is identical across ranks (required for DDP
       gradient synchronisation to deadlock-free).

    Properties:

    - **Patient leakage**: zero — a patient's slides never appear on multiple
      ranks within an epoch.
    - **Sample duplication**: limited to the padding needed to match the
      longest rank's slide count.

    Args:
        dataset:      An object exposing ``slide_feature_paths`` (a list of
                      ``Path`` objects with a ``.stem`` attribute) and a
                      ``slide_id_to_patient_id(slide_id) -> str`` method.
        num_replicas: DDP world size.
        rank:         Index of this process in ``[0, num_replicas)``.
        shuffle:      Whether to shuffle patients (and slides within each
                      patient) every epoch. Disable for deterministic eval.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        # patient -> list of slide indices into the dataset
        patient_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, slide_path in enumerate(dataset.slide_feature_paths):
            slide_id = slide_path.stem
            pid = dataset.slide_id_to_patient_id(slide_id)
            patient_to_indices[pid].append(idx)
        self.patient_to_indices = patient_to_indices

        # Stable round-robin: rank r owns patients i where i % num_replicas == r
        all_patients = sorted(patient_to_indices.keys())
        rank_to_patients: Dict[int, List[str]] = {r: [] for r in range(num_replicas)}
        for i, pid in enumerate(all_patients):
            rank_to_patients[i % num_replicas].append(pid)
        self.rank_patients = rank_to_patients[rank]
        self.epoch = 0

        rank_lengths = [
            sum(len(patient_to_indices[pid]) for pid in rank_to_patients[r])
            for r in range(num_replicas)
        ]
        self._raw_len = rank_lengths[rank]
        self._len = max(rank_lengths)   # all ranks pad up to this length

    def __len__(self) -> int:
        return self._len

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic per-epoch shuffling.

        Call once at the start of each training epoch (analogous to
        :class:`torch.utils.data.distributed.DistributedSampler`).
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(seed=self.epoch)
        pids = self.rank_patients.copy()
        if self.shuffle:
            rng.shuffle(pids)

        indices: List[int] = []
        for pid in pids:
            slides = self.patient_to_indices[pid].copy()
            if self.shuffle:
                rng.shuffle(slides)
            indices.extend(slides)

        # Pad up to the longest rank's length (DDP sync requires equal lengths).
        if len(indices) < self._len:
            indices = indices + indices[: self._len - len(indices)]
        return iter(indices)
