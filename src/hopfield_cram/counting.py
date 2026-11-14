"""Optimizer-free equal-value exclusion energy from completed training states."""
from __future__ import annotations

import torch
from torch import Tensor

from .energy import DensePairwiseEnergy


@torch.no_grad()
def fit_never_equal(labels: Tensor, n_values: int, *, chunk_size: int = 1024) -> tuple[DensePairwiseEnergy, Tensor]:
    """Connect cell pairs that never share a value in the supplied training rows.

    Return a frozen unit-weight energy and integer equality counts in its
    ``model.pairs`` order. No cell geometry or optimizer is used. A zero count
    is an empirical exclusion, not a guarantee about unseen valid states.
    Every supplied training state has zero energy by construction.

    ``labels`` is a nonempty (examples, cells) integer tensor, values 0..C-1.
    Memory for counting is bounded by ``chunk_size * number_of_pairs``.
    """
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError('labels must be a nonempty (examples, cells) tensor')
    if labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise ValueError('labels must have integer dtype')
    if n_values < 2 or chunk_size < 1:
        raise ValueError('n_values must be >=2 and chunk_size must be positive')
    if bool(((labels < 0) | (labels >= n_values)).any()):
        raise ValueError('labels outside 0..n_values-1')
    model = DensePairwiseEnergy(labels.shape[1], n_values).to(labels.device)
    left, right = model.pairs
    counts = torch.zeros(len(left), dtype=torch.long, device=labels.device)
    for batch in labels.split(chunk_size):
        counts += (batch[:, left] == batch[:, right]).sum(0)
    support = counts == 0
    model.pair_cost.copy_(support[:, None, None] * torch.eye(n_values, device=labels.device))
    model.unary_cost.zero_()
    model.requires_grad_(False)
    return model, counts
