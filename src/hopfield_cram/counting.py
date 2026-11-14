"""Optimizer-free equal-value exclusion energy from completed training states."""
from __future__ import annotations

import operator
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
    return fit_rarely_equal(labels, n_values, max_equal_count=0, chunk_size=chunk_size)


@torch.no_grad()
def fit_rarely_equal(labels: Tensor, n_values: int, *, max_equal_count: int,
                     chunk_size: int = 1024) -> tuple[DensePairwiseEnergy, Tensor]:
    """Connect pairs with at most ``max_equal_count`` observed equalities.

    Return a frozen unit-weight energy and integer equality counts, without
    supplying geometry or fitting an optimizer. A nonnegative integer threshold
    can tolerate some corrupted labels, but can also add false exclusions when
    valid co-occurrences are rare. It is an explicit prior, not a guarantee.

    Unlike ``fit_never_equal``, a positive threshold can give training states
    positive energy. Choose inference calibration separately; this function
    does not infer a stopping threshold from possibly corrupted labels.
    """
    try:
        threshold = operator.index(max_equal_count)
    except TypeError as exc:
        raise ValueError('max_equal_count must be a nonnegative integer') from exc
    if isinstance(max_equal_count, bool) or threshold < 0:
        raise ValueError('max_equal_count must be a nonnegative integer')
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
    support = counts <= threshold
    model.pair_cost.copy_(support[:, None, None] * torch.eye(n_values, device=labels.device))
    model.unary_cost.zero_()
    model.requires_grad_(False)
    return model, counts
