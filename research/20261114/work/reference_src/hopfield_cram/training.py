"""Local pseudolikelihood training: inference never enters the training graph."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .energy import PairwiseEnergy, SudokuEnergy


def pseudolikelihood(model: PairwiseEnergy, labels: Tensor, free: Tensor | None = None,
                     context: Tensor | None = None, *, reduction: str = "cells") -> Tensor:
    """Exact conditional cross-entropy, Eq. 5/6, on free cells.

    `cells` (the released implementation) gives each free cell equal weight.
    `boards` gives each nonempty board equal weight after its own |F| division,
    corresponding to averaging the displayed per-input Eq. 5. These differ
    when boards have unequal numbers of free cells. All-clamped batches fail
    explicitly rather than generating NaN or a misleading zero training loss.
    """
    if free is None:
        free = torch.ones_like(labels, dtype=torch.bool)
    if free.shape != labels.shape or free.dtype != torch.bool:
        raise ValueError("free must be a boolean tensor matching labels")
    if not bool(free.any()):
        raise ValueError("Pseudolikelihood requires at least one free cell")
    logits = model.conditional_logits(labels, context)
    nll = F.cross_entropy(logits.transpose(1, 2), labels.long(), reduction="none")
    if reduction == "cells":
        return (nll * free).sum() / free.sum()
    if reduction == "boards":
        count = free.sum(1)
        valid = count > 0
        return ((nll * free).sum(1)[valid] / count[valid]).mean()
    raise ValueError("reduction must be 'cells' or 'boards'")


@dataclass(frozen=True)
class TrainingConfig:
    updates: int
    learning_rate: float
    momentum: float = 0.5
    l1: float = 0.0
    batch_size: int | None = None
    digit_augmentation: bool = False
    seed: int = 0
    reduction: str = "cells"


def fit_pseudolikelihood(model: PairwiseEnergy, labels: Tensor, free: Tensor | None,
                        config: TrainingConfig, context: Tensor | None = None) -> list[float]:
    """SGD, heavy-ball momentum, cosine LR, then proximal L1 (Table 5).

    Mutates an existing model; callers use a fresh zero-initialized task model
    for the paper recipes. Returns one loss per update, for a readable reference
    fit. Minibatches are sampled without replacement within each update; this
    is reproducible for this implementation but not upstream RNG-bit-identical.
    Tensor/device transfer and timing are deliberately left to the caller.
    """
    if config.updates < 1 or config.learning_rate <= 0 or config.l1 < 0:
        raise ValueError("Require positive updates/LR and nonnegative L1")
    if not 0 <= config.momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if len(labels) == 0 or (config.batch_size is not None and config.batch_size < 1):
        raise ValueError("Need nonempty labels and positive batch_size")
    if config.digit_augmentation and not isinstance(model, SudokuEnergy):
        raise ValueError("Digit augmentation applies only to SudokuEnergy")
    generator = torch.Generator(device=labels.device).manual_seed(config.seed)
    params = list(model.parameters())
    velocity = [torch.zeros_like(p) for p in params]
    losses = []
    for step in range(config.updates):
        idx = (torch.randperm(len(labels), device=labels.device, generator=generator)[:config.batch_size]
               if config.batch_size is not None else slice(None))
        y = labels[idx]
        f = None if free is None else free[idx]
        ctx = None if context is None else context[idx]
        if config.digit_augmentation:
            relabel = torch.rand(len(y), model.n_values, device=y.device, generator=generator).argsort(1)
            y = relabel.gather(1, y.long())
        model.zero_grad(set_to_none=True)
        loss = pseudolikelihood(model, y, f, ctx, reduction=config.reduction)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite training loss at update {step}")
        loss.backward()
        model.precondition_gradients()
        lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * step / config.updates))
        with torch.no_grad():
            for p, v in zip(params, velocity):
                if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError(f"Missing/nonfinite gradient at update {step}")
                v.mul_(config.momentum).add_(p.grad)
                p.add_(v, alpha=-lr)
                if config.l1:
                    p.copy_(p.sign() * (p.abs() - lr * config.l1).clamp_min(0))
        losses.append(float(loss.detach()))
    return losses
