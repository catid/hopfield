"""Unoptimized, explicit inference oracles for the CRAM paper.

All functions are no-grad. Focused Gibbs recomputes the complete field after
one move, so it is suitable for correctness and small examples, not full test
sets. The pinned upstream implementation supplies sparse Triton search kernels.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .energy import GraphColoringEnergy, PairwiseEnergy


def _clamped_probabilities(model, logits, given, free, inverse_temperature):
    return torch.where(free[..., None], (inverse_temperature * logits).softmax(-1), model.one_hot(given))


@torch.no_grad()
def euler_step(model: PairwiseEnergy, logits: Tensor, given: Tensor, free: Tensor,
               context: Tensor | None = None, *, step_size: float = 0.5,
               inverse_temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    """Eq. 3 with hard givens: v <- (1-eta)v - eta(Wp+B).

    Returns new logits and clamped probabilities. The continuous-time descent
    theorem does not guarantee monotonic energy for a finite Euler step.
    """
    if step_size <= 0 or inverse_temperature <= 0:
        raise ValueError("step_size and inverse_temperature must be positive")
    if free.dtype != torch.bool or free.shape != given.shape:
        raise ValueError("free must be a boolean tensor matching given")
    p = _clamped_probabilities(model, logits, given, free, inverse_temperature)
    updated = (1 - step_size) * logits - step_size * model.fields(p, context)
    updated = torch.where(free[..., None], updated, logits)
    return updated, _clamped_probabilities(model, updated, given, free, inverse_temperature)


@dataclass
class RelaxationResult:
    probabilities: Tensor
    logits: Tensor
    energies: Tensor
    steps: int
    converged: Tensor


@torch.no_grad()
def relax(model: PairwiseEnergy, logits: Tensor, given: Tensor, free: Tensor,
          context: Tensor | None = None, *, steps: int = 1000, step_size: float = 0.5,
          inverse_temperature: float = 1.0, tolerance: float = 1e-6) -> RelaxationResult:
    """Deterministic Euler relaxation, with residual stopping and energy trace.

    A fixed point is not a certificate that a puzzle is solved. Cells in
    `given` matter only where `free` is false; no solution labels are read.
    """
    if steps < 0 or tolerance < 0 or step_size <= 0 or inverse_temperature <= 0:
        raise ValueError("Invalid relaxation budget, step size, temperature or tolerance")
    v = logits.detach().clone()
    p = _clamped_probabilities(model, v, given, free, inverse_temperature)
    energies = [model.energy(p, context, inverse_temperature=inverse_temperature)]
    converged = torch.zeros(len(v), dtype=torch.bool, device=v.device)
    completed = 0
    for step in range(steps + 1):
        residual = torch.where(free[..., None], -v - model.fields(p, context), 0).abs().amax((1, 2))
        converged = residual <= tolerance
        if bool(converged.all()) or step == steps:
            break
        new_v, _ = euler_step(model, v, given, free, context, step_size=step_size,
                              inverse_temperature=inverse_temperature)
        v = torch.where(converged[:, None, None], v, new_v)
        p = _clamped_probabilities(model, v, given, free, inverse_temperature)
        energies.append(model.energy(p, context, inverse_temperature=inverse_temperature))
        completed += 1
    return RelaxationResult(p, v, torch.stack(energies), completed, converged)


@dataclass(frozen=True)
class TrainingStatistics:
    energy_threshold: float
    mean_own_field: float
    mean_positive_cost: float
    minimum_margin: float
    binary_temperature_unit: float | None


@torch.no_grad()
def fit_statistics(model: PairwiseEnergy, labels: Tensor,
                   context: Tensor | None = None) -> TrainingStatistics:
    """Fit thresholds only from training solutions, never puzzle test labels."""
    if len(labels) == 0:
        raise ValueError("Need at least one training solution")
    fields = model.fields(model.one_hot(labels), context)
    own = fields.gather(-1, labels[..., None]).squeeze(-1)
    other = fields.scatter(-1, labels[..., None], torch.inf).amin(-1)
    K, _ = model.costs(context)
    positive = K[K > 0]
    return TrainingStatistics(
        float(model.board_energy(labels, context).max()), float(own.mean()),
        float(positive.mean()) if positive.numel() else float("nan"),
        float((other - own).min()),
        float((fields[..., 1] - fields[..., 0]).abs().mean()) if model.n_values == 2 else None,
    )


@dataclass
class ColoringThresholds:
    pair_by_edge: Tensor
    vertex: Tensor

    @torch.no_grad()
    def accepts(self, model: GraphColoringEnergy, board: Tensor, context: Tensor,
                tolerance: float = 1e-6) -> Tensor:
        """App. C.6: every pair and unary term is below its training maximum."""
        K, B = model.costs(context)
        i, j = model.pairs
        rows = torch.arange(len(board), device=board.device)[:, None]
        edges = torch.arange(len(i), device=board.device)[None]
        terms = K[rows, edges, board[:, i], board[:, j]]
        limit = self.pair_by_edge[edges, context.long()]
        unary = B[torch.arange(model.n_cells, device=board.device)[None], board]
        return (terms <= limit + tolerance).all(1) & (unary <= self.vertex + tolerance).all(1)


@torch.no_grad()
def fit_coloring_thresholds(model: GraphColoringEnergy, labels: Tensor,
                            context: Tensor) -> ColoringThresholds:
    """Per-pair maxima grouped by its own adjacency token, as in the release.

    A token state absent at a particular pair uses that state's largest value
    across observed pairs. If absent everywhere, its threshold remains -inf:
    there is no training evidence for accepting that state.
    """
    if not bool(((context == 0) | (context == 1)).all()):
        raise ValueError("Adjacency context must contain binary tokens")
    K, B = model.costs(context)
    i, j = model.pairs
    rows = torch.arange(len(labels), device=labels.device)[:, None]
    edges = torch.arange(len(i), device=labels.device)[None]
    held = K[rows, edges, labels[:, i], labels[:, j]]
    thresholds = []
    for value in (0, 1):
        maxima = held.masked_fill(context != value, -torch.inf).amax(0)
        thresholds.append(torch.where(torch.isneginf(maxima), maxima.max(), maxima))
    unary = B[torch.arange(model.n_cells, device=labels.device)[None], labels]
    return ColoringThresholds(torch.stack(thresholds, 1), unary.amax(0))


@dataclass(frozen=True)
class GeometricTemperature:
    high: float
    low: float
    period: int

    def __post_init__(self):
        if self.high <= 0 or self.low <= 0 or self.period < 1:
            raise ValueError("Require positive endpoint temperatures and period")

    def __call__(self, move: int) -> float:
        return self.high * (self.low / self.high) ** ((move % self.period) / self.period)


@dataclass
class SearchResult:
    boards: Tensor
    energies: Tensor
    moves: Tensor
    stopped: Tensor
    energy_trace: Tensor | None


@torch.no_grad()
def focused_gibbs(model: PairwiseEnergy, initial: Tensor, free: Tensor,
                  context: Tensor | None = None, *, max_moves: int = 1000,
                  temperature: float | Callable[[int], float] = 1.0,
                  focus_reference: float | None = None, focus_threshold: float = 0.0,
                  stop: str = "fixed", threshold: float = 0.0, tolerance: float = 0.0,
                  accept: Callable[[Tensor, Tensor | None], Tensor] | None = None,
                  restart_period: int | None = None, seed: int = 0,
                  record_trace: bool = False) -> SearchResult:
    """Focused single-cell heat-bath search, one independent chain per row.

    With `focus_reference`, flag cells whose own field exceeds the fitted mean
    plus `focus_threshold` (Sudoku/coloring). Without it, flag free cells whose
    current value is not minimal (N-Queens/maze). Pick uniformly among flagged
    cells, or all free cells if none are flagged. A selected value is sampled
    from softmax(-field/T), or set to argmin at T=0.

    `stop`: fixed, energy, margin, terms, or none. Energy/margin use `threshold`
    and `tolerance`; terms uses an `accept(board, context)` callback. Return
    the first accepted board; otherwise the lowest-energy board seen. No task
    validity checks or solution labels enter inference. Multi-chain callers
    repeat input/context rows and select their lowest-energy returned chain.
    """
    if max_moves < 0 or tolerance < 0 or (restart_period is not None and restart_period < 1):
        raise ValueError("Invalid search budget, tolerance or restart period")
    if stop not in {"fixed", "energy", "margin", "terms", "none"}:
        raise ValueError("Unknown stopping rule")
    if stop == "terms" and accept is None:
        raise ValueError("terms stopping needs an acceptance callback")
    if free.shape != initial.shape or free.dtype != torch.bool:
        raise ValueError("free must be a boolean tensor matching initial")
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    board = initial.long().clone()
    best = board.clone()
    best_e = model.board_energy(board, context)
    if not bool(torch.isfinite(best_e).all()):
        raise ValueError("Initial energy is nonfinite")
    stopped = torch.zeros(len(board), dtype=torch.bool, device=board.device)
    moves = torch.zeros(len(board), dtype=torch.long, device=board.device)
    traces = []
    for step in range(max_moves + 1):
        if restart_period is not None and step and step < max_moves and step % restart_period == 0:
            active = ~stopped & free.any(1)
            random = torch.randint(model.n_values, board.shape, device=board.device, generator=generator)
            board = torch.where(active[:, None] & free, random, board)
        # Exact conditionals also handle the latent extension's quadratic
        # self term, whose discrete and continuous derivatives differ.
        fields = -model.conditional_logits(board, context)
        energy = model.board_energy(board, context)
        own = fields.gather(-1, board[..., None]).squeeze(-1)
        bad = (own > fields.amin(-1)) & free
        lower = (energy < best_e) & ~stopped
        best = torch.where(lower[:, None], board, best)
        best_e = torch.where(lower, energy, best_e)
        if record_trace:
            traces.append(energy.clone())
        passed = torch.zeros_like(stopped)
        if stop == "fixed":
            passed = ~bad.any(1)
        elif stop == "energy":
            passed = energy <= threshold + tolerance
        elif stop == "margin":
            other = fields.scatter(-1, board[..., None], torch.inf).amin(-1)
            passed = ((other - own >= threshold - tolerance) | ~free).all(1)
        elif stop == "terms":
            passed = accept(board, context)
        newly = passed & ~stopped
        best = torch.where(newly[:, None], board, best)
        best_e = torch.where(newly, energy, best_e)
        stopped |= passed
        active = ~stopped & free.any(1)
        if step == max_moves or not bool(active.any()):
            break
        focus = bad if focus_reference is None else ((own > focus_reference + focus_threshold) & free)
        pool = torch.where(focus.any(1)[:, None], focus, free)
        rows = active.nonzero(as_tuple=True)[0]
        cells = torch.multinomial(pool[rows].float(), 1, generator=generator).squeeze(1)
        selected = fields[rows, cells]
        T = temperature(step) if callable(temperature) else temperature
        if T < 0 or not torch.isfinite(torch.tensor(T)):
            raise ValueError("Temperature must be finite and nonnegative")
        values = (selected.argmin(-1) if T == 0 else
                  torch.multinomial((-selected / T).softmax(-1), 1, generator=generator).squeeze(1))
        board[rows, cells] = values
        moves[rows] += 1
    return SearchResult(best, best_e, moves, stopped, torch.stack(traces) if record_trace else None)
