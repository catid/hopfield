"""Readable implementations of CRAM's energy (paper Eq. 1).

A single table is stored for each unordered pair i < j. Consequently the
interaction energy sums each pair once, and reverse fields use its transpose.
All states have shape (batch, cells, values); discrete values are zero based.
This implementation is an independent reference, not the paper's Triton path.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PairwiseEnergy(nn.Module):
    """Base class for parameterizations supplying pair tables and unary costs."""

    def __init__(self, n_cells: int, n_values: int, pairs: Tensor):
        super().__init__()
        if n_cells < 1 or n_values < 2:
            raise ValueError("Require at least one cell and two values")
        pairs = torch.as_tensor(pairs, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[0] != 2:
            raise ValueError("pairs must have shape (2, number_of_pairs)")
        if pairs.numel() and not bool(((pairs[0] >= 0) & (pairs[0] < pairs[1]) & (pairs[1] < n_cells)).all()):
            raise ValueError("Each pair must satisfy 0 <= i < j < n_cells")
        if torch.unique(pairs, dim=1).shape[1] != pairs.shape[1]:
            raise ValueError("Duplicate pairs would double-count the energy")
        self.n_cells, self.n_values = n_cells, n_values
        self.register_buffer("pairs", pairs)

    def costs(self, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return K (..., pairs, C, C), B (..., cells, C), with optional batch."""
        raise NotImplementedError

    def _check_states(self, p: Tensor) -> None:
        if p.ndim != 3 or p.shape[1:] != (self.n_cells, self.n_values):
            raise ValueError(f"states must be (batch, {self.n_cells}, {self.n_values})")

    def one_hot(self, labels: Tensor) -> Tensor:
        if labels.ndim != 2 or labels.shape[1] != self.n_cells:
            raise ValueError(f"labels must be (batch, {self.n_cells})")
        return F.one_hot(labels.long(), self.n_values).to(next(self.parameters()).dtype)

    def fields(self, p: Tensor, context: Tensor | None = None) -> Tensor:
        """Derivative of pair and unary energy, excluding entropy: Eq. 2's Wp+B."""
        self._check_states(p)
        K, B = self.costs(context)
        i, j = self.pairs
        left = torch.einsum("...eac,...ec->...ea", K, p[:, j])
        right = torch.einsum("...eac,...ea->...ec", K, p[:, i])
        return p.new_zeros(p.shape).index_add(1, i, left).index_add(1, j, right) + B

    def energy(self, p: Tensor, context: Tensor | None = None, *,
               inverse_temperature: float | None = None) -> Tensor:
        """Eq. 1 per board. None omits entropy; positive beta includes it.

        At one-hot states entropy is zero, so the discrete energy is independent
        of beta. Clamp cells by passing their one-hot probabilities.
        """
        self._check_states(p)
        K, B = self.costs(context)
        i, j = self.pairs
        pair = torch.einsum("...ea,...eac,...ec->...", p[:, i], K, p[:, j])
        result = pair + (p * B).sum((-2, -1))
        if inverse_temperature is not None:
            if inverse_temperature <= 0:
                raise ValueError("inverse_temperature must be positive")
            # Exact zero entropy at p=0, with finite arithmetic at hard clamps.
            entropy = p * p.clamp_min(torch.finfo(p.dtype).tiny).log()
            result = result + entropy.sum((-2, -1)) / inverse_temperature
        return result

    def board_energy(self, labels: Tensor, context: Tensor | None = None) -> Tensor:
        return self.energy(self.one_hot(labels), context)

    def conditional_logits(self, labels: Tensor, context: Tensor | None = None) -> Tensor:
        """Exact one-cell conditional logits at unit temperature (paper Eq. 6)."""
        return -self.fields(self.one_hot(labels), context)

    def dense_interactions(self, context: Tensor | None = None) -> Tensor:
        """Expand W (..., M, M, C, C) for inspection or an independent oracle."""
        K, _ = self.costs(context)
        W = K.new_zeros(*K.shape[:-3], self.n_cells, self.n_cells, self.n_values, self.n_values)
        i, j = self.pairs
        W[..., i, j, :, :] = K
        W[..., j, i, :, :] = K.transpose(-1, -2)
        return W

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def precondition_gradients(self) -> None:
        """Default: no preconditioner. N-Queens overrides this hook."""


class DensePairwiseEnergy(PairwiseEnergy):
    """Free C×C table per cell pair and free unary table, initialized to zero."""

    def __init__(self, n_cells: int, n_values: int):
        super().__init__(n_cells, n_values, torch.triu_indices(n_cells, n_cells, 1))
        self.pair_cost = nn.Parameter(torch.zeros(self.pairs.shape[1], n_values, n_values))
        self.unary_cost = nn.Parameter(torch.zeros(n_cells, n_values))

    def costs(self, context=None):
        if context is not None:
            raise ValueError("DensePairwiseEnergy has no input-dependent costs")
        return self.pair_cost, self.unary_cost


class SudokuEnergy(DensePairwiseEnergy):
    """Paper Sudoku: 81 opaque cells, 9 values, 263,169 parameters, no rule graph."""

    def __init__(self):
        super().__init__(81, 9)


class MazeEnergy(PairwiseEnergy):
    """Paper's 18 parameters: four 2×2 neighbor tables and two unary costs.

    Pairs use half of a 3×3 window, without wraparound. Optional context is a
    (batch, height*width) boolean mask of open, non-endpoint cells where the
    unary cost applies. All other cells have zero unary cost, matching the
    released code. Without context B is shared at every cell as in Table 1.
    The paper writes +B; upstream stores b with the opposite sign (B=-b).
    """
    offsets = ((0, 1), (1, -1), (1, 0), (1, 1))

    def __init__(self, height: int = 30, width: int = 30):
        if height < 1 or width < 1:
            raise ValueError("Grid dimensions must be positive")
        pairs, which = [], []
        for row in range(height):
            for col in range(width):
                for k, (dr, dc) in enumerate(self.offsets):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        pairs.append((row * width + col, rr * width + cc))
                        which.append(k)
        pair_tensor = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T
        super().__init__(height * width, 2, pair_tensor)
        self.height, self.width = height, width
        self.register_buffer("offset_index", torch.tensor(which, dtype=torch.long))
        self.tables = nn.Parameter(torch.zeros(4, 2, 2))
        self.unary_cost = nn.Parameter(torch.zeros(2))

    def costs(self, context=None):
        B = self.unary_cost.expand(self.n_cells, 2)
        if context is not None:
            if context.ndim != 2 or context.shape[-1] != self.n_cells:
                raise ValueError("Maze context must be (batch, cells) unary mask")
            B = context[..., None].to(B.dtype) * B
        return self.tables[self.offset_index], B


class NQueensEnergy(PairwiseEnergy):
    """Offset-shared Ising energy: sum J_d x_i x_j + h sum x_i.

    There are 2n(n-1)+1 parameters (113 at n=8, 181 at n=10). No attack rules
    enter this parameterization. Gradients of J_d are preconditioned by the
    number of cell pairs at that offset divided by the mean count (App. C.5).
    """

    def __init__(self, size: int = 8):
        if size < 2:
            raise ValueError("N-Queens requires size >= 2")
        pairs = torch.triu_indices(size * size, size * size, 1)
        super().__init__(size * size, 2, pairs)
        i, j = pairs
        offsets = torch.stack((j // size - i // size, j % size - i % size), 1)
        unique, index, counts = torch.unique(offsets, dim=0, return_inverse=True, return_counts=True)
        self.size = size
        self.register_buffer("offsets", unique)
        self.register_buffer("offset_index", index)
        self.register_buffer("preconditioner", counts.float() / counts.float().mean())
        self.coupling = nn.Parameter(torch.zeros(len(unique)))
        self.unary_cost = nn.Parameter(torch.zeros(()))

    def costs(self, context=None):
        if context is not None:
            raise ValueError("NQueensEnergy has no input-dependent costs")
        indicator = self.coupling.new_tensor([[0., 0.], [0., 1.]])
        K = self.coupling[self.offset_index, None, None] * indicator
        B = torch.stack((torch.zeros_like(self.unary_cost), self.unary_cost)).expand(self.n_cells, 2)
        return K, B

    def precondition_gradients(self):
        if self.coupling.grad is not None:
            self.coupling.grad.div_(self.preconditioner)


class GraphColoringEnergy(PairwiseEnergy):
    """Full affine-input model: each pair reads all adjacency tokens.

    K_p(a) = W0_p + sum_t a_t D_{p,t}; no graph edge is routed to its own
    pair by construction. There are 7,332 / 18,660 parameters for 8 / 10
    vertices. Context contains upper-triangle adjacency tokens in pairs order.
    """

    def __init__(self, n_vertices: int = 8, n_colors: int = 3):
        pairs = torch.triu_indices(n_vertices, n_vertices, 1)
        super().__init__(n_vertices, n_colors, pairs)
        edges = pairs.shape[1]
        self.constant = nn.Parameter(torch.zeros(edges, n_colors, n_colors))
        self.edge_response = nn.Parameter(torch.zeros(edges, edges, n_colors, n_colors))
        self.unary_cost = nn.Parameter(torch.zeros(n_vertices, n_colors))

    def costs(self, context=None):
        if context is None or context.ndim != 2 or context.shape[1] != self.pairs.shape[1]:
            raise ValueError("Graph context must be (batch, number_of_vertex_pairs)")
        K = self.constant + torch.einsum("bt,ptac->bpac", context.to(self.constant.dtype), self.edge_response)
        return K, self.unary_cost
