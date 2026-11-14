"""Maze-Hard extension: log-domain hidden fixed point and implicit adjoint.

Implements Appendix F (Eqs. 11–15) with 29 learned parameters. Hidden states
are stored as log(z), avoiding the underflow of values such as exp(-8000).
Neither the hidden fixed-point iterations nor answer relaxation are unrolled
through autograd. A custom backward solves the adjoint linear system.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from .energy import MazeEnergy, PairwiseEnergy
from .training import pseudolikelihood


class _LogHiddenFixedPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, open_cells, goal, target, inside, tolerance, max_steps):
        # Eq. 13: log z_i <- log(sum_neighbor z_j + 1_{i=G}) - log(m),
        # where the paper's m = 4 + exp(mu) > 4.
        log_mass = torch.logaddexp(mu, mu.new_tensor(math.log(4.0)))
        edges = open_cells[..., None] & open_cells[:, target] & inside
        log_source = torch.where(goal, mu.new_zeros(()), mu.new_full((), -torch.inf))
        value = torch.full(open_cells.shape, -torch.inf, dtype=mu.dtype, device=mu.device)
        for _ in range(max_steps):
            neighbors = value[:, target].masked_fill(~edges, -torch.inf)
            new = torch.logsumexp(torch.cat((neighbors, log_source[..., None]), -1), -1) - log_mass
            new = new.masked_fill(~open_cells, -torch.inf)
            finite = torch.isfinite(new) & torch.isfinite(value)
            diff = torch.where(finite, (new - value).abs(), 0)
            same_support = torch.equal(torch.isfinite(new), torch.isfinite(value))
            bound = tolerance * (1 + torch.where(torch.isfinite(new), new.abs(), 0))
            value = new
            if same_support and bool((diff <= bound).all()):
                break
        else:
            raise RuntimeError("Hidden log-domain Jacobi solve did not converge")
        live = edges & torch.isfinite(value[..., None]) & torch.isfinite(value[:, target])
        # Jacobian of the log-space fixed-point map, Q_ij = z_j/(m*z_i).
        exponent = torch.where(live, value[:, target] - value[..., None] - log_mass, 0)
        Q = exponent.exp() * live
        ctx.save_for_backward(Q, target, torch.isfinite(value), (mu - log_mass).exp())
        ctx.tolerance, ctx.max_steps = tolerance, max_steps
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_log_z):
        Q, target, finite, excess_fraction = ctx.saved_tensors
        rhs = torch.where(finite, grad_log_z, 0)
        adjoint = torch.zeros_like(rhs)
        indices = target.reshape(1, -1).expand(len(rhs), -1)
        # (I-Q)^T lambda = dL/dlog(z), solved without an autograd trajectory.
        for _ in range(ctx.max_steps):
            incoming = torch.zeros_like(rhs).scatter_add(1, indices, (Q * adjoint[..., None]).flatten(1))
            new = rhs + incoming
            bound = ctx.tolerance * (1 + new.abs().amax())
            converged = bool(((new - adjoint).abs() <= bound).all())
            adjoint = new
            if converged:
                break
        else:
            raise RuntimeError("Hidden implicit adjoint did not converge")
        grad_mu = -excess_fraction * adjoint.sum()
        return grad_mu, None, None, None, None, None, None


@dataclass
class MazeHardContext:
    """Input tokens and fixed hidden-derived attention, reusable during search."""
    tokens: Tensor
    log_hidden: Tensor
    attention: Tensor


class MazeHardEnergy(PairwiseEnergy):
    """29-parameter CRAM+z with the paper's goal-only source and shared mass.

    Token encoding is 0=wall, 1=open, 2=start, 3=goal. States and labels are
    flattened (batch, height*width[, 2]); tokens may also be (batch,H,W).
    `prepare(tokens)` settles the hidden layer once and returns a context to
    reuse during answer relaxation. `log_excess_mass` learns log(m-4).

    Self attention produces -kappa_0*alpha_ii*p_i^2. Its continuous derivative
    uses 2*p_i, but its exact binary conditional uses p_i^2=p_i. Therefore
    conditional_logits intentionally differs from -fields at hard states.
    """
    attention_offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, height=30, width=30, *, excess_mass=4 * math.exp(4),
                 tolerance=1e-6, max_hidden_steps=10000):
        maze = MazeEnergy(height, width)
        super().__init__(height * width, 2, maze.pairs)
        if excess_mass <= 0 or tolerance <= 0 or max_hidden_steps < 1:
            raise ValueError("Require positive mass, tolerance and iteration budget")
        self.maze = maze
        self.height, self.width = height, width
        self.tolerance, self.max_hidden_steps = tolerance, max_hidden_steps
        self.log_excess_mass = nn.Parameter(torch.tensor(math.log(excess_mass)))
        self.attention_bias = nn.Parameter(torch.zeros(5))
        self.link_strength = nn.Parameter(torch.zeros(5))
        targets, mask = [], []
        for row in range(height):
            for col in range(width):
                dest, valid = [], []
                for dr, dc in self.attention_offsets:
                    rr, cc = row + dr, col + dc
                    ok = 0 <= rr < height and 0 <= cc < width
                    dest.append(rr * width + cc if ok else row * width + col)
                    valid.append(ok)
                targets.append(dest)
                mask.append(valid)
        self.register_buffer('attention_target', torch.tensor(targets, dtype=torch.long))
        self.register_buffer('attention_inside', torch.tensor(mask, dtype=torch.bool))

    def _tokens(self, tokens):
        if tokens.ndim == 3 and tokens.shape[1:] == (self.height, self.width):
            tokens = tokens.flatten(1)
        if tokens.ndim != 2 or tokens.shape[1] != self.n_cells:
            raise ValueError("Tokens must be (batch,cells) or (batch,height,width)")
        if not bool(((tokens >= 0) & (tokens <= 3) & (tokens == tokens.long())).all()):
            raise ValueError("Maze tokens must be integers 0,1,2,3")
        if not bool(((tokens == 3).sum(1) == 1).all()):
            raise ValueError("Each maze requires exactly one goal token")
        return tokens.long()

    def settle_hidden(self, tokens: Tensor) -> Tensor:
        """Eq. 13 fixed point as log(z); backward runs its implicit adjoint."""
        tokens = self._tokens(tokens)
        return _LogHiddenFixedPoint.apply(self.log_excess_mass, tokens > 0, tokens == 3,
                                         self.attention_target[:, 1:], self.attention_inside[:, 1:],
                                         self.tolerance, self.max_hidden_steps)

    def prepare(self, tokens: Tensor) -> MazeHardContext:
        tokens = self._tokens(tokens)
        log_z = self.settle_hidden(tokens)
        keys = log_z[:, self.attention_target]
        live = self.attention_inside & torch.isfinite(keys)
        any_live = live.any(-1, keepdim=True)
        keys = torch.where(live, keys, -torch.inf)
        # The ratio in Eq. 14 is 0/0 in a disconnected window. The release's
        # explicit convention is attention from beta alone in that window.
        keys = torch.where(any_live, keys, torch.zeros_like(keys))
        scores = (keys + self.attention_bias).masked_fill(~self.attention_inside, -torch.inf)
        return MazeHardContext(tokens, log_z, scores.softmax(-1))

    def _context(self, context):
        if context is None:
            raise ValueError("MazeHardEnergy requires maze token context")
        return context if isinstance(context, MazeHardContext) else self.prepare(context)

    def _clamp(self, probabilities, context):
        self._check_states(probabilities)
        given = self.one_hot((context.tokens >= 2).long())
        return torch.where((context.tokens == 1)[..., None], probabilities, given)

    def _support(self, probability_on, context):
        strength = context.attention * self.link_strength
        targets = self.attention_target[:, 1:]
        outgoing = (strength[..., 1:] * probability_on[:, targets]).sum(-1)
        incoming = torch.zeros_like(probability_on).scatter_add(
            1, targets.flatten()[None].expand(len(probability_on), -1),
            (strength[..., 1:] * probability_on[..., None]).flatten(1))
        return outgoing + incoming, strength[..., 0]

    def energy(self, p, context=None, *, inverse_temperature=None):
        ctx = self._context(context)
        p = self._clamp(p, ctx)
        base = self.maze.energy(p, ctx.tokens == 1, inverse_temperature=inverse_temperature)
        on = p[..., 1]
        linked = ctx.attention * self.link_strength * on[..., None] * on[:, self.attention_target]
        return base - linked.sum((1, 2))

    def fields(self, p, context=None):
        """Continuous visible-state fields including the quadratic self term."""
        ctx = self._context(context)
        p = self._clamp(p, ctx)
        base = self.maze.fields(p, ctx.tokens == 1)
        neighbor, self_term = self._support(p[..., 1], ctx)
        shift = neighbor + 2 * self_term * p[..., 1]
        return base - torch.stack((torch.zeros_like(shift), shift), -1)

    def conditional_logits(self, labels, context=None):
        """Exact binary PL conditionals, including self cost only once."""
        ctx = self._context(context)
        p = self._clamp(self.one_hot(labels), ctx)
        base = self.maze.fields(p, ctx.tokens == 1)
        neighbor, self_term = self._support(p[..., 1], ctx)
        shift = neighbor + self_term
        return -base + torch.stack((torch.zeros_like(shift), shift), -1)

    def costs(self, context=None):
        raise NotImplementedError("The latent extension has a quadratic self term; use energy/fields")


def fit_maze_hard(model: MazeHardEnergy, tokens: Tensor, labels: Tensor, *,
                  updates: int = 4000, learning_rate: float = 0.1,
                  batch_size: int = 64, seed: int = 0) -> list[float]:
    """Reference PL+Adam recipe for Appendix F; no backprop through relaxation.

    Samples training mazes with replacement. Its torch RNG differs from the
    release's NumPy RNG, so exact checkpoint hashes are not expected to match.
    """
    if updates < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError("Require positive updates, learning rate and batch size")
    tokens = model._tokens(tokens)
    if labels.shape != tokens.shape or len(tokens) == 0:
        raise ValueError("Labels must match nonempty flattened maze tokens")
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(updates):
        idx = torch.randint(len(tokens), (batch_size,), generator=generator, device=tokens.device)
        t = tokens[idx]
        optimizer.zero_grad(set_to_none=True)
        loss = pseudolikelihood(model, labels[idx], t == 1, t)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite Maze-Hard pseudolikelihood")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@dataclass
class MazeHardSearchResult:
    boards: Tensor
    probabilities: Tensor
    steps: Tensor
    stopped: Tensor
    context: MazeHardContext


@torch.no_grad()
def solve_maze_hard(model: MazeHardEnergy, tokens: Tensor, *, steps: int = 2048,
                    step_size: float = 0.2, initial_logit: float = 3.0,
                    check_every: int = 16, tolerance: float = 1e-6) -> MazeHardSearchResult:
    """Paper's answer search: continuous flow, rounded-board local stopping.

    Settle the hidden layer once, start free cells at logit 3, update Eq. 15
    with eta=0.2, and every 16 steps stop if no binary flip lowers energy by
    more than 1e-6. This local certificate is not a puzzle-validity check.
    """
    if steps < 1 or step_size <= 0 or check_every < 1 or tolerance < 0:
        raise ValueError("Invalid answer-relaxation budget or stopping settings")
    context = model.prepare(tokens)
    tokens = context.tokens
    free = tokens == 1
    u = torch.full(tokens.shape, initial_logit, dtype=model.log_excess_mass.dtype, device=tokens.device)
    given = (tokens >= 2).long()
    stopped = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    used = torch.full((len(tokens),), steps, dtype=torch.long, device=tokens.device)
    for step in range(1, steps + 1):
        p = model._clamp(torch.stack((1 - u.sigmoid(), u.sigmoid()), -1), context)
        fields = model.fields(p, context)
        drive = fields[..., 0] - fields[..., 1]
        updated = u + step_size * (drive - u)
        u = torch.where(stopped[:, None] | ~free, u, updated)
        if step % check_every == 0:
            board = torch.where(free, (u > 0).long(), given)
            local = -model.conditional_logits(board, context)
            own = local.gather(-1, board[..., None]).squeeze(-1)
            other = local.gather(-1, (1 - board)[..., None]).squeeze(-1)
            passed = ((other - own >= -tolerance) | ~free).all(1)
            used = torch.where(passed & ~stopped, torch.full_like(used, step), used)
            stopped |= passed
            if bool(stopped.all()):
                break
    p = model._clamp(torch.stack((1 - u.sigmoid(), u.sigmoid()), -1), context)
    board = torch.where(free, (u > 0).long(), given)
    return MazeHardSearchResult(board, p, used, stopped, context)
