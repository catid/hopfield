"""Independent mathematical oracles; no released predictions are test targets."""
from itertools import product

import pytest
import torch
from torch.nn import functional as F

from hopfield_cram import (
    DensePairwiseEnergy, GraphColoringEnergy, MazeEnergy, NQueensEnergy,
    SudokuEnergy, TrainingConfig, euler_step, fit_coloring_thresholds,
    fit_pseudolikelihood, fit_statistics, focused_gibbs, pseudolikelihood, relax,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(123)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def random_model(cells=4, values=3):
    model = DensePairwiseEnergy(cells, values).double()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.4)
    return model


def brute_energy(model, probabilities, context=None):
    """Literal Eq. 1 pair sum, avoiding vectorized model fields/energy."""
    K, B = model.costs(context)
    result = []
    for b in range(len(probabilities)):
        terms = probabilities.new_zeros(())
        for edge, (i, j) in enumerate(model.pairs.T.tolist()):
            table = K[edge] if K.ndim == 3 else K[b, edge]
            for a in range(model.n_values):
                for c in range(model.n_values):
                    terms = terms + probabilities[b, i, a] * table[a, c] * probabilities[b, j, c]
        for i in range(model.n_cells):
            for a in range(model.n_values):
                bias = B[i, a] if B.ndim == 2 else B[b, i, a]
                terms = terms + bias * probabilities[b, i, a]
        result.append(terms)
    return torch.stack(result)


def test_energy_and_fields_against_independent_pair_sum_and_autograd():
    model = random_model()
    p = torch.rand(2, 4, 3, dtype=torch.double, requires_grad=True)
    expected = brute_energy(model, p)
    torch.testing.assert_close(model.energy(p), expected)
    derivative, = torch.autograd.grad(expected.sum(), p)
    torch.testing.assert_close(model.fields(p), derivative)
    W = model.dense_interactions()
    torch.testing.assert_close(W, W.permute(1, 0, 3, 2))
    assert torch.count_nonzero(W[torch.arange(4), torch.arange(4)]) == 0
    dense = 0.5 * torch.einsum('bia,ijac,bjc->b', p, W, p) + (p * model.unary_cost).sum((1, 2))
    torch.testing.assert_close(dense, expected)


def test_exact_conditionals_and_pl_parameter_gradients_by_enumeration():
    model = random_model(3, 2)
    labels = torch.tensor([[0, 1, 0], [1, 1, 0]])
    free = torch.tensor([[True, False, True], [False, True, True]])
    all_logits = []
    for batch in range(2):
        logits = []
        for cell in range(3):
            energies = []
            for value in range(2):
                changed = labels[batch:batch+1].clone()
                changed[0, cell] = value
                energies.append(brute_energy(model, F.one_hot(changed, 2).double())[0])
            logits.append(-torch.stack(energies))
        all_logits.append(torch.stack(logits))
    oracle = torch.stack(all_logits)
    actual = model.conditional_logits(labels)
    torch.testing.assert_close(actual.log_softmax(-1), oracle.log_softmax(-1))
    brute_loss = -(oracle.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1) * free).sum() / free.sum()
    actual_loss = pseudolikelihood(model, labels, free)
    torch.testing.assert_close(actual_loss, brute_loss)
    actual_grads = torch.autograd.grad(actual_loss, tuple(model.parameters()))
    oracle_grads = torch.autograd.grad(brute_loss, tuple(model.parameters()))
    for a, b in zip(actual_grads, oracle_grads):
        torch.testing.assert_close(a, b)


def test_cell_and_board_weighted_objectives_are_explicitly_different():
    model = random_model(3, 2)
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    free = torch.tensor([[True, False, False], [True, True, True]])
    one = pseudolikelihood(model, labels[:1], free[:1])
    three = pseudolikelihood(model, labels[1:], free[1:])
    torch.testing.assert_close(pseudolikelihood(model, labels, free), (one + 3 * three) / 4)
    torch.testing.assert_close(pseudolikelihood(model, labels, free, reduction='boards'), (one + three) / 2)
    with pytest.raises(ValueError, match='at least one free'):
        pseudolikelihood(model, labels, torch.zeros_like(free))


@pytest.mark.parametrize('factory,count', [
    (SudokuEnergy, 263169), (MazeEnergy, 18),
    (lambda: NQueensEnergy(8), 113), (lambda: NQueensEnergy(10), 181),
    (lambda: GraphColoringEnergy(8), 7332), (lambda: GraphColoringEnergy(10), 18660),
])
def test_paper_parameter_counts_and_zero_initialization(factory, count):
    model = factory()
    assert model.parameter_count == count
    assert all(torch.count_nonzero(p) == 0 for p in model.parameters())


def test_maze_stencil_boundaries_transposes_and_unary_mask():
    model = MazeEnergy(2, 3).double()
    with torch.no_grad():
        model.tables.copy_(torch.arange(16, dtype=torch.double).reshape(4, 2, 2) / 9)
        model.unary_cost.copy_(torch.tensor([-0.3, 0.8]))
    p = torch.rand(2, 6, 2, dtype=torch.double)
    context = torch.tensor([[False, True, True, True, True, False], [True] * 6])
    assert model.pairs.shape[1] == 11
    # Geometric enumeration does not use model offsets/pairs to construct fields.
    expected = torch.zeros_like(p)
    for b, row, col, a in product(range(2), range(2), range(3), range(2)):
        i = row * 3 + col
        expected[b, i, a] += model.unary_cost[a] * context[b, i]
        for rr, cc in product(range(2), range(3)):
            dr, dc = rr - row, cc - col
            if (dr, dc) == (0, 0) or max(abs(dr), abs(dc)) > 1:
                continue
            if dr > 0 or (dr == 0 and dc > 0):
                table = model.tables[((0, 1), (1, -1), (1, 0), (1, 1)).index((dr, dc))]
            else:
                table = model.tables[((0, 1), (1, -1), (1, 0), (1, 1)).index((-dr, -dc))].T
            expected[b, i, a] += (table[a] * p[b, rr * 3 + cc]).sum()
    torch.testing.assert_close(model.fields(p, context), expected)
    torch.testing.assert_close(model.energy(p, context), brute_energy(model, p, context))


def test_queens_offsets_and_preconditioner_match_geometric_pair_counts():
    model = NQueensEnergy(3).double()
    with torch.no_grad():
        model.coupling.copy_(torch.arange(len(model.coupling), dtype=torch.double) / 7)
        model.unary_cost.fill_(-0.2)
    board = torch.tensor([[1, 0, 1, 0, 1, 0, 0, 0, 1]])
    offsets = {tuple(x): k for k, x in enumerate(model.offsets.tolist())}
    expected = -0.2 * board.sum().double()
    counts = torch.zeros(len(offsets), dtype=torch.double)
    for i in range(9):
        for j in range(i + 1, 9):
            k = offsets[j // 3 - i // 3, j % 3 - i % 3]
            counts[k] += 1
            expected += model.coupling[k] * board[0, i] * board[0, j]
    torch.testing.assert_close(model.board_energy(board)[0], expected)
    torch.testing.assert_close(model.preconditioner, counts / counts.mean(), rtol=1e-6, atol=1e-7)
    model.coupling.grad = torch.ones_like(model.coupling)
    model.precondition_gradients()
    torch.testing.assert_close(model.coupling.grad, 1 / model.preconditioner)


def test_coloring_all_tokens_can_affect_each_pair_and_term_stopping():
    model = GraphColoringEnergy(3, 3).double()
    with torch.no_grad():
        model.edge_response[0, 2, 1, 2] = 5  # Edge (1,2) controls the cost for pair (0,1).
        for edge in range(3):
            model.edge_response[edge, edge] += torch.eye(3, dtype=torch.double)
    context = torch.tensor([[1., 1., 1.], [1., 1., 0.]], dtype=torch.double)
    K, _ = model.costs(context)
    assert K[0, 0, 1, 2] == 5 and K[1, 0, 1, 2] == 0
    labels = torch.tensor([[0, 1, 2], [0, 1, 2]])
    p = model.one_hot(labels)
    torch.testing.assert_close(model.energy(p, context), brute_energy(model, p, context))
    thresholds = fit_coloring_thresholds(model, labels, context)
    assert thresholds.accepts(model, labels, context).all()
    invalid = torch.zeros_like(labels)
    assert not thresholds.accepts(model, invalid, context).any()


def test_continuous_energy_lyapunov_identity_at_arbitrary_beta():
    model = random_model(4, 3)
    beta = 1.7
    v = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
    p = (beta * v).softmax(-1)
    energy = model.energy(p, inverse_temperature=beta)
    gradient, = torch.autograd.grad(energy.sum(), v)
    velocity = -v - model.fields(p)
    actual = (gradient * velocity).sum()
    jacobian = beta * (torch.diag_embed(p) - p[..., :, None] * p[..., None, :])
    expected = -torch.einsum('bia,biac,bic->', velocity, jacobian, velocity)
    torch.testing.assert_close(actual, expected)
    assert actual < 0
    hard = F.one_hot(torch.zeros(2, 4, dtype=torch.long), 3).double()
    torch.testing.assert_close(model.energy(hard, inverse_temperature=beta), model.energy(hard))


def test_euler_clamps_and_small_step_descent_without_training_graph():
    model = random_model(3, 2)
    logits = torch.randn(2, 3, 2, dtype=torch.double, requires_grad=True)
    given = torch.tensor([[0, 1, 1], [1, 0, 0]])
    free = torch.tensor([[False, True, True], [True, False, True]])
    v, p = euler_step(model, logits, given, free, step_size=0.01)
    assert not v.requires_grad and not p.requires_grad
    torch.testing.assert_close(p[~free], model.one_hot(given)[~free])
    torch.testing.assert_close(v[~free], logits[~free])
    result = relax(model, logits, given, free, steps=50, step_size=0.01)
    assert torch.all(result.energies.diff(dim=0) <= 1e-12)
    torch.testing.assert_close(result.probabilities[~free], model.one_hot(given)[~free])


def test_zero_temperature_gibbs_clamps_and_does_not_increase_energy():
    model = random_model(4, 3)
    board = torch.randint(3, (5, 4))
    free = torch.tensor([[False, True, True, True]] * 5)
    result = focused_gibbs(model, board, free, max_moves=100, temperature=0, record_trace=True)
    assert result.stopped.all()
    assert torch.all(result.energy_trace.diff(dim=0) <= 1e-12)
    assert torch.equal(result.boards[:, 0], board[:, 0])
    fields = model.fields(model.one_hot(result.boards))
    own = fields.gather(-1, result.boards[..., None]).squeeze(-1)
    assert torch.all((own <= fields.amin(-1) + 1e-12) | ~free)
    torch.testing.assert_close(result.energies, model.board_energy(result.boards))


def test_gibbs_reproducibility_best_board_and_all_clamped_inputs():
    model = random_model(3, 2)
    initial = torch.tensor([[0, 1, 0], [1, 0, 1]])
    free = torch.tensor([[False, True, True], [False, False, False]])
    a = focused_gibbs(model, initial, free, max_moves=20, stop='none', seed=9, record_trace=True)
    b = focused_gibbs(model, initial, free, max_moves=20, stop='none', seed=9, record_trace=True)
    assert torch.equal(a.boards, b.boards)
    assert torch.equal(a.energy_trace, b.energy_trace)
    assert a.moves.tolist() == [20, 0]
    torch.testing.assert_close(a.energies, a.energy_trace.amin(0))
    assert torch.equal(a.boards[1], initial[1])
    fixed = focused_gibbs(model, initial[1:], free[1:], max_moves=5)
    assert fixed.stopped.all() and fixed.moves.item() == 0


def test_local_training_learns_two_solution_distribution_and_energy_stop():
    model = DensePairwiseEnergy(2, 2).double()
    labels = torch.tensor([[0, 1], [1, 0]])
    free = torch.ones_like(labels, dtype=torch.bool)
    before = float(pseudolikelihood(model, labels, free).detach())
    losses = fit_pseudolikelihood(model, labels, free, TrainingConfig(40, 1.0, seed=3))
    assert losses[-1] < before / 10
    stats = fit_statistics(model, labels)
    invalid = torch.tensor([[0, 0], [1, 1]])
    assert torch.all(model.board_energy(invalid) > stats.energy_threshold + 0.1)
    result = focused_gibbs(model, invalid, free, temperature=0, stop='energy',
                          threshold=stats.energy_threshold, tolerance=1e-12)
    assert result.stopped.all()
    assert torch.all(result.boards[:, 0] != result.boards[:, 1])
    assert stats.minimum_margin > 0


def test_no_interaction_single_cell_and_validation():
    model = DensePairwiseEnergy(1, 2).double()
    with torch.no_grad():
        model.unary_cost.copy_(torch.tensor([[0., 2.]]))
    labels = torch.tensor([[1]])
    torch.testing.assert_close(model.board_energy(labels), torch.tensor([2.], dtype=torch.double))
    result = focused_gibbs(model, labels, torch.ones_like(labels, dtype=torch.bool), temperature=0)
    assert result.boards.item() == 0 and result.moves.item() == 1
    with pytest.raises(ValueError, match='Temperature'):
        focused_gibbs(model, labels, torch.ones_like(labels, dtype=torch.bool), temperature=-1)


def _dense_hidden_oracle(tokens, height, width, mu):
    """Solve Eq. 11 directly in float64 on tiny graphs, independent of Jacobi."""
    matrices = []
    for board in tokens:
        matrix = torch.eye(height * width, dtype=mu.dtype) * (4 + mu.exp())
        for row, col in product(range(height), range(width)):
            i = row * width + col
            if board[i] == 0:
                continue
            for rr, cc in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                if 0 <= rr < height and 0 <= cc < width and board[rr*width+cc] > 0:
                    matrix[i, rr*width+cc] = -1
        matrices.append(matrix)
    return torch.linalg.solve(torch.stack(matrices), (tokens == 3).to(mu.dtype).unsqueeze(-1)).squeeze(-1)


def test_hidden_log_fixed_point_and_implicit_gradient_match_dense_linear_solve():
    from hopfield_cram import MazeHardEnergy
    model = MazeHardEnergy(3, 3, excess_mass=2.0, tolerance=1e-13).double()
    tokens = torch.tensor([[2, 1, 0, 0, 1, 3, 1, 0, 1]])
    logz = model.settle_hidden(tokens)
    mu = model.log_excess_mass.detach().clone().requires_grad_()
    expected = _dense_hidden_oracle(tokens, 3, 3, mu)
    finite = expected > 0
    torch.testing.assert_close(logz[finite], expected[finite].log(), rtol=1e-11, atol=1e-11)
    assert torch.isneginf(logz[~finite]).all()
    weights = torch.arange(9, dtype=torch.double).reshape(1, 9) / 9 - 0.2
    actual_grad, = torch.autograd.grad((logz[finite] * weights[finite]).sum(), model.log_excess_mass)
    expected_grad, = torch.autograd.grad((expected[finite].log() * weights[finite]).sum(), mu)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
    assert model.parameter_count == 29


def test_hidden_log_domain_survives_when_float64_z_underflows():
    from hopfield_cram import MazeHardEnergy
    model = MazeHardEnergy(1, 100, excess_mass=22000., tolerance=1e-12).double()
    tokens = torch.ones(1, 100, dtype=torch.long)
    tokens[0, 0], tokens[0, -1] = 2, 3
    context = model.prepare(tokens)
    assert torch.isfinite(context.log_hidden).all()
    assert context.log_hidden[0, 0] < -900
    assert context.log_hidden[0, 0].exp() == 0  # The log-state still exists accurately.
    torch.testing.assert_close(context.attention.sum(-1), torch.ones(1, 100, dtype=torch.double))
    gradient, = torch.autograd.grad(context.log_hidden[0, 0], model.log_excess_mass)
    assert torch.isfinite(gradient) and gradient < -99


def test_latent_visible_energy_fields_and_exact_pl_handle_quadratic_self_cost():
    from hopfield_cram import MazeHardEnergy
    model = MazeHardEnergy(2, 3, excess_mass=3., tolerance=1e-13).double()
    with torch.no_grad():
        model.maze.tables.normal_(0, 0.2)
        model.maze.unary_cost.normal_(0, 0.2)
        model.link_strength.copy_(torch.tensor([2., 0.1, -0.3, 0.4, 0.5]))
        model.attention_bias.normal_(0, 0.3)
    tokens = torch.tensor([[2, 1, 0, 1, 1, 3]])
    context = model.prepare(tokens)
    on = torch.rand(1, 6, dtype=torch.double, requires_grad=True)
    p = torch.stack((1 - on, on), -1)
    grad, = torch.autograd.grad(model.energy(p, context).sum(), on, retain_graph=True)
    field = model.fields(p, context)
    torch.testing.assert_close(grad[tokens == 1], (field[..., 1] - field[..., 0])[tokens == 1])
    assert torch.count_nonzero(grad[tokens != 1]) == 0
    label = torch.tensor([[1, 0, 0, 1, 1, 1]])
    logits = model.conditional_logits(label, context)
    naive = -model.fields(model.one_hot(label), context)
    assert not torch.allclose(logits[tokens == 1], naive[tokens == 1])
    for cell in (tokens[0] == 1).nonzero(as_tuple=True)[0]:
        values = label.repeat(2, 1)
        values[0, cell], values[1, cell] = 0, 1
        energies = torch.stack([model.board_energy(v[None], context)[0] for v in values])
        torch.testing.assert_close(logits[0, cell].log_softmax(-1), (-energies).log_softmax(-1))
    # Check the complete visible PL-to-hidden-mass gradient by central
    # differences, independently of the custom adjoint backward.
    model.zero_grad(set_to_none=True)
    actual = pseudolikelihood(model, label, tokens == 1, tokens)
    actual.backward()
    analytic = model.log_excess_mass.grad.clone()
    saved = model.log_excess_mass.detach().clone()
    eps = 1e-5
    with torch.no_grad():
        model.log_excess_mass.copy_(saved + eps)
        upper = pseudolikelihood(model, label, tokens == 1, tokens)
        model.log_excess_mass.copy_(saved - eps)
        lower = pseudolikelihood(model, label, tokens == 1, tokens)
        model.log_excess_mass.copy_(saved)
    torch.testing.assert_close(analytic, (upper - lower) / (2 * eps), rtol=2e-6, atol=1e-9)


def test_latent_disconnected_attention_and_small_adam_fit():
    from hopfield_cram import MazeHardEnergy, fit_maze_hard
    model = MazeHardEnergy(3, 3, excess_mass=5., tolerance=1e-12).double()
    tokens = torch.tensor([[2, 1, 0, 0, 1, 3, 1, 0, 1]])
    labels = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0, 0]])
    ctx = model.prepare(tokens)
    assert torch.isneginf(ctx.log_hidden[0, 6])  # Disconnected open cell.
    assert torch.isfinite(ctx.attention).all()
    torch.testing.assert_close(ctx.attention.sum(-1), torch.ones(1, 9, dtype=torch.double))
    base = model.maze.board_energy(labels, tokens == 1)
    torch.testing.assert_close(model.board_energy(labels, ctx), base)  # kappa=0.
    losses = fit_maze_hard(model, tokens, labels, updates=5, batch_size=2, learning_rate=0.1)
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_pinned_upstream_sudoku_energy_pl_and_gradients():
    import importlib
    import sys
    from pathlib import Path
    pytest.importorskip('cram')
    experiments = Path(__file__).resolve().parents[1] / 'upstream' / 'experiments'
    if not experiments.is_dir():
        pytest.skip('Pinned upstream checkout not present')
    sys.path.insert(0, str(experiments))
    try:
        upstream = importlib.import_module('sudoku')
    finally:
        sys.path.remove(str(experiments))
    model = SudokuEnergy()
    with torch.no_grad():
        model.pair_cost.normal_(0, 0.02)
        model.unary_cost.normal_(0, 0.02)
    labels = torch.randint(9, (3, 81))
    free = torch.rand(3, 81) > 0.25
    W = model.pair_cost.detach().clone().requires_grad_()
    B = model.unary_cost.detach().clone().requires_grad_()
    S = upstream.full_S(W, upstream.pair_index('cpu'))
    torch.testing.assert_close(model.board_energy(labels), upstream.board_energy(S, B, labels), atol=2e-6, rtol=2e-6)
    ours = pseudolikelihood(model, labels, free)
    theirs = upstream.pl_loss(S, B, labels, free)
    torch.testing.assert_close(ours, theirs)
    actual = torch.autograd.grad(ours, tuple(model.parameters()))
    expected = torch.autograd.grad(theirs, (W, B))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-5)


def test_pinned_upstream_maze_sign_mapping_energy_pl_and_gradients():
    upstream = pytest.importorskip('cram.cram_maze')
    maze = pytest.importorskip('cram.maze_unique')
    model = MazeEnergy(2, 3)
    theirs = upstream.CRAMMazeFreeEnergy()
    with torch.no_grad():
        model.tables.normal_(0, 0.1)
        model.unary_cost.normal_(0, 0.1)
        theirs.T.copy_(model.tables)
        theirs.b.copy_(-model.unary_cost)
    tokens = torch.tensor([[[2, 1, 0], [0, 1, 3]], [[2, 1, 1], [0, 1, 3]]])
    labels = torch.tensor([[[1, 1, 0], [0, 1, 1]], [[1, 1, 0], [0, 1, 1]]])
    free = tokens == 1
    y = labels.flatten(1)
    s = model.one_hot(y).transpose(1, 2).reshape(2, 2, 2, 3)
    torch.testing.assert_close(model.board_energy(y, free.flatten(1)), theirs.energy(s, tokens, free))
    ours = pseudolikelihood(model, y, free.flatten(1), free.flatten(1))
    expected = maze.pl_loss(theirs, tokens, labels, free)
    torch.testing.assert_close(ours, expected)
    g_t, g_b = torch.autograd.grad(ours, (model.tables, model.unary_cost))
    u_t, u_b = torch.autograd.grad(expected, (theirs.T, theirs.b))
    torch.testing.assert_close(g_t, u_t, atol=5e-8, rtol=1e-6)
    torch.testing.assert_close(g_b, -u_b, atol=5e-8, rtol=1e-6)


def test_pinned_upstream_coloring_full_affine_energy_and_gradients():
    upstream = pytest.importorskip('cram.cram_graph_coloring')
    model = GraphColoringEnergy(4).double()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    context = torch.randint(2, (3, 6)).double()
    labels = torch.randint(3, (3, 4))
    params = {k: v.detach().clone().requires_grad_() for k, v in
              zip(('W0', 'D', 'B'), (model.constant, model.edge_response, model.unary_cost))}
    K = upstream.assemble(params, context)
    S = upstream.pair_tensor(K, 4)
    torch.testing.assert_close(model.board_energy(labels, context), upstream.energy_t(S, params['B'], labels))
    ours = pseudolikelihood(model, labels, context=context)
    theirs = upstream.pl_loss(params, context, labels, 4)
    torch.testing.assert_close(ours, theirs)
    for a, b in zip(torch.autograd.grad(ours, tuple(model.parameters())),
                    torch.autograd.grad(theirs, tuple(params.values()))):
        torch.testing.assert_close(a, b)


def test_pinned_upstream_latent_attention_energy_pl_and_parameter_gradients():
    from hopfield_cram import MazeHardEnergy
    upstream = pytest.importorskip('cram.cram_maze_hard_graph')
    ours = MazeHardEnergy(2, 3, excess_mass=10., tolerance=1e-12).double()
    theirs = upstream.GraphLatentEnergyMU(mass='shared', mass_fn='exp', source='goal', mass0=10.).double()
    theirs.degree_in_diag = False
    theirs.adjoint_tol = 1e-12
    permutation = [ours.attention_offsets.index(d) for d in theirs.aoffs]
    with torch.no_grad():
        ours.maze.tables.normal_(0, 0.2)
        ours.maze.unary_cost.normal_(0, 0.2)
        ours.attention_bias.normal_(0, 0.3)
        ours.link_strength.normal_(0, 0.5)
        theirs.T.copy_(ours.maze.tables)
        theirs.b.copy_(-ours.maze.unary_cost)
        theirs.mu.copy_(ours.log_excess_mass)
        theirs.beta.copy_(ours.attention_bias[permutation])
        theirs.kappa.copy_(ours.link_strength[permutation])
    tokens = torch.tensor([[[2, 1, 0], [1, 1, 3]]])
    labels = torch.tensor([[[1, 1, 0], [0, 1, 1]]])
    context = ours.prepare(tokens)
    psi, _ = theirs.settle(tokens, grad=True, tol=1e-12)
    attention = theirs.couplings(tokens, psi)
    torch.testing.assert_close(context.log_hidden.reshape_as(psi), -psi, equal_nan=True)
    a = context.attention[..., permutation].transpose(1, 2).reshape_as(attention)
    torch.testing.assert_close(a, attention)
    torch.testing.assert_close(ours.board_energy(labels.flatten(1), context),
                               theirs.energy(tokens, labels.double(), attention), atol=1e-10, rtol=1e-10)
    ours_loss = pseudolikelihood(ours, labels.flatten(1), tokens.flatten(1) == 1, context)
    theirs_loss = theirs.pl_loss(tokens, labels.double(), attention)
    torch.testing.assert_close(ours_loss, theirs_loss)
    ours_loss.backward()
    theirs_loss.backward()
    torch.testing.assert_close(ours.maze.tables.grad, theirs.T.grad)
    torch.testing.assert_close(-ours.maze.unary_cost.grad, theirs.b.grad)
    torch.testing.assert_close(ours.attention_bias.grad[permutation], theirs.beta.grad)
    torch.testing.assert_close(ours.link_strength.grad[permutation], theirs.kappa.grad)
    torch.testing.assert_close(ours.log_excess_mass.grad.reshape(1), theirs.mu.grad, atol=1e-9, rtol=1e-7)


def test_maze_hard_paper_relaxation_and_local_stopping_match_upstream():
    from hopfield_cram import MazeHardEnergy, solve_maze_hard
    upstream = pytest.importorskip('cram.cram_maze_hard_graph')
    search = pytest.importorskip('cram.maze_h3')
    ours = MazeHardEnergy(3, 3, excess_mass=10., tolerance=1e-12).double()
    theirs = upstream.GraphLatentEnergyMU(mass='shared', mass_fn='exp', source='goal', mass0=10.).double()
    theirs.degree_in_diag = False
    permutation = [ours.attention_offsets.index(d) for d in theirs.aoffs]
    with torch.no_grad():
        ours.maze.tables.normal_(0, 0.3)
        ours.maze.unary_cost.normal_(0, 0.3)
        ours.link_strength.normal_(0, 1.0)
        theirs.T.copy_(ours.maze.tables)
        theirs.b.copy_(-ours.maze.unary_cost)
        theirs.mu.copy_(ours.log_excess_mass)
        theirs.beta.copy_(ours.attention_bias[permutation])
        theirs.kappa.copy_(ours.link_strength[permutation])
    tokens = torch.tensor([[[2, 1, 0], [1, 1, 3], [0, 1, 0]],
                           [[2, 1, 1], [0, 0, 1], [1, 1, 3]]])
    actual = solve_maze_hard(ours, tokens, steps=128)
    psi, _ = theirs.settle(tokens, tol=1e-12)
    attention = theirs.couplings(tokens, psi)
    board, steps, stopped = search.descend(theirs, tokens, attention, steps=128, eta=0.2)
    assert torch.equal(actual.boards, board.flatten(1).long())
    assert torch.equal(actual.steps, steps)
    assert torch.equal(actual.stopped, stopped)
    assert torch.equal(actual.boards[tokens.flatten(1) != 1], (tokens.flatten(1)[tokens.flatten(1) != 1] >= 2).long())
