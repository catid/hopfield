"""Literal state-space checks for the public optimizer-free baseline."""
import itertools
import pytest
import torch

from hopfield_cram import fit_never_equal


def test_exhaustive_binary_chain_energy_and_fields():
    train = torch.tensor([[0, 1, 0], [1, 0, 1]])
    model, counts = fit_never_equal(train, 2, chunk_size=1)
    assert counts.tolist() == [0, 2, 0]
    states = torch.tensor(list(itertools.product(range(2), repeat=3)))
    expected = torch.tensor([int(a == b) + int(b == c) for a, b, c in states.tolist()])
    assert torch.equal(model.board_energy(states), expected.float())
    fields = model.fields(model.one_hot(states))
    for row, board in enumerate(states):
        for cell in range(3):
            changed = board.repeat(2, 1)
            changed[:, cell] = torch.arange(2)
            energies = model.board_energy(changed)
            assert torch.equal(fields[row, cell] - fields[row, cell, 0], energies - energies[0])
    assert (model.board_energy(train) == 0).all()
    assert not any(p.requires_grad for p in model.parameters())


def test_count_chunking_and_value_permutation_preserve_model():
    rng = torch.Generator().manual_seed(19)
    labels = torch.randint(4, (31, 7), generator=rng)
    a, ca = fit_never_equal(labels, 4, chunk_size=1)
    b, cb = fit_never_equal(labels, 4, chunk_size=100)
    c, cc = fit_never_equal(torch.tensor([2, 0, 3, 1])[labels], 4)
    assert torch.equal(ca, cb) and torch.equal(ca, cc)
    assert torch.equal(a.pair_cost, b.pair_cost) and torch.equal(a.pair_cost, c.pair_cost)


@pytest.mark.parametrize('labels', [torch.empty(0, 3, dtype=torch.long), torch.tensor([[0., 1., 2.]]), torch.tensor([[0, -1, 1]]), torch.tensor([[0, 3, 1]])])
def test_reject_invalid_training_data(labels):
    with pytest.raises(ValueError):
        fit_never_equal(labels, 3)
