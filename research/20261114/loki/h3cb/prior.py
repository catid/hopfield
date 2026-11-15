"""Actual uniform truth sampling is separate from the fixed fitted hierarchy."""
import itertools
import math
import numpy as np


def fitted_prior_rows(cardinalities):
    orbits = list(itertools.combinations(range(1, 12), 2))
    choices = [chosen for k in [0, 1, 2] for chosen in itertools.combinations(range(55), k)]
    assert len(cardinalities) == len(choices) == 1541
    return [dict(candidate=cid, orbits=[list(orbits[j]) for j in chosen], source_population=int(M),
                 eligible=int(M) >= 8,
                 prior_integer_weight=1485//math.comb(55, len(chosen)) if int(M) >= 8 else 0)
            for cid, (chosen, M) in enumerate(zip(choices, cardinalities))]


def sampling_prior_rows(cardinalities):
    return [dict(candidate=row['candidate'], orbits=row['orbits'],
                 source_population=row['source_population'], eligible=row['eligible'],
                 sampling_integer_weight=int(row['eligible']))
            for row in fitted_prior_rows(cardinalities)]


def truth_from_ticket(rows, ticket):
    cumulative = np.cumsum([row['sampling_integer_weight'] for row in rows], dtype=np.int64)
    assert int(cumulative[-1]) == 1522 and 0 <= ticket < 1522
    candidate = int(np.searchsorted(cumulative, ticket, side='right'))
    assert rows[candidate]['eligible']
    return rows[candidate]['candidate']


def draw_truth(rows, seed):
    ticket = int(np.random.default_rng(seed).integers(1522))
    return truth_from_ticket(rows, ticket), ticket
