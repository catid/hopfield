"""Label-only three-query decoder and exact symmetric-channel enumeration."""
from fractions import Fraction
from itertools import product
import numpy as np
NOISE_P = float.fromhex('0x1.3333333333333p-4')


def decode(first, second, third):
    assert first.shape == second.shape == third.shape
    assert all(((a >= 0) & (a <= 8)).all() for a in [first, second, third])
    return np.where((first == second) | (first == third), first,
                    np.where(second == third, second, first)).astype(np.int8)


def corrupted(clean, noise):
    uniforms, offsets = noise['iid_uniforms'], noise['iid_offsets']
    assert uniforms.shape == offsets.shape == clean.shape
    assert ((offsets >= 1) & (offsets <= 8)).all()
    return np.where(uniforms < NOISE_P, (clean.astype(np.int16)+offsets) % 9, clean).astype(np.int8)


def count_values(given, labels):
    assert given.shape == labels.shape and given.shape[1] == 81
    values = np.full(labels.shape, -1, np.int8)
    values[given] = labels[given]
    ii, jj = np.triu_indices(81, 1)
    common = given[:, ii] & given[:, jj]
    return dict(given=given, values=values, joint_count=common.sum(0),
                equal_count=(common & (values[:, ii] == values[:, jj])).sum(0))


def observations(mask, clean, noise):
    first = corrupted(clean, {key: noise['first_'+key] for key in ['iid_uniforms', 'iid_offsets']})
    repeats = [corrupted(clean[:90], {key: noise[f'repeat{r}_'+key] for key in ['iid_uniforms', 'iid_offsets']}) for r in [1, 2]]
    decoded = decode(first[:90], *repeats)
    return dict(single270=count_values(mask, first), triple90=count_values(mask[:90], decoded))


def channel(probability):
    """Exhaustive rational channel for all9 true digits and all729 observed triples."""
    result = []
    for truth in range(9):
        outputs = [Fraction(0) for _ in range(9)]
        ties = Fraction(0)
        for labels in product(range(9), repeat=3):
            weight = Fraction(1)
            for label in labels:
                weight *= 1-probability if label == truth else probability/8
            counts = [labels.count(label) for label in range(9)]
            maximum = max(counts)
            output = labels[0] if maximum == 1 else counts.index(maximum)
            outputs[output] += weight
            if maximum == 1:
                ties += weight
        assert sum(outputs) == 1
        assert len({outputs[d] for d in range(9) if d != truth}) == 1
        result.append(dict(truth=truth, probabilities=[str(value) for value in outputs],
                           error=str(1-outputs[truth]), error_decimal=float(1-outputs[truth]),
                           all_distinct_probability=str(ties), all_distinct_decimal=float(ties)))
    assert len({row['error'] for row in result}) == 1
    return dict(input_error=str(probability), input_error_decimal=float(probability), rows=result,
                label_permutation_symmetric=True, temporal_tie_preference='first query', outcomes_enumerated=9*729)


def exact_channels():
    threshold = Fraction.from_float(NOISE_P)
    scaled = threshold*(1 << 53)
    # NumPy's standard float64 uniform is k/2^53, 0<=k<2^53; strict U<threshold.
    accepted = (scaled.numerator+scaled.denominator-1)//scaled.denominator
    actual = Fraction(accepted, 1 << 53)
    return dict(noise_threshold_hex=NOISE_P.hex(), noise_threshold_rational=str(threshold),
                uniform_support_size=1 << 53, accepted_uniform_values=accepted,
                probability_scope='Exact ideal uniform53bit marginal with independent uniform offsets; not proof of deterministic PRNG independence.',
                nominal=channel(Fraction(3, 40)), actual_uniform53=channel(actual))
