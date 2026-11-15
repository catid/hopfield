"""Synthetic H1bw decoder/channel, paired-endpoint, mask, and query-union gates."""
import json, platform, time, statistics
from fractions import Fraction
from itertools import product
import numpy as np
from repeat_query_tradeoff_study import ROOT, HERE, ARMS, ARM_ROWS, sha, save, make_masks
from repeat_query_math import decode, observations, count_values, exact_channels
from audit_repeat_query_tradeoff import reference_masks, reference_decode, reference_observations, complete_gate, literal_account, aggregate
from audit_laplace_rate import independent_rate_costs
from laplace_rate_math import rate_costs
from fixed_270_confirm_study import make_masks as parent_masks
from audit_balanced_mask_math import verify_incidence


def seed_audit(design):
    values = set()
    files = {}
    def scan(value, context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                scan(item, context or 'seed' in key.lower())
        elif isinstance(value, list):
            for item in value:
                scan(item, context)
        elif context and isinstance(value, int) and not isinstance(value, bool):
            values.add(value)
    for path in sorted((ROOT/'research').rglob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and 'frozen_utc' in data:
            files[str(path.relative_to(ROOT))] = sha(path)
            scan(data)
    proposed = {name: set(range(design['seeds'][name], design['seeds'][name]+64)) for name in ['source', 'balanced_mask', 'noise', 'repeat1', 'repeat2']}
    proposed['solver'] = {design['seeds']['solver_base']+41*index+degree for index in range(128) for degree in range(1, 41)}
    assert sum(len(v) for v in proposed.values()) == len(set().union(*proposed.values())) == 5440
    assert not values & set().union(*proposed.values())
    result = dict(status='passed', frozen_protocol_sha256=files, historical_explicit_seed_count=len(values),
                  new_ranges={name: [min(v), max(v)] for name, v in proposed.items()}, new_counts={name: len(v) for name, v in proposed.items()}, collisions=[],
                  scope='Disjoint from all explicit integer seed-valued fields in frozen protocol snapshots; does not expand every historical formula or prove PRNG independence.')
    save(HERE/'repeat-query-tradeoff-seed-audit.json', result)
    return result


def fixture(pairs):
    assert len(pairs) == 64
    rows = []
    for bank, pair in enumerate(pairs):
        for index, arm in enumerate(ARMS):
            kind = pair[index]
            rows.append(dict(index=2*bank+index, seed=bank, arm=arm, rows=ARM_ROWS[arm], classification=kind,
                             selected_degree=20, fixed_degree20=dict(classification=kind), top810=dict(classification=kind)))
    return aggregate(rows)


def endpoint_gates():
    good, bad, frac, unknown = 'success', 'resolved_failure_nonexact', 'resolved_failure_fractional', 'unresolved'
    states = [good, bad, frac, unknown]
    # Exhaust the16 logical paired classifications against literal binary truth completions.
    for first, second in product(states, repeat=2):
        answer = fixture([(good, good)]*63+[(first, second)])['endpoints']['paired_loss']
        options = lambda kind: [0, 1] if kind == unknown else [int(kind == good)]
        actual = [a*(1-b) for a, b in product(options(first), options(second))]
        assert (answer['lower'], answer['upper']) == (min(actual), max(actual))
    cases = [
        ([(good, good)]*61+[(good, bad)]*3, ('supported', 'supported')),
        ([(good, good)]*60+[(good, frac)]*4, ('falsified', 'falsified')),
        ([(good, good)]*60+[(good, bad)]*3+[(good, unknown)], ('inconclusive', 'inconclusive')),
        ([(good, good)]*60+[(good, bad)]*3+[(unknown, bad)], ('falsified', 'inconclusive')),
        ([(good, good)]*60+[(bad, good)]*4, ('supported', 'supported')),
        ([(good, good)]*62+[(good, bad), (bad, good)], ('supported', 'supported')),
    ]
    receipts = []
    for pairs, decisions in cases:
        result = fixture(pairs)
        assert tuple(result['endpoints'][key]['decision'] for key in ['triple_exact', 'paired_loss']) == decisions
        receipts.append(result)
    assert receipts[-1]['paired_changes']['single270_wins'] == [62]
    assert receipts[-1]['paired_changes']['triple90_wins'] == [63]
    return receipts


def main():
    assert platform.node() == 'ripper'
    path = HERE/'repeat-query-tradeoff-preflight.json'
    assert not path.exists() and not (HERE/'protocol-repeat-query-tradeoff.json').exists()
    design = json.loads((HERE/'repeat-query-tradeoff-proposal.json').read_text())
    seeds = seed_audit(design)
    incidence = verify_incidence()
    triples = np.array(list(product(range(9), repeat=3)), np.int8).T
    actual = decode(*triples)
    assert np.array_equal(actual, reference_decode(*triples))
    for digit in range(8):
        permutation = np.arange(9, dtype=np.int8)
        permutation[digit], permutation[digit+1] = permutation[digit+1], permutation[digit]
        assert np.array_equal(decode(*permutation[triples]), permutation[actual])
    assert decode(np.array([0, 0, 0, 1]), np.array([0, 1, 2, 2]), np.array([2, 1, 1, 3])).tolist() == [0, 1, 0, 1]
    channels = exact_channels()
    for key in ['nominal', 'actual_uniform53']:
        channel = channels[key]
        p = Fraction(channel['input_error'])
        q = 1-p
        correct = q**3+3*q*q*p+q*p*p*Fraction(7, 8)
        tie = 3*q*p*p*Fraction(7, 8)+p**3*Fraction(21, 32)
        for row in channel['rows']:
            assert Fraction(row['error']) == 1-correct and Fraction(row['all_distinct_probability']) == tie
    assert channels['nominal']['rows'][0]['error'] == '5877/512000'
    threshold = Fraction(channels['noise_threshold_rational'])
    count, support = channels['accepted_uniform_values'], channels['uniform_support_size']
    assert Fraction(count-1, support) < threshold <= Fraction(count, support)
    save(HERE/'repeat-query-channel.json', dict(status='passed', **channels))
    for seed in [986001, 986002, 986003]:
        mask = make_masks(seed)
        assert np.array_equal(mask, reference_masks(seed)) and np.array_equal(mask, parent_masks(seed))
        assert mask.shape == (270, 81) and (mask.sum(1) == 72).all()
        for n in [90, 270]:
            exposure = mask[:n].astype(np.int64).T @ mask[:n].astype(np.int64)
            k = n//90
            assert np.array_equal(exposure, 71*k*np.ones((81, 81), np.int64)+9*k*np.eye(81, dtype=np.int64))
        assert literal_account(mask) == dict(full_physical_noisy_union=32400, single270_deployment=19440, triple90_deployment=19440, shared_first_query90=6480, extra_repeat_queries=12960)
    rng = np.random.default_rng(986004)
    clean = rng.integers(0, 9, (270, 81), dtype=np.int8)
    noise = {}
    for name, n in [('first', 270), ('repeat1', 90), ('repeat2', 90)]:
        noise[name+'_iid_uniforms'] = rng.random((n, 81))
        noise[name+'_iid_offsets'] = rng.integers(1, 9, (n, 81), dtype=np.int8)
    produced = observations(mask, clean, noise)
    expected, queries, decoded = reference_observations(mask, clean, noise)
    for arm in ARMS:
        for key, value in expected[arm].items():
            assert np.array_equal(produced[arm][key], value)
        assert np.array_equal(rate_costs(produced[arm]['joint_count'], produced[arm]['equal_count']),
                              independent_rate_costs(expected[arm]['joint_count'], expected[arm]['equal_count']))
    assert np.array_equal(produced['single270']['values'][:90][mask[:90]], queries['first'][:90][mask[:90]])
    gates = complete_gate(mask, clean, noise, expected)
    missing = mask.copy()
    missing[0] = False
    counted = count_values(missing, clean)
    assert (counted['values'][0] == -1).all()
    ii, jj = np.triu_indices(81, 1)
    keep = missing[:, ii] & missing[:, jj]
    assert np.array_equal(counted['equal_count'], (keep & (clean[:, ii] == clean[:, jj])).sum(0))
    fixtures = endpoint_gates()
    toy = rng.integers(0, 9, (3, 64, 81), dtype=np.int8)
    timings = {}
    outputs = {}
    for name, function in [('scalar', reference_decode), ('vectorized', decode)]:
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            output = function(*toy)
            samples.append(time.perf_counter()-start)
        outputs[name] = output
        timings[name] = dict(samples_wall_seconds=samples, median_wall_seconds=statistics.median(samples))
    assert np.array_equal(outputs['scalar'], outputs['vectorized'])
    partitions = [list(range(worker, 64, 8)) for worker in range(8)]
    assert sorted(bank for part in partitions for bank in part) == list(range(64))
    dependencies = json.loads((HERE/'fixed-270-confirm-preflight.json').read_text())['source_sha256']
    files = {ROOT/rel for rel in dependencies}
    files |= {HERE/name for name in ['repeat_query_tradeoff_study.py', 'audit_repeat_query_tradeoff.py', 'repeat_query_math.py',
                                     'preflight_repeat_query_tradeoff.py', 'run_repeat_query_tradeoff.py']}
    result = dict(status='passed', host=platform.node(), scope='Synthetic only; no selected source IDs, study masks/noise, or LP outcomes.',
                  incidence=incidence, exact_channel_sha256=sha(HERE/'repeat-query-channel.json'), decoder_cases=729,
                  adjacent_digit_swap_checks=8, decoded_channel_error_nominal='5877/512000', exact_exposure_matrices=True,
                  shared_first_query90_verified=True, query_union_per_bank=32400, full_label_gates=gates, endpoint_cases=fixtures,
                  endpoint_state_combinations=16, decoder_timing=timings, decoder_timing_scope='Unrelated64x81 toy, three repetitions; no full-study speed claim.',
                  worker_bank_partition=partitions, seed_audit_sha256=sha(HERE/'repeat-query-tradeoff-seed-audit.json'),
                  source_sha256={str(f.relative_to(ROOT)): sha(f) for f in sorted(files)})
    save(path, result)
    print(json.dumps(dict(status='passed', source_files=len(files), seed_ranges=seeds['new_ranges'], decoder_timing=timings)))


if __name__ == '__main__':
    main()


