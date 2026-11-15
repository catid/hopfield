"""Synthetic-only H1bt p-value/endpoint/provenance tests; no study draws or LP outcomes."""
from collections import Counter
from fractions import Fraction
import itertools, json, platform, time
import numpy as np
from signal_detection_study import ROOT, HERE, ARMS, sha, save, observed_histograms, shuffle_observed, observe, concentration, randomization_test, grid_index, shuffle_seed
from audit_signal_detection import reference_test, joint_decision, aggregate
from audit_marginal_shuffle import reference_histograms, reference_shuffle, reference_observe, reference_concentration, label_gate
from balanced_mask_math import make_balanced_masks


def ratio(n, d=1):
    return dict(numerator=n, denominator=d, all_zero_curvature=n == 0)


def seed_audit(design):
    frozen = {}
    values = set()
    def scan(value, seed_context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                scan(item, seed_context or 'seed' in key.lower())
        elif isinstance(value, list):
            for item in value:
                scan(item, seed_context)
        elif seed_context and isinstance(value, int) and not isinstance(value, bool):
            values.add(value)
    for path in sorted((ROOT/'research').rglob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and 'frozen_utc' in data:
            frozen[str(path.relative_to(ROOT))] = sha(path)
            scan(data)
    shuffles = {shuffle_seed(design, case, ai, draw) for case in range(64) for ai in range(2) for draw in range(1, 32)}
    nulls = set(range(design['null_seed_base'], design['null_seed_base']+32))
    solver = {design['solver_seed_base']+grid_index(case, ai, draw)*41+degree for case in range(64) for ai in range(2) for draw in range(0 if case >= 32 else 1, 32) for degree in range(1, 41)}
    assert len(shuffles) == 3968 and len(nulls) == 32 and len(solver) == 161280
    assert not (shuffles & nulls or shuffles & solver or nulls & solver)
    assert not ((shuffles | nulls | solver) & values)
    receipt = dict(status='passed', scope='Disjoint from all explicit integer seed-valued fields in snapshotted research JSON objects with frozen_utc. Does not establish absence from undocumented streams or unexpanded historical formulas.',
                   historical_explicit_seed_count=len(values), frozen_protocol_sha256=frozen,
                   shuffle_range=[min(shuffles), max(shuffles)], null_range=[min(nulls), max(nulls)], solver_range=[min(solver), max(solver)],
                   unique_shuffle_streams=len(shuffles), unique_null_streams=len(nulls), unique_solver_seeds=len(solver), collisions=[])
    save(HERE/'signal-detection-seed-audit.json', receipt)
    return receipt


def main():
    assert platform.node() == 'ripper'
    path = HERE/'signal-detection-preflight.json'
    assert not path.exists()
    design = json.loads((HERE/'signal-detection-proposal.json').read_text())
    seeds = seed_audit(design)
    rng = np.random.default_rng(982001)
    labels = rng.integers(0, 9, (180, 81), dtype=np.int8)
    uniform = np.zeros((180, 81), bool)
    for row in range(180):
        uniform[row, rng.permutation(81)[:72]] = True
    masks = dict(uniform=uniform, balanced=make_balanced_masks(982002))
    gate_labels = {}
    gate_expected = {}
    for ai, arm in enumerate(ARMS):
        original = np.where(masks[arm], labels, -1).astype(np.int8)
        hist = observed_histograms(masks[arm], original)
        assert np.array_equal(hist, reference_histograms(masks[arm], original))
        changed = shuffle_observed(masks[arm], hist, 982003+ai)
        assert np.array_equal(changed, reference_shuffle(masks[arm], hist, 982003+ai))
        assert np.array_equal(hist, reference_histograms(masks[arm], changed))
        actual = observe(masks[arm], changed)
        expected = reference_observe(masks[arm], changed)
        for key, value in actual.items():
            assert np.array_equal(value, expected[key])
        gate_labels[arm] = changed
        gate_expected[arm] = expected
    gates = label_gate(masks, gate_labels, gate_expected)
    permutation_cases = []
    for multiset in [[0, 0, 1], [0, 1, 2, 2]]:
        outputs = Counter(tuple(multiset[i] for i in permutation) for permutation in itertools.permutations(range(len(multiset))))
        assert len(set(outputs.values())) == 1
        permutation_cases.append(dict(multiset=multiset, distinct_arrangements=len(outputs), multiplicity=next(iter(outputs.values()))))
    rank_cases = []
    for scores in [list(range(32)), [0]*32, [0]*16+[1]*16, [0]*30+[1, 2]]:
        numerators = []
        for which in range(32):
            observed = ratio(scores[which])
            others = [ratio(value) for i, value in enumerate(scores) if i != which]
            actual = randomization_test(observed, others)
            assert actual == reference_test(observed, others)
            numerators.append(actual['numerator'])
        # Conditional on an exchangeable score multiset, at most k of 32
        # positions can have conservative rank <=k, even with ties.
        for k in range(1, 33):
            assert sum(n <= k for n in numerators) <= k
        rank_cases.append(dict(scores=scores, p_numerators=numerators))
    assert randomization_test(ratio(2), [ratio(1)]*31)['signal'] is True
    assert randomization_test(ratio(2), [ratio(1)]*30+[ratio(2)])['signal'] is False
    assert randomization_test(ratio(2), [ratio(1)]*30+[None])['signal'] is None
    assert randomization_test(None, [ratio(1)]*31)['signal'] is None
    huge = ratio(10**40+1, 10**40)
    assert randomization_test(huge, [ratio(1)]*31) == reference_test(huge, [ratio(1)]*31)
    assert randomization_test(huge, [ratio(1)]*31)['signal'] is True
    cases = []
    for real_signals, real_unknown, null_signals, null_unknown, real_decision, null_decision in [
            (29, 0, 2, 0, 'supported', 'supported'), (28, 1, 2, 1, 'inconclusive', 'inconclusive'),
            (28, 0, 3, 0, 'falsified', 'falsified'), (32, 0, 0, 2, 'supported', 'supported')]:
        rows = []
        for case in range(64):
            signals, unknown = (real_signals, real_unknown) if case < 32 else (null_signals, null_unknown)
            pair = [True, True] if case % 32 < signals else [True, None] if case % 32 < signals+unknown else [False, None]
            for ai, arm in enumerate(ARMS):
                rows.append(dict(case=case, cohort='real' if case < 32 else 'null', arm=arm, test=dict(signal=pair[ai]), observed_outcome=dict(classification='resolved_failure_nonexact')))
        result = aggregate(rows)
        assert result['endpoints']['real']['decision'] == real_decision and result['endpoints']['null']['decision'] == null_decision
        assert result['endpoints']['real']['lower'] == real_signals and result['endpoints']['real']['upper'] == real_signals+real_unknown
        assert result['endpoints']['null']['lower'] == null_signals and result['endpoints']['null']['upper'] == null_signals+null_unknown
        cases.append(result['endpoints'])
    assert joint_decision([False, None]) == 'no_signal' and joint_decision([None, True]) == 'unresolved'
    # Timing is synthetic count construction only, with no scientific LPs or pilot outcomes.
    timings = {}
    for name, function in [('producer', observe), ('independent_reference', reference_observe)]:
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            function(masks['uniform'], gate_labels['uniform'])
            samples.append(time.perf_counter()-start)
        timings[name+'_median_seconds'] = float(np.median(samples))
    dependencies = json.loads((HERE/'marginal-shuffle-preflight.json').read_text())['source_sha256']
    files = {ROOT/rel for rel in dependencies}
    files |= {HERE/name for name in ['signal_detection_study.py', 'audit_signal_detection.py', 'preflight_signal_detection.py', 'run_signal_detection.py', 'marginal_shuffle_study.py', 'audit_marginal_shuffle.py']}
    result = dict(status='passed', host=platform.node(), scope='Synthetic labels, toy score ranks and endpoint counts only; no H1bt study draws or LP outcomes.',
                  seed_audit_sha256=sha(HERE/'signal-detection-seed-audit.json'), conditional_uniform_multiset_cases=permutation_cases,
                  conservative_exchangeable_rank_cases=rank_cases, ties_fail_minimum_p_threshold=True, unresolved_propagation=True,
                  exact_large_integer_comparison=True, endpoint_cases=cases, complete_label_gates=gates, synthetic_count_timing=timings,
                  source_sha256={str(f.relative_to(ROOT)): sha(f) for f in sorted(files)})
    save(path, result)
    print(json.dumps(dict(status='passed', source_files=len(files), seed_ranges={key: seeds[key] for key in ['shuffle_range', 'null_range', 'solver_range']}, synthetic_count_timing=timings)))


if __name__ == '__main__':
    main()
