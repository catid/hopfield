"""Unrelated synthetic checks and explicit seed audit before H4ar scores."""
from pathlib import Path
from fractions import Fraction
import json
import time
import numpy as np
import normalized_deficit as S
from audit_normalized_deficit import reference_scores, aggregate
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def seeds():
    values, files = set(), {}
    def scan(value, context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                scan(item, context or 'seed' in key.lower())
        elif isinstance(value, list):
            for item in value:
                scan(item, context)
        elif context and isinstance(value, int) and not isinstance(value, bool):
            values.add(value)
    for path in sorted((S.ROOT/'research').rglob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and 'frozen_utc' in data:
            files[str(path.relative_to(S.ROOT))] = S.sha(path)
            scan(data)
    new = set(range(S.SOLVER_SEED_BASE, S.SOLVER_SEED_BASE+256))
    assert not new & values
    result = dict(status='passed', explicit_seed_values=len(values), frozen_protocol_sha256=files,
                  new_solver_seeds=256, solver_seed_range=[min(new), max(new)], collisions=[],
                  scope='All explicit integer seed-valued fields in snapshotted frozen protocols. Does not expand every formula or establish undocumented-stream independence.')
    S.save(S.HERE/'normalized-deficit-seed-audit.json', result)


def main():
    assert not S.PROTOCOL.exists()
    started = time.perf_counter()
    seeds()
    rng = np.random.default_rng(982011)
    for numerator in list(range(-15, 16)) + [2**55-1, -2**55+1]:
        for denominator in range(1, 12):
            assert S.half_even(numerator, denominator) == round(Fraction(numerator, denominator))
    hand = np.array([0, 600, 600, 600, 600, 0], np.int64)
    h = S.scores(4, 1, hand)
    assert h['scales'].tolist() == [600]*4 and h['cost'].tolist() == [-1667, 13333, 13333, 13333, 13333, -1667]
    all_high = S.scores(4, 2, np.full(6, 600, np.int64))
    assert all_high['scales'].tolist() == [2]*4 and (all_high['cost'] == 4800000000).all()
    assert all_high['cost'].dtype == np.int64
    for n in [4, 9, 81]:
        for degree in [1, min(3, n-1)]:
            count = rng.integers(0, 601, n*(n-1)//2, dtype=np.int64)
            produced, reference = S.scores(n, degree, count), reference_scores(n, degree, count)
            assert all(np.array_equal(produced[k], reference[k]) for k in produced)
    n, degree = 81, 8
    count = rng.integers(0, 601, n*(n-1)//2, dtype=np.int64)
    values = S.scores(n, degree, count)
    ii, jj = np.triu_indices(n, 1)
    matrix = np.zeros((n, n), np.int64)
    matrix[ii, jj] = count
    matrix[jj, ii] = count
    permutation = rng.permutation(n)
    permuted = S.scores(n, degree, matrix[permutation][:, permutation][ii, jj])
    cost_matrix = np.zeros_like(matrix)
    cost_matrix[ii, jj] = values['cost']
    cost_matrix[jj, ii] = values['cost']
    assert np.array_equal(permuted['scales'], values['scales'][permutation])
    assert np.array_equal(permuted['cost'], cost_matrix[permutation][:, permutation][ii, jj])
    timing = {}
    for name, function in [('producer', S.scores), ('independent_Fraction', reference_scores)]:
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            function(n, degree, count)
            elapsed.append(time.perf_counter()-start)
        timing[name+'_median_seconds'] = float(np.median(elapsed))
    folder = S.ROOT/'artifacts/research/20261114/foureyes/normalized_deficit_preflight'
    folder.mkdir(parents=True, exist_ok=True)
    ii, jj = np.triu_indices(4, 1)
    toy_cost = np.array([-3, 4, 2, 2, 4, -3], np.int64)
    x, y, metadata = solve(4, 1, ii, jj, toy_cost, 982012-2026111502, folder/'negative-cost.log')
    cert, proof = attempt_certificate(4, 1, ii, jj, toy_cost, x, y)
    assert cert['certified'] and metadata['solver_optimal'] and cert['primal_objective_twice'] == -12
    assert check_certificate(4, 1, ii, jj, toy_cost, *[proof[k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']])
    # Full synthetic denominators exercise independent loss and recovery rules.
    endpoints = []
    for successes, losses, expected in [(145, 8, ('supported', 'supported')), (144, 9, ('falsified', 'falsified'))]:
        rows = []
        retained = 129-losses
        gained = successes-retained
        for i in range(256):
            baseline = i < 129
            good = i < retained or 129 <= i < 129+gained
            rows.append(dict(index=i, family=S.FAMILIES[i % 4], baseline_exact=baseline,
                             classification='success' if good else 'resolved_failure_nonexact',
                             certificates_checked=2, mutations_rejected=6))
        result = aggregate(rows)
        assert (result['primary']['recovery']['decision'], result['primary']['paired_loss']['decision']) == expected
        endpoints.append(result['primary'])
    # Last fixture has 144 successes / 9 losses; one uncertain baseline loss
    # straddles both separate thresholds without changing the denominator.
    rows[120]['classification'] = 'unresolved'
    uncertain = aggregate(rows)['primary']
    assert uncertain['recovery'] == dict(lower=144, upper=145, required=145, decision='inconclusive')
    assert uncertain['paired_loss'] == dict(lower=8, upper=9, maximum=8, decision='inconclusive')
    endpoints.append(uncertain)
    files = [Path(__file__), S.HERE/'normalized_deficit.py', S.HERE/'audit_normalized_deficit.py',
             S.HERE/'audit_objective_gap.py']
    files += [S.HERE.parent/'ripper'/name for name in ['degree_prior_solver.py', 'degree_prior_math.py', 'audit_degree_prior.py']]
    S.save(S.HERE/'normalized-deficit-preflight.json', dict(status='passed', study_scores_read=False,
           synthetic_negative_cost_certificate=cert, synthetic_int32_overflow_case_passed=True,
           signed_half_even_and_permutation_gates=True, endpoint_gates=endpoints,
           score_timings=timing, gate_wall_seconds=time.perf_counter()-started,
           seed_audit_sha256=S.sha(S.HERE/'normalized-deficit-seed-audit.json'),
           source_sha256={str(path.relative_to(S.ROOT)): S.sha(path) for path in files}))
    print('PREFLIGHT_PASSED', json.dumps(timing), flush=True)


if __name__ == '__main__':
    main()
