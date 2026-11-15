"""Synthetic channel, audit optimization, mask streams, proofs and endpoints."""
from pathlib import Path
from fractions import Fraction
import json
import time
import numpy as np
import oracle_attenuation as S
from oracle_attenuation_math import score, reference_score, independent_score, mask_from_indices, channel_gate
from normalized_deficit import half_even
from audit_oracle_attenuation import aggregate, selection_stream
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
    S.save(S.HERE/'oracle-attenuation-seed-audit.json', result)


def main():
    assert not S.PROTOCOL.exists() and not S.OUT.exists()
    started = time.perf_counter()
    seeds()
    channel = channel_gate()
    rng = np.random.default_rng(982030)
    for numerator in list(range(-15, 16)) + [307200000000, -38400000000, 2**55-1]:
        for denominator in range(1, 12):
            assert half_even(numerator, denominator) == round(Fraction(numerator, denominator))
    for n, rows in [(4, 31), (11, 81), (81, 600)]:
        mask = rng.random((rows, n)) < .12
        counts = rng.integers(0, rows+1, n*(n-1)//2, dtype=np.int64)
        a, b, c = score(counts, mask), reference_score(counts, mask), independent_score(counts, mask)
        assert all(np.array_equal(a[k], b[k]) and np.array_equal(a[k], c[k]) for k in a)
    # Independent full 2,400-row replay and original stable sort on unrelated streams.
    hot = np.random.default_rng(982031).choice(81, 18, replace=False)
    uniform = 1-np.random.default_rng(982032).random((2400, 81))
    weights = np.ones(81); weights[hot] = 32
    expected = np.argsort(-np.log(uniform)/weights, axis=1, kind='stable')[:600, :9]
    assert np.array_equal(selection_stream(982031, 982032), expected)
    rebuilt = mask_from_indices(expected)
    assert rebuilt.shape == (600, 81) and (rebuilt.sum(1) == 9).all()
    timing = {}
    for name, function in [('producer', score), ('literal_pair_sum', reference_score), ('expanded_boolean_identity', independent_score)]:
        samples = []
        for _ in range(3):
            start = time.perf_counter(); result = function(counts, mask); samples.append(time.perf_counter()-start)
            assert all(np.array_equal(result[k], a[k]) for k in a)
        timing[name+'_median_seconds'] = float(np.median(samples))
    timing['literal_over_expanded'] = timing['literal_pair_sum_median_seconds']/timing['expanded_boolean_identity_median_seconds']
    # Nonpositive banks return no costs. No per-pair fallback or clipping.
    for rows in [72, 600]:
        bad = np.zeros((rows, 4), bool); bad[(8 if rows == 72 else 0):, 0] = True
        r = score(np.zeros(6, np.int64), bad)
        assert len(r['nonpositive_pairs']) == 3 and r['cost'].size == 0
        assert all(np.array_equal(r[k], independent_score(np.zeros(6, np.int64), bad)[k]) for k in r)
    folder = S.ROOT/'artifacts/research/20261114/foureyes/oracle_attenuation_preflight'
    folder.mkdir(parents=True, exist_ok=True)
    ii, jj = np.triu_indices(4, 1)
    certificates = []
    for suffix, toy_cost, expected in [('negative', np.array([-3,4,2,2,4,-3], np.int64), -12),
                                      ('range', np.array([-38400000000,307200000000,2,2,307200000000,-38400000000], np.int64), -153600000000)]:
        x, y, metadata = solve(4, 1, ii, jj, toy_cost, 982033-2026111502, folder/(suffix+'.log'))
        cert, proof = attempt_certificate(4, 1, ii, jj, toy_cost, x, y)
        assert cert['certified'] and metadata['solver_optimal'] and cert['primal_objective_twice'] == expected
        w = [proof[k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
        assert check_certificate(4, 1, ii, jj, toy_cost, *w)
        for k in range(3):
            mutated = [v.copy() for v in w]; mutated[k][0] += 1
            assert not check_certificate(4, 1, ii, jj, toy_cost, *mutated)
        certificates.append(cert)
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
    files = [Path(__file__), S.HERE/'oracle_attenuation.py', S.HERE/'oracle_attenuation_math.py',
             S.HERE/'audit_oracle_attenuation.py', S.HERE/'normalized_deficit.py', S.HERE/'audit_objective_gap.py']
    files += [S.HERE.parent/'ripper'/name for name in ['degree_prior_solver.py', 'degree_prior_math.py', 'audit_degree_prior.py']]
    S.save(S.HERE/'oracle-attenuation-preflight.json', dict(status='passed', study_masks_or_scores_read=False,
           conditional_channel_gate=channel, synthetic_certificates=certificates,
           full2400_stream_then600_prefix_verified=True, independent_nonpositive_bank_skip=True,
           signed_half_even_and_range_gates=True, endpoint_gates=endpoints,
           score_timings=timing, gate_wall_seconds=time.perf_counter()-started,
           seed_audit_sha256=S.sha(S.HERE/'oracle-attenuation-seed-audit.json'),
           source_sha256={str(path.relative_to(S.ROOT)): S.sha(path) for path in files}))
    print('PREFLIGHT_PASSED', json.dumps(timing), flush=True)


if __name__ == '__main__':
    main()
