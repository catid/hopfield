"""H4av synthetic math, lifecycle, seed and endpoint gates; no cohort reads."""
from pathlib import Path
import tempfile
import time
import numpy as np
import conflict_mask as S
from conflict_mask_math import (sparse_scores, dense_scores, literal_scores,
    detector, independent_detector, random_mask, independent_random_mask)
from corruption_censor_math import score, independent_score, literal_score
from audit_conflict_mask import aggregate
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def rejected(fn, *args):
    try:
        fn(*args)
    except (AssertionError, FileNotFoundError):
        return
    raise AssertionError('Invalid input/lifecycle accepted')


def seeds(design):
    values, files = set(), {}
    def scan(v, context=False):
        if isinstance(v, dict):
            for key, item in v.items():
                scan(item, context or 'seed' in key.lower())
        elif isinstance(v, list):
            for item in v:
                scan(item, context)
        elif context and type(v) is int:
            values.add(v)
    for path in sorted((S.ROOT / 'research').rglob('*.json')):
        try:
            data = S.read(path)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and data.get('frozen_utc'):
            files[str(path.relative_to(S.ROOT))] = S.sha(path)
            scan(data)
    expanded, ranges = set(), {}
    for filename, count in [('corruption-censor.json', 256), ('partial-mask-recall.json', 1280)]:
        p = S.read(S.HERE / filename)
        base = p['solver']['random_seed_base']
        expanded.update(range(base, base + count))
        ranges[filename] = [base, base + count - 1]
        files[str((S.HERE / filename).relative_to(S.ROOT))] = S.sha(S.HERE / filename)
    random = set(range(design['random_seed_base'], design['random_seed_base'] + 256))
    solver = set(range(design['solver_seed_base'], design['solver_seed_base'] + 512))
    assert len(random | solver) == 768 and not (random | solver) & (values | expanded)
    assert min(random | solver) >= 0 and max(random | solver) < 2**31
    S.save(S.HERE / 'conflict-mask-seed-audit.json', dict(status='passed',
        frozen_protocol_sha256=files, historical_explicit_seed_values=len(values),
        expanded_ancestor_ranges=ranges, random_range=[min(random), max(random)],
        solver_range=[min(solver), max(solver)], collisions=[],
        scope='Explicit integer seed fields in frozen_utc protocols plus all actual H4at/H4au solver seeds. Other historical seed formulas not exhaustively expanded; no proof of stream independence.'))


def endpoint_fixture(d, du, r, ru):
    rows = []
    for index in range(256):
        for arm, exact, unknown in [('detector', d, du), ('random', r, ru)]:
            kind = ('success' if index < exact else 'unresolved'
                    if index < exact + unknown else 'resolved_failure_nonexact')
            rows.append(dict(index=index, arm=arm, classification=kind,
                certified=kind != 'unresolved', baseline_exact=index < 129,
                family=S.FAMILIES[index % 4], true_positive_locations=600,
                false_positive_locations=4800, false_negative_locations=4800))
    return aggregate(rows)


def lifecycle():
    saved_out, saved_protocol = S.OUT, S.PROTOCOL
    with tempfile.TemporaryDirectory(prefix='h4av-synthetic-seal-') as tmp:
        try:
            S.OUT = Path(tmp) / 'fits'; S.OUT.mkdir()
            S.PROTOCOL = Path(tmp) / 'protocol.json'
            S.save(S.PROTOCOL, dict(source_sha256={}))
            rejected(S.seal_fits, {})
            rejected(S.audit_worker, 0)
            ph = S.sha(S.PROTOCOL)
            for index in range(256):
                folder = S.OUT / f'b{index:03d}'; folder.mkdir()
                np.savez(folder / 'detector-scores.npz', scores=np.zeros((1, 2), np.int64))
                for arm in S.ARMS:
                    np.savez(folder / f'{arm}.npz', fixture=np.zeros(1))
                    S.save(folder / f'{arm}.json', dict(index=index, arm=arm,
                        protocol_sha256=ph, models_sha256=S.sha(folder / f'{arm}.npz'),
                        fit_completed_monotonic_ns=time.monotonic_ns()))
            S.seal_fits({})
            seal = S.check_seal(S.OUT)
            assert seal['fits'] == 512 and len(seal['files']) == 1280
            rejected(S.seal_fits, {})
            target = S.OUT / 'b000/detector.npz'
            target.write_bytes(target.read_bytes() + b'changed')
            rejected(S.check_seal, S.OUT)
        finally:
            S.OUT, S.PROTOCOL = saved_out, saved_protocol
    return dict(missing_fits_rejected=True, early_audit_rejected=True,
                all512_sealed=True, overwrite_rejected=True, post_seal_mutation_rejected=True)


def main():
    assert not S.PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    target = S.HERE / 'conflict-mask-preflight.json'
    assert not target.exists()
    started = time.perf_counter()
    seeds(S.read(S.HERE / 'conflict-mask-proposal.json'))
    rng = np.random.default_rng(982060)
    comparisons = 0
    for n in [4, 9, 13]:
        noisy = rng.integers(0, 9, (17, n), dtype=np.int8)
        for fractional in [False, True]:
            primal = (rng.integers(0, 3, n*(n-1)//2, dtype=np.int64) if fractional
                      else 2*rng.integers(0, 2, n*(n-1)//2, dtype=np.int64))
            expected = literal_scores(noisy, primal)
            assert np.array_equal(sparse_scores(noisy, primal), expected)
            assert np.array_equal(dense_scores(noisy, primal), expected)
            digit_permutation = rng.permutation(9)
            assert np.array_equal(sparse_scores(digit_permutation[noisy], primal), expected)
            matrix = np.zeros((n, n), np.int64); ii, jj = np.triu_indices(n, 1)
            matrix[ii, jj] = matrix[jj, ii] = primal
            order = rng.permutation(n)
            assert np.array_equal(sparse_scores(noisy[:, order], matrix[order][:, order][ii, jj]), expected[:, order])
            for k in [0, min(9, n), n]:
                a, mask = detector(noisy, primal, k)
                b, other = independent_detector(noisy, primal, k)
                assert np.array_equal(a, b) and np.array_equal(mask, other) and (mask.sum(1) == k).all()
                for key, value in score(noisy, mask).items():
                    assert np.array_equal(value, independent_score(noisy, mask)[key])
                    assert np.array_equal(value, literal_score(noisy, mask)[key])
                if k == n:
                    assert not score(noisy, mask)['exposure'].any() and not score(noisy, mask)['cost'].any()
                comparisons += 1
    zeros = np.zeros(13*12//2, np.int64)
    tie_input = np.zeros((3, 13), np.int8)
    for fn in [detector, independent_detector]:
        _, mask = fn(tie_input, zeros)
        assert mask[:, :9].all() and not mask[:, 9:].any()
    for rows, n, k in [(1, 13, 0), (7, 13, 9), (600, 81, 9), (4, 13, 13)]:
        a = random_mask(rows, n, 982061, k)
        assert np.array_equal(a, independent_random_mask(rows, n, 982061, k))
        assert (a.sum(1) == k).all()
    bad = [(tie_input.astype(float), zeros), (tie_input - 1, zeros),
           (tie_input + 9, zeros), (tie_input, zeros.astype(np.int32)),
           (tie_input, zeros[:-1]), (tie_input, zeros + 3),
           (tie_input, zeros, -1), (tie_input, zeros, 14), (tie_input, zeros, 9.)]
    for args in bad:
        rejected(detector, *args)
    folder = S.ROOT / 'artifacts/research/20261114/foureyes/conflict_mask_preflight'
    folder.mkdir(parents=True, exist_ok=True)
    ii, jj = np.triu_indices(4, 1); cost = np.ones(6, np.int64)
    x, y, meta = solve(4, 1, ii, jj, cost, 982062-2026111502, folder / 'positive.log')
    cert, proof = attempt_certificate(4, 1, ii, jj, cost, x, y)
    w = [proof[k] for k in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']]
    assert meta['solver_optimal'] and cert['certified'] and cert['primal_objective_twice'] == 4
    assert check_certificate(4, 1, ii, jj, cost, *w)
    for k in range(3):
        mutated = [v.copy() for v in w]; mutated[k][0] += 1
        assert not check_certificate(4, 1, ii, jj, cost, *mutated)
    ii, jj = np.triu_indices(6, 1); triangle = (ii//3 == jj//3)
    assert check_certificate(6, 1, ii, jj, (~triangle).astype(np.int64),
                             triangle.astype(np.int64), np.zeros(6, np.int64), np.zeros(15, np.int64))
    endpoints = []
    for d, du, r, ru, absolute, advantage in [
        (200,0,168,0,'supported','supported'), (199,0,168,0,'falsified','falsified'),
        (199,1,168,0,'inconclusive','inconclusive'), (200,0,168,1,'supported','inconclusive'),
        (0,256,0,256,'inconclusive','inconclusive'), (256,0,256,0,'supported','falsified')]:
        result = endpoint_fixture(d, du, r, ru)['primary']
        assert result['detector_exact']['decision'] == absolute
        assert result['advantage_over_random']['decision'] == advantage
        endpoints.append(result)
    life = lifecycle()
    noisy = rng.integers(0, 9, (600, 81), dtype=np.int8)
    ii, jj = np.triu_indices(81, 1)
    delta = (jj-ii) % 81
    primal = 2*((delta <= 4) | (delta >= 77)).astype(np.int64)
    timings = {}
    reference = dense_scores(noisy, primal)
    for fn in [sparse_scores, dense_scores]:
        samples = []
        for _ in range(7):
            before = time.perf_counter(); answer = fn(noisy, primal)
            samples.append(time.perf_counter() - before)
            assert np.array_equal(answer, reference)
        timings[fn.__name__] = float(np.median(samples))
    sources = S.read(S.PARENT_PROTOCOL)['source_sha256'].copy()
    for name in ['conflict_mask_math.py', 'conflict_mask.py', 'audit_conflict_mask.py', 'preflight_conflict_mask.py']:
        path = S.HERE / name; sources[str(path.relative_to(S.ROOT))] = S.sha(path)
    S.save(target, dict(status='passed', study_masks_scores_or_LPs_read=False,
        math_cases=comparisons, integral_fractional_scalar_dense_sparse_parity=True,
        digit_and_vertex_score_permutation=True, stable_index_ties=True,
        random_mask_replay=True, zero_exposure_cost_identity=True,
        invalid_inputs_rejected=len(bad), positive_objective_certificate=cert,
        fractional_triangle_certificate=True, certificate_mutations_rejected=3,
        primary_boundaries=endpoints, lifecycle=life, score_timings_seconds=timings,
        selected_implementation='sparse_scores',
        performance_scope='Unrelated600x81 degree8 fixture only; complete experiment not timed.',
        source_sha256=sources, seed_audit_sha256=S.sha(S.HERE / 'conflict-mask-seed-audit.json'),
        gate_wall_seconds=time.perf_counter()-started))
    print('PREFLIGHT_PASSED', timings, flush=True)


if __name__ == '__main__':
    main()
