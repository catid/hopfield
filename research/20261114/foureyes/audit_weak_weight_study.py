"""Independent H4z source reconstruction and complete returned-board audit.

The source reconstruction does not call the preparation weight builder. The
board auditor computes Sudoku validity and dyadic energies directly on CPU.
"""
import argparse
import hashlib
import json
import os
import sys
for _name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_name] = '1'
import numpy as np
from weak_weight_study import ROOT, HERE, RIPPER, PARENT, OUT, PUBLIC, PROTOCOL, sha, save, verify

ARMS = {'weak16': 16, 'weak64': 4, 'weak256': 1}
KWARGS = dict(temp=.4, theta=.45, focused=True, C=4, tau=0., tol=.001,
              ebar=0., cap=4e11, check=4096, seed=0, anneal=[.4, .31, 524288])


def digest_bytes(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def geometry():
    ii, jj = np.triu_indices(81, 1)
    truth = (ii//9 == jj//9) | (ii%9 == jj%9) | ((ii//27 == jj//27) & (ii%9//3 == jj%9//3))
    assert int(truth.sum()) == 810
    return ii, jj, truth


def load_model(row):
    path = ROOT/row['model_path']
    assert sha(path) == row['model_sha256']
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'support', 'weight256'}
        w, support = z['weight256'].copy(), z['support'].copy()
    assert w.dtype == np.uint16 and w.shape == (3240,)
    assert support.dtype == bool and np.array_equal(support, w > 0)
    assert digest_bytes(w) == row['weight256_sha256']
    return w


def identity(w, p):
    # A shared support alone is insufficient to reuse a search.
    return ((w.astype(np.float32)/256).tobytes(), tuple(p['rows_half_open']),
            p['search_seed'], json.dumps(KWARGS, sort_keys=True), 262144)


def sources():
    p = json.loads(PROTOCOL.read_text()); verify(p)
    assert p['arms'] == list(ARMS) and p['banks'] == 64
    assert p['rows_half_open'] == [410000, 415000] and p['search_seed'] == 0
    mapping = json.loads((OUT/'mapping.json').read_text())
    assert mapping['protocol_sha256'] == sha(PROTOCOL)
    parent = json.loads((PARENT/'summary.json').read_text())
    assert parent['protocol_sha256'] == sha(HERE/'canonical_confirm.json')
    assert parent['weight_audit']['models_sha256'] == sha(PARENT/'models.npz')
    assert parent['outcome_bits_sha256'] == sha(PARENT/'audited-outcome-bits.npz')
    with np.load(PARENT/'models.npz', allow_pickle=False) as z:
        j = z['canonical_weak_coefficients'].copy()
        support = z['canonical_weak_support'].copy()
    assert j.shape == (64, 3240) and j.dtype == np.float64 and np.isfinite(j).all()
    assert support.dtype == bool and np.array_equal(support, j > 0)
    ii, jj, truth = geometry()
    assert support[:, truth].all()
    rows = []; evidence = {}; identities = {}; reverse = {}; seen = set()
    for row in mapping['rows']:
        b, arm = row['bank'], row['arm']
        assert 0 <= b < 64 and arm in ARMS and (b, arm) not in seen
        seen.add((b, arm))
        positive = sorted(float(v) for v in j[b] if v > 0)
        n = len(positive); assert n
        median = positive[n//2] if n % 2 else (positive[n//2-1] + positive[n//2])/2
        cut = median/4
        expected = np.array([0 if v <= 0 else 256 if v >= cut else ARMS[arm] for v in j[b]], np.uint16)
        w = load_model(row)
        assert np.array_equal(w, expected) and np.array_equal(w > 0, support[b])
        assert w[truth].min() > 0
        key = (row['kind'], row['job']); ident = identity(w, p)
        assert row['kind'] in {'archived', 'new'}
        if key in identities:
            assert identities[key] == ident
        if ident in reverse:
            assert reverse[ident] == key
        identities[key] = ident; reverse[ident] = key
        path = ROOT/row['model_path']; evidence[str(path.relative_to(ROOT))] = sha(path)
        if arm == 'weak16':
            assert row['kind'] == 'archived'
        if row['kind'] == 'archived':
            prior = [r for r in parent['rows'] if r['arm'] == 'magnitude' and r['job'] == row['job']]
            assert prior
            old = ROOT/prior[0]['model_path']; assert sha(old) == prior[0]['model_sha256']
            with np.load(old, allow_pickle=False) as z:
                oldw = z['weight16']; assert oldw.dtype == np.uint8
                assert digest_bytes(oldw) == prior[0]['weight16_sha256']
                assert np.array_equal(oldw.astype(np.uint16)*16, w)
                assert np.array_equal(z['support'], w > 0)
            evidence[str(old.relative_to(ROOT))] = sha(old)
        rows.append(dict(bank=b, arm=arm, threshold=cut, retained=int((w > 0).sum()),
                         missing_true=int((w[truth] == 0).sum()), weak_true=int((w[truth] == ARMS[arm]).sum()),
                         weak_extra=int((w[~truth] == ARMS[arm]).sum()), strong_extra=int((w[~truth] == 256).sum())))
    assert seen == {(b, a) for b in range(64) for a in ARMS}
    q = json.loads((OUT/'queue.json').read_text())
    new = {key[1] for key in identities if key[0] == 'new'}
    assert len(q) == len(new) == mapping['unique_new_searches']
    assert {r['job_id'] for r in q} == new
    for job in q:
        row = next(r for r in mapping['rows'] if r['kind'] == 'new' and r['job'] == job['job_id'])
        assert job['seed'] == 0 and job['n'] == 600 and job['arm'] == row['arm']
        assert job['model'] == dict(path=row['model_path'], sha256=row['model_sha256'])
    result = dict(status='validated', protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__),
                  mapping_sha256=sha(OUT/'mapping.json'), rows=rows, model_sha256=evidence,
                  logical_cells=len(rows), unique_new_searches=len(q), unique_archived_searches=len(identities)-len(q))
    save(OUT/'source-audit.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'logical_cells', 'unique_new_searches', 'unique_archived_searches']}))


def outcomes():
    import torch
    sys.path.insert(0, str(RIPPER))
    from bootstrap_two_way import resample_means
    p = json.loads(PROTOCOL.read_text()); verify(p)
    source = json.loads((OUT/'source-audit.json').read_text())
    assert source['status'] == 'validated' and source['protocol_sha256'] == sha(PROTOCOL)
    assert source['auditor_sha256'] == sha(__file__) and source['mapping_sha256'] == sha(OUT/'mapping.json')
    for path, digest in source['model_sha256'].items():
        assert sha(ROOT/path) == digest
    mapping = json.loads((OUT/'mapping.json').read_text()); q = json.loads((OUT/'queue.json').read_text())
    assert len(q) == mapping['unique_new_searches'] and all(r['status'] == 'complete' for r in q)
    parent = json.loads((PARENT/'summary.json').read_text())
    with np.load(PARENT/'audited-outcome-bits.npz', allow_pickle=False) as z:
        archived = {str(job): tuple(np.unpackbits(z[k][i], count=5000).astype(bool) for k in ['exact', 'valid', 'stopped']) for i, job in enumerate(z['jobs'])}
    lo, hi = p['rows_half_open']; count = hi-lo; assert count == 5000
    with np.load(ROOT/'upstream/data/full-test/full-all.npz', allow_pickle=False) as z:
        target = z['solutions'][lo:hi].astype(np.int8)-1
        free = ~z['clue_mask'][lo:hi].astype(bool)
    assert target.shape == free.shape == (count, 81)
    ii, jj, truth = geometry()
    units = [[r*9+c for c in range(9)] for r in range(9)]
    units += [[r*9+c for r in range(9)] for c in range(9)]
    units += [[(br*3+r)*9+bc*3+c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
    assert all((np.sort(target[:, cells], axis=1) == np.arange(9)).all() for cells in units)
    outputs = {}; diagnostics = {}; evidence = {}; all_weights = {}
    for key in sorted({(r['kind'], r['job']) for r in mapping['rows']}):
        kind, job = key
        mapped = [r for r in mapping['rows'] if (r['kind'], r['job']) == key]
        w = load_model(mapped[0]); assert all(np.array_equal(load_model(r), w) for r in mapped)
        all_weights[key] = w
        d = OUT/'jobs'/job if kind == 'new' else ROOT/'artifacts/research/20261114/foureyes/canonical_confirm/jobs'/job
        receipt = d/'receipt.json'; r = json.loads(receipt.read_text())
        assert r['status'] == 'complete' and r['job_id'] == job and r['seed'] == 0
        assert r['rows_half_open'] == [lo, hi] and r['test_rows'] == count and r['search_kwargs'] == KWARGS
        assert r['move_cap'] == 262144 and r['chains'] == 4
        assert r['used_unit'] == 1 and r['tau'] == r['ebar'] == 0
        assert r['preflight_state_comparisons'] == 150 and r['hidden_target_state_comparisons'] == 132
        assert r['integer_field_checks'] == 6 and r['integer_chain_states'] >= 64
        assert r['backend_sha256'] == sha(RIPPER/'sparse_backend.py') and r['capture_sha256'] == sha(RIPPER/'run.py')
        assert r['support_bytes_sha256'] == digest_bytes(w > 0) and r['edges'] == int((w > 0).sum())
        if kind == 'new':
            assert r['protocol_sha256'] == sha(PROTOCOL) and r['wrapper_sha256'] == sha(HERE/'weak_weight_search.py')
            assert r['integer_audit_sha256'] == sha(HERE/'weak_weight_math.py')
            assert r['weight256_bytes_sha256'] == digest_bytes(w)
            assert r['support_file_sha256'] == mapped[0]['model_sha256']
            denominator, field = 256, 'weight256'
        else:
            assert r['protocol_sha256'] == parent['protocol_sha256']
            assert r['wrapper_sha256'] == sha(RIPPER/'confidence_weight_search.py')
            assert r['integer_audit_sha256'] == sha(RIPPER/'confidence_integer_audit.py')
            assert np.all(w % 16 == 0) and r['weight16_bytes_sha256'] == digest_bytes((w//16).astype(np.uint8))
            prior = next(x for x in parent['rows'] if x['job'] == job and x['arm'] == 'magnitude')
            assert r['support_file_sha256'] == prior['model_sha256']
            for file in [receipt, d/'boards.npz']:
                assert sha(file) == parent['input_sha256'][str(file.relative_to(ROOT))]
            denominator, field = 16, 'weight16'
        assert set(r['output_sha256']) == {'boards.npz', 'result.json', 'searched_model.pt'}
        for name, digest in r['output_sha256'].items():
            assert sha(d/name) == digest
            evidence[str((d/name).relative_to(ROOT))] = digest
        evidence[str(receipt.relative_to(ROOT))] = sha(receipt)
        model = torch.load(d/'searched_model.pt', map_location='cpu', weights_only=True)
        assert set(model) == {'W', 'B', field}
        expected_w = torch.from_numpy(w.astype(np.float32)/256)[:, None, None]*torch.eye(9)
        assert model['W'].dtype == torch.float32 and torch.equal(model['W'], expected_w)
        assert model['B'].shape == (81, 9) and not model['B'].any()
        assert np.array_equal(model[field].numpy().astype(np.uint16)*(256//denominator), w)
        with np.load(d/'boards.npz', allow_pickle=False) as z:
            board = z['boards']; stopped = z['stopped'].copy(); moves = z['moves']
            assert board.shape == (count, 81) and board.dtype.kind in 'iu'
            assert stopped.shape == moves.shape == (count,) and stopped.dtype == bool
            assert ((board >= 0) & (board < 9)).all()
            assert np.array_equal(z['solutions'], target) and np.array_equal(z['free'], free)
            assert np.array_equal(z['puzzle'], np.arange(lo, hi)) and np.array_equal(board[~free], target[~free])
            assert moves.dtype.kind in 'iu' and ((moves > 0) & (moves <= 262144)).all()
            assert (moves[~stopped] == 262144).all() and (moves[stopped] % 4096 == 0).all()
            valid = np.ones(count, bool)
            for cells in units:
                valid &= (np.sort(board[:, cells], axis=1) == np.arange(9)).all(1)
            exact = (board == target).all(1)
            assert np.array_equal(z['exact'], exact) and np.array_equal(z['valid'], valid)
            e256 = ((board[:, ii] == board[:, jj])*w.astype(np.int64)).sum(1)
            target_e256 = ((target[:, ii] == target[:, jj])*w.astype(np.int64)).sum(1)
            given_e256 = ((target[:, ii] == target[:, jj]) & ~free[:, ii] & ~free[:, jj])*w.astype(np.int64)
            given_e256 = given_e256.sum(1)
            assert not e256[stopped].any() and (e256 >= given_e256).all()
            assert np.all(~exact | valid)
            if kind == 'archived':
                assert all(np.array_equal(a, b) for a, b in zip((exact, valid, stopped), archived[job]))
            diag = dict(correct_certified=int((exact & stopped).sum()), correct_capped=int((exact & ~stopped).sum()),
                        valid_certified=int((valid & stopped).sum()), valid_capped=int((valid & ~stopped).sum()),
                        wrong_stops=int((stopped & ~valid).sum()), stopped=int(stopped.sum()), capped=int((~stopped).sum()),
                        noncanonical_valid=int((valid & ~exact).sum()), noncanonical_valid_certified=int((valid & ~exact & stopped).sum()),
                        canonical_compatible=int((target_e256 == 0).sum()), given_given_infeasible=int((given_e256 > 0).sum()))
            assert r['solved'] == int(exact.sum()) and r['valid'] == int(valid.sum()) and r['wrong_stops'] == diag['wrong_stops']
            assert r['budget_failures'] == diag['capped'] and r['mean_moves'] == float(moves.mean())
            assert r['noncanonical_valid_returns'] == diag['noncanonical_valid']
            assert r['noncanonical_valid_certificates'] == diag['noncanonical_valid_certified']
            outputs[key] = (exact.copy(), valid.copy(), stopped.copy()); diagnostics[key] = diag
    rows = []; arrays = {}
    for row in mapping['rows']:
        key = row['kind'], row['job']; exact = outputs[key][0]
        arrays[row['bank'], row['arm']] = exact.astype(np.int8)
        rows.append(dict(**row, accuracy_pct=float(exact.mean()*100), **diagnostics[key]))
    grouped = []
    for arm in ARMS:
        chosen = [r for r in rows if r['arm'] == arm]
        totals = {k: sum(r[k] for r in chosen) for k in next(iter(diagnostics.values()))}
        grouped.append(dict(arm=arm, accuracy_pct=float(np.mean([r['accuracy_pct'] for r in chosen])),
                            **totals, **{k+'_pct': value/(64*count)*100 for k, value in totals.items()}))
    endpoints = []
    for arm, seed in [('weak256', 20261116031), ('weak64', 20261116032)]:
        delta = np.stack([arrays[b, arm]-arrays[b, 'weak16'] for b in range(64)])
        interval = np.quantile(resample_means(delta, seed, 5000), [.025, .975]).tolist()
        wrong = sum(r['wrong_stops'] for r in rows if r['arm'] == arm)
        endpoint = dict(arm=arm, reference='weak16', gain_pp=float(delta.mean()*100), gain95_interval_pp=interval,
                        wrong_stops=wrong, bootstrap_draws=5000, bootstrap_seed=seed)
        if arm == 'weak256':
            endpoint['decision'] = 'falsified' if wrong or interval[1] < 5 else 'supported' if interval[0] >= 5 else 'inconclusive'
        else:
            endpoint['role'] = 'descriptive'
        endpoints.append(endpoint)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    keys = sorted(outputs)
    packed = dict(kinds=np.array([k[0] for k in keys]), jobs=np.array([k[1] for k in keys]))
    for i, name in enumerate(['exact', 'valid', 'stopped']):
        matrix = np.stack([outputs[k][i] for k in keys]); packed[name] = np.packbits(matrix, axis=1)
        assert np.array_equal(np.unpackbits(packed[name], axis=1, count=count).astype(bool), matrix)
    np.savez_compressed(PUBLIC/'audited-outcome-bits.npz', **packed)
    result = dict(status='validated', protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__), scope=p['scope'],
                  primary=endpoints[0], descriptive=endpoints[1], grouped=grouped, rows=rows, source_audit=source,
                  input_sha256=evidence, boards_audited=len(outputs)*count, new_searches=len(q),
                  archived_searches=len(outputs)-len(q), logical_cells=len(rows), unique_inputs=count, source_banks=64,
                  outcome_bits_sha256=sha(PUBLIC/'audited-outcome-bits.npz'),
                  gate_launch=json.loads((OUT/'gate-launch.json').read_text()), search_launch=json.loads((OUT/'search-launch.json').read_text()))
    save(PUBLIC/'summary.json', result)
    print(json.dumps(dict(primary=result['primary'], descriptive=result['descriptive'], grouped=grouped)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['sources', 'outcomes'])
    globals()[parser.parse_args().action]()
