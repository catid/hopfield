"""Audit transformed raw fits with independent float64 coefficient reconstruction."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/research/20261114/ripper'
OUT = ROOT / 'artifacts/research/20261114/foureyes/raw_transform'
PUBLIC = ROOT / 'results/research/foureyes/raw_transform'


def main():
    evidence = {}
    def verify(path, expected=None):
        key = str(path.relative_to(ROOT))
        if key not in evidence:
            with path.open('rb') as f:
                evidence[key] = hashlib.file_digest(f, 'sha256').hexdigest()
        if expected is not None:
            assert evidence[key] == expected, path
        return evidence[key]

    protocol = HERE / 'raw_transform.json'
    p = json.loads(protocol.read_text())
    protocol_sha = verify(protocol)
    wrapper_sha = verify(HERE / 'raw_transform.py')
    verify(Path(__file__))
    erratum = json.loads((HERE / 'RAW_TRANSFORM_METADATA_ERRATUM.json').read_text())
    verify(HERE / 'RAW_TRANSFORM_METADATA_ERRATUM.json')
    assert erratum['frozen_protocol_sha256'] == protocol_sha and erratum['runner_sha256'] == wrapper_sha
    for name, digest in (p['source_sha256'] | p['frozen_models']).items():
        verify(ROOT / name, digest)
    jobs = json.loads((OUT / 'queue.json').read_text())
    assert len(jobs) == 48 and all(j['status'] == 'complete' for j in jobs)
    datafile = ROOT / 'upstream/data/full-test/full-all.npz'
    prior = json.loads((ROOT / 'results/research/ripper/raw_compare.json').read_text())
    verify(datafile, prior['input_sha256'][str(datafile.relative_to(ROOT))])
    lo, hi = p['rows_half_open']
    source = np.load(datafile)
    target = source['solutions'][lo:hi].astype(np.int8) - 1
    free = ~source['clue_mask'][lo:hi].astype(bool)
    ii, jj = np.triu_indices(81, 1)
    eye = np.eye(9, dtype=bool)
    row, col = np.arange(81) // 9, np.arange(81) % 9
    truth = ((row[ii] == row[jj]) | (col[ii] == col[jj]) |
             ((row[ii]//3 == row[jj]//3) & (col[ii]//3 == col[jj]//3)))
    arrays, rows, graph_rows, overlaps = {}, [], [], []
    test_set = set(map(bytes, np.ascontiguousarray(target)))
    for job in jobs:
        directory = OUT / 'jobs' / job['job_id']
        r = json.loads((directory / 'receipt.json').read_text())
        verify(directory / 'receipt.json')
        n, seed, arm = r['k'], r['seed'], r['arm']
        assert r['status'] == 'complete' and r['protocol_sha256'] == protocol_sha and r['wrapper_sha256'] == wrapper_sha
        assert r['rows_half_open'] == [lo, hi] and r['test_rows'] == hi-lo and r['training_rows'] == n
        assert r['preflight_checks'] == 5 and r['preflight_state_fields'] == 25
        verify(ROOT / 'research/20261114/ripper/sparse_backend.py', r['backend_sha256'])
        for name, digest in r['source_sha256'].items():
            verify(ROOT / name, digest)
        for name, digest in r['output_sha256'].items():
            verify(directory / name, digest)
        entry = next(e for e in p['training_inputs'] if e['seed'] == seed)
        trainfile = ROOT / entry['path']
        verify(trainfile, entry['sha256'])
        assert r['training_sha256'] == entry['sha256']
        train = np.load(trainfile)['solutions'][:n]
        overlap = sum(bytes(v) in test_set for v in np.ascontiguousarray(train, dtype=np.int8))
        overlaps.append(dict(n=n, seed=seed, arm=arm, training_boards_in_test=overlap))
        assert overlap == 0
        model = torch.load(directory / 'searched_model.pt', weights_only=True, map_location='cpu')
        fit = torch.load(directory / 'fitted_model.pt', weights_only=True, map_location='cpu')
        W, B = model['W'].numpy(), model['B'].numpy()
        assert not B.any()
        support = W[:, 0, 0] > 0
        level = float(W[:, 0, 0][support][0])
        assert np.array_equal(W, support[:, None, None].astype(np.float32) * np.float32(level) * eye)
        if arm == 'count':
            counted = ~(train[:, ii] == train[:, jj]).any(0)
            assert np.array_equal(support, counted) and level == 1 and r['used_hbar'] == 1
            assert all(torch.equal(fit[k], model[k]) for k in model)
        else:
            original_path = BASE / f'raw_compare/jobs/k{n}-s{seed}-{arm}/fitted_model.pt'
            original = torch.load(original_path, weights_only=True, map_location='cpu')
            assert all(torch.equal(original[k], fit[k]) for k in fit)
            assert r['reused_prior_fit'] and r['fit_s'] == 0
            verify(original_path, r['transform']['source_model_sha256'])
            # Direct float64 digit-orbit averages, independently of the GPU helper.
            f = fit['W'].numpy().astype(np.float64)
            j = np.maximum(f[:, eye].mean(1) - f[:, ~eye].mean(1), 0)
            positive = np.sort(j[j > 0])
            gap_index = int(np.diff(positive).argmax())
            threshold = (positive[gap_index] + positive[gap_index+1]) / 2
            reconstructed = j > threshold
            assert np.array_equal(support, reconstructed)
            assert np.isclose(level, j[support].mean(), rtol=1e-6, atol=1e-8)
            assert gap_index == r['transform']['gap_index'] and r['used_hbar'] == level
        train_energy = (train[:, ii[support]] == train[:, jj[support]]).sum(1) * level
        assert np.isclose(r['statistics']['tau_train'], train_energy.max(), rtol=1e-5, atol=1e-6)
        assert np.isclose(r['statistics']['ebar'], 2*train_energy.mean()/81, rtol=1e-5, atol=1e-6)
        graph_rows.append(dict(n=n, seed=seed, arm=arm, edges=int(support.sum()),
            missing_true_edges=int((truth & ~support).sum()), extra_edges=int((support & ~truth).sum()),
            exact_peer_graph=bool(np.array_equal(support, truth)), coefficient=level))
        z = np.load(directory / 'boards.npz')
        b, st, moves = z['boards'], z['stopped'], z['moves']
        assert np.array_equal(z['solutions'], target) and np.array_equal(z['free'], free)
        assert np.array_equal(z['puzzle'], np.arange(lo, hi)) and b.shape == (10000,81)
        board = b.reshape(-1,9,9)
        box = board.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        valid = ((b == target) | free).all(1)
        for digit in range(9):
            valid &= ((board == digit).sum(1) == 1).all(1)
            valid &= ((board == digit).sum(2) == 1).all(1)
            valid &= ((box == digit).sum(2) == 1).all(1)
        exact = (b == target).all(1)
        assert np.array_equal(valid, exact) and np.array_equal(z['valid'], valid) and np.array_equal(z['exact'], exact)
        assert int(valid.sum()) == r['solved'] and int((st & ~valid).sum()) == r['wrong_stops']
        assert np.all((moves > 0) & (moves <= 262144)) and (moves[~st] == 262144).all()
        result = json.loads((directory / 'result.json').read_text())['per_puzzle']
        assert np.array_equal(result['solved'], valid) and np.array_equal(result['stopped'], st)
        if arm == 'count':
            assert not (st & ~valid).any()
        arrays[n, seed, arm] = valid.astype(float)
        rows.append({key:r[key] for key in ['job_id','k','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s']})
    reuse = []
    for seed in range(6):
        a = np.load(BASE / f'raw/counts/n100-s{seed}/support.npz')['support']
        b = np.load(BASE / f'raw/counts/n300-s{seed}/support.npz')['support']
        assert np.array_equal(a, b)
        arrays[300, seed, 'count'] = arrays[100, seed, 'count']
        reuse.append(dict(requested=f'k300-s{seed}-count', actual=f'k100-s{seed}-count', reason='Identical support, unit, tau, ebar, cohort and seed'))
    grouped = []
    for n in (30,100,300):
        for arm in p['arms']:
            rs = [r for r in rows if r['k'] == (100 if n == 300 and arm == 'count' else n) and r['arm'] == arm]
            grouped.append(dict(n=n, arm=arm, mean_accuracy_pct=float(np.mean([r['solved']/100 for r in rs])),
                solved_per_seed=[r['solved'] for r in rs], wrong_stops=sum(r['wrong_stops'] for r in rs),
                reused_count_n100=n == 300 and arm == 'count'))
    delta = np.stack([arrays[100,s,'pl_digit']-arrays[100,s,'count'] for s in range(6)])
    rng = np.random.default_rng(20261114)
    boots = np.array([100*delta[rng.integers(6,size=6)][:,rng.integers(10000,size=10000)].mean() for _ in range(5000)])
    low, high = map(float,np.quantile(boots,[.025,.975]))
    wrong = next(r['wrong_stops'] for r in grouped if r['n'] == 100 and r['arm'] == 'pl_digit')
    decision = 'falsified' if wrong or high < -2 else 'supported' if low >= -2 else 'inconclusive'
    primary = dict(n=100, contrast=['pl_digit','count'], gain_pp=float(delta.mean()*100),
        paired95_interval_pp=[low,high], noninferiority_margin_pp=2, wrong_stops=wrong, decision=decision)
    rec = dict(status='validated', utc=datetime.now(timezone.utc).isoformat(), primary=primary,
        grouped=grouped, rows=rows, graph_diagnostics=graph_rows, reused_searches=reuse,
        training_test_overlap=overlaps, saved_boards_verified=480000,
        original_fits_unchanged=True, independent_float64_support_reconstruction=True,
        input_sha256=evidence, limitations=p['limits'], metadata_erratum=erratum)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for path in (OUT/'audit.json',PUBLIC/'summary.json'):
        path.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
    for name, data in [('per_seed.csv',rows),('graph_diagnostics.csv',graph_rows)]:
        with (PUBLIC/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    print(json.dumps(primary))


if __name__ == '__main__':
    main()
