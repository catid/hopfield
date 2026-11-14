"""Independent audit of periodic unfocused exploration."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/research/20261114/foureyes/focus'
BASE = ROOT / 'artifacts/research/20261114/ripper'
PUBLIC = ROOT / 'results/research/foureyes/focus'


def bounds(k, n, alpha=.05):
    """Invert exact binomial tails; two separate one-sided bounds, not a 95% CI."""
    def cdf(j, probability):
        if j < 0:
            return 0.0
        if j >= n:
            return 1.0
        r = np.arange(j + 1, dtype=float)
        logs = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(j + 1)])
        logs += r * math.log(probability) + (n - r) * math.log1p(-probability)
        peak = logs.max()
        return min(1.0, float(math.exp(peak) * np.exp(logs - peak).sum()))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        if cdf(n - k, 1 - p) < alpha:
            lo = p
        else:
            hi = p
    lower = (lo + hi) / 2 if k else 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        if cdf(k, p) > alpha:
            lo = p
        else:
            hi = p
    upper = (lo + hi) / 2 if k < n else 1.0
    return lower, upper


def validity(boards, solutions, free):
    n = len(boards)
    b = boards.reshape(n, 9, 9)
    expected = np.arange(9)
    rows = (np.sort(b, axis=2) == expected).all((1, 2))
    cols = (np.sort(b, axis=1) == expected[None, :, None]).all((1, 2))
    boxes = b.reshape(n, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(n, 9, 9)
    blocks = (np.sort(boxes, axis=2) == expected).all((1, 2))
    clues = ((boards == solutions) | free).all(1)
    return rows & cols & blocks & clues


def main():
    protocol_path = ROOT / 'research/20261114/foureyes/focus.json'
    protocol = json.loads(protocol_path.read_text())
    evidence = {}
    def verify(path, expected=None):
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if expected is not None:
            assert digest == expected, path
        evidence[str(path.relative_to(ROOT))] = digest
        return digest
    verify(protocol_path)
    verify(Path(__file__))
    prior = json.loads((ROOT / 'results/research/ripper/finaltest.json').read_text())
    verify(ROOT / 'results/research/ripper/finaltest.json')
    dataset = ROOT / 'upstream/data/full-test/full-all.npz'
    verify(dataset, prior['input_sha256'][str(dataset.relative_to(ROOT))])
    sys.path[:0] = [str(ROOT / 'upstream')]
    from cram.cram_sudoku_search import load_puzzles
    sol, free, idx = load_puzzles(protocol['split'], protocol['rows_half_open'][0], protocol['rows_half_open'][1]-protocol['rows_half_open'][0])
    assert validity(sol, sol, free).all()
    caps = protocol['checkpoints'][1:]
    matrices = {(arm,cap): [] for arm in protocol['arms'] for cap in caps}
    support = np.load(BASE / 'count/k10-s0/counts.npz')['support']
    ii, jj = np.triu_indices(81, 1)
    adjacency = np.zeros((81,81), dtype=bool)
    adjacency[ii[support], jj[support]] = True
    adjacency |= adjacency.T
    neighbors = np.stack([np.flatnonzero(row) for row in adjacency])
    assert neighbors.shape == (81,20)
    support_sha = verify(BASE / 'count/k10-s0/counts.npz')
    rows = []
    checked = 0
    for arm in protocol['arms']:
        for seed in protocol['search_seeds']:
            dest = OUT / f's{seed}-{arm}'
            rec = json.loads((dest / 'receipt.json').read_text())
            assert rec['status'] == 'complete' and rec['bounded_native_capture_sparse_parity']
            verify(dest / 'receipt.json')
            for path, expected in (rec['input_sha256'] | rec['source_sha256']).items():
                verify(ROOT / path, expected)
            previous = None
            for envelope in caps:
                cap = envelope
                directory = dest / f'moves-{cap}'
                receipt = next(r for r in rec['checkpoints'] if r['cap'] == cap)
                for name, expected in receipt['output_sha256'].items():
                    verify(directory / name, expected)
                verify(directory / 'boards.npz')
                z = np.load(directory / 'boards.npz')
                assert np.array_equal(z['solutions'], sol) and np.array_equal(z['free'], free) and np.array_equal(z['puzzle'], idx)
                b = z['boards']
                valid = validity(b, sol, free)
                exact = (b == sol).all(1)
                assert np.array_equal(valid, exact) and np.array_equal(exact, z['exact']) and np.array_equal(valid, z['valid'])
                assert not (z['stopped'] & ~valid).any()
                assert (z['moves'] <= cap).all()
                assert (z['moves'][~z['stopped']] == cap).all()
                assert valid.sum() == receipt['solved']
                if previous is not None:
                    stopped, boards, moves = previous
                    assert z['stopped'][stopped].all()
                    assert np.array_equal(b[stopped], boards[stopped]) and np.array_equal(z['moves'][stopped], moves[stopped])
                state_name = 'state.pt'
                capture_name = 'capture.pt'
                state = torch.load(directory / state_name, weights_only=False, map_location='cpu')
                capture = torch.load(directory / capture_name, weights_only=True, map_location='cpu')
                returned = torch.where(state['stopped'][:, None], capture['cert_boards'], capture['best_boards']).numpy()
                assert np.array_equal(returned, b) and np.array_equal(state['stopped'].numpy(), z['stopped'])
                assert np.array_equal(torch.where(state['stopped'], state['ok'], state['bestok']).numpy(), valid)
                assert state['t'] == receipt['terminal_move']
                y = state['y'].numpy()
                adjacent_digits = y[:, neighbors]
                fields = np.stack([(adjacent_digits == digit).sum(2) for digit in range(9)], axis=2)
                energy = (y[:, ii[support]] == y[:, jj[support]]).sum(1)
                assert np.array_equal(state['G'].numpy(), fields)
                assert np.array_equal(state['P'].numpy(), energy)
                assert receipt['integer_fields_and_energy_exact']
                expected_flag=(np.take_along_axis(fields,y[:,:,None],axis=2).squeeze(2)>.45)&state['fr'].numpy()
                assert np.array_equal(state['flag'].numpy(),expected_flag)
                blocks=state['t']//protocol['check']
                uniform=0 if arm=='focused' else (blocks if arm=='uniform' else blocks//16)
                assert receipt['uniform_blocks']==uniform and receipt['focused_blocks']==blocks-uniform
                previous = (z['stopped'].copy(), b.copy(), z['moves'].copy())
                assert receipt['maximum_chain_moves'] == envelope * protocol['chains_by_arm'][arm] and receipt['chains'] == protocol['chains_by_arm'][arm]
                matrices[arm,envelope].append(valid)
                rows.append(dict(arm=arm, seed=seed, cap=cap, maximum_chain_moves=envelope*protocol['chains_by_arm'][arm], chains=protocol['chains_by_arm'][arm], n=len(b), solved=int(valid.sum()), accuracy_pct=float(valid.mean() * 100),
                                 stopped=int(z['stopped'].sum()), wrong_stops=0, mean_moves=float(z['moves'].mean()),
                                 restart_count_sum=receipt['restart_count_sum'],restart_count_mean=receipt['restart_count_mean'],continuation_wall_s=receipt.get('continuation_wall_s', 0.0)))
                checked += len(b)
    grouped = []
    for (arm,cap), entries in matrices.items():
        a = np.stack(entries)
        joint = a.all(0)
        lower, upper = bounds(int(joint.sum()), len(joint))
        grouped.append(dict(arm=arm,maximum_chain_moves=cap*protocol['chains_by_arm'][arm], chains=protocol['chains_by_arm'][arm],cap=cap, unique_puzzles=len(joint), mean_accuracy_pct=float(a.mean() * 100),
                            per_seed_solved=[int(v.sum()) for v in a], joint_solved=int(joint.sum()), joint_accuracy_pct=float(joint.mean() * 100),
                            one_sided95_lower_pct=lower * 100, one_sided95_upper_pct=upper * 100,
                            joint_statistics_role='descriptive_only', wrong_stops=0))
    comparisons=[]
    for envelope in caps:
        for arm in ('mixed16','uniform'):
            a=np.asarray(matrices[arm,envelope],dtype=float)
            b=np.asarray(matrices['focused',envelope],dtype=float)
            delta=a-b
            rng=np.random.default_rng(20261114)
            boots=np.array([delta[rng.integers(6,size=6)][:,rng.integers(10000,size=10000)].mean()*100 for _ in range(5000)])
            lower,upper=np.quantile(boots,[.025,.975])
            primary = arm == 'mixed16' and envelope == protocol['primary_endpoint']
            decision='supported' if lower>=protocol['primary_gain_pp'] else ('falsified' if upper<protocol['primary_gain_pp'] else 'inconclusive')
            comparisons.append(dict(cap=envelope,arm=arm,control='focused',primary=primary,
                accuracy_pct=float(a.mean()*100),control_accuracy_pct=float(b.mean()*100),gain_pp=float(delta.mean()*100),
                paired95_interval_pp=[float(lower),float(upper)],threshold_pp=protocol['primary_gain_pp'],
                decision=decision if primary else 'secondary_descriptive',
                regressions=int((delta<0).sum()),improvements=int((delta>0).sum())))
    summary = dict(status='validated', utc=datetime.now(timezone.utc).isoformat(), protocol=str(protocol_path.relative_to(ROOT)),
                   primary=next(c for c in comparisons if c['primary']), comparisons=comparisons,
                   rows=rows, grouped=grouped, saved_boards_verified=checked,
                   bounded_native_capture_sparse_checks_per_job=33, integer_fields_and_energy_independently_recomputed=True, stopped_prefix_preservation=True,
                   input_sha256=evidence, limitations=protocol['limits'] + ['Ten unique base puzzles with99stored variants each;1000augmented training examples.', 'One optimizer-free binary graph with six search seeds; no trained-model replication.', 'One-sided95% endpoints are not a two-sided95% confidence interval.'])
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'audit.json', PUBLIC / 'summary.json'):
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    with (PUBLIC / 'per_seed.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: summary[k] for k in ('status', 'primary', 'saved_boards_verified')}))


if __name__ == '__main__':
    main()
