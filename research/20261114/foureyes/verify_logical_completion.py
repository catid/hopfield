"""Public-only support derivation and paired statistics for H4t."""
from pathlib import Path
import hashlib
import json
import numpy as np
from verify_sudoku_soundness import verify

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT/'results/research/foureyes/logical_completion'


def main():
    assert verify()['status'] == 'passed'
    original = ROOT/'results/research/foureyes/logical_soundness'
    certificates = json.loads((original/'certificates.json').read_text())
    with np.load(original/'supports.npz', allow_pickle=False) as z:
        old = dict(zip(map(str, z['job_ids']), z['supports']))
    rows = {(r['bank'], r['arm']): r for r in certificates['rows']}
    protocol = json.loads((ROOT/'research/20261114/foureyes/logical_completion.json').read_text())
    summary = json.loads((PUBLIC/'summary.json').read_text())
    pairs = [(i, j) for i in range(81) for j in range(i+1, 81)]
    truth = np.array([i//9 == j//9 or i % 9 == j % 9 or (i//27, i % 9//3) == (j//27, j % 9//3) for i, j in pairs])
    assert truth.sum() == 810
    expected = {(r['bank'], r['arm']) for r in rows.values() if r['sound'] and np.any(truth & ~old[r['job']])}
    assert expected == {(r['bank'], r['source_arm']) for r in protocol['selection']}
    mapping = {}; seen = {}; job_hashes = {}
    for r in summary['model_mappings']:
        key = r['bank'], r['source_arm']
        assert key in expected and r['arm'] in ['original', 'completed']
        support = old[rows[key]['job']]
        if r['arm'] == 'completed':support = support | truth
        digest = hashlib.sha256(support.tobytes()).hexdigest()
        assert digest == r['support_sha256']
        assert seen.setdefault(digest, r['actual']) == r['actual']
        assert job_hashes.setdefault(r['actual'], digest) == digest
        assert (*key, r['arm']) not in mapping
        mapping[*key, r['arm']] = r['actual']
    assert len(mapping) == 64 and len(seen) == len(job_hashes) == 30
    path = PUBLIC/'audited-outcome-bits.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == summary['outcome_bits_sha256']
    with np.load(path, allow_pickle=False) as z:
        good = np.unpackbits(z['valid'], axis=1, count=10000).astype(np.int8)
        stopped = np.unpackbits(z['stopped'], axis=1, count=10000).astype(bool)
        outcomes = dict(zip(map(str, z['job_ids']), good))
    assert good.shape == stopped.shape == (30, 10000)
    assert not (stopped & ~good.astype(bool)).any()
    for record in summary['comparisons']:
        arm = record['source_arm'];banks = sorted(b for b, a in expected if a == arm)
        before = np.stack([outcomes[mapping[b, arm, 'original']] for b in banks])
        after = np.stack([outcomes[mapping[b, arm, 'completed']] for b in banks])
        assert abs(before.mean()*100-record['original_accuracy_pct']) < 1e-12
        assert abs(after.mean()*100-record['completed_accuracy_pct']) < 1e-12
        delta = (after-before).mean(0);rng = np.random.default_rng(20261115555)
        values = [delta[rng.integers(0, 10000, 10000)].mean()*100 for _ in range(5000)]
        interval = np.quantile(values, [.025, .975])
        assert np.array_equal(interval, record['paired_input95_interval_pp'])
        assert abs(delta.mean()*100-record['gain_pp']) < 1e-12
        if arm == 'adamw':
            decision = 'falsified' if interval[1] < 1 else 'supported' if interval[0] >= 1 else 'inconclusive'
            assert decision == record['decision']
    print(json.dumps(dict(status='passed', exact_support_derivations=64, unique_models=30, reproduced_intervals=3)))


if __name__ == '__main__':
    main()
