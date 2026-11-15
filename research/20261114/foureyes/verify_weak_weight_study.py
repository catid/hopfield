"""Public-only reconstruction of H4z weights, outcomes and paired intervals."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT/'results/research/foureyes/weak_weight_study'
PARENT = ROOT/'results/research/foureyes/canonical_confirm'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    s = json.loads((PUBLIC/'summary.json').read_text())
    protocol = ROOT/'research/20261114/foureyes/weak_weight_study.json'
    p = json.loads(protocol.read_text())
    assert s['status'] == 'validated' and sha(protocol) == s['protocol_sha256']
    for path, digest in p['source_sha256'].items():
        # Upstream source is pinned separately and not vendored in Git.
        if not path.startswith('upstream/'):
            assert sha(ROOT/path) == digest, path
    for path, digest in p['input_sha256'].items():
        if not path.startswith('upstream/'):
            assert sha(ROOT/path) == digest, path
    assert sha(PUBLIC/'audited-outcome-bits.npz') == s['outcome_bits_sha256']
    with np.load(PARENT/'models.npz', allow_pickle=False) as z:
        coeff = z['canonical_weak_coefficients']
    with np.load(PUBLIC/'audited-outcome-bits.npz', allow_pickle=False) as z:
        outputs = {(str(k), str(j)): tuple(np.unpackbits(z[n][i], count=5000).astype(bool)
                   for n in ['exact', 'valid', 'stopped'])
                   for i, (k, j) in enumerate(zip(z['kinds'], z['jobs']))}
    with np.load(PARENT/'audited-outcome-bits.npz', allow_pickle=False) as z:
        parent = {str(j): tuple(np.unpackbits(z[n][i], count=5000).astype(bool)
                  for n in ['exact', 'valid', 'stopped']) for i, j in enumerate(z['jobs'])}
    ii, jj = np.triu_indices(81, 1)
    true = (ii//9 == jj//9) | (ii%9 == jj%9) | ((ii//27 == jj//27) & (ii%9//3 == jj%9//3))
    arrays = {}; identities = {}; reverse = {}
    for row in s['rows']:
        bank, arm = row['bank'], row['arm']; j = coeff[bank]
        vals = sorted(float(x) for x in j if x > 0); n = len(vals)
        median = vals[n//2] if n % 2 else (vals[n//2-1]+vals[n//2])/2
        weak = {'weak16': 16, 'weak64': 4, 'weak256': 1}[arm]
        w = np.array([0 if x <= 0 else 256 if x >= median/4 else weak for x in j], np.uint16)
        assert hashlib.sha256(w.tobytes()).hexdigest() == row['weight256_sha256']
        assert (w[true] > 0).all()
        key = row['kind'], row['job']; ident = (w.astype(np.float32)/256).tobytes()
        if key in identities:
            assert identities[key] == ident
        if ident in reverse:
            assert reverse[ident] == key
        identities[key] = ident; reverse[ident] = key
        exact, valid, stopped = outputs[key]
        assert np.all(~exact | valid)
        if key[0] == 'archived':
            assert all(np.array_equal(a, b) for a, b in zip(outputs[key], parent[key[1]]))
        arrays[bank, arm] = exact.astype(np.int8)
        assert abs(exact.mean()*100-row['accuracy_pct']) < 1e-12
        metrics = dict(correct_certified=(exact & stopped).sum(), correct_capped=(exact & ~stopped).sum(),
                       valid_certified=(valid & stopped).sum(), valid_capped=(valid & ~stopped).sum(),
                       wrong_stops=(stopped & ~valid).sum(), stopped=stopped.sum(), capped=(~stopped).sum(),
                       noncanonical_valid=(valid & ~exact).sum(), noncanonical_valid_certified=(valid & ~exact & stopped).sum())
        assert all(row[k] == int(v) for k, v in metrics.items())
    assert len(arrays) == 192 and len(outputs) == 192
    for group in s['grouped']:
        rows = [r for r in s['rows'] if r['arm'] == group['arm']]
        assert abs(np.mean([r['accuracy_pct'] for r in rows])-group['accuracy_pct']) < 1e-12
        for metric in metrics:
            total = sum(r[metric] for r in rows)
            assert total == group[metric] and abs(total/320000*100-group[metric+'_pct']) < 1e-12
    for endpoint in [s['primary'], s['descriptive']]:
        delta = np.stack([arrays[b, endpoint['arm']]-arrays[b, 'weak16'] for b in range(64)])
        assert delta.mean()*100 == endpoint['gain_pp']
        rng = np.random.default_rng(endpoint['bootstrap_seed']); draws = []
        for draw in range(5000):
            banks = rng.integers(0, 64, 64); inputs = rng.integers(0, 5000, 5000)
            total = delta[banks].sum(axis=0, dtype=np.int64)
            value = 100.*total[inputs].sum()/320000
            if draw < 16:
                assert value == 100.*delta[banks[:, None], inputs].sum()/320000
            draws.append(value)
        assert np.array_equal(np.quantile(draws, [.025, .975]), endpoint['gain95_interval_pp'])
        assert endpoint['wrong_stops'] == sum(r['wrong_stops'] for r in s['rows'] if r['arm'] == endpoint['arm'])
    e = s['primary']; lo, hi = e['gain95_interval_pp']
    decision = 'falsified' if e['wrong_stops'] or hi < 5 else 'supported' if lo >= 5 else 'inconclusive'
    assert decision == e['decision']
    assert s['gate_launch']['wall_seconds'] <= 45 and s['search_launch']['wall_seconds'] <= 255
    print(json.dumps(dict(status='passed', logical_cells=192, audited_outcomes=960000,
                         bootstrap_draws=10000, primary=e,
                         scope='Public packed outcomes and weight reconstruction; raw board validity remains the separately hashed full CPU audit.')))


if __name__ == '__main__':
    main()
