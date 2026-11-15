"""Public-only verification of H4u's inherited supports and fresh interval."""
from pathlib import Path
import hashlib
import json
import numpy as np
from verify_logical_completion import main as verify_parent

ROOT = Path(__file__).resolve().parents[3]
PUB = ROOT/'results/research/foureyes/logical_completion_confirm'


def main():
    verify_parent()
    p = ROOT/'research/20261114/foureyes/logical_completion_confirm.json'
    protocol = json.loads(p.read_text());result = json.loads((PUB/'summary.json').read_text())
    assert result['protocol_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert protocol['rows_half_open'] == [340000,380000] and protocol['search_seed'] == 0
    assert protocol['primary']['target_gain_pp'] == 1
    parent = json.loads((ROOT/'results/research/foureyes/logical_completion/summary.json').read_text())
    inherited = [r for r in parent['model_mappings'] if r['source_arm'] == 'adamw']
    assert inherited == result['model_mappings']
    mapping = {(r['bank'],r['arm']):r['actual'] for r in inherited}
    banks = sorted({b for b,a in mapping});assert len(banks) == 13 and len(mapping) == 26
    path = PUB/'audited-outcome-bits.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result['outcome_bits_sha256']
    with np.load(path,allow_pickle=False) as z:
        good = np.unpackbits(z['valid'],axis=1,count=40000).astype(np.int8)
        stopped = np.unpackbits(z['stopped'],axis=1,count=40000).astype(bool)
        outcomes = dict(zip(map(str,z['job_ids']),good))
    assert good.shape == stopped.shape == (18,40000)
    assert set(outcomes) == set(mapping.values())
    assert int((stopped & ~good.astype(bool)).sum()) == result['unique_wrong_stops'] == 0
    before = np.stack([outcomes[mapping[b,'original']] for b in banks])
    after = np.stack([outcomes[mapping[b,'completed']] for b in banks])
    assert abs(before.mean()*100-result['original_accuracy_pct']) < 1e-12
    assert abs(after.mean()*100-result['completed_accuracy_pct']) < 1e-12
    delta = (after-before).mean(0);rng = np.random.default_rng(20261115666)
    values = [delta[rng.integers(0,40000,40000)].mean()*100 for _ in range(5000)]
    interval = np.quantile(values,[.025,.975])
    assert np.array_equal(interval,result['paired_input95_interval_pp'])
    assert abs(delta.mean()*100-result['gain_pp']) < 1e-12
    decision = 'falsified' if interval[1]<1 else 'supported' if interval[0]>=1 else 'inconclusive'
    assert decision == result['decision']
    # Cross-check the first16 draws as literal two-dimensional column sampling.
    rng = np.random.default_rng(20261115666);difference = after-before
    for i in range(16):
        sample = rng.integers(0,40000,40000)
        assert abs(difference[:,sample].mean()*100-values[i]) < 1e-12
    print(json.dumps(dict(status='passed',inherited_model_mappings=26,unique_supports=18,unique_inputs=40000,
        interval_pp=interval.tolist(),literal_draws_checked=16,decision=decision)))


if __name__ == '__main__':main()
