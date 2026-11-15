"""Recompute H4r's primary interval using only public compact outcome bits."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/20261114/ripper'))
from bootstrap_two_way import resample_means


def main():
    public=ROOT/'results/research/foureyes/noisy_pl'
    s=json.loads((public/'summary.json').read_text());export=json.loads((public/'outcome-export.json').read_text())
    path=public/'audited-outcome-bits.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==export['sha256']
    with np.load(path,allow_pickle=False) as z:
        valid=np.unpackbits(z['valid'],axis=1,count=10000).astype(np.int8)
        stopped=np.unpackbits(z['stopped'],axis=1,count=10000).astype(bool)
        ids=list(z['job_ids'])
    assert valid.shape==(53,10000) and stopped.shape==valid.shape
    lookup={str(job):i for i,job in enumerate(ids)}
    mapping={(r['bank'],r['arm']):lookup[r['actual']] for r in s['model_mappings']}
    for arm in ['adamw','paper_pl','restricted','chernoff9']:
        indices=[mapping[b,arm] for b in range(32)]
        row=next(r for r in s['grouped'] if r['arm']==arm)
        assert abs(float(valid[indices].mean()*100)-row['mean_accuracy_pct'])<1e-10
        assert int((stopped[indices] & ~valid[indices].astype(bool)).sum())==row['logical_wrong_stops']
    delta=np.stack([valid[mapping[b,'paper_pl']]-valid[mapping[b,'restricted']] for b in range(32)])
    boot=resample_means(delta,20261115333,draws=5000)
    interval=np.quantile(boot,[.025,.975])
    assert np.array_equal(interval,np.asarray(s['primary']['paired_two_way95_interval_pp']))
    # Literal indexed resampling independently checks the multiplicity algorithm.
    rng=np.random.default_rng(20261115333)
    for k in range(16):
        rows=rng.integers(0,32,32);cols=rng.integers(0,10000,10000)
        expected=float(delta[rows][:,cols].mean()*100)
        assert abs(expected-boot[k])<1e-12
    print(json.dumps(dict(status='passed',gain_pp=float(delta.mean()*100),interval_pp=interval.tolist(),literal_draws_checked=16)))


if __name__=='__main__':main()
