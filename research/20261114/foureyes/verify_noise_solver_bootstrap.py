"""Reproduce primary statistics from public outcome bits without raw archives."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'research/20261114/ripper'))
from bootstrap_two_way import resample_means
PUBLIC=ROOT/'results/research/foureyes/noise_solver';s=json.loads((PUBLIC/'summary.json').read_text());receipt=json.loads((PUBLIC/'outcome-export-provenance.json').read_text())
path=PUBLIC/'audited-outcome-bits.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['export_sha256']
z=np.load(path,allow_pickle=False);assert z['valid'].shape==z['stopped'].shape==(70,10000)
lookup={name:row for name,row in zip(z['job_ids'],z['valid'])};mapped={(r['bank'],r['arm']):lookup[r['actual']] for r in s['model_mappings']}
delta=np.stack([mapped[b,'chernoff9'].astype(np.int8)-mapped[b,'gap'].astype(np.int8) for b in range(512)])
boot=resample_means(delta,20261115222,5000);interval=np.quantile(boot,[.025,.975]);assert np.allclose(interval,s['primary']['paired_two_way95_interval_pp'],atol=1e-12,rtol=0)
assert abs(100*delta.mean()-s['primary']['gain_pp'])<1e-12
# Independent naive index replication checks the optimized integer formulation.
rng=np.random.default_rng(20261115222)
for index in range(16):
    banks=rng.integers(512,size=512);puzzles=rng.integers(10000,size=10000)
    direct=100*delta[banks][:,puzzles].mean();assert abs(direct-boot[index])<1e-12
print(json.dumps(dict(status='passed',gain_pp=float(100*delta.mean()),interval=interval.tolist(),bootstrap_draws=len(boot),naive_resamples_checked=16,public_data_only=True)))
