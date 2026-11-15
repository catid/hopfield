"""Prespecified descriptive fallback comparisons from independently audited H3w arrays."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3w';OUT=ROOT/'artifacts/research/20261114/loki/h3w';PUBLIC=ROOT/'results/research/loki/h3w'
spec=importlib.util.spec_from_file_location('loki_bootstrap',SRC.parent/'analyze.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    summary=json.loads((PUBLIC/'summary.json').read_text());assert summary['status']=='validated';records={};hashes={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),PUBLIC/'summary.json',SRC.parent/'analyze.py']};denom=None
    for row in summary['rows']:
        arm=row['arm'];z=[]
        for seed in [0,1,2]:
            path=OUT/f'search/{arm}/s{seed}/records.npz';rel=str(path.relative_to(ROOT));assert sha(path)==summary['input_sha256'][rel];hashes[rel]=sha(path);z.append(dict(np.load(path)))
        records[arm]=dict(n20=np.stack([x['n20_certificate']for x in z]),found=np.stack([x['found']for x in z]));denom=z[0]['denom']
    comparisons=[dict(arm=a,control=b,metric='certificate-first-with-best-energy-fallback accuracy',primary=False,**B.bootstrap(records[a],records[b],denom))for a,b in [('bp-adamw-permutation','uniform-filtered-permutation'),('bp-adamw-permutation','bp-adamw-arbitrary'),('bp-adamw-permutation','oracle-permutation')]]
    out=dict(status='validated_descriptive',primary_decision_unchanged=True,comparisons=comparisons,input_sha256=hashes);(PUBLIC/'fallback-comparisons.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(comparisons))
if __name__=='__main__':main()
