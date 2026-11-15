"""Fit count supports from a prespecified label-noise curve; no Sudoku geometry."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/research/20261114/foureyes/noise_curve'


def main():
    p = json.loads((HERE/'noise_curve.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'prepared.json').exists()
    ii,jj = np.triu_indices(81,1)
    jobs, mappings, manifest = [], [], []
    for seed in p['search_seeds']:
        source = ROOT / f'artifacts/research/20261114/ripper/raw/inputs/s{seed}/train.npz'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == p['frozen_inputs'][str(source.relative_to(ROOT))]
        z=np.load(source)
        clean=z['solutions'][:300].astype(np.int64)
        free=z['free'][:300].astype(bool)
        noisy=clean.copy()
        rng=np.random.default_rng(p['noise_rng_base']+seed)
        changed=[]
        permutation=rng.permutation(300)
        for row in permutation:
            cell=int(rng.choice(np.flatnonzero(free[row])))
            value=int((clean[row,cell]+rng.integers(1,9))%9)
            changed.append(dict(row=int(row),cell=cell,old=int(clean[row,cell]),new=value))
        seen={}
        for arm in p['arms']:
            recipe=p['recipes'][arm]
            train=clean.copy()
            for change in changed[:recipe['corrupted_rows']]:
                train[change['row'],change['cell']]=change['new']
            counts=(train[:,ii]==train[:,jj]).sum(0,dtype=np.int64)
            threshold=recipe['threshold']
            if threshold=='largest_gap':
                values=np.unique(counts)
                ix=int(np.diff(values).argmax()) if len(values)>1 else None
                threshold=int((values[ix]+values[ix+1])//2) if ix is not None else 0
            support=counts<=threshold
            name=f's{seed}-{arm}'
            model=OUT/'models'/f'{name}.npz'
            model.parent.mkdir(exist_ok=True)
            np.savez_compressed(model,support=support,counts=counts,train=train,free=free,clean=clean)
            digest=hashlib.sha256(support.tobytes()).hexdigest()
            row=dict(job_id=name,seed=seed,arm=arm,threshold=threshold,model_path=str(model.relative_to(ROOT)),
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),support_sha256=digest,
                corrupted_rows=recipe['corrupted_rows'],corruption=changed[:recipe['corrupted_rows']])
            manifest.append(row)
            if digest not in seen:
                seen[digest]=name
                jobs.append(dict(row,status='pending'))
            mappings.append(dict(requested=name,actual=seen[digest],seed=seed,arm=arm,
                reason='Identical support, literal unit1, tau0, ebar0, test inputs and search seed'))
    result=dict(jobs=jobs,mappings=mappings,models=manifest,numpy=np.__version__,
        protocol_sha256=hashlib.sha256((HERE/'noise_curve.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Prepared',len(manifest),'model requests and',len(jobs),'unique searches')


if __name__=='__main__':
    main()
