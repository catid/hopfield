"""Train-only fixed count-gap boundary with nested sample sizes and cell corruptions."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/noise_boundary'


def main():
    p=json.loads((HERE/'noise_boundary.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'prepared.json').exists()
    ii,jj=np.triu_indices(81,1)
    jobs,mappings,models=[],[],[]
    for seed in p['search_seeds']:
        complete=ROOT/f'artifacts/research/20261114/ripper/clues/completion_controls/s{seed}.npz'
        clues=ROOT/f'artifacts/research/20261114/ripper/clues/clue_inputs/s{seed}.npz'
        for path in (complete,clues):assert hashlib.sha256(path.read_bytes()).hexdigest()==p['frozen_inputs'][str(path.relative_to(ROOT))]
        a,b=np.load(complete),np.load(clues)
        clean=a['solutions'].astype(np.int64);free=~b['given']
        assert np.array_equal(a['source_row'],b['source_row']) and np.array_equal(clean[~free],b['values'][~free])
        assert free.sum(1).min()>=9
        rng=np.random.default_rng(p['noise_rng_base']+seed)
        cells=np.empty((3000,9),dtype=np.int64);values=np.empty_like(cells)
        for row in range(3000):
            cells[row]=rng.choice(np.flatnonzero(free[row]),size=9,replace=False)
            values[row]=(clean[row,cells[row]]+rng.integers(1,9,size=9))%9
        seen={}
        for arm,recipe in p['recipes'].items():
            n,c=recipe['n'],recipe['corrupted_cells_per_board']
            train=clean[:n].copy()
            if c:train[np.arange(n)[:,None],cells[:n,:c]]=values[:n,:c]
            assert np.count_nonzero(train!=clean[:n])==n*c
            counts=(train[:,ii]==train[:,jj]).sum(0,dtype=np.int64)
            threshold=0
            if c:
                levels=np.unique(counts)
                index=int(np.diff(levels).argmax()) if len(levels)>1 else None
                threshold=int((levels[index]+levels[index+1])//2) if index is not None else 0
            support=counts<=threshold
            name=f's{seed}-{arm}';path=OUT/'models'/f'{name}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,train=train,clean=clean[:n],free=free[:n],counts=counts,support=support,
                source_row=a['source_row'][:n],changed_cells=cells[:n,:c],changed_values=values[:n,:c])
            digest=hashlib.sha256(support.tobytes()).hexdigest()
            item=dict(job_id=name,seed=seed,arm=arm,n=n,corrupted_cells_per_board=c,threshold=threshold,
                model_path=str(path.relative_to(ROOT)),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),support_sha256=digest)
            models.append(item)
            if digest not in seen:seen[digest]=name;jobs.append(dict(item,status='pending'))
            mappings.append(dict(requested=name,actual=seen[digest],seed=seed,arm=arm,
                reason='Identical support and all fixed inference settings, including seed and cohort'))
    out=dict(jobs=jobs,mappings=mappings,models=models,numpy=np.__version__,
        protocol_sha256=hashlib.sha256((HERE/'noise_boundary.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'prepared.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Prepared',len(models),'requested models and',len(jobs),'unique searches')


if __name__=='__main__':main()
