"""Independent direct-equality CPU audit of every archived corruption draw.

Run with upstream/.venv/bin/python; no SciPy required. The producer uses GPU
one-hot dot products; this audit reconstructs changes and compares raw digits.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/20261114/foureyes/noise_replication'
PUBLIC = ROOT/'results/research/foureyes/noise_replication'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def binomial_tail(k, n, p, upper):
    if p == 0:
        return float(0 >= k if upper else 0 <= k)
    if p == 1:
        return float(n >= k if upper else n <= k)
    indices = range(k, n+1) if upper else range(k+1)
    return math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
        +j*math.log(p)+(n-j)*math.log1p(-p)) for j in indices)


def exact_bounds(k, n, alpha):
    """Separate one-sided Clopper-Pearson bounds by direct tail inversion."""
    lower, upper = 0., 1.
    if k:
        lo, hi = 0., 1.
        for _ in range(65):
            mid = (lo+hi)/2
            if binomial_tail(k,n,mid,True) < alpha: lo=mid
            else: hi=mid
        lower=(lo+hi)/2
    if k<n:
        lo, hi=0.,1.
        for _ in range(65):
            mid=(lo+hi)/2
            if binomial_tail(k,n,mid,False)>alpha: lo=mid
            else: hi=mid
        upper=(lo+hi)/2
    return lower,upper


def audit_bank(seed):
    start=time.monotonic()
    p=json.loads((HERE/'noise_replication.json').read_text())
    dest=OUT/f's{seed}'
    receipt=json.loads((dest/'receipt.json').read_text())
    evidence={}
    def verify(path, expected=None):
        actual=digest(path)
        if expected is not None: assert actual==expected,path
        evidence[str(path.relative_to(ROOT))]=actual
    verify(dest/'receipt.json')
    verify(HERE/'noise_replication.json',receipt['protocol_sha256'])
    verify(HERE/'noise_replication.py',receipt['source_sha256'])
    for path,expected in receipt['input_sha256'].items(): verify(ROOT/path,expected)
    for name,expected in receipt['output_sha256'].items(): verify(dest/name,expected)
    assert receipt['status']=='complete' and receipt['draws']==p['draws']==512
    assert receipt['sample_sizes']==[300,3000] and receipt['per_row_cells_changed']==9
    sources=ROOT/f'artifacts/research/20261114/ripper/clues/completion_controls/s{seed}.npz'
    clues=ROOT/f'artifacts/research/20261114/ripper/clues/clue_inputs/s{seed}.npz'
    for path in [sources,clues]: verify(path,p['frozen_inputs'][str(path.relative_to(ROOT))])
    a,b=np.load(sources),np.load(clues)
    clean=a['solutions'].astype(np.uint8);free=~b['given']
    assert clean.shape==free.shape==(3000,81)
    assert np.array_equal(a['source_row'],b['source_row'])
    assert np.all((clean>=0)&(clean<9)) and np.array_equal(clean[~free],b['values'][~free])
    rows,cols=np.arange(81)//9,np.arange(81)%9
    ii,jj=np.triu_indices(81,1)
    truth=(rows[ii]==rows[jj])|(cols[ii]==cols[jj])|((rows[ii]//3==rows[jj]//3)&(cols[ii]//3==cols[jj]//3))
    assert truth.sum()==810 and not (clean[:,ii[truth]]==clean[:,jj[truth]]).any()
    changes=np.load(dest/'corruptions.npz');stored=np.load(dest/'counts.npz')
    cells,values=changes['cells'],changes['values']
    assert cells.shape==values.shape==(512,3000,9)
    assert (cells<81).all() and (values<9).all()
    assert (np.diff(np.sort(cells,axis=2),axis=2)>0).all()
    rr=np.arange(3000)[:,None]
    records=[]
    for draw in range(p['draws']):
        cc,vv=cells[draw],values[draw]
        assert free[rr,cc].all() and (clean[rr,cc]!=vv).all()
        noisy=clean.copy();noisy[rr,cc]=vv
        assert ((noisy!=clean).sum(1)==9).all() and np.array_equal(noisy[~free],clean[~free])
        # Direct equality, no one-hot representation or matrix multiplication.
        comparisons=noisy[:,ii]==noisy[:,jj]
        for n in p['sample_sizes']:
            counts=comparisons[:n].sum(0,dtype=np.int64)
            assert np.array_equal(counts,stored[f'n{n}'][draw]),(seed,n,draw)
            distinct=np.array(sorted(set(map(int,counts))))
            gaps=list(map(int,np.diff(distinct)))
            ix=gaps.index(max(gaps)) if gaps else None
            threshold=int((distinct[ix]+distinct[ix+1])//2) if ix is not None else 0
            assert threshold==stored[f't{n}'][draw]
            support=counts<=threshold
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            max_true=int(counts[truth].max());min_other=int(counts[~truth].min())
            records.append(dict(seed=seed,n=n,draw=draw,threshold=threshold,
                missing_true_edges=missing,extra_edges=extra,exact_graph=missing==extra==0,
                max_true_count=max_true,min_nonpeer_count=min_other,
                scalar_separator_exists=max_true<min_other))
        if (draw+1)%128==0:
            print(json.dumps(dict(seed=seed,draws_audited=draw+1,wall_s=time.monotonic()-start)),flush=True)
    return dict(seed=seed,records=records,input_sha256=evidence,wall_s=time.monotonic()-start)


def main():
    start=time.monotonic();p=json.loads((HERE/'noise_replication.json').read_text())
    jobs=json.loads((OUT/'queue.json').read_text())
    assert len(jobs)==6 and all(j['status']=='complete' for j in jobs)
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(audit_bank,p['source_seeds']))
    rows=[r for result in results for r in result['records']]
    grouped=[];alpha=.05/6
    for n in p['sample_sizes']:
        for seed in p['source_seeds']:
            group=[r for r in rows if r['n']==n and r['seed']==seed]
            k=sum(r['exact_graph'] for r in group);lo,hi=exact_bounds(k,len(group),alpha)
            grouped.append(dict(seed=seed,n=n,draws=len(group),exact_graphs=k,
                recovery_fraction=k/len(group),lower_one_sided=lo,upper_one_sided=hi,
                per_bank_alpha=alpha,separable_draws=sum(r['scalar_separator_exists'] for r in group),
                threshold_min=min(r['threshold'] for r in group),threshold_max=max(r['threshold'] for r in group)))
    primary_rows=[r for r in grouped if r['n']==p['primary_n']]
    supported=all(r['lower_one_sided']>=p['primary_fraction'] for r in primary_rows)
    falsified=any(r['upper_one_sided']<p['primary_fraction'] for r in primary_rows)
    evidence={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),str(OUT.relative_to(ROOT) / 'queue.json'):digest(OUT/'queue.json')}
    for result in results:evidence.update(result['input_sha256'])
    summary=dict(status='complete',audited_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(HERE/'noise_replication.json'),id=p['id'],models_audited=len(rows),
        corrupted_rows_audited=6*512*3000,changed_cells_audited=6*512*3000*9,
        all_counts_rebuilt_independently_on_cpu=True,all_clues_preserved=True,
        primary=dict(n=3000,target_fraction=.99,decision='supported' if supported else 'falsified' if falsified else 'inconclusive',
            family='Six separate banks; one-sided exact bounds each alpha=.05/6; Bonferroni simultaneous lower coverage>=95%. Upper bounds form a separate one-sided family.',
            min_lower_bound=min(r['lower_one_sided'] for r in primary_rows)),
        grouped=grouped,audit_wall_s=time.monotonic()-start,worker_wall_s={str(r['seed']):r['wall_s'] for r in results},
        input_sha256=evidence,limitations=p['limits'],
        compute_scope='Model construction only. No inference reruns for models identical to previously audited energies. CPU audit times include direct reconstruction of every archived noisy row.')
    PUBLIC.mkdir(exist_ok=True,parents=True)
    for name,table in [('per_draw.csv',rows),('per_bank.csv',grouped)]:
        with (PUBLIC/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    write(PUBLIC/'summary.json',summary);write(OUT/'audit.json',summary)
    print(json.dumps(dict(primary=summary['primary'],grouped=grouped,audit_wall_s=summary['audit_wall_s'])))


if __name__=='__main__':main()
