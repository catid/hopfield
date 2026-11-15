"""Privileged clean-input-only Chernoff bands, not an estimator guarantee."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'results/research/foureyes/noise_replication'
RAW=ROOT/'artifacts/research/20261114/foureyes/noise_replication'


def tail_bounds(k,n,mean,upper):
    """P(C>=k) or P(C<=k), via Bernoulli KL using mean row probability."""
    if (upper and k<=0) or (not upper and k>=n):return np.ones_like(mean)
    if (upper and k>n) or (not upper and k<0):return np.zeros_like(mean)
    q=k/n
    out=np.ones_like(mean)
    tail=(mean<q) if upper else (mean>q)
    inside=tail&(mean>0)&(mean<1)
    p=mean[inside]
    divergence=np.zeros_like(p)
    if q>0:divergence+=q*np.log(q/p)
    if q<1:divergence+=(1-q)*np.log((1-q)/(1-p))
    out[inside]=np.exp(-n*divergence)
    out[tail&((mean==0)|(mean==1))]=0
    return out


def main():
    ii,jj=np.triu_indices(81,1);r,c=np.arange(81)//9,np.arange(81)%9
    truth=(r[ii]==r[jj])|(c[ii]==c[jj])|((r[ii]//3==r[jj]//3)&(c[ii]//3==c[jj]//3))
    records=[];evidence={}
    def record(path):evidence[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(PUBLIC/'summary.json');record(PUBLIC/'per_draw.csv')
    rows=list(csv.DictReader((PUBLIC/'per_draw.csv').open()))
    for seed in range(6):
        sp=ROOT/f'artifacts/research/20261114/ripper/clues/completion_controls/s{seed}.npz'
        cp=ROOT/f'artifacts/research/20261114/ripper/clues/clue_inputs/s{seed}.npz'
        record(sp);record(cp)
        clean=np.load(sp)['solutions'];free=~np.load(cp)['given']
        m=free.sum(1);a=(9/m)[:,None];b=(72/(m*(m-1)))[:,None]
        equal=clean[:,ii]==clean[:,jj]
        both=free[:,ii]&free[:,jj];one=free[:,ii]^free[:,jj]
        probabilities=np.where(both,np.where(equal,1-2*a+9*b/8,a/4-9*b/64),
            np.where(one,np.where(equal,1-a,a/8),equal.astype(float)))
        assert ((probabilities>=0)&(probabilities<=1)).all()
        for n in [300,3000]:
            mean=probabilities[:n].mean(0)
            # Choose deterministic endpoints using clean data/noise law alone.
            lower=next(t for t in range(n+1) if tail_bounds(t+1,n,mean[truth],True).sum()<=.005)
            upper=next(t for t in range(n,-2,-1) if tail_bounds(t,n,mean[~truth],False).sum()<=.005)
            bound=float(tail_bounds(lower+1,n,mean[truth],True).sum()+tail_bounds(upper,n,mean[~truth],False).sum())
            group=[z for z in rows if int(z['seed'])==seed and int(z['n'])==n]
            nonempty=lower<=upper
            separated=sum(int(z['max_true_count'])<=lower and int(z['min_nonpeer_count'])>upper for z in group)
            membership=sum(lower<=int(z['threshold'])<=upper for z in group)
            records.append(dict(seed=seed,n=n,lower=lower,upper=upper,nonempty_band=nonempty,
                separation_failure_union_bound=bound,draws=len(group),separation_event_draws=separated,
                threshold_inside_band_draws=membership,
                true_pair_mean_count_max=float(n*mean[truth].max()),nonpeer_mean_count_min=float(n*mean[~truth].min())))
    with (PUBLIC/'conditional_bands.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    record(Path(__file__))
    record(ROOT/'research/20261114/ripper/NOISE_PAIR_BOUND_REVIEW.md')
    (PUBLIC/'conditional-band-provenance.json').write_text(json.dumps(dict(
        role='Post-fit clean-input-only privileged diagnostic. Does not enter estimator.',
        endpoint_rule='Smallest L with true upper-tail union<=.005, largest U with nonpeer lower-tail union<=.005.',
        theory='Independent row corruption gives Poisson-binomial counts. Jensen MGF gives binomial-form Chernoff KL bounds, summed across pairs.',
        limitation='Nonempty band guarantees any threshold inside it is correct on the separation event. The largest-gap selection need not lie inside; observed membership does not imply the same conditional error bound. Empty bands provide no valid threshold.',
        input_sha256=evidence),indent=2,allow_nan=False)+'\n')
    print(json.dumps(records))


if __name__=='__main__':main()
