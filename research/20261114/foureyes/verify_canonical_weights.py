"""Public-only independent H4x weight mapping and paired-interval verifier."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'results/research/foureyes/canonical_weights'
PARENT=ROOT/'results/research/foureyes/canonical_study'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    s=json.loads((PUBLIC/'summary.json').read_text());path=ROOT/'research/20261114/foureyes/canonical_weights.json';p=json.loads(path.read_text());assert sha(path)==s['protocol_sha256']
    assert sha(PARENT/'models.npz')==p['input_sha256'][str((PARENT/'models.npz').relative_to(ROOT))]
    assert sha(PUBLIC/'audited-outcome-bits.npz')==s['outcome_bits_sha256']
    with np.load(PARENT/'models.npz',allow_pickle=False)as z:coeff=z['canonical_weak_coefficients']
    with np.load(PARENT/'audited-outcome-bits.npz',allow_pickle=False)as z:
        inherited={str(k):(v,t)for k,v,t in zip(z['job_ids'],np.unpackbits(z['valid'],axis=1,count=10000),np.unpackbits(z['stopped'],axis=1,count=10000))}
    with np.load(PUBLIC/'audited-outcome-bits.npz',allow_pickle=False)as z:
        exact=np.unpackbits(z['exact'],axis=1,count=10000).astype(np.int8);stopped=np.unpackbits(z['stopped'],axis=1,count=10000)
        outputs={(str(kind),str(job)):(v,t)for kind,job,v,t in zip(z['kinds'],z['jobs'],exact,stopped)}
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));arrays={};identities={}
    for row in s['rows']:
        b,arm=row['bank'],row['arm'];j=coeff[b];vals=sorted(j[j>0]);n=len(vals)
        median=vals[n//2]if n%2 else(vals[n//2-1]+vals[n//2])/2
        w=np.array([0 if v<=0 else 16 if v>=median/4 else 1 for v in j],np.uint8)
        if arm=='uniform':w=(16*(j>0)).astype(np.uint8)
        if arm=='shuffled':
            pos=np.flatnonzero(j>0);w[pos]=w[pos][np.random.default_rng(p['shuffle_seed_base']+b).permutation(len(pos))]
        assert hashlib.sha256(w.tobytes()).hexdigest()==row['weight16_sha256']
        assert (w[truth]>0).all() and np.array_equal(w>0,j>0)
        key=(row['kind'],row['job']);v,t=outputs[key];arrays[b,arm]=v
        if row['kind']=='archived':
            assert all(np.array_equal(a,b)for a,b in zip((v,t),inherited[row['job']]))
        if key in identities:assert identities[key]==w.tobytes()
        identities[key]=w.tobytes()
        assert abs(v.mean()*100-row['accuracy_pct'])<1e-12
        assert (v*t).sum()==row['correct_certified'] and (v*(1-t)).sum()==row['correct_capped']
        # These observed exports have no noncanonical valid returns. The proof
        # of zero-stop validity itself follows from retaining all true edges.
        assert row['noncanonical_valid']==row['wrong_stops']==0 and not(t&~v.astype(bool)).any()
    assert len(arrays)==192
    for group in s['grouped']:
        rows=[r for r in s['rows']if r['arm']==group['arm']]
        assert abs(np.mean([r['accuracy_pct']for r in rows])-group['accuracy_pct'])<1e-12
        assert sum(r['correct_certified']for r in rows)/6400==group['correct_certified_pct']
        assert sum(r['correct_capped']for r in rows)/6400==group['correct_capped_pct']
    for endpoint,seed in zip(s['endpoints'],[20261116011,20261116012],strict=True):
        delta=np.stack([arrays[b,'magnitude']-arrays[b,endpoint['reference']]for b in range(64)]);rng=np.random.default_rng(seed);draws=[]
        for draw in range(5000):
            banks=rng.integers(0,64,64);inputs=rng.integers(0,10000,10000)
            sums=delta[banks].sum(axis=0,dtype=np.int64);value=100.*sums[inputs].sum()/640000
            if draw<16:assert value==100.*delta[banks[:,None],inputs].sum()/640000
            draws.append(value)
        interval=np.quantile(draws,[.025,.975]);assert np.array_equal(interval,endpoint['paired_two_way95_interval_pp'])
        assert abs(delta.mean()*100-endpoint['gain_pp'])<1e-12
        decision='supported'if interval[0]>=10 else'falsified'if interval[1]<10 else'inconclusive';assert decision==endpoint['decision']
    conjunction='falsified'if any(e['decision']=='falsified'for e in s['endpoints'])else'supported'if all(e['decision']=='supported'for e in s['endpoints'])else'inconclusive'
    assert conjunction==s['decision']
    print(json.dumps(dict(status='passed',logical_models=192,unique_outcomes=len(outputs),decision=conjunction,endpoints=s['endpoints'])))


if __name__=='__main__':main()
