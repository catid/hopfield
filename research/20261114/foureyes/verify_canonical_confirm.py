"""Independent public-only check of fresh H4y mappings and endpoints."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'results/research/foureyes/canonical_confirm'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    s=json.loads((PUBLIC/'summary.json').read_text());p=ROOT/'research/20261114/foureyes/canonical_confirm.json'
    assert sha(p)==s['protocol_sha256']
    assert sha(PUBLIC/'models.npz')==s['weight_audit']['models_sha256']
    assert sha(PUBLIC/'audited-outcome-bits.npz')==s['outcome_bits_sha256']
    with np.load(PUBLIC/'models.npz',allow_pickle=False)as z:models={k:z[k]for k in z.files}
    with np.load(PUBLIC/'audited-outcome-bits.npz',allow_pickle=False)as z:
        exact=np.unpackbits(z['exact'],axis=1,count=5000).astype(np.int8)
        valid=np.unpackbits(z['valid'],axis=1,count=5000).astype(bool)
        stopped=np.unpackbits(z['stopped'],axis=1,count=5000).astype(bool)
        outputs={str(k):(e,v,t)for k,e,v,t in zip(z['jobs'],exact,valid,stopped)}
    assert len(outputs)==s['unique_searches'];arrays={};identities={};wrong={}
    for row in s['rows']:
        b,arm=row['bank'],row['arm'];j=models['canonical_weak_coefficients'][b]
        if arm=='magnitude':
            vals=sorted(j[j>0]);n=len(vals);median=vals[n//2]if n%2 else(vals[n//2-1]+vals[n//2])/2
            w=np.array([0 if v<=0 else 16 if v>=median/4 else 1 for v in j],np.uint8)
        else:w=(16*((j>0)if arm=='uniform'else models[arm+'_support'][b])).astype(np.uint8)
        assert hashlib.sha256(w.tobytes()).hexdigest()==row['weight16_sha256']
        if row['job']in identities:assert identities[row['job']]==w.tobytes()
        identities[row['job']]=w.tobytes()
        e,v,t=outputs[row['job']];arrays[b,arm]=e
        assert np.all(~e.astype(bool)|v)
        assert abs(e.mean()*100-row['accuracy_pct'])<1e-12
        assert int((t&~v).sum())==row['wrong_stops'];wrong[b,arm]=row['wrong_stops']
        assert int((e.astype(bool)&t).sum())==row['correct_certified']
        assert int((e.astype(bool)&~t).sum())==row['correct_capped']
        assert int((v&~e.astype(bool)).sum())==row['noncanonical_valid']
    assert len(arrays)==256
    for group in s['grouped']:
        rows=[r for r in s['rows']if r['arm']==group['arm']]
        assert abs(np.mean([r['accuracy_pct']for r in rows])-group['accuracy_pct'])<1e-12
        assert sum(r['wrong_stops']for r in rows)==group['wrong_stops']
        for metric in ['correct_certified','correct_capped']:
            assert abs(sum(r[metric]for r in rows)/320000*100-group[metric+'_pct'])<1e-12
    primary=s['primary'];mag=np.stack([arrays[b,'magnitude']for b in range(64)]);delta=mag-np.stack([arrays[b,'uniform']for b in range(64)])
    assert abs(mag.mean()*100-primary['accuracy_pct'])<1e-12 and abs(delta.mean()*100-primary['gain_pp'])<1e-12
    for data,seed,key in [(delta,20261116021,'gain95_interval_pp'),(mag,20261116022,'accuracy95_interval_pct')]:
        rng=np.random.default_rng(seed);draws=[]
        for draw in range(5000):
            banks=rng.integers(0,64,64);inputs=rng.integers(0,5000,5000)
            totals=data[banks].sum(axis=0,dtype=np.int64);value=100.*totals[inputs].sum()/320000
            if draw<16:assert value==100.*data[banks[:,None],inputs].sum()/320000
            draws.append(value)
        assert np.array_equal(np.quantile(draws,[.025,.975]),primary[key])
    assert sum(wrong[b,'magnitude']for b in range(64))==primary['wrong_stops']
    g,a=primary['gain95_interval_pp'],primary['accuracy95_interval_pct']
    decision='falsified'if primary['wrong_stops']or g[1]<50 or a[1]<60 else'supported'if g[0]>=50 and a[0]>=60 else'inconclusive'
    assert decision==primary['decision'];print(json.dumps(dict(status='passed',logical_cells=256,unique_searches=len(outputs),primary=primary)))


if __name__=='__main__':main()
