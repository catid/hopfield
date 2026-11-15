"""Public H4aa weights, archived identity, audit correction and interval check."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'research/20261114/foureyes'
PUBLIC=ROOT/'results/research/foureyes/strong_support_study'
PARENT=ROOT/'results/research/foureyes/weak_weight_study'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    s=json.loads((PUBLIC/'summary.json').read_text());p=json.loads((HERE/'strong_support_study.json').read_text())
    assert s['status']=='validated'and s['protocol_sha256']==sha(HERE/'strong_support_study.json')
    for rel,h in (p['source_sha256']|p['input_sha256']).items():
        if not rel.startswith('upstream/'):assert sha(ROOT/rel)==h,rel
    correction=json.loads((HERE/'strong-support-audit-correction.json').read_text())
    assert correction['original_auditor_sha256']==sha(HERE/'audit_strong_support_study.py')
    assert correction['corrected_auditor_sha256']==s['auditor_sha256']==sha(HERE/'audit_strong_support_study_corrected.py')
    original=(HERE/'audit_strong_support_study.py').read_text()
    corrected=original.replace("source['auditor_sha256']==sha(__file__)","source['auditor_sha256']==sha(HERE/'audit_strong_support_study.py')")
    corrected=corrected.replace("assert t.dtype==bool and ((moves>=0)&(moves<=262144)).all()and(moves[~t]==262144).all()and(moves[t]%4096==0).all()",
        "assert t.dtype==bool and moves.dtype.kind in 'iu' and ((moves>0)&(moves<=262144)).all()and(moves[~t]==262144).all()")
    assert corrected==(HERE/'audit_strong_support_study_corrected.py').read_text()
    assert not correction['inference_rerun']
    assert sha(PUBLIC/'audited-outcome-bits.npz')==s['outcome_bits_sha256']
    with np.load(ROOT/'results/research/foureyes/canonical_confirm/models.npz',allow_pickle=False)as z:coeff=z['canonical_weak_coefficients']
    with np.load(PUBLIC/'audited-outcome-bits.npz',allow_pickle=False)as z:
        outputs={(str(k),str(j)):tuple(np.unpackbits(z[n][i],count=5000).astype(bool)for n in ['exact','valid','stopped'])for i,(k,j)in enumerate(zip(z['kinds'],z['jobs']))}
    with np.load(PARENT/'audited-outcome-bits.npz',allow_pickle=False)as z:
        archived={str(j):tuple(np.unpackbits(z[n][i],count=5000).astype(bool)for n in ['exact','valid','stopped'])for i,(k,j)in enumerate(zip(z['kinds'],z['jobs']))if str(k)=='new'}
    arrays={};identity={};reverse={};ii,jj=np.triu_indices(81,1)
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    structure={(r['bank'],r['arm']):r for r in s['source_audit']['rows']}
    for r in s['rows']:
        b,a=r['bank'],r['arm'];j=coeff[b];vals=sorted(j[j>0]);n=len(vals);cut=(vals[n//2]if n%2 else(vals[n//2-1]+vals[n//2])/2)/4
        weak={'weak256':1,'strongonly':0}[a];w=np.array([0 if x<=0 else 256 if x>=cut else weak for x in j],np.uint16)
        assert hashlib.sha256(w.tobytes()).hexdigest()==r['weight256_sha256']
        key=r['kind'],r['job'];v=w.tobytes()
        if key in identity:assert identity[key]==v
        if v in reverse:assert reverse[v]==key
        identity[key]=v;reverse[v]=key
        st=structure[b,a];assert st['retained']==int((w>0).sum())and st['missing_true']==int((w[truth]==0).sum())and st['extra']==int((w[~truth]>0).sum())
        exact,valid,stopped=outputs[key];assert np.all(~exact|valid)
        if key[0]=='archived':assert all(np.array_equal(x,y)for x,y in zip(outputs[key],archived[key[1]]))
        arrays[b,a]=exact.astype(np.int8)
        metrics=dict(correct_certified=(exact&stopped).sum(),correct_capped=(exact&~stopped).sum(),valid_certified=(valid&stopped).sum(),valid_capped=(valid&~stopped).sum(),wrong_stops=(stopped&~valid).sum(),stopped=stopped.sum(),noncanonical_valid=(valid&~exact).sum(),noncanonical_valid_certificates=(valid&~exact&stopped).sum())
        assert abs(exact.mean()*100-r['accuracy_pct'])<1e-12 and all(r[k]==int(v)for k,v in metrics.items())
    assert len(arrays)==128 and len(outputs)==123 and s['boards_audited']==615000
    for g in s['grouped']:
        chosen=[r for r in s['rows']if r['arm']==g['arm']]
        assert abs(np.mean([r['accuracy_pct']for r in chosen])-g['accuracy_pct'])<1e-12
        assert all(sum(r[k]for r in chosen)==g[k]for k in metrics)
    delta=np.stack([arrays[b,'strongonly']-arrays[b,'weak256']for b in range(64)])
    rng=np.random.default_rng(20261116041);draws=[]
    for draw in range(5000):
        banks=rng.integers(0,64,64);inputs=rng.integers(0,5000,5000);totals=delta[banks].sum(0,dtype=np.int64)
        value=100.*totals[inputs].sum()/320000
        if draw<16:assert value==100.*delta[banks[:,None],inputs].sum()/320000
        draws.append(value)
    e=s['primary'];assert np.array_equal(np.quantile(draws,[.025,.975]),e['gain95_interval_pp'])and e['gain_pp']==delta.mean()*100
    assert e['wrong_stops']==sum(r['wrong_stops']for r in s['rows']if r['arm']=='strongonly')
    lo,hi=e['gain95_interval_pp'];decision='falsified'if e['wrong_stops']or hi<5 else'supported'if lo>=5 else'inconclusive';assert decision==e['decision']
    assert s['search_launch']['wall_seconds']<=180
    print(json.dumps(dict(status='passed',logical_cells=128,unique_outcomes=615000,bootstrap_draws=5000,primary=e,scope='Public weight/outcome reconstruction. Full board validity audited separately against retained raw files.')))


if __name__=='__main__':main()
