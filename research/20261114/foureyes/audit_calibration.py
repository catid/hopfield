"""Audit H4b saved predictions; undefined conditional rates remain null."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/research/20261114/foureyes/calibration'
P=ROOT/'research/20261114/foureyes/calibration.json'
protocol=json.loads(P.read_text());evidence={};records=[];arrays={}
def sha(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    evidence[str(p.relative_to(ROOT))]=h;return h
sha(P);sha(Path(__file__))
for arm in protocol['arms']:
    values=[];stops=[]
    for k in protocol['training_k']:
        kv=[];ks=[]
        for seed in protocol['training_seeds']:
            d=OUT/'jobs'/f'k{k}-s{seed}-{arm}';sha(d/'receipt.json');r=json.loads((d/'receipt.json').read_text())
            assert r['status']=='complete' and r['protocol_sha256']==sha(P)
            for name,h in r['output_sha256'].items():assert sha(d/name)==h
            for name,h in r['source_sha256'].items():assert sha(ROOT/name)==h
            z=np.load(d/'boards.npz');n=len(z['boards']);b=z['boards'].reshape(n,9,9);valid=np.ones(n,bool)
            for i in range(9):
                valid&=(np.sort(b[:,i,:],axis=1)==np.arange(9)).all(1)
                valid&=(np.sort(b[:,:,i],axis=1)==np.arange(9)).all(1)
            for i in range(3):
                for j in range(3):valid&=(np.sort(b[:,3*i:3*i+3,3*j:3*j+3].reshape(n,9),axis=1)==np.arange(9)).all(1)
            valid&=((z['boards']==z['solutions'])|z['free']).all(1)
            assert np.array_equal(valid,z['valid']) and np.array_equal(valid,z['exact'])
            assert valid.sum()==r['solved'] and ((~valid)&z['stopped']).sum()==r['wrong_stops']
            assert n==2048 and (z['moves']<=262144).all() and (z['moves'][~z['stopped']]==262144).all()
            kv.append(valid);ks.append(z['stopped'])
            records.append({**{key:r[key] for key in ['arm','k','seed','solved','stopped','wrong_stops','mean_moves','wall_s']},'n':n,
                            **{key:r['training_statistics'][key] for key in ['tau_train','tau_min','tau_p05']}})
        values.append(kv);stops.append(ks)
    arrays[arm]=np.asarray(values),np.asarray(stops)
rng=np.random.default_rng(20261114);bs=rng.integers(6,size=(5000,6));bp=rng.integers(2048,size=(5000,2048));stats=[]
for arm in protocol['arms']:
    ok,st=arrays[arm];delta=ok.astype(float)-arrays['energy_AND_valid'][0];gain=[];wrong=[]
    for s,p in zip(bs,bp):
        o=ok[:,s][:,:,p];t=st[:,s][:,:,p];gain.append(delta[:,s][:,:,p].mean()*100)
        if t.any():wrong.append(((~o)&t).sum()/t.sum()*100)
    lo,hi=np.quantile(gain,[.05,.95]);wl,wh=np.quantile(wrong,[.05,.95]) if wrong else (None,None)
    decision='inconclusive'
    if hi < -5 or (wl is not None and wl>1):decision='falsified'
    elif lo>=-5 and wh is not None and wh<=1:decision='supported'
    stats.append({'arm':arm,'accuracy_pct':float(ok.mean()*100),'accuracy_difference_vs_validity_pp':float(delta.mean()*100),
                  'one_sided95_bounds_pp':[float(lo),float(hi)],'stopped':int(st.sum()),
                  'wrong_halts_pct':float(((~ok)&st).sum()/st.sum()*100) if st.any() else None,
                  'one_sided95_wrong_halt_bounds_pct':[float(wl),float(wh)] if wrong else None,
                  'decision':decision,'conditional_rate_note':'Undefined when no puzzle halted; never interpreted as zero error.'})
audit={'status':'validated','utc':datetime.now(timezone.utc).isoformat(),'study':'H4b','rows':records,'comparisons':stats,
       'primary_decision':next(r['decision'] for r in stats if r['arm']=='energy_min'),'saved_boards':98304,'input_sha256':evidence,
       'limitations':['Disjoint development cohort; reused fitted models.','Six training-seed clusters; bootstrap approximate.',
                     'Conditional wrong-halt rate is undefined for the zero-halt minimum-threshold arm.',
                     'No proof that all training-only thresholds fail; only the frozen minimum and5percent choices were tested.']}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':'validated','comparisons':stats}))
