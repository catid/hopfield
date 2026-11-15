"""Independent CSV/noise, NumPy objective/support, and saved-state H4w audit."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import json
import sys
import time
import numpy as np
from canonical_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,ARMS,sha,save,verify_frozen


def sources():
    import torch
    start=time.monotonic();p=json.loads(PROTOCOL.read_text());verify_frozen(p)
    manifest=json.loads((OUT/'inputs.json').read_text());assert manifest['protocol_sha256']==sha(PROTOCOL)
    excluded=np.zeros(3831994,bool)
    for spec in p['exclusions']:
        path=ROOT/spec['path'];assert sha(path)==spec['sha256']
        for entry in json.loads(path.read_text())['inputs']:
            f=ROOT/entry['completion_path'];assert sha(f)==entry['completion_sha256'];excluded[np.load(f)['source_row']]=True
    eligible=np.flatnonzero(~excluded);banks={};wanted=set();evidence={}
    for entry in manifest['banks']:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'];evidence[str(path.relative_to(ROOT))]=sha(path)
        data=np.load(path,allow_pickle=False);banks[entry['bank']]=data;wanted.update(map(int,data['source_row']))
        assert np.array_equal(np.random.default_rng(p['bank_rng_base']+entry['bank']).choice(eligible,600,replace=False),data['source_row'])
    original={}
    with (ROOT/p['source']).open(newline='') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if i in wanted:original[i]=(np.array([int(c)-1 for c in row['answer']],np.int8),np.array([c not in '123456789' for c in row['question']],bool))
    assert len(original)==len(wanted)
    ii,jj=np.triu_indices(81,1);r,c=np.arange(81)//9,np.arange(81)%9
    truth=(r[ii]==r[jj])|(c[ii]==c[jj])|((r[ii]//3==r[jj]//3)&(c[ii]//3==c[jj]//3))
    rows=[];stationarity=[]
    for bank,data in banks.items():
        clean=np.stack([original[int(i)][0] for i in data['source_row']]);free=np.stack([original[int(i)][1] for i in data['source_row']])
        assert np.array_equal(clean,data['solutions']) and np.array_equal(free,data['free'])
        rng=np.random.default_rng(p['noise_rng_base']+bank);cells=np.stack([rng.choice(np.flatnonzero(f),9,replace=False) for f in free]);values=(np.take_along_axis(clean,cells,1)+rng.integers(1,9,cells.shape))%9
        assert np.array_equal(cells,data['cells']) and np.array_equal(values,data['values'])
        noisy=clean.copy();np.put_along_axis(noisy,cells,values,1);assert np.array_equal(noisy,data['noisy']) and np.array_equal(noisy[~free],clean[~free])
        dest=OUT/'fits'/f'b{bank:02d}';receipt=json.loads((dest/'receipt.json').read_text());assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['input_sha256']==sha(ROOT/manifest['banks'][bank]['path'])
        for name,digest in receipt['output_sha256'].items():assert sha(dest/name)==digest;evidence[str((dest/name).relative_to(ROOT))]=digest
        models=np.load(dest/'models.npz',allow_pickle=False);counts=np.array([np.count_nonzero(noisy[:,i]==noisy[:,j]) for i,j in zip(ii,jj)])
        assert np.array_equal(counts,models['counts']);distinct=sorted(set(map(int,counts)));choices=[(distinct[k+1]-distinct[k],k) for k in range(len(distinct)-1) if distinct[k+1]<=np.median(counts)]
        k=max(choices,key=lambda x:x[0])[1] if choices else None;cut=(distinct[k]+distinct[k+1])//2 if k is not None else 0
        inferred={'restricted':counts<=cut}
        checkpoint=torch.load(dest/'adamw.pt',map_location='cpu',weights_only=True);w=checkpoint['W'].numpy().astype(np.float64)
        diag=np.stack([w[:,d,d] for d in range(9)],1).mean(1);off=np.stack([w[:,a,b] for a in range(9) for b in range(9) if a!=b],1).mean(1);coeff=np.maximum(diag-off,0)
        assert np.max(np.abs(coeff-models['adamw_coefficients']))<1e-14
        positive=np.sort(coeff[coeff>0]);k=int(np.diff(positive).argmax());inferred['adamw']=coeff>(positive[k]+positive[k+1])/2
        y=np.eye(9)[noisy]
        for arm,raw_lambda in p['raw_penalties'].items():
            j=models[arm+'_coefficients'];assert np.isfinite(j).all();inferred[arm]=j>0
            adjacency=np.zeros((81,81));adjacency[ii,jj]=j;adjacency[jj,ii]=j
            fields=np.stack([y[:,:,v]@adjacency.T for v in range(9)],2)
            logits=-fields;shift=logits.max(2,keepdims=True);ex=np.exp(logits-shift);prob=ex/ex.sum(2,keepdims=True)
            logprob=logits-shift-np.log(ex.sum(2,keepdims=True))
            loss=float(-(np.take_along_axis(logprob,noisy[:,:,None],2)[:,:,0]*free).sum()/free.sum())
            residue=(y-prob)*free[:,:,None]/free.sum();directed=sum(residue[:,:,v].T@y[:,:,v] for v in range(9));g=directed[ii,jj]+directed[jj,ii]
            penalty=9*raw_lambda;kkt=np.where(j>0,g+penalty,np.where(j<0,g-penalty,np.maximum(np.abs(g)-penalty,0)))
            fit=json.loads((dest/(arm+'-fit.json')).read_text());assert abs(loss-fit['history'][-1]['loss'])<1e-12 and abs(float(np.abs(kkt).max())-fit['kkt_linf'])<1e-12
            objectives=[q['objective'] for q in fit['history']];assert max(np.diff(objectives))<=4e-12
            stationarity.append(dict(bank=bank,arm=arm,numpy_loss=loss,numpy_kkt_linf=float(np.abs(kkt).max()),j_l1=float(np.abs(j).sum()),positive=int((j>0).sum())))
        for arm,support in inferred.items():
            assert np.array_equal(support,models[arm+'_support'])
            import hashlib
            rows.append(dict(bank=bank,arm=arm,retained=int(support.sum()),missing=int((truth&~support).sum()),extra=int((support&~truth).sum()),support_sha256=hashlib.sha256(support.tobytes()).hexdigest()))
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),models=rows,stationarity=stationarity,unique_csv_rows=len(wanted),input_sha256=evidence,auditor_sha256=sha(__file__),seconds=time.monotonic()-start)
    save(OUT/'source-audit.json',result);print(json.dumps(dict(status='validated',unique_csv_rows=len(wanted),models=len(rows),max_kkt=max(r['numpy_kkt_linf'] for r in stationarity),seconds=result['seconds'])))


def outcomes():
    import audit_noisy_pl_long as auditor
    sys.path.insert(0,str(HERE.parent/'ripper'));from bootstrap_two_way import resample_means
    auditor.OUT=OUT;p=json.loads(PROTOCOL.read_text());verify_frozen(p)
    source=json.loads((OUT/'source-audit.json').read_text());assert source['protocol_sha256']==sha(PROTOCOL) and source['auditor_sha256']==sha(__file__)
    for path,digest in source['input_sha256'].items():assert sha(ROOT/path)==digest
    prepared=json.loads((OUT/'prepared.json').read_text());assert prepared['protocol_sha256']==sha(PROTOCOL) and not prepared['ineligible']
    q=json.loads((OUT/'queue.json').read_text());assert all(r['status']=='complete' for r in q)
    with np.load(ROOT/'upstream/data/full-test/full-all.npz',allow_pickle=False) as z:
        lo,hi=p['rows_half_open'];np.savez_compressed(OUT/'audit-target.npz',solutions=z['solutions'][lo:hi].astype(np.int8)-1,free=~z['clue_mask'][lo:hi].astype(bool),puzzle=np.arange(lo,hi))
    with ProcessPoolExecutor(max_workers=6) as pool:result=list(pool.map(auditor.audit_job,q))
    lookup={r['job_id']:r for r in result};models={(r['bank'],r['arm']):r['support_sha256'] for r in source['models']};rows=[];arrays={}
    for r in prepared['mappings']:
        got=lookup[r['actual']];assert got['support_sha256']==models[r['bank'],r['arm']]==r['support_sha256']
        arrays[r['bank'],r['arm']]=got['good'].astype(np.int8)
        rows.append(dict(**r,accuracy_pct=float(got['good'].mean()*100),wrong_stops=int((got['stopped']&~got['good']).sum())))
    grouped=[]
    for arm in ARMS:
        chosen=[r for r in rows if r['arm']==arm];structure=[r for r in source['models'] if r['arm']==arm]
        grouped.append(dict(arm=arm,accuracy_pct=float(np.mean([r['accuracy_pct'] for r in chosen])),wrong_stops=sum(r['wrong_stops'] for r in chosen),exact_graphs=sum(r['missing']==r['extra']==0 for r in structure),all_true_retained=sum(r['missing']==0 for r in structure)))
    delta=np.stack([arrays[b,'canonical_half']-arrays[b,'restricted'] for b in range(64)]);interval=np.quantile(resample_means(delta,20261115999,draws=5000),[.025,.975]).tolist();wrong=next(r['wrong_stops'] for r in grouped if r['arm']=='canonical_half')
    primary=dict(gain_pp=float(delta.mean()*100),paired_two_way95_interval_pp=interval,wrong_stops=wrong,decision='falsified' if wrong or interval[1]<-2 else 'supported' if interval[0]>=-2 else 'inconclusive')
    PUBLIC.mkdir(parents=True,exist_ok=True);ids=sorted(lookup)
    np.savez_compressed(PUBLIC/'audited-outcome-bits.npz',job_ids=np.array(ids),valid=np.packbits(np.stack([lookup[k]['good'] for k in ids]),axis=1),stopped=np.packbits(np.stack([lookup[k]['stopped'] for k in ids]),axis=1))
    evidence=dict(source['input_sha256'])
    for r in result:evidence.update(r['input_sha256'])
    summary=dict(status='validated',protocol_sha256=sha(PROTOCOL),primary=primary,grouped=grouped,source_audit=source,model_mappings=prepared['mappings'],outcomes=rows,unique_inputs=10000,source_banks=64,unique_searches=len(q),logical_cells=len(rows),boards_audited=len(q)*10000,input_sha256=evidence,outcome_bits_sha256=sha(PUBLIC/'audited-outcome-bits.npz'),fit_launch=json.loads((OUT/'fit-launch.json').read_text()),search_launch=json.loads((OUT/'search-launch.json').read_text()))
    save(PUBLIC/'summary.json',summary);print(json.dumps(dict(primary=primary,grouped=grouped,boards_audited=summary['boards_audited'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['sources','outcomes']);args=parser.parse_args()
    if args.action=='sources':sources()
    else:outcomes()
