"""Fresh H4y source, weight-assignment and saved-board audits."""
import argparse
import hashlib
import json
import sys
import numpy as np
from canonical_confirm import ROOT,HERE,RIPPER,OUT,PUBLIC,PROTOCOL,ARMS,sha,save,verify


def sources():
    import audit_canonical_study as source
    source.OUT=OUT;source.PROTOCOL=PROTOCOL
    source.sources()


def weights():
    p=json.loads(PROTOCOL.read_text());verify(p);a=json.loads((OUT/'source-audit.json').read_text())
    assert a['protocol_sha256']==sha(PROTOCOL) and a['auditor_sha256']==sha(HERE/'audit_canonical_study.py')
    for path,digest in a['input_sha256'].items():assert sha(ROOT/path)==digest
    mapping=json.loads((OUT/'mapping.json').read_text());assert mapping['protocol_sha256']==sha(PROTOCOL)
    models={};rows=[];arrays={}
    for b in range(64):
        with np.load(OUT/'fits'/f'b{b:02d}'/'models.npz',allow_pickle=False)as z:
            models[b]={k:z[k]for k in z.files}
            for k in z.files:arrays.setdefault(k,[]).append(z[k])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    for r in mapping['rows']:
        b,arm=r['bank'],r['arm'];z=models[b];j=z['canonical_weak_coefficients'];positive=j>0
        if arm=='magnitude':
            ordered=sorted(j[positive]);n=len(ordered);median=ordered[n//2]if n%2 else(ordered[n//2-1]+ordered[n//2])/2
            expected=np.array([0 if x<=0 else 16 if x>=median/4 else 1 for x in j],np.uint8)
            assert median/4==mapping['thresholds'][b]
        else:expected=(16*(positive if arm=='uniform'else z[arm+'_support'])).astype(np.uint8)
        path=ROOT/r['model_path'];assert sha(path)==r['model_sha256']
        with np.load(path,allow_pickle=False)as data:
            w=data['weight16'];assert np.array_equal(w,expected) and np.array_equal(data['support'],w>0)
            assert hashlib.sha256(w.tobytes()).hexdigest()==r['weight16_sha256']
            rows.append(dict(bank=b,arm=arm,missing=int((truth&(w==0)).sum()),extra=int((~truth&(w>0)).sum()),weak_true=int((truth&(w==1)).sum()),weak_extra=int((~truth&(w==1)).sum())))
    PUBLIC.mkdir(parents=True,exist_ok=True);np.savez_compressed(PUBLIC/'models.npz',**{k:np.stack(v)for k,v in arrays.items()})
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),rows=rows,models_sha256=sha(PUBLIC/'models.npz'),source_audit_sha256=sha(OUT/'source-audit.json'),mapping_sha256=sha(OUT/'mapping.json'),auditor_sha256=sha(__file__))
    save(OUT/'weight-audit.json',result);print(json.dumps(dict(status='validated',models=len(rows))))


def outcomes():
    import torch
    sys.path.insert(0,str(RIPPER));from bootstrap_two_way import resample_means
    p=json.loads(PROTOCOL.read_text());verify(p);w_audit=json.loads((OUT/'weight-audit.json').read_text());assert w_audit['mapping_sha256']==sha(OUT/'mapping.json') and w_audit['auditor_sha256']==sha(__file__)
    mapping=json.loads((OUT/'mapping.json').read_text());queue=json.loads((OUT/'queue.json').read_text());assert all(r['status']=='complete'for r in queue)
    lo,hi=p['rows_half_open'];n=hi-lo
    with np.load(ROOT/'upstream/data/full-test/full-all.npz',allow_pickle=False)as z:target=z['solutions'][lo:hi].astype(np.int8)-1;free=~z['clue_mask'][lo:hi].astype(bool)
    units=[[x for x in range(81)if x//9==k]for k in range(9)]+[[x for x in range(81)if x%9==k]for k in range(9)]+[[x for x in range(81)if (x//27)*3+x%9//3==k]for k in range(9)]
    ii,jj=np.triu_indices(81,1);outputs={};evidence={}
    for job in queue:
        d=OUT/'jobs'/job['job_id'];r=json.loads((d/'receipt.json').read_text());model=ROOT/job['model']['path'];assert sha(model)==job['model']['sha256'];w=np.load(model,allow_pickle=False)['weight16']
        assert r['status']=='complete' and r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(RIPPER/'confidence_weight_search.py')
        assert r['preflight_state_comparisons']==150 and r['hidden_target_state_comparisons']==132 and r['integer_field_checks']==6
        assert r['search_kwargs']==dict(temp=.4,theta=.45,focused=True,C=4,tau=0.,tol=.001,ebar=0.,cap=4e11,check=4096,seed=0,anneal=[.4,.31,524288])
        assert r['rows_half_open']==[lo,hi] and r['move_cap']==262144
        for name,digest in r['output_sha256'].items():assert sha(d/name)==digest
        fitted=torch.load(d/'searched_model.pt',map_location='cpu',weights_only=True)
        assert torch.equal(fitted['W'],torch.from_numpy(w.astype(np.float32)/16)[:,None,None]*torch.eye(9)) and not fitted['B'].any()
        assert np.array_equal(fitted['weight16'].numpy(),w)
        with np.load(d/'boards.npz',allow_pickle=False)as z:
            board=z['boards'];stopped=z['stopped'];assert board.dtype.kind in'iu' and board.shape==(n,81) and ((board>=0)&(board<9)).all()
            assert np.array_equal(z['solutions'],target) and np.array_equal(z['free'],free) and np.array_equal(z['puzzle'],np.arange(lo,hi))
            assert np.array_equal(board[~free],target[~free])
            valid=np.ones(n,bool)
            for cells in units:valid&=(np.sort(board[:,cells],axis=1)==np.arange(9)).all(1)
            exact=(board==target).all(1);assert np.array_equal(valid,z['valid']) and np.array_equal(exact,z['exact'])
            energy=((board[:,ii]==board[:,jj])*w.astype(np.int64)).sum(1);assert not energy[stopped].any()
            assert ((z['moves']>0)&(z['moves']<=262144)).all()
            assert int(exact.sum())==r['solved'] and int((stopped&~valid).sum())==r['wrong_stops']
            outputs[job['job_id']]=(exact.copy(),valid.copy(),stopped.copy(),hashlib.sha256(w.tobytes()).hexdigest())
        for path in [d/'receipt.json',d/'boards.npz',model]:evidence[str(path.relative_to(ROOT))]=sha(path)
    rows=[];arrays={}
    for row in mapping['rows']:
        exact,valid,stopped,digest=outputs[row['job']];assert row['weight16_sha256']==digest
        arrays[row['bank'],row['arm']]=exact.astype(np.int8)
        rows.append(dict(**row,accuracy_pct=float(exact.mean()*100),wrong_stops=int((stopped&~valid).sum()),correct_certified=int((exact&stopped).sum()),correct_capped=int((exact&~stopped).sum()),noncanonical_valid=int((valid&~exact).sum())))
    grouped=[]
    for arm in ARMS:
        chosen=[r for r in rows if r['arm']==arm];grouped.append(dict(arm=arm,accuracy_pct=float(np.mean([r['accuracy_pct']for r in chosen])),wrong_stops=sum(r['wrong_stops']for r in chosen),correct_certified_pct=sum(r['correct_certified']for r in chosen)/(64*n)*100,correct_capped_pct=sum(r['correct_capped']for r in chosen)/(64*n)*100))
    mag=np.stack([arrays[b,'magnitude']for b in range(64)]);delta=mag-np.stack([arrays[b,'uniform']for b in range(64)])
    gain_ci=np.quantile(resample_means(delta,20261116021,5000),[.025,.975]).tolist();acc_ci=np.quantile(resample_means(mag,20261116022,5000),[.025,.975]).tolist()
    wrong=next(r['wrong_stops']for r in grouped if r['arm']=='magnitude')
    decision='falsified'if wrong or gain_ci[1]<50 or acc_ci[1]<60 else'supported'if gain_ci[0]>=50 and acc_ci[0]>=60 else'inconclusive'
    primary=dict(gain_pp=float(delta.mean()*100),gain95_interval_pp=gain_ci,accuracy_pct=float(mag.mean()*100),accuracy95_interval_pct=acc_ci,wrong_stops=wrong,decision=decision)
    ids=sorted(outputs);np.savez_compressed(PUBLIC/'audited-outcome-bits.npz',jobs=np.array(ids),exact=np.packbits(np.stack([outputs[j][0]for j in ids]),axis=1),valid=np.packbits(np.stack([outputs[j][1]for j in ids]),axis=1),stopped=np.packbits(np.stack([outputs[j][2]for j in ids]),axis=1))
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),primary=primary,grouped=grouped,rows=rows,source_audit=json.loads((OUT/'source-audit.json').read_text()),weight_audit=w_audit,input_sha256=evidence,
        unique_searches=len(queue),boards_audited=len(queue)*n,unique_inputs=n,source_banks=64,outcome_bits_sha256=sha(PUBLIC/'audited-outcome-bits.npz'),fit_launch=json.loads((OUT/'fit-launch.json').read_text()),search_launch=json.loads((OUT/'search-launch.json').read_text()))
    save(PUBLIC/'summary.json',result);print(json.dumps(dict(primary=primary,grouped=grouped,boards_audited=len(queue)*n)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['sources','weights','outcomes']);a=ap.parse_args();globals()[a.action]()
