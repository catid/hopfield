"""Independent source-weight and all-returned-board H4x audits."""
import argparse
import hashlib
import json
import sys
import numpy as np
from canonical_weights import ROOT,HERE,RIPPER,PARENT,OUT,PUBLIC,PROTOCOL,sha,save,verify


def sources():
    p=json.loads(PROTOCOL.read_text());verify(p)
    mapping=json.loads((OUT/'mapping.json').read_text());assert mapping['protocol_sha256']==sha(PROTOCOL)
    with np.load(PARENT/'models.npz',allow_pickle=False) as z:j=z['canonical_weak_coefficients'].copy()
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    rows=[];hashes={}
    for row in mapping['rows']:
        b,arm=row['bank'],row['arm'];values=sorted(float(x)for x in j[b] if x>0);n=len(values)
        median=values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2;cut=median/4
        expected=np.array([0 if x<=0 else 16 if x>=cut else 1 for x in j[b]],np.uint8)
        if arm=='uniform':expected=np.array([16 if x>0 else 0 for x in j[b]],np.uint8)
        if arm=='shuffled':
            positions=np.flatnonzero(j[b]>0);order=np.random.default_rng(p['shuffle_seed_base']+b).permutation(len(positions));expected[positions]=expected[positions][order]
        path=ROOT/row['model_path'];assert sha(path)==row['model_sha256'];hashes[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            w=z['weight16'];assert w.dtype==np.uint8 and np.array_equal(w,expected) and np.array_equal(z['support'],j[b]>0)
            assert np.array_equal(w>0,j[b]>0) and (w[truth]>0).all()
            assert hashlib.sha256(w.tobytes()).hexdigest()==row['weight16_sha256'] and cut==row['threshold']
            rows.append(dict(bank=b,arm=arm,weak_true=int((truth&(w==1)).sum()),weak_extra=int((~truth&(w==1)).sum()),strong_extra=int((~truth&(w==16)).sum()),retained=int((w>0).sum())))
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,model_sha256=hashes))
    print(json.dumps(dict(status='validated',models=len(rows),weak_true_magnitude=sum(r['weak_true']for r in rows if r['arm']=='magnitude'),weak_extra_magnitude=sum(r['weak_extra']for r in rows if r['arm']=='magnitude'))))


def outcomes():
    import torch
    sys.path.insert(0,str(RIPPER));from bootstrap_two_way import resample_means
    p=json.loads(PROTOCOL.read_text());verify(p);source=json.loads((OUT/'source-audit.json').read_text())
    assert source['protocol_sha256']==sha(PROTOCOL) and source['auditor_sha256']==sha(__file__)
    for path,digest in source['model_sha256'].items():assert sha(ROOT/path)==digest
    mapping=json.loads((OUT/'mapping.json').read_text());q=json.loads((OUT/'queue.json').read_text())
    assert len(q)==mapping['unique_new_searches'] and all(r['status']=='complete'for r in q)
    lo,hi=p['rows_half_open']
    with np.load(ROOT/'upstream/data/full-test/full-all.npz',allow_pickle=False)as z:
        target=z['solutions'][lo:hi].astype(np.int8)-1;free=~z['clue_mask'][lo:hi].astype(bool)
    ii,jj=np.triu_indices(81,1);outputs={};evidence={};diagnostics={}
    units=[[x for x in range(81)if x//9==k]for k in range(9)]+[[x for x in range(81)if x%9==k]for k in range(9)]+[[x for x in range(81)if (x//27)*3+x%9//3==k]for k in range(9)]
    parent=json.loads((PARENT/'summary.json').read_text())
    for kind,job in sorted({(r['kind'],r['job'])for r in mapping['rows']}):
        row=next(r for r in mapping['rows']if(r['kind'],r['job'])==(kind,job));w=np.load(ROOT/row['model_path'])['weight16']
        if kind=='archived':
            path=ROOT/'artifacts/research/20261114/foureyes/canonical_study'/job/'moves-262144/boards.npz'
            assert sha(path)==parent['input_sha256'][str(path.relative_to(ROOT))]
        else:
            d=OUT/'jobs'/job;r=json.loads((d/'receipt.json').read_text());path=d/'boards.npz'
            assert r['status']=='complete' and r['protocol_sha256']==sha(PROTOCOL)
            assert r['wrapper_sha256']==sha(RIPPER/'confidence_weight_search.py')
            assert r['preflight_state_comparisons']==150 and r['hidden_target_state_comparisons']==132 and r['integer_field_checks']==6
            assert r['search_kwargs']==dict(temp=.4,theta=.45,focused=True,C=4,tau=0.,tol=.001,ebar=0.,cap=4e11,check=4096,seed=0,anneal=[.4,.31,524288])
            assert r['move_cap']==262144 and r['rows_half_open']==[lo,hi]
            assert r['weight16_bytes_sha256']==hashlib.sha256(w.tobytes()).hexdigest()
            for name,digest in r['output_sha256'].items():assert sha(d/name)==digest
            searched=torch.load(d/'searched_model.pt',map_location='cpu',weights_only=True)
            assert torch.equal(searched['W'],torch.from_numpy(w.astype(np.float32)/16)[:,None,None]*torch.eye(9))
            assert not searched['B'].any() and np.array_equal(searched['weight16'].numpy(),w)
            evidence[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        evidence[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path,allow_pickle=False)as z:
            board=z['boards'];stopped=z['stopped'];assert board.shape==(10000,81) and board.dtype.kind in 'iu'
            assert ((board>=0)&(board<9)).all() and np.array_equal(z['solutions'],target) and np.array_equal(z['free'],free) and np.array_equal(z['puzzle'],np.arange(lo,hi))
            assert np.array_equal(board[~free],target[~free])
            valid=np.ones(len(board),bool)
            for cells in units:valid&=(np.sort(board[:,cells],axis=1)==np.arange(9)).all(1)
            exact=(board==target).all(1);assert np.array_equal(valid,z['valid']) and np.array_equal(exact,z['exact'])
            assert ((z['moves']>0)&(z['moves']<=262144)).all()
            e16=((board[:,ii]==board[:,jj])*w.astype(np.int64)).sum(1)
            assert not e16[stopped].any() and not(stopped&~valid).any()
            outputs[kind,job]=(exact.copy(),stopped.copy())
            diagnostics[kind,job]=dict(correct_certified=int((exact&stopped).sum()),correct_capped=int((exact&~stopped).sum()),wrong_stops=int((stopped&~valid).sum()),noncanonical_valid=int((valid&~exact).sum()))
    arrays={};rows=[]
    for row in mapping['rows']:
        exact,stopped=outputs[row['kind'],row['job']];arrays[row['bank'],row['arm']]=exact.astype(np.int8)
        rows.append(dict(**row,accuracy_pct=float(exact.mean()*100),**diagnostics[row['kind'],row['job']]))
    grouped=[]
    for arm in p['arms']:
        chosen=[r for r in rows if r['arm']==arm]
        grouped.append(dict(arm=arm,accuracy_pct=float(np.mean([r['accuracy_pct']for r in chosen])),wrong_stops=sum(r['wrong_stops']for r in chosen),correct_certified_pct=sum(r['correct_certified']for r in chosen)/6400,correct_capped_pct=sum(r['correct_capped']for r in chosen)/6400))
    endpoints=[]
    for ref,seed in [('uniform',20261116011),('shuffled',20261116012)]:
        delta=np.stack([arrays[b,'magnitude']-arrays[b,ref]for b in range(64)])
        interval=np.quantile(resample_means(delta,seed,5000),[.025,.975]).tolist()
        endpoints.append(dict(reference=ref,gain_pp=float(delta.mean()*100),paired_two_way95_interval_pp=interval,decision='supported'if interval[0]>=10 else'falsified'if interval[1]<10 else'inconclusive'))
    decision='falsified'if any(r['wrong_stops']for r in rows)or any(e['decision']=='falsified'for e in endpoints)else'supported'if all(e['decision']=='supported'for e in endpoints)else'inconclusive'
    PUBLIC.mkdir(parents=True,exist_ok=True)
    keys=sorted(outputs);np.savez_compressed(PUBLIC/'audited-outcome-bits.npz',kinds=np.array([k[0]for k in keys]),jobs=np.array([k[1]for k in keys]),exact=np.packbits(np.stack([outputs[k][0]for k in keys]),axis=1),stopped=np.packbits(np.stack([outputs[k][1]for k in keys]),axis=1))
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),scope=p['scope'],grouped=grouped,endpoints=endpoints,decision=decision,rows=rows,source_audit=source,
        input_sha256=evidence,boards_audited=len(outputs)*10000,new_searches=len(q),logical_cells=len(rows),unique_inputs=10000,outcome_bits_sha256=sha(PUBLIC/'audited-outcome-bits.npz'),launch=json.loads((OUT/'launch.json').read_text()))
    save(PUBLIC/'summary.json',result);print(json.dumps(dict(grouped=grouped,endpoints=endpoints,decision=decision)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['sources','outcomes']);a=ap.parse_args();globals()[a.action]()
