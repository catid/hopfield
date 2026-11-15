"""Independent integer, full-board and archived-identity audit for H4aa."""
import argparse
import hashlib
import json
import os
import sys
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import numpy as np
from strong_support_study import ROOT,HERE,RIPPER,OUT,PUBLIC,PROTOCOL,PARENT,MODELS,sha,save,verify


def sources():
    p=json.loads(PROTOCOL.read_text());verify(p)
    m=json.loads((OUT/'mapping.json').read_text());assert m['protocol_sha256']==sha(PROTOCOL)
    parent=json.loads((PARENT/'summary.json').read_text());assert parent['status']=='validated'
    with np.load(MODELS,allow_pickle=False)as z:coeff=z['canonical_weak_coefficients']
    ii,jj=np.triu_indices(81,1);true=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    rows=[];seen=set();identities={};reverse={}
    for r in m['rows']:
        b,arm=r['bank'],r['arm'];assert (b,arm)not in seen;seen.add((b,arm))
        j=coeff[b];vals=sorted(float(x)for x in j if x>0);n=len(vals);cut=(vals[n//2]if n%2 else(vals[n//2-1]+vals[n//2])/2)/4
        weak={'weak256':1,'strongonly':0}[arm]
        expected=np.array([0 if x<=0 else 256 if x>=cut else weak for x in j],np.uint16)
        path=ROOT/r['model_path'];assert sha(path)==r['model_sha256']
        with np.load(path,allow_pickle=False)as z:
            assert set(z.files)=={'support','weight256'} and z['weight256'].shape==(3240,) and z['weight256'].dtype==np.uint16 and z['support'].dtype==bool
            assert np.array_equal(z['weight256'],expected)and np.array_equal(z['support'],expected>0)
        assert hashlib.sha256(expected.tobytes()).hexdigest()==r['weight256_sha256']
        key=r['kind'],r['job'];ident=expected.tobytes()
        if key in identities:assert identities[key]==ident
        if ident in reverse:assert reverse[ident]==key
        identities[key]=ident;reverse[ident]=key
        if arm=='weak256':
            assert r['kind']=='archived';old=next(x for x in parent['rows']if x['bank']==b and x['arm']=='weak256')
            assert r['job']==old['job'] and r['weight256_sha256']==old['weight256_sha256']
        else:assert r['kind']=='new'
        rows.append(dict(bank=b,arm=arm,retained=int((expected>0).sum()),missing_true=int((expected[true]==0).sum()),extra=int((expected[~true]>0).sum())))
    assert seen=={(b,a)for b in range(64)for a in p['arms']}
    q=json.loads((OUT/'queue.json').read_text());new={k[1]for k in identities if k[0]=='new'}
    assert len(q)==len(new) and {j['job_id']for j in q}==new
    for job in q:
        r=next(x for x in m['rows']if x['kind']=='new'and x['job']==job['job_id'])
        assert job['seed']==0 and job['n']==600 and job['arm']=='strongonly'
        assert job['model']==dict(path=r['model_path'],sha256=r['model_sha256'])
    save(OUT/'source-audit.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),mapping_sha256=sha(OUT/'mapping.json'),rows=rows,unique_new_searches=len(q)))
    print(json.dumps(dict(status='validated',logical_cells=len(rows),unique_new_searches=len(q))))


def outcomes():
    import torch
    sys.path.insert(0,str(RIPPER));from bootstrap_two_way import resample_means
    p=json.loads(PROTOCOL.read_text());verify(p);source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated'and source['auditor_sha256']==sha(HERE/'audit_strong_support_study.py')and source['protocol_sha256']==sha(PROTOCOL)and source['mapping_sha256']==sha(OUT/'mapping.json')
    mapping=json.loads((OUT/'mapping.json').read_text())['rows'];q=json.loads((OUT/'queue.json').read_text());assert all(x['status']=='complete'for x in q)
    assert len(q)==source['unique_new_searches'] and {x['job_id']for x in q}=={r['job']for r in mapping if r['kind']=='new'}
    parent=json.loads((PARENT/'summary.json').read_text())
    with np.load(PARENT/'audited-outcome-bits.npz',allow_pickle=False)as z:
        old={str(j):tuple(np.unpackbits(z[n][i],count=5000).astype(bool)for n in ['exact','valid','stopped'])for i,(k,j)in enumerate(zip(z['kinds'],z['jobs']))if str(k)=='new'}
    with np.load(ROOT/'upstream/data/full-test/full-all.npz',allow_pickle=False)as z:
        target=z['solutions'][410000:415000].astype(np.int8)-1;free=~z['clue_mask'][410000:415000].astype(bool)
    units=[np.arange(81).reshape(9,9)[r]for r in range(9)]+[np.arange(81).reshape(9,9)[:,c]for c in range(9)]
    units += [np.array([(br+r)*9+bc+c for r in range(3)for c in range(3)])for br in [0,3,6]for bc in [0,3,6]]
    assert all((np.sort(target[:,u],axis=1)==np.arange(9)).all()for u in units)
    ii,jj=np.triu_indices(81,1);outputs={};diagnostics={};hashes={}
    for key in sorted({(r['kind'],r['job'])for r in mapping}):
        kind,job=key;rows=[r for r in mapping if(r['kind'],r['job'])==key];row=rows[0]
        for rr in rows:assert sha(ROOT/rr['model_path'])==rr['model_sha256'] and rr['weight256_sha256']==row['weight256_sha256']
        with np.load(ROOT/row['model_path'],allow_pickle=False)as z:w=z['weight256'].copy()
        d=(OUT if kind=='new'else ROOT/'artifacts/research/20261114/foureyes/weak_weight_study')/'jobs'/job
        r=json.loads((d/'receipt.json').read_text());assert r['status']=='complete'and r['job_id']==job and r['seed']==0
        assert r['protocol_sha256']==(sha(PROTOCOL)if kind=='new'else parent['protocol_sha256'])
        assert r['wrapper_sha256']==sha(HERE/'weak_weight_search.py')and r['backend_sha256']==sha(RIPPER/'sparse_backend.py')and r['capture_sha256']==sha(RIPPER/'run.py')
        assert r['integer_audit_sha256']==sha(HERE/'weak_weight_math.py')and r['integer_field_checks']==6 and r['integer_chain_states']>=64
        assert r['preflight_state_comparisons']==150 and r['hidden_target_state_comparisons']==132
        assert r['weight256_bytes_sha256']==row['weight256_sha256']and r['support_bytes_sha256']==hashlib.sha256((w>0).tobytes()).hexdigest()
        expected_path=row if kind=='new'else next(x for x in parent['rows']if x['kind']=='new'and x['job']==job)
        assert r['support_file_sha256']==expected_path['model_sha256']
        assert r['rows_half_open']==[410000,415000]and r['test_rows']==5000 and r['chains']==4 and r['move_cap']==262144
        assert r['used_unit']==1 and r['tau']==r['ebar']==0
        assert r['search_kwargs']==dict(temp=.4,theta=.45,focused=True,C=4,tau=0.,tol=.001,ebar=0.,cap=4e11,check=4096,seed=0,anneal=[.4,.31,524288])
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        assert set(r['output_sha256'])=={'boards.npz','result.json','searched_model.pt'}
        for n,h in r['output_sha256'].items():assert sha(d/n)==h;hashes[str((d/n).relative_to(ROOT))]=h
        if kind=='archived':
            for f in [d/'receipt.json',d/'boards.npz']:assert sha(f)==parent['input_sha256'][str(f.relative_to(ROOT))]
        model=torch.load(d/'searched_model.pt',map_location='cpu',weights_only=True)
        assert set(model)=={'W','B','weight256'} and model['W'].dtype==model['B'].dtype==torch.float32 and model['B'].shape==(81,9) and model['weight256'].numpy().dtype==np.uint16
        assert torch.equal(model['W'],torch.from_numpy(w.astype(np.float32)/256)[:,None,None]*torch.eye(9))and not model['B'].any()
        assert np.array_equal(model['weight256'].numpy(),w)
        with np.load(d/'boards.npz',allow_pickle=False)as z:
            b=z['boards'];t=z['stopped'].copy();moves=z['moves'];assert b.shape==(5000,81)and((b>=0)&(b<9)).all()
            assert np.array_equal(z['solutions'],target)and np.array_equal(z['free'],free)and np.array_equal(z['puzzle'],np.arange(410000,415000))
            assert np.array_equal(b[~free],target[~free]);e=(b==target).all(1);v=np.ones(5000,bool)
            for u in units:v&=(np.sort(b[:,u],axis=1)==np.arange(9)).all(1)
            assert np.array_equal(e,z['exact'])and np.array_equal(v,z['valid'])
            assert t.dtype==bool and moves.dtype.kind in 'iu' and ((moves>0)&(moves<=262144)).all()and(moves[~t]==262144).all()
            energy=((b[:,ii]==b[:,jj])*w.astype(np.int64)).sum(1);assert not energy[t].any()
            target_energy=((target[:,ii]==target[:,jj])*w.astype(np.int64)).sum(1)
            given=(((target[:,ii]==target[:,jj])&~free[:,ii]&~free[:,jj])*w.astype(np.int64)).sum(1);assert(energy>=given).all()
            diag=dict(correct_certified=int((e&t).sum()),correct_capped=int((e&~t).sum()),valid_certified=int((v&t).sum()),valid_capped=int((v&~t).sum()),wrong_stops=int((t&~v).sum()),stopped=int(t.sum()),noncanonical_valid=int((v&~e).sum()),noncanonical_valid_certificates=int((v&~e&t).sum()),canonical_compatible=int((target_energy==0).sum()),given_given_infeasible=int((given>0).sum()))
            assert r['solved']==int(e.sum())and r['valid']==int(v.sum())and r['wrong_stops']==diag['wrong_stops']and r['mean_moves']==float(moves.mean())and r['budget_failures']==int((~t).sum())
            assert r['noncanonical_valid_returns']==diag['noncanonical_valid']and r['noncanonical_valid_certificates']==diag['noncanonical_valid_certificates']
            if kind=='archived':assert all(np.array_equal(a,b)for a,b in zip((e,v,t),old[job]))
            outputs[key]=(e.copy(),v.copy(),t.copy());diagnostics[key]=diag
    rows=[];arrays={}
    for r in mapping:
        key=r['kind'],r['job'];arrays[r['bank'],r['arm']]=outputs[key][0].astype(np.int8)
        rows.append(dict(**r,accuracy_pct=float(outputs[key][0].mean()*100),**diagnostics[key]))
    grouped=[dict(arm=a,accuracy_pct=float(np.mean([r['accuracy_pct']for r in rows if r['arm']==a])),**{k:sum(r[k]for r in rows if r['arm']==a)for k in next(iter(diagnostics.values()))})for a in p['arms']]
    delta=np.stack([arrays[b,'strongonly']-arrays[b,'weak256']for b in range(64)]);ci=np.quantile(resample_means(delta,20261116041,5000),[.025,.975]).tolist();wrong=grouped[1]['wrong_stops']
    endpoint=dict(gain_pp=float(delta.mean()*100),gain95_interval_pp=ci,wrong_stops=wrong,decision='falsified'if wrong or ci[1]<5 else'supported'if ci[0]>=5 else'inconclusive')
    PUBLIC.mkdir(parents=True,exist_ok=True);keys=sorted(outputs);packed=dict(kinds=np.array([k[0]for k in keys]),jobs=np.array([k[1]for k in keys]))
    for i,n in enumerate(['exact','valid','stopped']):
        matrix=np.stack([outputs[k][i]for k in keys]);packed[n]=np.packbits(matrix,axis=1)
        assert np.array_equal(np.unpackbits(packed[n],axis=1,count=5000).astype(bool),matrix)
    np.savez_compressed(PUBLIC/'audited-outcome-bits.npz',**packed)
    save(PUBLIC/'summary.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),scope=p['scope'],primary=endpoint,grouped=grouped,rows=rows,source_audit=source,boards_audited=len(keys)*5000,input_sha256=hashes,outcome_bits_sha256=sha(PUBLIC/'audited-outcome-bits.npz'),search_launch=json.loads((OUT/'search-launch.json').read_text())))
    print(json.dumps(dict(primary=endpoint,grouped=grouped)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['sources','outcomes']);globals()[ap.parse_args().action]()
