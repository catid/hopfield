"""Independent CSV/noise replay and NumPy graph reconstruction for H4r."""
import csv
import hashlib
import json
import time
import numpy as np
import torch
from noisy_pl import ROOT, HERE, OUT, PROTOCOL, sha


def audit_sources():
    start=time.monotonic();p=json.loads(PROTOCOL.read_text());m=json.loads((OUT/'inputs.json').read_text())
    assert m['protocol_sha256']==sha(PROTOCOL)
    assert sha(ROOT/p['source'])==p['source_sha256_value']
    excluded=np.zeros(3831994,bool)
    for spec in p['exclusions']:
        manifest=ROOT/spec['path'];assert sha(manifest)==spec['sha256']
        for e in json.loads(manifest.read_text())['inputs']:
            file=ROOT/e['completion_path'];assert sha(file)==e['completion_sha256']
            excluded[np.load(file)['source_row']]=True
    eligible=np.flatnonzero(~excluded);assert len(eligible)==m['eligible_rows']
    banks={};wanted=set();evidence={}
    for entry in m['banks']:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'];evidence[entry['path']]=sha(path)
        z=np.load(path);banks[entry['bank']]=z;wanted.update(map(int,z['source_row']))
        rows=np.random.default_rng(p['bank_rng_base']+entry['bank']).choice(eligible,600,replace=False)
        assert np.array_equal(rows,z['source_row']) and len(set(rows))==600
    originals={}
    with (ROOT/p['source']).open(newline='') as f:
        for index,row in enumerate(csv.DictReader(f)):
            if index in wanted:
                answer=np.array([int(c)-1 for c in row['answer']],dtype=np.int8)
                given=np.array([c in '123456789' for c in row['question']],dtype=bool)
                assert np.array_equal(np.array([int(c)-1 for c in row['question'] if c in '123456789']),answer[given])
                originals[index]=(answer,~given)
    assert len(originals)==len(wanted)
    ii,jj=np.triu_indices(81,1);r,c=np.arange(81)//9,np.arange(81)%9
    truth=(r[ii]==r[jj])|(c[ii]==c[jj])|((r[ii]//3==r[jj]//3)&(c[ii]//3==c[jj]//3))
    assert truth.sum()==810
    records=[];count_checks=0
    for bank,z in banks.items():
        solutions=np.stack([originals[int(row)][0] for row in z['source_row']]);free=np.stack([originals[int(row)][1] for row in z['source_row']])
        assert np.array_equal(solutions,z['solutions']) and np.array_equal(free,z['free'])
        rng=np.random.default_rng(p['noise_rng_base']+bank)
        cells=np.stack([rng.choice(np.flatnonzero(f),9,replace=False) for f in free])
        values=(np.take_along_axis(solutions,cells,1)+rng.integers(1,9,cells.shape))%9
        assert np.array_equal(cells,z['cells']) and np.array_equal(values,z['values'])
        noisy=solutions.copy();np.put_along_axis(noisy,cells,values,1)
        assert np.array_equal(noisy,z['noisy']) and ((noisy!=solutions).sum(1)==9).all()
        assert np.array_equal(noisy[~free],solutions[~free])
        dest=OUT/'fits'/f'b{bank:02d}';rec=json.loads((dest/'receipt.json').read_text());models=np.load(dest/'models.npz')
        assert rec['protocol_sha256']==sha(PROTOCOL) and rec['input_sha256']==sha(ROOT/m['banks'][bank]['path'])
        for name,digest in rec['output_sha256'].items():assert sha(dest/name)==digest;evidence[str((dest/name).relative_to(ROOT))]=digest
        counts=np.array([np.count_nonzero(noisy[:,a]==noisy[:,b]) for a,b in zip(ii,jj)])
        assert np.array_equal(counts,models['counts']);count_checks+=len(counts)
        distinct=sorted(set(map(int,counts)));candidates=[(distinct[k+1]-distinct[k],k) for k in range(len(distinct)-1) if distinct[k+1]<=np.median(counts)]
        index=max(candidates,key=lambda pair:pair[0])[1] if candidates else None
        cutoff=(distinct[index]+distinct[index+1])//2 if index is not None else 0
        inferred={'restricted':counts<=cutoff}
        assert cutoff==rec['arms']['restricted']['threshold']
        # Reuse the separately reviewed Chernoff helper; independent rowwise means
        # and all minimality conditions are checked below without fit_counts.
        from noise_chernoff import chernoff_upper
        means=np.zeros(3240)
        for f in free:
            a=9/int(f.sum());b=72/(int(f.sum())*(int(f.sum())-1))
            means+=(f[ii].astype(int)+f[jj].astype(int))*a/8-(f[ii]&f[jj])*9*b/64
        thresholds=models['chernoff9_thresholds'];alpha=.01/3240
        for threshold in np.unique(thresholds):
            selected=thresholds==threshold
            assert (chernoff_upper(int(threshold)+1,600,means[selected])<=alpha).all()
            if threshold>0:assert (chernoff_upper(int(threshold),600,means[selected])>alpha).all()
        inferred['chernoff9']=counts<=thresholds
        for arm in ['adamw','paper_pl']:
            fit=torch.load(dest/f'{arm}.pt',map_location='cpu',weights_only=True)
            w=fit['W'].numpy().astype(np.float64);b=fit['B'].numpy();assert np.isfinite(w).all() and np.isfinite(b).all()
            diag=np.stack([w[:,d,d] for d in range(9)],1).mean(1)
            off=np.stack([w[:,a,b] for a in range(9) for b in range(9) if a!=b],1).mean(1)
            coefficients=np.maximum(diag-off,0)
            assert np.allclose(coefficients,models[f'{arm}_coefficients'],rtol=0,atol=1e-14)
            positive=sorted(coefficients[coefficients>0]);gaps=np.diff(positive);index=int(gaps.argmax())
            threshold=(positive[index]+positive[index+1])/2
            inferred[arm]=coefficients>threshold
            assert abs(threshold-rec['arms'][arm]['threshold'])<1e-14
        for arm,support in inferred.items():
            assert np.array_equal(support,models[f'{arm}_support'])
            missing=int((truth&~support).sum());extra=int((support&~truth).sum())
            records.append(dict(bank=bank,arm=arm,retained=int(support.sum()),missing=missing,extra=extra,exact=missing==extra==0,
                                support_sha256=hashlib.sha256(support.tobytes()).hexdigest()))
    summary=[]
    for arm in p['arms']:
        rows=[r for r in records if r['arm']==arm]
        summary.append(dict(arm=arm,exact_banks=sum(r['exact'] for r in rows),all_true_retained=sum(r['missing']==0 for r in rows),
                            total_missing=sum(r['missing'] for r in rows),total_extra=sum(r['extra'] for r in rows)))
    return dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),models=records,grouped=summary,banks=len(banks),unique_csv_rows=len(wanted),
                equal_counts_checked=count_checks,input_sha256=evidence,wall_s=time.monotonic()-start)


if __name__=='__main__':
    result=audit_sources();(OUT/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['grouped']))
