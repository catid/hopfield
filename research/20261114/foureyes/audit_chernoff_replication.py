"""Independent fresh-source replay and direct-equality model audit."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import csv
import hashlib
import json
import time
import numpy as np
from audit_noise_replication import exact_bounds
from noise_conditional_bands import tail_bounds
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/chernoff_replication'
PUBLIC=ROOT/'results/research/foureyes/chernoff_replication'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def audit_sources(p,manifest):
    start=time.monotonic();source=ROOT/p['source'];assert sha(source)==p['source_sha256']==manifest['source_sha256']
    exclusion=np.zeros(p['pool_size'],bool);evidence={}
    for spec in p['exclusions']:
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];evidence[spec['path']]=sha(path)
        for item in json.loads(path.read_text())['inputs']:
            file=ROOT/item['completion_path'];assert sha(file)==item['completion_sha256'];exclusion[np.load(file)['source_row']]=True
    eligible=np.flatnonzero(~exclusion);assert len(eligible)==manifest['eligible_rows']
    entries=[];rows=[]
    for entry in manifest['shards']:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'];evidence[entry['path']]=sha(path);z=np.load(path)
        for i,bank in enumerate(z['banks']):
            expected=np.random.default_rng(p['bank_rng_base']+int(bank)).choice(eligible,p['rows_per_bank'],replace=False)
            assert np.array_equal(z['source_row'][i],expected)
            assert len(set(expected))==1000 and not exclusion[expected].any()
        entries.append(z);rows.append(z['source_row'].ravel())
    selected=np.unique(np.concatenate(rows));assert len(selected)==manifest['unique_selected_rows']
    wanted=np.zeros(p['pool_size'],bool);wanted[selected]=True
    answers=np.zeros((p['pool_size'],81),dtype=np.uint8);givens=np.zeros_like(answers,dtype=bool)
    # Independent parsing uses explicit per-character integers, not raw byte subtraction.
    found=0
    with source.open(newline='') as f:
        for row_index,row in enumerate(csv.DictReader(f)):
            if not wanted[row_index]:continue
            answer=np.array([int(char)-1 for char in row['answer']],dtype=np.uint8)
            given=np.array([char in '123456789' for char in row['question']])
            assert all(int(row['question'][c])-1==int(answer[c]) for c in np.flatnonzero(given))
            answers[row_index]=answer;givens[row_index]=given;found+=1
    assert found==len(selected)
    ii,jj=np.triu_indices(81,1);r,c=np.arange(81)//9,np.arange(81)%9
    truth=(r[ii]==r[jj])|(c[ii]==c[jj])|((r[ii]//3==r[jj]//3)&(c[ii]//3==c[jj]//3))
    for z in entries:
        assert np.array_equal(z['solutions'],answers[z['source_row']]) and np.array_equal(z['given'],givens[z['source_row']])
        for clean in z['solutions']:
            assert not (clean[:,ii[truth]]==clean[:,jj[truth]]).any()
    return dict(status='passed',banks=512,rows_per_bank=1000,selected_occurrences=512*1000,
        unique_selected_rows=found,independent_sampling_replayed=True,all_selected_rows_reread_from_csv=True,
        all_selected_boards_valid=True,input_sha256=evidence,wall_s=time.monotonic()-start)

def audit_shard(entry):
    start=time.monotonic();p=json.loads((HERE/'chernoff_replication.json').read_text())
    source=ROOT/entry['path'];assert sha(source)==entry['sha256'];z=np.load(source)
    dest=OUT/f'shard-{entry["shard"]:02d}';receipt=json.loads((dest/'receipt.json').read_text())
    assert receipt['status']=='complete' and receipt['banks']==entry['banks']
    assert receipt['helper_sha256']==p['helper_sha256']==sha(HERE/'noise_chernoff.py')
    assert receipt['producer_sha256']==sha(HERE/'chernoff_replication.py') and receipt['protocol_sha256']==sha(HERE/'chernoff_replication.json')
    evidence={str((dest/'receipt.json').relative_to(ROOT)):sha(dest/'receipt.json')}
    for path,expected in receipt['input_sha256'].items():assert sha(ROOT/path)==expected;evidence[path]=expected
    for name,expected in receipt['output_sha256'].items():assert sha(dest/name)==expected;evidence[str((dest/name).relative_to(ROOT))]=expected
    changes=np.load(dest/'corruptions.npz');stored=np.load(dest/'counts.npz')
    assert changes['cells'].shape==changes['values'].shape==(len(z['banks']),1000,9)
    assert (changes['cells']<81).all() and (changes['values']<9).all()
    assert (np.diff(np.sort(changes['cells'],axis=2),axis=2)>0).all()
    ii,jj=np.triu_indices(81,1);r,c=np.arange(81)//9,np.arange(81)%9
    truth=(r[ii]==r[jj])|(c[ii]==c[jj])|((r[ii]//3==r[jj]//3)&(c[ii]//3==c[jj]//3));rr=np.arange(1000)[:,None]
    records=[]
    for k,bank in enumerate(z['banks']):
        clean=z['solutions'][k];free=~z['given'][k];cells=changes['cells'][k];values=changes['values'][k]
        assert free[rr,cells].all() and (clean[rr,cells]!=values).all()
        noisy=clean.copy();noisy[rr,cells]=values
        assert ((noisy!=clean).sum(1)==9).all() and np.array_equal(noisy[~free],clean[~free])
        comparisons=noisy[:,ii]==noisy[:,jj]
        for n in p['sample_sizes']:
            counts=comparisons[:n].sum(0,dtype=np.int64);assert np.array_equal(counts,stored[f'n{n}'][k])
            distinct=sorted(set(map(int,counts)));gaps=[b-a for a,b in zip(distinct,distinct[1:])]
            ix=gaps.index(max(gaps)) if gaps else None;threshold=(distinct[ix]+distinct[ix+1])//2 if ix is not None else 0
            assert threshold==stored[f't{n}'][k]
            alternatives={'gap':np.full(3240,threshold)}
            median=np.median(counts);eligible=[i for i in range(len(gaps)) if distinct[i+1]<=median]
            best=max(eligible,key=lambda i:gaps[i]) if eligible else None
            restricted=(distinct[best]+distinct[best+1])//2 if best is not None else 0
            assert restricted==stored[f'restricted_t{n}'][k]
            alternatives['restricted_gap']=np.full(3240,restricted)
            ff=free[:n];m=ff.sum(1);both=ff[:,ii]&ff[:,jj];one=ff[:,ii]^ff[:,jj]
            for assumed in p['assumed_noise']:
                # Literal row probabilities independent of producer's matrix formula.
                a=assumed/m;b=assumed*(assumed-1)/(m*(m-1))
                probabilities=np.where(both,(a/4-9*b/64)[:,None],np.where(one,(a/8)[:,None],0.))
                mean=probabilities.sum(0)
                assert np.allclose(mean,stored[f'law{assumed}_mean{n}'][k],atol=1e-12,rtol=0)
                thresholds=stored[f'law{assumed}_t{n}'][k]
                assert ((thresholds>=0)&(thresholds<=n)).all()
                for t in np.unique(thresholds):
                    indices=thresholds==t
                    bound=tail_bounds(int(t)+1,n,mean[indices]/n,True)
                    assert (bound<=p['threshold_alpha']/3240).all()
                    if t>0:assert (tail_bounds(int(t),n,mean[indices]/n,True)>p['threshold_alpha']/3240).all()
                alternatives[f'chernoff{assumed}']=thresholds
            for arm,cut in alternatives.items():
                support=counts<=cut;missing=int((truth&~support).sum());extra=int((~truth&support).sum())
                records.append(dict(bank=int(bank),n=n,arm=arm,threshold_min=int(cut.min()),threshold_max=int(cut.max()),missing_true_edges=missing,extra_edges=extra,
                    exact_graph=missing==extra==0,all_true_edges_retained=missing==0,
                    support_sha256=hashlib.sha256(support.tobytes()).hexdigest()))
    print(json.dumps(dict(shard=entry['shard'],models_audited=len(records),wall_s=time.monotonic()-start)),flush=True)
    return dict(records=records,input_sha256=evidence,wall_s=time.monotonic()-start)

def main():
    start=time.monotonic();p=json.loads((HERE/'chernoff_replication.json').read_text());manifest=json.loads((OUT/'inputs.json').read_text())
    assert manifest['protocol_sha256']==sha(HERE/'chernoff_replication.json') and manifest['producer_sha256']==sha(HERE/'chernoff_replication.py')
    jobs=json.loads((OUT/'queue.json').read_text());assert len(jobs)==16 and all(j['status']=='complete' for j in jobs)
    inputs=audit_sources(p,manifest);write(OUT/'input-audit.json',inputs);print(json.dumps({k:v for k,v in inputs.items() if k!='input_sha256'}),flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:results=list(pool.map(audit_shard,manifest['shards']))
    records=[r for a in results for r in a['records']];assert len(records)==4096
    grouped=[]
    for n in p['sample_sizes']:
        for arm in ['gap','restricted_gap','chernoff3','chernoff9']:
            rows=[r for r in records if r['n']==n and r['arm']==arm];k=sum(r['exact_graph'] for r in rows);lo,hi=exact_bounds(k,len(rows),.05)
            grouped.append(dict(n=n,arm=arm,banks=len(rows),exact_graphs=k,recovery_fraction=k/len(rows),
                lower_one_sided95=lo,upper_one_sided95=hi,all_true_edges_retained=sum(r['all_true_edges_retained'] for r in rows),
                total_missing_edges=sum(r['missing_true_edges'] for r in rows),total_extra_edges=sum(r['extra_edges'] for r in rows)))
    primary=next(r for r in grouped if r['n']==p['primary_n'] and r['arm']=='chernoff9').copy()
    primary.update(target=p['primary_target'],decision='supported' if primary['lower_one_sided95']>=p['primary_target'] else 'falsified' if primary['upper_one_sided95']<p['primary_target'] else 'inconclusive')
    evidence={str(x.relative_to(ROOT)):sha(x) for x in [HERE/'chernoff_replication.json',HERE/'chernoff_replication.py',Path(__file__),HERE/'audit_noise_replication.py',HERE/'noise_chernoff.py',HERE/'noise_conditional_bands.py',OUT/'inputs.json',OUT/'input-audit.json',OUT/'queue.json']}
    for r in results:evidence.update(r['input_sha256'])
    summary=dict(status='complete',audited_utc=datetime.now(timezone.utc).isoformat(),id=p['id'],primary=primary,grouped=grouped,
        models_audited=4096,independent_count_arrays_audited=1024,noisy_rows_audited=512*1000,changed_cells_audited=512*1000*9,input_audit=inputs,
        all_counts_independently_rebuilt=True,all_clues_preserved=True,cross_bank_overlap_allowed=True,
        source_sampling='512 independent size1000 samples without replacement within each bank; fixed eligible official corpus. Cross-bank overlap retained.',
        uncertainty='Separate one-sided exact95% binomial endpoints. Only correct-law N600 primary; others descriptive. Not a joint two-sided95% interval.',
        audit_wall_s=time.monotonic()-start,input_sha256=evidence,limitations=p['limits'])
    PUBLIC.mkdir(parents=True,exist_ok=True);write(PUBLIC/'summary.json',summary);write(OUT/'audit.json',summary)
    for name,rows in [('per_bank.csv',records),('recovery.csv',grouped)]:
        with (PUBLIC/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(primary=primary,grouped=grouped,audit_wall_s=summary['audit_wall_s'])),flush=True)

if __name__=='__main__':main()
