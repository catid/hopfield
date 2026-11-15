"""Audit stored input identities, exact counts, paired times and budget."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import csv
import hashlib
import itertools
import shutil
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
PUBLIC=ROOT/'results/research/loki/h3ah'
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def main():
    start=time.perf_counter();p=frozen();g=read(OUT/'benchmark-gate.json');r=read(OUT/'benchmark-result.json');assert g['status']==r['status']=='complete_verified'and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and not g['monitor_expired'];assert g['protocol_sha256']==sha(SRC/'protocol.json')and g['source_sha256']==sha(SRC/'benchmark.py')and g['rows']==r['rows'];assert len(g['rows'])==8
    PUBLIC.mkdir(parents=True,exist_ok=True);files=['benchmark-gate.json','benchmark-result.json','benchmark.log'];audit=[];seedrows=[]
    for d in g['datasets']:
        bank=d['bank'];name=d['name'];folder=OUT/f'b{bank}/{name}';z=np.load(folder/'inputs.npz');boards=z['boards'];ready=z['ready'];counts=np.load(folder/'conflict-counts.npy');cfg=p['models'][bank];F=np.load(ROOT/cfg['model_path'])['F'];assert sha(folder/'inputs.npz')==d['input_npz_sha256']and digest(boards)==d['boards_sha256']and digest(ready)==d['ready_sha256']and sha(ROOT/cfg['model_path'])==d['model_sha256'];assert boards.dtype==np.bool_ and(boards.sum(1)==15).all()and len(counts)==len(boards)==d['states'];assert counts.dtype==np.int64 and((counts>=0)&(counts<=105)).all()
        if name=='complete_cache':
            old=np.load(ROOT/cfg['cache_path']);assert np.array_equal(old['boards'],boards)and np.array_equal(old['columns'],ready)and not counts.any();cells=15*np.arange(15)[None,:]+ready
            # Independent count via ordered-row interactions, avoiding the benchmark's triangular advanced indexing.
            recomputed=np.zeros(len(cells),np.int64)
            for i in range(15):
                for j in range(i+1,15):recomputed+=F[cells[:,i],cells[:,j]]
            assert np.array_equal(counts,recomputed)
        else:
            rng=np.random.default_rng(p['random_grid']['seeds'][bank]);expected=np.stack([np.sort(rng.choice(225,15,replace=False))for _ in range(8192)]);expected=np.concatenate([expected,np.array([np.arange(15),15*np.arange(15),16*np.arange(15)])]);assert np.array_equal(expected,ready)
            cells=ready;literal=np.array([sum(int(F[a,b])for a,b in itertools.combinations(np.flatnonzero(board),2))for board in boards],np.int64);assert np.array_equal(literal,counts)and np.array_equal(counts[-3:],[105,105,105])
        reconstructed=np.zeros_like(boards);reconstructed[np.arange(len(boards))[:,None],cells]=True;assert np.array_equal(reconstructed,boards)
        subset=[row for row in g['rows']if row['bank']==bank and row['dataset']==name];assert len(subset)==2
        for row in subset:
            assert row['all_integer_counts_equal']and row['count_sha256']==digest(counts);assert row['zero_conflict_states']==int((counts==0).sum())and row['nonzero_conflict_states']==int((counts>0).sum())and row['maximum_conflicts']==int(counts.max());assert len(row['trials'])==3
            for trial in row['trials']:
                k=trial['trial'];parity=(k+bank+int(name=='arbitrary_cardinality15')+int(row['input_path']=='full_board'))%2;assert trial['order']==(['dense','occupied']if parity==0 else['occupied','dense'])and trial['all_integer_counts_equal']
                for arm in ['dense','occupied']:
                    t=trial['times'][arm];assert t['integer_count_sha256']==digest(counts)and t['wall_seconds']>0 and t['cpu_seconds']>0;seedrows.append(dict(bank=bank,dataset=name,input_path=row['input_path'],trial=k,order=','.join(trial['order']),method=arm,**t))
            dense=float(np.median([x['times']['dense']['wall_seconds']for x in row['trials']]));fast=float(np.median([x['times']['occupied']['wall_seconds']for x in row['trials']]));assert dense==row['dense_median_seconds']and fast==row['occupied_median_seconds']and dense/fast==row['speed_ratio']and row['decision']==('supported'if dense/fast>=20 else'falsified')
        audit.append(dict(bank=bank,dataset=name,all_input_identities_verified=True,all_integer_counts_independently_recomputed=True,random_generation_replayed=name=='arbitrary_cardinality15',complete_cache_identity_checked=name=='complete_cache',states=len(boards)));files += [f'b{bank}/{name}/inputs.npz',f'b{bank}/{name}/conflict-counts.npy']
    for bank,cfg in enumerate(p['models']):
        target=PUBLIC/f'b{bank}/model.npz';target.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/cfg['model_path'],target)
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    for filename,rows in [('comparisons.csv',[{k:v for k,v in row.items()if k!='trials'}for row in g['rows']]),('paired_trials.csv',seedrows)]:
        with(PUBLIC/filename).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='validated',utc=now(),hypothesis='H3ah',decision='supported'if all(row['decision']=='supported'for row in g['rows'])else'falsified',rows=g['rows'],datasets=g['datasets'],audit=audit,distinct_model_state_pairs=sum(d['states']for d in g['datasets']),paired_comparisons=len(g['rows'])*3,method_calls=len(seedrows),all_integer_counts_equal=True,minimum_speed_ratio=min(row['speed_ratio']for row in g['rows']),maximum_speed_ratio=max(row['speed_ratio']for row in g['rows']),aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,budget=p['budget'],overheads=p['overheads'],scope=p['scope'],protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),public_sha256={rel:sha(PUBLIC/rel)for rel in files+[f'b{b}/model.npz'for b in range(2)]},post_run_audit_wall_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary);print({k:summary[k]for k in ['decision','distinct_model_state_pairs','minimum_speed_ratio','maximum_speed_ratio','aggregate_CPU_seconds','wall_seconds','post_run_audit_wall_seconds']})
if __name__=='__main__':main()
