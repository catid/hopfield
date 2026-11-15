"""Audit sampled positive evidence, every deletion, complete supports and capacities."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import csv
import itertools
import hashlib
import shutil
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
PUBLIC=ROOT/'results/research/loki/h3ak'
def hist(columns):
    h=np.zeros(226**3,np.int64)
    for k in [1,2,3]:
        ix=np.array(list(itertools.combinations(range(15),k)));powers=np.array([226**(2-j)for j in range(k)],np.int64)
        for lo in range(0,len(columns),2048):
            cells=15*np.arange(15)[None,:]+columns[lo:lo+2048];keys=((cells[:,ix]+1)*powers).sum(-1);h+=np.bincount(keys.reshape(-1),minlength=226**3)
    assert h.sum()==575*len(columns);return h

def main():
    start=time.perf_counter();p=frozen('target');g=read(OUT/'capacity-gate.json');raw=read(OUT/'study-result.json');acq=read(OUT/'source-positive-seal.json');repair=read(OUT/'repair-seal.json');compiled=read(OUT/'learned-enumeration-seal.json');validation=read(OUT/'compiler-validation.json');assert g['status']==raw['status']=='complete_verified'and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and not g['monitor_expired'];assert raw['source_sha256']==sha(SRC/'study.py')and raw['protocol_sha256']==sha(SRC/'protocol.json')and raw['rows']==g['rows'];assert sha(OUT/'repair-seal.json')==raw['repair_sha256']and sha(OUT/'source-positive-seal.json')==raw['source_positive_seal_sha256'];assert acq['source_sha256']==sha(SRC/'acquire.py')and repair['source_sha256']==sha(SRC/'repair.py')and repair['artifact_read_allowlist_enforced'];assert acq['utc']<repair['utc']<compiled['utc'];assert len(validation['cases'])==42 and all(r['all_four_equal']for r in validation['cases']);population=np.load(ROOT/p['acquisition']['population_path']);models={};inputids=[];files=['capacity-gate.json','study-result.json','source-positive-seal.json','repair-seal.json','learned-enumeration-seal.json','compiler-validation.json','campaign.log'];allowed=set()
    for cfg in p['models']:
        bank=cfg['bank'];base=np.load(ROOT/cfg['model_path']);d=np.load(ROOT/cfg['source_design_path']);finite={tuple(x)for x in d['offsets']};folder=OUT/f'b{bank}';indices=np.load(folder/'positive-population-indices.npy');boards=np.load(folder/'positive-boards.npy');expected=np.random.default_rng(p['acquisition']['seeds'][bank]).choice(1932,64,replace=False);assert np.array_equal(indices,expected)and np.array_equal(boards,population[indices])and len(set(indices))==64;assert sha(folder/'positive-boards.npy')==acq['banks'][bank]['boards_sha256']and sha(folder/'positive-population-indices.npy')==acq['banks'][bank]['indices_sha256'];inputids.append(indices)
        for board in boards:
            assert board.sum()==13
            for a,b in itertools.combinations(np.flatnonzero(board),2):
                dr=abs(int(a)//13-int(b)//13);dc=abs(int(a)%13-int(b)%13);assert dr and dc and dr!=dc and(dr,dc)not in[(1,3),(3,1)]
        previous=base['F'];allowed|={cfg['model_path'],cfg['source_design_path'],str((folder/'positive-boards.npy').relative_to(ROOT)),str((folder/'positive-population-indices.npy').relative_to(ROOT))};files += [f'b{bank}/positive-boards.npy',f'b{bank}/positive-population-indices.npy']
        for dose in p['acquisition']['doses']:
            dest=folder/f'd{dose}';m=np.load(dest/'model.npz');r=read(dest/'repair.json');assert r['model_sha256']==sha(dest/'model.npz');observed={}
            for position,board in enumerate(boards[:dose]):
                for a,b in itertools.combinations(np.flatnonzero(board),2):
                    o=(int(b)//13-int(a)//13,int(b)%13-int(a)%13)
                    if o in finite and o not in observed:observed[o]=(position,int(a),int(b))
            F=base['F'].copy();J=base['J'].copy();removed=set()
            for a,b in itertools.combinations(range(225),2):
                o=(b//15-a//15,b%15-a%15)
                if F[a,b]and o in observed:F[a,b]=F[b,a]=0;J[a,b]=J[b,a]=0;removed.add(o)
            assert np.array_equal(F,m['F'])and np.array_equal(J,m['J'])and np.array_equal(m['unit'],base['unit'])and np.array_equal(m['h'],base['h'])and np.array_equal(m['q'],base['q'])and(F<=previous).all();previous=F;assert r['removed_finite_offsets']==[list(x)for x in sorted(removed)]and r['removed_target_pairs']==int((base['F'].sum()-F.sum())//2);assert len(r['witnesses'])==len(removed)
            for w in r['witnesses']:
                o=tuple(w['offset']);pos,a,b=observed[o];assert w['positive_position']==pos<dose and w['population_index']==indices[pos]and w['source_cells']==[a,b]and boards[pos,a]and boards[pos,b]
            models[bank,dose]=F;allowed.add(str((dest/'model.npz').relative_to(ROOT)));files += [f'b{bank}/d{dose}/model.npz',f'b{bank}/d{dose}/repair.json']
    assert set(repair['artifact_read_paths'])<=allowed and p['acquisition']['population_path']not in repair['artifact_read_paths'];assert acq['overlap_counts']==[[len(set(a)&set(b))for b in inputids]for a in inputids];ref=ROOT/p['evaluation']['reference_root'];Ftrue=np.load(ref/'truth/F.npy');truth=np.load(ref/'truth/cache.npz')['columns'];truthkeys={row.tobytes()for row in truth};u=np.load(ref/'input-universe.npz');keys=u['eligible_keys'];clues=u['eligible_clues'];truehist=hist(truth);assert np.array_equal(truehist[keys],u['eligible_true_counts']);audits=[]
    for comp in compiled['supports']:
        group=comp['group'];folder=OUT/f'support{group}';F=models[comp['representative_bank'],comp['representative_dose']];assert hashlib.sha256(F.tobytes()).hexdigest()==comp['support_sha256'];cache=np.load(folder/'cache.npz');columns=cache['columns'];boards=cache['boards'];assert sha(folder/'cache.npz')==comp['cache_sha256']and len(columns)==comp['states']and len(np.unique(columns,axis=0))==len(columns)and np.array_equal(columns,np.load(folder/'unverified-forward-columns.npy'))and np.array_equal(columns,np.fromfile(OUT/f'verifier-io/N15-support{group}.columns.bin',np.uint8).reshape(-1,15));cells=15*np.arange(15)[None,:]+columns;rebuilt=np.zeros_like(boards);rebuilt[np.arange(len(columns))[:,None],cells]=True;assert np.array_equal(rebuilt,boards)and np.array_equal(np.packbits(boards.T,axis=1,bitorder='little'),cache['index']);count=np.zeros(len(columns),np.int64)
        for i in range(15):
            for j in range(i+1,15):count+=F[cells[:,i],cells[:,j]]
        assert not count.any();valid=np.array([row.tobytes()in truthkeys for row in columns]);h=hist(columns[valid]);cap=np.load(folder/'capacity.npz');kh=np.load(folder/'key-histogram.npz');assert np.array_equal(cap['eligible_keys'],keys)and np.array_equal(cap['compatible_valid_zero_F'],h[keys])and np.array_equal(cap['valid_state_mask'],valid)and np.array_equal(kh['keys'],np.flatnonzero(h))and np.array_equal(kh['counts'],h[h>0]);info=next(r for r in g['support_results']if r['group']==group);assert info['capacity_pct']==100*float((h[keys]>0).mean())and info['valid_state_retention_pct']==100*float(valid.sum()/len(truth))and info['invalid_zero_F_states']==int((~valid).sum());i,j=np.triu_indices(225,1);actual=F[i,j].astype(bool);gt=Ftrue[i,j].astype(bool);assert info['false_forbidden_edges']==int((actual&~gt).sum())and info['missed_true_edges']==int((~actual&gt).sum())
        for k in [1,2,3]:v=info['per_clue'][str(k)];part=clues==k;assert v['eligible_inputs']==int(part.sum())and v['inputs_with_valid_zero_F_completion']==int((h[keys][part]>0).sum())and v['capacity_pct']==100*float((h[keys][part]>0).mean())
        for member in comp['members']:
            bank,dose=member['bank'],member['dose'];assert np.array_equal(models[bank,dose],F);r=next(r for r in g['rows']if r['bank']==bank and r['dose']==dose);assert r['capacity_pct']==info['capacity_pct']and r['primary']==(dose==64)and r['decision']==('supported'if r['capacity_pct']>=95 and r['invalid_zero_F_states']==0 else'falsified')
        audits.append(dict(group=group,all_members_identical_full_support=True,complete_cache_forward_reverse_identity=True,all_integer_certificates_independently_recomputed=True,all_true_memberships_and_key_histograms_recomputed=True));files += [f'support{group}/{name}'for name in ['cache.npz','compilation.json','capacity.npz','key-histogram.npz']]
    assert len(g['rows'])==16 and len(g['primary'])==4 and g['decision']==('supported'if all(r['decision']=='supported'for r in g['primary'])else'falsified');PUBLIC.mkdir(parents=True,exist_ok=True)
    for rel in files:
        dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest)
    shutil.copyfile(ROOT/p['acquisition']['population_path'],PUBLIC/'source-population-n13.npy')
    public_rows=[dict(r,threshold_met=(r['capacity_pct']>=95 and r['invalid_zero_F_states']==0),decision=r['decision']if r['primary']else'descriptive')for r in g['rows']]
    rows=[{k:v for k,v in r.items()if k!='per_clue'}for r in public_rows]
    with(PUBLIC/'capacity.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='validated',utc=now(),hypothesis='H3ak',decision=g['decision'],rows=public_rows,primary=[r for r in public_rows if r['primary']],support_results=g['support_results'],distinct_full_supports=len(compiled['supports']),all_four_nested_source_draws_replayed=True,all_selected_source_positives_independently_valid=True,every_deletion_reconstructed_from_selected_positive_witnesses=True,repair_artifact_read_allowlist_verified=True,repair_artifact_read_paths=repair['artifact_read_paths'],all_16_models_and_doses_retained=True,audit=audits,source_positive_overlap_counts=acq['overlap_counts'],same_seen_eligible_inputs=len(keys),reference_true_states=len(truth),aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,target_cohorts_created=0,theorem_scope=p['theorem'],scope=p['scope'],protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),public_sha256={rel:sha(PUBLIC/rel)for rel in files+['source-population-n13.npy']},post_run_audit_wall_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary);print(dict(decision=summary['decision'],primary=[r['capacity_pct']for r in summary['primary']],supports=summary['distinct_full_supports'],CPU=summary['aggregate_CPU_seconds'],wall=summary['wall_seconds'],audit=summary['post_run_audit_wall_seconds']))
if __name__=='__main__':main()
