"""Independently audit complete universes, bank transfers and exact clue capacities."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import itertools
import math
import shutil
import time
import csv
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
PUBLIC=ROOT/'results/research/loki/h3aj'
def hist(columns):
    result=np.zeros(226**3,np.int64)
    # Different loop nesting from the experiment; all clue sizes independently accumulated.
    for k in [1,2,3]:
        ix=np.array(list(itertools.combinations(range(15),k)));powers=np.array([226**(2-j)for j in range(k)],np.int64)
        for lo in range(0,len(columns),2048):
            cells=15*np.arange(15)[None,:]+columns[lo:lo+2048];keys=((cells[:,ix]+1)*powers).sum(-1);result+=np.bincount(keys.reshape(-1),minlength=226**3)
    assert result.sum()==len(columns)*575;return result

def audit_cache(folder,F):
    z=np.load(folder/'cache.npz');a=z['columns'];boards=z['boards'];cells=15*np.arange(15)[None,:]+a;rebuilt=np.zeros_like(boards);rebuilt[np.arange(len(boards))[:,None],cells]=True;assert np.array_equal(rebuilt,boards)and len(np.unique(a,axis=0))==len(a)and np.array_equal(np.packbits(boards.T,axis=1,bitorder='little'),z['index']);counts=np.zeros(len(a),np.int64)
    for i in range(15):
        for j in range(i+1,15):counts+=F[cells[:,i],cells[:,j]]
    assert not counts.any();return a,boards

def main():
    start=time.perf_counter();p=frozen();g=read(OUT/'capacity-gate.json');raw=read(OUT/'study-result.json');assert g['status']==raw['status']=='complete_verified'and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and not g['monitor_expired']and g['rows']==raw['rows'];assert g['protocol_sha256']==sha(SRC/'protocol.json')and raw['source_sha256']==sha(SRC/'study.py');seal=read(OUT/'enumeration-seal.json');assert seal['status']=='all_complete'and len(g['rows'])==4;PUBLIC.mkdir(parents=True,exist_ok=True);Ftrue=np.load(OUT/'truth/F.npy');q=15
    for i,j in itertools.combinations(range(225),2):
        ar,ac=divmod(i,q);br,bc=divmod(j,q);truth=ar==br or ac==bc or abs(ar-br)==abs(ac-bc)or(abs(ar-br),abs(ac-bc))in[(1,3),(3,1)];assert Ftrue[i,j]==Ftrue[j,i]==truth
    truecols,trueboards=audit_cache(OUT/'truth',Ftrue);assert len(truecols)==g['universe']['true_states'];assert np.array_equal(truecols,np.fromfile(OUT/'verifier-io/N15-truth.columns.bin',np.uint8).reshape(-1,15))and np.array_equal(truecols,np.fromfile(OUT/'verifier-io/N15-truth.truth-columns.bin',np.uint8).reshape(-1,15));truehist=hist(truecols);universe=np.load(OUT/'input-universe.npz');allkeys=np.flatnonzero(truehist);prior=np.load(ROOT/'results/research/loki/h3ag/per-input-metrics.npz');excluded=np.union1d(prior['b0_keys'],prior['b1_keys']);eligible=np.setdiff1d(allkeys,excluded);clues=sum((eligible//226**power%226)>0 for power in [2,1,0]);assert np.array_equal(allkeys,universe['all_keys'])and np.array_equal(excluded,universe['excluded_keys'])and np.array_equal(eligible,universe['eligible_keys'])and np.array_equal(clues,universe['eligible_clues'])and np.array_equal(truehist[eligible],universe['eligible_true_counts']);assert len(excluded)==8192 and len(eligible)==g['universe']['remaining_eligible_keys'];sourcepop=np.load(ROOT/p['prior_true_solutions']['12']);models=[];weights=[];supportkeys=[];files=['capacity-gate.json','study-result.json','verification.json','enumeration-seal.json','input-universe.npz','campaign.log','truth/cache.npz','truth/F.npy']
    for bank,cfg in enumerate(p['models']):
        folder=OUT/f'b{bank}';m=np.load(folder/'model.npz');d=np.load(ROOT/cfg['source_design_path']);params=np.load(ROOT/cfg['parameters_path']);indices=np.sort(np.random.default_rng(2026112700+bank).choice(176,128,replace=False));assert np.array_equal(indices,d['population_indices'])and np.array_equal(sourcepop[indices],d['labels']);coefficient=params['A'][d['groups']];mask=(d['counts']==0)&(d['exposure']>=math.log(264/.01));mask|=(d['offsets']@d['lines'].T==0).any(1);F=np.zeros((225,225),np.int32);J=np.zeros((225,225),np.float32)
        for i,j in itertools.combinations(range(225),2):
            displacement=np.array([j//15-i//15,j%15-i%15]);lines=np.flatnonzero(d['lines']@displacement==0);offset=np.flatnonzero((d['offsets']==displacement).all(1));inside=bool(len(lines)or(len(offset)and mask[offset[0]]));value=float(params['A'][lines[0]])if len(lines)else float(coefficient[offset[0]])if inside else 0.;F[i,j]=F[j,i]=inside;J[i,j]=J[j,i]=value
        assert np.array_equal(F,m['F'])and np.array_equal(J,m['J'])and not m['h'].any()and float(m['unit'])==float(coefficient[mask].mean())and(J[F.astype(bool)]>0).all();models.append(F);weights.append(sha(folder/'model.npz'));supportkeys.append(F.tobytes());files += [f'b{bank}/model.npz',f'b{bank}/transfer.json']
    audited=[]
    for item in seal['supports']:
        group=item['group'];bank=item['representative_bank'];F=models[bank];folder=OUT/f'support{group}';columns,boards=audit_cache(folder,F);assert sha(folder/'cache.npz')==item['cache_sha256']and np.array_equal(columns,np.load(folder/'unverified-forward-columns.npy'))and np.array_equal(columns,np.fromfile(OUT/f'verifier-io/N15-support{group}.columns.bin',np.uint8).reshape(-1,15));truekeys={row.tobytes()for row in truecols};valid=np.array([row.tobytes()in truekeys for row in columns]);h=hist(columns[valid]);assert(h<=truehist).all();cap=np.load(folder/'capacity.npz');savedhist=np.load(folder/'key-histogram.npz');assert np.array_equal(cap['eligible_keys'],eligible)and np.array_equal(cap['valid_state_mask'],valid)and np.array_equal(cap['compatible_true_zero_F'],h[eligible])and np.array_equal(savedhist['keys'],np.flatnonzero(h))and np.array_equal(savedhist['counts'],h[h>0]);info=next(r for r in g['support_results']if r['group']==group);assert info['capacity_pct']==100*float((h[eligible]>0).mean())and info['valid_state_retention_pct']==100*float(valid.sum()/len(truecols))and info['invalid_zero_F_states']==int((~valid).sum());i,j=np.triu_indices(225,1);actual=F[i,j].astype(bool);truth=Ftrue[i,j].astype(bool);assert info['false_forbidden_edges']==int((actual&~truth).sum())and info['missed_true_edges']==int((~actual&truth).sum())
        for k in [1,2,3]:part=clues==k;assert info['per_clue'][str(k)]['eligible_inputs']==int(part.sum())and info['per_clue'][str(k)]['inputs_with_valid_zero_F_completion']==int((h[eligible][part]>0).sum())and info['per_clue'][str(k)]['capacity_pct']==100*float((h[eligible][part]>0).mean())
        for b in item['banks']:
            assert np.array_equal(models[b],F)and seal['bank_to_support'][b]==group;row=g['rows'][b];assert row['bank']==b and row['support_group']==group and row['capacity_pct']==info['capacity_pct']and row['decision']==('supported'if info['capacity_pct']>=95 and info['invalid_zero_F_states']==0 else'falsified')
        audited.append(dict(group=group,complete_forward_reverse_cache_identity=True,all_integer_certificates_independently_recomputed=True,validity_membership_independently_recomputed=True,full_key_histogram_and_per_clue_capacities_recomputed=True,banks=item['banks']));files += [f'support{group}/{name}'for name in ['cache.npz','compilation.json','capacity.npz','key-histogram.npz']]
    assert len(set(supportkeys))==len(seal['supports']);decision='supported'if all(r['decision']=='supported'for r in g['rows'])else'falsified';assert decision==g['decision']
    for rel in files:
        dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest)
    rows=[{k:v for k,v in row.items()if k not in ['per_clue','banks']}for row in g['rows']]
    with(PUBLIC/'capacity.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    byclue=[dict(bank=r['bank'],method='BP AdamW',clues=k,**v)for r in g['rows']for k,v in r['per_clue'].items()]
    with(PUBLIC/'per_clue_capacity.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(byclue[0]));w.writeheader();w.writerows(byclue)
    summary=dict(status='validated',utc=now(),hypothesis='H3aj',decision=decision,rows=g['rows'],support_results=g['support_results'],universe=g['universe'],source_banks=4,distinct_model_files=len(set(weights)),distinct_full_supports=len(set(supportkeys)),audit=audited,all_source_bank_draws_replayed=True,all_transferred_models_independently_reconstructed=True,true_graph_and_three_complete_enumerators_verified=True,all_full_input_histograms_independently_recomputed=True,prior_N15_exclusions_replayed_from_public_H3ag_keys=True,aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,target_cohorts_created=0,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),scope=p['scope'],public_sha256={rel:sha(PUBLIC/rel)for rel in files},post_run_audit_wall_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary);print({k:summary[k]for k in ['decision','source_banks','distinct_model_files','distinct_full_supports','aggregate_CPU_seconds','wall_seconds','post_run_audit_wall_seconds']})
if __name__=='__main__':main()
