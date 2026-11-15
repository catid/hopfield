"""Positive-only finite-candidate likelihood with all tied maximizers retained."""
import itertools
import math
import subprocess
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load
def rule_graph(n,orbits):
    F=np.zeros((n*n,n*n),np.int32)
    for a,b in itertools.combinations(range(n*n),2):
        dr,dc=abs(a//n-b//n),abs(a%n-b%n);F[a,b]=F[b,a]=not dr or not dc or dr==dc or any((dr,dc)in(o,o[::-1])for o in orbits)
    return F
def incidence(columns,orbits):
    i,j=np.triu_indices(columns.shape[1],1);dr=j-i;dc=abs(columns[:,j].astype(np.int16)-columns[:,i].astype(np.int16));return np.stack([(((dr==a)&(dc==b))|((dr==b)&(dc==a))).any(1)for a,b in orbits],axis=1)
def main():
    p=frozen();helper=load(SRC.parent/'h3ad/generate.py','h3ar_source_ordinary');ordinary=helper.queens(12);assert ordinary.shape==(14200,12);orbits=[tuple(o)for o in p['candidate_prior']['orbits']];hit=incidence(ordinary,orbits);binary=OUT/'source_reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True);candidates=[]
    for count in(0,1,2):
        for selected in itertools.combinations(range(10),count):
            cid=len(candidates);members=[orbits[i]for i in selected];allowed=ordinary[~hit[:,selected].any(1)]if selected else ordinary.copy();F=rule_graph(12,members);folder=OUT/f'candidate{cid}';folder.mkdir(parents=True,exist_ok=True);inp=folder/'F.bin';out=folder/'columns.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),'12',str(inp),str(out)],capture_output=True,text=True,check=True);assert int(result.stdout)==len(allowed)and np.array_equal(allowed,np.fromfile(out,np.uint8).reshape(-1,12));np.save(folder/'source-columns.npy',allowed);np.save(folder/'source-F.npy',F);candidates.append(dict(candidate=cid,orbit_indices=list(selected),orbits=[list(o)for o in members],complete_source_cardinality=len(allowed),source_columns_sha256=sha(folder/'source-columns.npy'),source_F_sha256=sha(folder/'source-F.npy')))
    assert len(candidates)==56;supports=[];arrays=[];banks=[]
    def register(index,arm,cid,chosen):
        F=rule_graph(13,chosen);family=index//4;matches=[r['group']for r in supports if r['family']==family and np.array_equal(arrays[r['group']],F)]
        if matches:group=matches[0]
        else:
            assert len(arrays)<p['budget']['distinct_target_supports_max'],'Support cap exceeded';group=len(arrays);arrays.append(F);folder=OUT/f'support{group}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'model.npz',F=F,J=F.astype(np.float32),h=np.zeros(169,np.float32),q=np.array(13),unit=np.array(1.));supports.append(dict(group=group,family=family,model_sha256=sha(folder/'model.npz'),members=[]))
        supports[group]['members'].append(dict(index=index,arm=arm,candidate=cid));return group
    ordinary_keys={row.tobytes()for row in ordinary}
    for index,cfg in enumerate(p['banks']):
        y=np.load(ROOT/cfg['design_path'])['labels'];assert y.shape==(16,144)and(y.sum(1)==12).all();cells=np.flatnonzero(y).reshape(-1,12)%144;assert np.array_equal(cells//12,np.broadcast_to(np.arange(12),cells.shape));columns=(cells%12).astype(np.uint8);assert len({r.tobytes()for r in columns})==16 and all(r.tobytes()in ordinary_keys for r in columns);sample_hit=incidence(columns,orbits);scores=[]
        for c in candidates:
            selected=c['orbit_indices'];violations=int(sample_hit[:,selected].any(1).sum())if selected else 0;M=c['complete_source_cardinality'];consistent=violations==0;assert not consistent or M>=16;scores.append(dict(candidate=c['candidate'],sample_violations=violations,consistent=consistent,complete_source_cardinality=M,likelihood_numerator='1'if consistent else'0',likelihood_denominator=str(math.comb(M,16))if consistent else None))
        best=min(r['complete_source_cardinality']for r in scores if r['consistent']);winners=[r['candidate']for r in scores if r['consistent']and r['complete_source_cardinality']==best];selected_supports=[dict(candidate=cid,group=register(index,'ML_maximizer',cid,[tuple(o)for o in candidates[cid]['orbits']]))for cid in winners];unwitnessed=np.flatnonzero(~sample_hit.any(0)).tolist();control=register(index,'consistency_envelope',None,[orbits[i]for i in unwitnessed]);folder=OUT/f'b{index}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'source-evidence.npz',sample_columns=columns,sample_orbit_incidence=sample_hit);banks.append(dict(index=index,family=index//4,bank=index%4,scores=scores,maximizers=selected_supports,unique_maximum=len(winners)==1,best_source_cardinality=best,consistency_control_group=control,consistency_orbit_indices=unwitnessed,source_evidence_sha256=sha(folder/'source-evidence.npz')));print('SOURCE_SELECTION',index,winners,best,unwitnessed,flush=True)
    save(OUT/'source-selection.json',dict(status='complete_source_only',utc=now(),candidates=candidates,banks=banks,supports=supports,all56_normalizations_independently_verified=True,all_tied_maximizers_retained=True,no_observed_source_completeness_assumed=True,no_target_truth_labels_or_scores_read=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
