"""Expanded 55-orbit vocabulary with all 1,541 exact normalizers and all ties."""
import hashlib
import itertools
import math
import subprocess
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,save,sha,now,load


def rule_graph(n,orbits):
    cells=np.arange(n*n);dr=abs(cells[:,None]//n-cells[None,:]//n);dc=abs(cells[:,None]%n-cells[None,:]%n)
    F=(dr==0)|(dc==0)|(dr==dc)
    for a,b in orbits:F|=((dr==a)&(dc==b))|((dr==b)&(dc==a))
    np.fill_diagonal(F,False);return F.astype(np.int32)


def literal_graph(n,orbits):
    F=np.zeros((n*n,n*n),np.int32)
    for a,b in itertools.combinations(range(n*n),2):
        dr,dc=abs(a//n-b//n),abs(a%n-b%n);F[a,b]=F[b,a]=not dr or not dc or dr==dc or any((dr,dc) in (o,o[::-1]) for o in orbits)
    return F


def incidence(columns,orbits):
    i,j=np.triu_indices(columns.shape[1],1);dr=j-i;dc=abs(columns[:,j].astype(np.int16)-columns[:,i].astype(np.int16))
    return np.stack([(((dr==a)&(dc==b))|((dr==b)&(dc==a))).any(1) for a,b in orbits],axis=1)


def main():
    p=frozen();fixtures=[(n,o) for n in (12,13) for o in ([],[(2,3)],[(1,7),(5,11)])];t=time.perf_counter();fast=[rule_graph(n,o) for n,o in fixtures];fastseconds=time.perf_counter()-t;t=time.perf_counter();slow=[literal_graph(n,o) for n,o in fixtures];slowseconds=time.perf_counter()-t;assert all(np.array_equal(a,b) for a,b in zip(fast,slow));save(OUT/'graph-preflight.json',dict(fixtures=[dict(n=n,orbits=[list(x) for x in o]) for n,o in fixtures],complete_dtype_shape_bytes_identical=True,vector_wall_seconds=fastseconds,literal_wall_seconds=slowseconds,vector_speedup=slowseconds/fastseconds,charged_to_study=True))
    ordinary=load(SRC.parent/'h3ad/generate.py','h3at_source_ordinary').queens(12);assert ordinary.shape==(14200,12);np.save(OUT/'source-ordinary-columns.npy',ordinary)
    orbits=[tuple(o) for o in p['candidate_prior']['orbits']];assert len(orbits)==55;hit=incidence(ordinary,orbits);binary=OUT/'source_reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True);candidates=[];validity=[]
    for count in (0,1,2):
        for selected in itertools.combinations(range(55),count):
            cid=len(candidates);members=[orbits[i] for i in selected];mask=~hit[:,selected].any(1) if selected else np.ones(len(ordinary),bool);allowed=ordinary[mask];F=rule_graph(12,members);inp=OUT/'candidate-scratch-F.bin';out=OUT/'candidate-scratch-columns.bin';F.astype(np.uint8).tofile(inp);result=subprocess.run([str(binary),'12',str(inp),str(out)],capture_output=True,text=True,check=True);literal=np.fromfile(out,np.uint8).reshape(-1,12);assert int(result.stdout)==len(allowed) and np.array_equal(allowed,literal)
            candidates.append(dict(candidate=cid,orbit_indices=list(selected),orbits=[list(o) for o in members],complete_source_cardinality=len(allowed),source_columns_bytes_sha256=hashlib.sha256(allowed.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest()));validity.append(np.packbits(mask,bitorder='little'))
            if cid%100==0:print('CANDIDATE_PARITY',cid,len(allowed),flush=True)
    assert len(candidates)==1541;np.savez_compressed(OUT/'candidate-validity-bits.npz',bits=np.stack(validity),population_size=np.array(len(ordinary)))
    supports=[];groups={};banks=[]
    def register(index,arm,cid,chosen):
        F=rule_graph(13,chosen);family=index//4;key=(family,F.tobytes())
        if key in groups:group=groups[key]
        else:
            assert len(supports)<p['budget']['distinct_target_supports_max'],'Support cap exceeded';group=len(supports);groups[key]=group;folder=OUT/f'support{group}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'model.npz',F=F,J=F.astype(np.float32),h=np.zeros(169,np.float32),q=np.array(13),unit=np.array(1.));supports.append(dict(group=group,family=family,model_sha256=sha(folder/'model.npz'),members=[]))
        supports[group]['members'].append(dict(index=index,arm=arm,candidate=cid));return group
    ordinary_keys={row.tobytes() for row in ordinary}
    for index,cfg in enumerate(p['banks']):
        y=np.load(ROOT/cfg['design_path'])['labels'];assert y.shape==(16,144) and (y.sum(1)==12).all();cells=np.flatnonzero(y).reshape(-1,12)%144;assert np.array_equal(cells//12,np.broadcast_to(np.arange(12),cells.shape));columns=(cells%12).astype(np.uint8);assert len({r.tobytes() for r in columns})==16 and all(r.tobytes() in ordinary_keys for r in columns);sample_hit=incidence(columns,orbits);scores=[]
        for c in candidates:
            selected=c['orbit_indices'];violations=int(sample_hit[:,selected].any(1).sum()) if selected else 0;M=c['complete_source_cardinality'];consistent=violations==0;assert not consistent or M>=16,'Distinct sixteen-positive sample cannot fit candidate cardinality below16';scores.append(dict(candidate=c['candidate'],sample_violations=violations,consistent=consistent,complete_source_cardinality=M,likelihood_numerator='1' if consistent else '0',likelihood_denominator=str(math.comb(M,16)) if consistent else None))
        best=min(r['complete_source_cardinality'] for r in scores if r['consistent']);winners=[r['candidate'] for r in scores if r['consistent'] and r['complete_source_cardinality']==best];maximizers=[dict(candidate=cid,group=register(index,'ML_maximizer',cid,[tuple(o) for o in candidates[cid]['orbits']])) for cid in winners];unwitnessed=np.flatnonzero(~sample_hit.any(0)).tolist();control=register(index,'consistency_envelope',None,[orbits[i] for i in unwitnessed]);folder=OUT/f'b{index}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'source-evidence.npz',sample_columns=columns,sample_orbit_incidence=sample_hit);banks.append(dict(index=index,family=index//4,bank=index%4,scores=scores,maximizers=maximizers,unique_maximum=len(winners)==1,best_source_cardinality=best,consistency_control_group=control,consistency_orbit_indices=unwitnessed,source_evidence_sha256=sha(folder/'source-evidence.npz')));print('SOURCE_SELECTION',index,winners,best,unwitnessed,flush=True)
    save(OUT/'source-selection.json',dict(status='complete_source_only',utc=now(),candidates=candidates,banks=banks,supports=supports,all1541_normalizations_independently_verified=True,all_tied_maximizers_retained=True,no_observed_source_completeness_assumed=True,no_target_truth_labels_or_scores_read=True,source_ordinary_columns_sha256=sha(OUT/'source-ordinary-columns.npy'),candidate_validity_bits_sha256=sha(OUT/'candidate-validity-bits.npz'),graph_preflight_sha256=sha(OUT/'graph-preflight.json'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))


if __name__=='__main__':main()
