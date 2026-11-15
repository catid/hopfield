"""One declared hierarchical complexity prior, exact posterior scores and all ties."""
import hashlib
import itertools
import math
import shutil
import subprocess
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load


def main():
    p=frozen();helper=load(SRC.parent/'h3at/source.py','h3au_frozen_graph_helpers');old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);packed=np.load(ROOT/p['normalizers']['bits_path']);orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[chosen for size in (0,1,2) for chosen in itertools.combinations(range(55),size)];hit=helper.incidence(ordinary,orbits);candidates=[]
    assert ordinary.shape==(14200,12) and packed['bits'].shape==(1541,1775) and packed['population_size']==14200
    for cid,chosen in enumerate(choices):
        mask=~hit[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[mask];F=helper.rule_graph(12,[orbits[j] for j in chosen]);record=old['candidates'][cid]
        assert np.array_equal(np.packbits(mask,bitorder='little'),packed['bits'][cid])
        assert record==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest())
        candidates.append(dict(record,orbit_count=len(chosen),prior_class_probability_numerator='1',prior_class_probability_denominator='3',models_in_class=math.comb(55,len(chosen)),prior_per_model_denominator=str(3*math.comb(55,len(chosen)))))
    assert len(candidates)==1541 and sum(1/math.comb(55,len(chosen)) for chosen in choices)>2.99999999999
    shutil.copyfile(ROOT/p['normalizers']['ordinary_path'],OUT/'source-ordinary-columns.npy');shutil.copyfile(ROOT/p['normalizers']['bits_path'],OUT/'candidate-validity-bits.npz')
    binary=OUT/'source_reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True)
    supports=[];groups={};banks=[]
    def register(index,arm,cid,chosen):
        F=helper.rule_graph(13,chosen);family=index//4;key=(family,F.tobytes())
        if key in groups:group=groups[key]
        else:
            assert len(supports)<p['budget']['distinct_target_supports_max'],'Support cap exceeded';group=len(supports);groups[key]=group;folder=OUT/f'support{group}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'model.npz',F=F,J=F.astype(np.float32),h=np.zeros(169,np.float32),q=np.array(13),unit=np.array(1.));supports.append(dict(group=group,family=family,model_sha256=sha(folder/'model.npz'),members=[]))
        supports[group]['members'].append(dict(index=index,arm=arm,candidate=cid));return group
    ordinary_keys=set(map(bytes,ordinary))
    for index,cfg in enumerate(p['banks']):
        labels=np.load(ROOT/cfg['design_path'])['labels'];assert labels.shape==(16,144) and (labels.sum(1)==12).all();cells=np.flatnonzero(labels).reshape(-1,12)%144;assert np.array_equal(cells//12,np.broadcast_to(np.arange(12),cells.shape));columns=(cells%12).astype(np.uint8);assert len(set(map(bytes,columns)))==16 and all(bytes(row) in ordinary_keys for row in columns);hits=helper.incidence(columns,orbits);scores=[]
        for candidate in candidates:
            selected=candidate['orbit_indices'];violations=int(hits[:,selected].any(1).sum()) if selected else 0;M=candidate['complete_source_cardinality'];consistent=violations==0;assert not consistent or M>=16
            denom=math.comb(M,16) if consistent else None;score=denom*candidate['models_in_class'] if consistent else None
            scores.append(dict(candidate=candidate['candidate'],sample_violations=violations,consistent=consistent,complete_source_cardinality=M,orbit_count=candidate['orbit_count'],models_in_class=candidate['models_in_class'],likelihood_numerator='1' if consistent else '0',likelihood_denominator=str(denom) if consistent else None,posterior_weight_numerator='1' if consistent else '0',posterior_weight_denominator=str(3*score) if consistent else None,MAP_integer_score_without_common3=str(score) if consistent else None))
        best=min(int(row['MAP_integer_score_without_common3']) for row in scores if row['consistent']);winners=[row['candidate'] for row in scores if row['consistent'] and int(row['MAP_integer_score_without_common3'])==best];maximizers=[dict(candidate=cid,group=register(index,'MAP_maximizer',cid,[tuple(o) for o in candidates[cid]['orbits']])) for cid in winners];unwitnessed=np.flatnonzero(~hits.any(0)).tolist();control=register(index,'consistency_envelope',None,[orbits[j] for j in unwitnessed]);folder=OUT/f'b{index}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'source-evidence.npz',sample_columns=columns,sample_orbit_incidence=hits);banks.append(dict(index=index,family=index//4,bank=index%4,scores=scores,maximizers=maximizers,unique_maximum=len(winners)==1,best_MAP_integer_score_without_common3=str(best),consistency_control_group=control,consistency_orbit_indices=unwitnessed,source_evidence_sha256=sha(folder/'source-evidence.npz')));print('SOURCE_MAP',index,winners,[(candidates[cid]['orbits'],candidates[cid]['complete_source_cardinality']) for cid in winners],flush=True)
    save(OUT/'source-selection.json',dict(status='complete_source_only',utc=now(),candidates=candidates,banks=banks,supports=supports,all1541_frozen_normalizations_rechecked=True,independent_literal_normalizer_verification_reused_from='H3at',all_tied_maximizers_retained=True,no_observed_source_completeness_assumed=True,no_target_truth_labels_or_scores_read=True,source_ordinary_columns_sha256=sha(OUT/'source-ordinary-columns.npy'),candidate_validity_bits_sha256=sha(OUT/'candidate-validity-bits.npz'),inherited_source_selection_sha256=sha(ROOT/p['normalizers']['selection_path']),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))


if __name__=='__main__':main()
