"""Seal all hypothetical updated-MAP choices before reading acquisition populations."""
import hashlib
import itertools
import math
import subprocess
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load


def main():
    p=frozen();helper=load(SRC.parent/'h3at/source.py','h3av_rule_helpers');old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);bits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[chosen for k in (0,1,2) for chosen in itertools.combinations(range(55),k)];hits=helper.incidence(ordinary,orbits);candidates=[]
    for cid,chosen in enumerate(choices):
        mask=~hits[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[mask];F=helper.rule_graph(12,[orbits[j] for j in chosen]);c=old['candidates'][cid];assert np.array_equal(np.packbits(mask,bitorder='little'),bits[cid]);assert c==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest());M=len(columns)
        score=math.comb(55,len(chosen))*math.comb(M,32) if M>=32 else None
        if M>=32:assert math.comb(M,16)*math.comb(M-16,16)==math.comb(M,32)*math.comb(32,16)
        candidates.append(dict(c,orbit_count=len(chosen),models_in_class=math.comb(55,len(chosen)),sample32_eligible=M>=32,MAP32_integer_score_without_common3=str(score) if score is not None else None))
    binary=OUT/'reverse';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True);supports=[];groups={};banks=[]
    def register(index,mask,cid):
        F=helper.rule_graph(13,[tuple(o) for o in candidates[cid]['orbits']]);key=(index//4,F.tobytes())
        if key in groups:group=groups[key]
        else:
            assert len(supports)<p['budget']['distinct_target_supports_max'],'Support cap exceeded';group=len(supports);groups[key]=group;folder=OUT/f'support{group}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'model.npz',F=F,J=F.astype(np.float32),h=np.zeros(169,np.float32),q=np.array(13),unit=np.array(1.));supports.append(dict(group=group,family=index//4,model_sha256=sha(folder/'model.npz'),members=[]))
        supports[group]['members'].append(dict(index=index,mask=mask,candidate=cid));return group
    for index,cfg in enumerate(p['banks']):
        labels=np.load(ROOT/cfg['design_path'])['labels'];assert labels.shape==(16,144) and (labels.sum(1)==12).all();cells=np.flatnonzero(labels).reshape(-1,12)%144;assert np.array_equal(cells//12,np.broadcast_to(np.arange(12),cells.shape));columns=(cells%12).astype(np.uint8);assert len(set(map(bytes,columns)))==16;seen=helper.incidence(columns,orbits);unwitnessed=np.flatnonzero(~seen.any(0)).tolist();initial=[cid for cid,chosen in enumerate(choices) if not seen[:,chosen].any()];assert all(candidates[cid]['complete_source_cardinality']>=16 for cid in initial);patterns=[]
        for mask in range(1<<len(unwitnessed)):
            newlyseen={unwitnessed[bit] for bit in range(len(unwitnessed)) if mask&(1<<bit)};consistent=[cid for cid in initial if newlyseen.isdisjoint(choices[cid])];eligible=[cid for cid in consistent if candidates[cid]['sample32_eligible']];assert eligible;best=min(int(candidates[cid]['MAP32_integer_score_without_common3']) for cid in eligible);winners=[cid for cid in eligible if int(candidates[cid]['MAP32_integer_score_without_common3'])==best]
            patterns.append(dict(mask=mask,newly_witnessed_orbit_indices=sorted(newlyseen),consistent_candidates=consistent,eligible32_candidates=eligible,ineligible32_candidates=[cid for cid in consistent if not candidates[cid]['sample32_eligible']],best_MAP32_integer_score_without_common3=str(best),maximizers=[dict(candidate=cid,group=register(index,mask,cid)) for cid in winners],unique=len(winners)==1))
        folder=OUT/f'b{index}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'initial-evidence.npz',columns=columns,orbit_incidence=seen);banks.append(dict(index=index,family=index//4,bank=index%4,unwitnessed_orbit_indices=unwitnessed,initial_consistent_candidates=initial,patterns=patterns,initial_evidence_sha256=sha(folder/'initial-evidence.npz')));print('SOURCE_PATTERN_SEAL',index,len(patterns),[r['maximizers'] for r in patterns],flush=True)
    save(OUT/'source-selection.json',dict(status='complete_source_only',utc=now(),candidates=candidates,banks=banks,supports=supports,all1541_normalizers_rechecked=True,all_eligible_binomial_sequential_identities_verified=True,all_abstract_masks_and_tied_maximizers_retained=True,no_true_source_acquisition_population_or_target_truth_read=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))


if __name__=='__main__':main()
