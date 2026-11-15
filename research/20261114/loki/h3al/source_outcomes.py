"""Source-only joint witness statistics and fixed deletion outcomes, no target access."""
import hashlib
import itertools
import math
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
from distribution import union_mass,validate

def main():
    p=frozen();assert not(OUT/'source-outcomes.json').exists();validation=validate();save(OUT/'distribution-validation.json',validation);population=np.load(ROOT/p['source_population']['path']);assert population.shape==(1932,169);observed=[]
    for board in population:
        assert board.sum()==13;offsets=set()
        for a,b in itertools.combinations(np.flatnonzero(board),2):
            dr=int(b)//13-int(a)//13;dc=int(b)%13-int(a)%13;assert dr and dc and dr!=abs(dc)and(dr,abs(dc))not in[(1,3),(3,1)];offsets.add((dr,dc))
        observed.append(offsets)
    outcomes=[];supports=[];full_arrays=[];total_patterns=0
    for bank,cfg in enumerate(p['models']):
        m=np.load(ROOT/cfg['model_path']);d=np.load(ROOT/cfg['source_design_path']);base=m['F'];assert not m['h'].any()and(m['J'][base.astype(bool)]>0).all()and not m['J'][~base.astype(bool)].any();finite={tuple(x)for x in d['offsets']};seen=set().union(*observed);classes=[]
        for dr,dc in sorted(finite&seen):
            a=max(0,-dc);b=15*dr+max(0,dc)
            if base[a,b]:classes.append((dr,dc))
        count=len(classes);assert count<=p['complexity']['classes_per_bank_max'],'Class-count gate exceeded';total_patterns+=1<<count;assert total_patterns<=p['complexity']['sum_bank_masks_max'],'Total bank-mask gate exceeded';lookup={o:k for k,o in enumerate(classes)};masks=np.array([sum(1<<lookup[o]for o in offsets if o in lookup)for offsets in observed],np.uint16);frequency=np.bincount(masks,minlength=1<<count).astype(np.int64);assert frequency.sum()==1932;mass={str(k):union_mass(frequency.tolist(),k)for k in p['doses']};folder=OUT/f'b{bank}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'source-patterns.npz',masks=masks,frequency=frequency,classes=np.array(classes,np.int16).reshape(-1,2));mapping=[]
        for mask in range(1<<count):
            F=base.copy();deleted={classes[k]for k in range(count)if mask&(1<<k)}
            for a,b in itertools.combinations(range(225),2):
                if(b//15-a//15,b%15-a%15)in deleted:F[a,b]=F[b,a]=0
            key=hashlib.sha256(F.tobytes()).hexdigest();matches=[i for i,s in enumerate(supports)if s['support_sha256']==key]
            if matches:group=matches[0];assert np.array_equal(full_arrays[group],F)
            else:
                group=len(supports);assert group<p['complexity']['unique_full_supports_max'],'Unique support gate exceeded';dest=OUT/f'support{group}';dest.mkdir(exist_ok=True);np.savez_compressed(dest/'F.npz',F=F);supports.append(dict(group=group,support_sha256=key,F_npz_sha256=sha(dest/'F.npz'),members=[]));full_arrays.append(F)
            supports[group]['members'].append(dict(bank=bank,mask=mask));mapping.append(group)
        outcomes.append(dict(bank=bank,classes=[list(o)for o in classes],class_count=count,frequency=frequency.tolist(),mask_to_support=mapping,draw_masses={k:[str(x)for x in values]for k,values in mass.items()},total_masses={str(k):str(math.comb(1932,k))for k in p['doses']},source_patterns_sha256=sha(folder/'source-patterns.npz'),base_model_sha256=sha(ROOT/cfg['model_path'])))
    save(OUT/'source-outcomes.json',dict(status='complete_source_only',utc=now(),banks=outcomes,supports=supports,all_masks_retained_including_unchanged_and_zero_mass=True,total_bank_masks=total_patterns,unique_supports=len(supports),class_complexity_gate_passed=True,exact_integer_joint_union_mass=True,no_target_truth_keys_or_scores_used=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
