"""Exact source-complement paired acquisition outcomes, sealed before target reads."""
import hashlib
import itertools
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from learner import views,offsets,repair
from distribution import paired_mass,validate

def valid(board,leaper):
    assert board.sum()==12
    for a,b in itertools.combinations(np.flatnonzero(board),2):
        dr,dc=abs(int(b)//12-int(a)//12),abs(int(b)%12-int(a)%12)
        assert dr and dc and dr!=dc and(dr,dc)not in(leaper,leaper[::-1])

def main():
    p=frozen();save(OUT/'distribution-validation.json',validate());rows=[];supports=[];arrays=[];family_counts=[0,0];fixtures=[]
    for cell in range(144):
        x=np.zeros(144,bool);x[cell]=True;r,c=divmod(cell,12);mapped={(r,c),(c,11-r),(11-r,11-c),(11-c,r),(r,11-c),(11-c,11-r),(11-r,c),(c,r)};assert{int(np.flatnonzero(v)[0])for v in views(x)}=={12*a+b for a,b in mapped}
    for index in range(8):
        folder=OUT/f'b{index}';fit=read(folder/'model.json');family=index//4
        if not fit['source_eligible']:
            rows.append(dict(index=index,family=family,status='source_fit_ineligible',fit_sha256=sha(folder/'model.json')));continue
        population=np.load(folder/'acquisition-population.npy');model=np.load(folder/'model.npz');F=model['F'];finite={tuple(o)for o in np.load(folder/'source-design.npz')['offsets']};leaper=tuple(p['families'][family]['leaper']);rawseen=[];augseen=[];viewcounts=[]
        for board in population:
            valid(board,leaper);transformed=views(board)
            for view in transformed:valid(view,leaper)
            rawseen.append(offsets([board]));augseen.append(offsets(transformed));viewcounts.append(len(transformed))
        seen=set().union(*augseen);classes=sorted((dr,dc)for dr,dc in seen&finite if F[max(0,-dc),13*dr+max(0,dc)]);m=len(classes);assert m<=p['complexity']['observable_classes_per_fit_max'],'Observable class gate exceeded'
        raw_masks=np.array([sum(1<<j for j,o in enumerate(classes)if o in v)for v in rawseen],np.int64);aug_masks=np.array([sum(1<<j for j,o in enumerate(classes)if o in v)for v in augseen],np.int64);pair_masks=raw_masks|(aug_masks<<m);types,counts=np.unique(pair_masks,return_counts=True);frequency={int(k):int(v)for k,v in zip(types,counts)};mass,receipt=paired_mass(frequency,32,p['complexity']['reachable_paired_unions_per_dose_fit_max']);assert all((pair&((1<<m)-1))|(pair>>m)==pair>>m for pair in mass)
        needed={0}|{pair&((1<<m)-1)for pair in mass}|{pair>>m for pair in mass};mapping={}
        for mask in sorted(needed):
            repaired=F.copy();deleted={classes[j]for j in range(m)if mask&(1<<j)}
            for a,b in itertools.combinations(range(169),2):
                if(b//13-a//13,b%13-a%13)in deleted:repaired[a,b]=repaired[b,a]=0
            digest=hashlib.sha256(repaired.tobytes()).hexdigest();existing=[r['group']for r in supports if r['family']==family and r['support_sha256']==digest]
            if existing:group=existing[0];assert np.array_equal(repaired,arrays[group])
            else:
                assert family_counts[family]<p['complexity']['distinct_full_supports_per_family_max'],'Support complexity gate exceeded'
                group=len(arrays);arrays.append(repaired);family_counts[family]+=1;dest=OUT/f'support{group}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'F.npz',F=repaired);supports.append(dict(group=group,family=family,support_sha256=digest,F_npz_sha256=sha(dest/'F.npz'),members=[]))
            mapping[str(mask)]=group;supports[group]['members'].append(dict(index=index,mask=mask))
        seed=p['validation']['learner_call_seeds'][index];indices=np.random.default_rng(seed).choice(len(population),32,replace=False)
        for augment,masks in[(False,raw_masks),(True,aug_masks)]:
            rebuilt,cost=repair(population[indices],F,finite,augment);mask=int(np.bitwise_or.reduce(masks[indices]));assert np.array_equal(rebuilt,arrays[mapping[str(mask)]]);fixtures.append(dict(index=index,seed=seed,indices=indices.tolist(),augment=augment,union_mask=mask,cost=cost))
        np.savez_compressed(folder/'source-patterns.npz',raw_masks=raw_masks,augmented_masks=aug_masks,paired_masks=pair_masks,classes=np.array(classes,np.int16).reshape(-1,2),unique_views_per_board=np.array(viewcounts,np.uint8))
        rows.append(dict(index=index,family=family,status='complete_source_only',class_count=m,classes=[list(o)for o in classes],source_complement_size=len(population),frequency={str(k):v for k,v in frequency.items()},draw_masses={str(k):str(v)for k,v in sorted(mass.items())},mask_to_support=mapping,distribution_receipt=receipt,source_patterns_sha256=sha(folder/'source-patterns.npz'),model_sha256=sha(folder/'model.npz')));print('SOURCE_OUTCOME',index,m,len(mass),len(needed),flush=True)
    save(OUT/'source-validation.json',dict(status='passed',basis_cells=144,all_D4_source_views_literal_valid=True,all_concrete_learner_calls=fixtures,no_target_truth_labels_or_scores_read=True))
    save(OUT/'source-outcomes.json',dict(status='complete_source_only',utc=now(),banks=rows,supports=supports,distinct_supports_per_family=family_counts,all_positive_mass_paired_outcomes_retained=True,baseline_zero_mask_retained=True,source_validation_sha256=sha(OUT/'source-validation.json'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
