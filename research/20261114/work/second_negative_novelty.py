"""H2ad predeclared exact-row novelty and previous-witness margin endpoint."""
from fractions import Fraction as Q
import hashlib
import json
import time
import numpy as np
import study as S
import feasibility_features as F
import verify_training_feasibility as V
from mine_second_hard_negatives import PP,DEST,TRAIN,PRIOR,protocol


def prefix(row):
    a=np.asarray(row[:18],np.float64).copy();a[a==0]=0;return a.tobytes()


def exact_negative(t,y,z,a,muoffs,aoffs,h):
    def feature(board):
        p=np.where(t==1,board,t>=2).astype(np.int8);f=[Q(int(v)) for v in F.energy_features(t,board,a,muoffs,aoffs)[:18]]
        for k,(dr,dc) in enumerate(aoffs):
            l,r=F.slices(t.shape,int(dr),int(dc));mask=(p[l]*p[r]).astype(bool);f.append(-sum((Q.from_float(float(v)) for v in a[k][l][mask]),Q(0)))
        return f
    return tuple((b-c)/int(h) for b,c in zip(feature(z),feature(y)))


def main():
    started=time.perf_counter();p=protocol();audit=json.loads((DEST/'mining-audit.json').read_text());records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    for seed in range(3):
        with np.load(DEST/f'negatives-seed{seed}.npz') as z:ix=z['indices'];boards=z['boards'];ham=z['hamming']
        with np.load(PRIOR/f'negatives-seed{seed}.npz') as z:old_ix=z['indices'];old_boards=z['boards'];old_ham=z['hamming']
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        with np.load(S.ROOT/p['initial_models'][seed]['path']) as z:muoffs=z['mu_offsets'];aoffs=z['aoffs']
        previous=np.load(PRIOR/f'seed{seed}/candidate-theta.npy');assert S.sha(PRIOR/f'seed{seed}/candidate-theta.npy')==p['previous_witnesses'][str(seed)]['sha256'];theta=previous.astype(V.LD)
        new_rows=[exact_negative(tokens[i],labels[i],z,attention[i],muoffs,aoffs,h) for i,z,h in zip(ix,boards,ham)];keys={prefix([float(v) for v in row]) for row in new_rows};prior_rows=set();old_constraints=0;exact_old_comparisons=0
        # Exact equality implies identical rounded first18 rational coordinates.
        # Float64 keys only filter candidates; full Fraction equality decides novelty.
        for i,(t,y,a) in enumerate(zip(tokens,labels,attention)):
            rows,kind,flat=F.constraints(t,y,a,muoffs,aoffs);old_constraints+=len(rows)
            for j,row in enumerate(rows):
                if prefix(row) in keys:prior_rows.add(tuple(F.exact_row(t,y,a,muoffs,aoffs,int(kind[j]),int(flat[j]))));exact_old_comparisons+=1
        for i,z,h in zip(old_ix,old_boards,old_ham):
            row=(F.energy_features(tokens[i],z,attention[i],muoffs,aoffs)-F.energy_features(tokens[i],labels[i],attention[i],muoffs,aoffs))/h
            if prefix(row) in keys:prior_rows.add(exact_negative(tokens[i],labels[i],z,attention[i],muoffs,aoffs,h));exact_old_comparisons+=1
        old_board_keys={(int(i),b.tobytes()) for i,b in zip(old_ix,old_boards)};seen_rows=set(prior_rows);seen_boards=set(old_board_keys);mapping=[]
        norm=abs(theta).max();m=V.LD(256*900+1000);u=V.LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(V.LD(1),norm)+1)
        for j,(i,z,h,row) in enumerate(zip(ix,boards,ham,new_rows)):
            key=(int(i),z.tobytes());bn=key not in seen_boards;rn=row not in seen_rows;seen_boards.add(key);seen_rows.add(row)
            t=tokens[i:i+1];y=labels[i:i+1];a=attention[i:i+1].astype(V.LD);margin=(V.energy(t,z[None],a,theta,muoffs,aoffs)-V.energy(t,y,a,theta,muoffs,aoffs))[0]/h
            exact_margin=sum(v*Q.from_float(float(w)) for v,w in zip(row,previous));assert abs(float(exact_margin)-float(margin))<float(allowance)
            rational_hash=hashlib.sha256(json.dumps([[v.numerator,v.denominator] for v in row],separators=(',',':')).encode()).hexdigest()
            mapping.append({'negative_row':j,'training_index':int(i),'new_board_vs_all_previous':bn,'new_exact_normalized_constraint_vs_all_previous':rn,'exact_constraint_sha256':rational_hash,'previous_unscaled_normalized_margin':float(margin),'previous_exact_margin_numerator':str(exact_margin.numerator),'previous_exact_margin_denominator':str(exact_margin.denominator),'violates_previous_unit_margin_exact':exact_margin<1,'direct_margin_allowance':float(allowance),'violates_with_conservative_direct_allowance':bool(margin+allowance<1)})
        path=DEST/f'all-negatives-seed{seed}.npz';np.savez_compressed(path,indices=np.concatenate([old_ix,ix]),boards=np.concatenate([old_boards,boards]),hamming=np.concatenate([old_ham,ham]),source_round=np.concatenate([np.ones(len(old_ix),np.int8),np.full(len(ix),2,np.int8)]))
        records.append({'seed':seed,'previous_negative_rows':len(old_ix),'new_mined_negative_rows':len(ix),'genuinely_new_boards':sum(r['new_board_vs_all_previous'] for r in mapping),'genuinely_new_exact_constraint_rows':sum(r['new_exact_normalized_constraint_vs_all_previous'] for r in mapping),'previous_unit_margin_violations':sum(r['violates_previous_unit_margin_exact'] for r in mapping),'conservatively_verified_violations':sum(r['violates_with_conservative_direct_allowance'] for r in mapping),'retained_old_base_constraints':old_constraints,'retained_combined_negative_rows':len(old_ix)+len(ix),'all_negative_archive_sha256':S.sha(path),'prior_exact_candidate_comparisons':exact_old_comparisons,'mapping':mapping})
    S.save(DEST/'novelty-audit.json',{'status':'passed','utc':S.utc(),'protocol_sha256':S.sha(PP),'records':records,'new_independent_test_inputs':0,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__),'scope':'Board novelty keyed by original training index and full board. Constraint novelty by exact Fraction equality against every H2w row, every H2ab negative and earlier new rows. Rounded first18 coordinates are only a conservative candidate filter. Margin violations evaluated exactly under previous UNSCALED H2ab feasibility witnesses; direct extended-precision energies independently cross-check them. All negative rows retained, no deduplication. Absence of violated rows is reported explicitly and cannot motivate a claim of repairing ranking violations.'})


if __name__=='__main__':main()
