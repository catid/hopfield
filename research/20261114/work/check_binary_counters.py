"""Source-only independent bit-transition counting for frozen H2aa helper."""
import itertools
import time
import numpy as np
import study as S
import binary_coordinate as C
from gate_lp_relaxation import model_for
from gate_binary_coordinate import DEST


def main():
    torch,_,_=S.imports();started=time.perf_counter();rows=[];tok=np.array([[2,1,0],[1,1,3],[0,1,1]],np.int64);free=tok==1
    boards=np.repeat((tok>0)[None],32,axis=0).astype(np.float64)
    for b,bits in zip(boards,itertools.product([0,1],repeat=5)):b[free]=bits
    for seed in range(3):
        rng=np.random.default_rng(2026111511+seed);raw=rng.random((5,3,3)).astype(np.float32);raw/=raw.sum(0,keepdims=True)
        model=model_for(seed,'cpu',torch.float64);t=torch.tensor(np.repeat(tok[None],32,0));a=torch.tensor(np.repeat(raw[None],32,0),dtype=torch.float64);p=torch.tensor(boards);cache=C.BinaryQuadratic(model,t,a);out=C.solve(cache,initial=p,trace=True)
        totals=torch.zeros(32,dtype=torch.int64);adds=totals.clone();removes=totals.clone()
        for k,event in enumerate(out['history']):
            before=event['before'];after=out['history'][k+1]['before'] if k+1<len(out['history']) else out['prediction'];actual=before!=after
            assert torch.equal(actual,event['flip']) and torch.all((before==0)|(before==1)) and torch.all((after==0)|(after==1))
            totals+=actual.sum((1,2));adds+=(actual&(before==0)).sum((1,2));removes+=(actual&(before==1)).sum((1,2))
        assert torch.equal(totals,out['flip_count']) and torch.equal(totals,adds+removes)
        assert torch.equal(p.sum((1,2))-out['prediction'].sum((1,2)),(removes-adds).double())
        rows.append({'seed':seed,'source_boards':32,'phase_board_transition_checks':32*len(out['history']),'actual_bit_transitions':int(totals.sum()),'added_bits':int(adds.sum()),'removed_bits':int(removes.sum()),'recorded_counters_equal_independently_counted_actual_transitions':True})
    S.save(DEST/'source-counter-verification.json',{'status':'passed','rows':rows,'seconds':time.perf_counter()-started,'helper_sha256':S.sha(C.__file__),'source_sha256':S.sha(__file__),'scope':'Source-only96exhaustiveboardstates; transitionsreadfromsuccessiveactualphaseboards independentlyofhelpercounter. No targettrajectoryreplay. Frozenhelper incrementscountbychangedmask andappliesexact1-p atthosebinarycells, establishingF=R+A.'})


if __name__=='__main__':main()
