"""Ask the board rejecting most alternatives if the current MAP guide is true."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import sys
import numpy as np


def main():
    request=json.loads(sys.stdin.read());assert set(request)=={'bits_path','scores_path','board_count','candidates','queried'}
    with open(request['scores_path']) as f:scores=json.load(f)
    candidates=request['candidates'];best=min(int(scores[cid]) for cid in candidates);map_ties=[cid for cid in candidates if int(scores[cid])==best];guide=min(map_ties)
    packed=np.load(request['bits_path'],mmap_mode='r');predictions=np.unpackbits(packed[candidates],axis=1,count=request['board_count'],bitorder='little').astype(bool)
    disagreements=(predictions!=predictions[candidates.index(guide)]).sum(0,dtype=np.int64);allowed=np.ones(request['board_count'],bool);allowed[request['queried']]=False;best_disagreement=int(disagreements[allowed].max());ties=np.flatnonzero(allowed & (disagreements==best_disagreement))
    result=dict(MAP_tied_candidates=map_ties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=best_disagreement,optimal_board_count=len(ties))
    if best_disagreement==0:result.update(query_index=None,candidate_predictions=None,stop_reason='no_informative_unqueried_board_remaining_identity_unresolved')
    else:
        index=int(ties[0]);result.update(query_index=index,candidate_predictions={str(cid):bool(predictions[j,index]) for j,cid in enumerate(candidates)},stop_reason=None)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
