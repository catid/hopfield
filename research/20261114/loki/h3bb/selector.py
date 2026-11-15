"""Query selector with no true-ID or oracle argument; JSON-only process interface."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import sys
import numpy as np


def choose(predictions, candidate_ids, candidates, queried):
    rows=[candidate_ids.index(cid) for cid in candidates]
    positives=predictions[rows].sum(0,dtype=np.int64)
    imbalance=abs(2*positives-len(rows))
    allowed=np.ones(predictions.shape[1],bool)
    allowed[queried]=False
    best=int(imbalance[allowed].min())
    ties=np.flatnonzero(allowed & (imbalance==best))
    index=int(ties[0])
    return dict(query_index=index,imbalance=best,optimal_board_count=len(ties),
                candidate_predictions={str(cid):bool(predictions[row,index]) for cid,row in zip(candidates,rows)})


def main():
    request=json.loads(sys.stdin.read())
    assert set(request)=={'predictions_path','candidate_ids','candidates','queried'}
    predictions=np.load(request['predictions_path'])
    result=choose(predictions,request['candidate_ids'],request['candidates'],request['queried'])
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
