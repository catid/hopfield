"""Truth-blind balancing with memory-mapped packed candidate predictions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import sys
import numpy as np


def main():
    request=json.loads(sys.stdin.read())
    assert set(request)=={'bits_path','candidate_ids','board_count','candidates','queried'}
    rows=[request['candidate_ids'].index(cid) for cid in request['candidates']]
    packed=np.load(request['bits_path'],mmap_mode='r')
    predictions=np.unpackbits(packed[rows],axis=1,count=request['board_count'],bitorder='little').astype(bool)
    counts=predictions.sum(0,dtype=np.int64);imbalance=abs(2*counts-len(rows));allowed=np.ones(request['board_count'],bool);allowed[request['queried']]=False
    best=int(imbalance[allowed].min());ties=np.flatnonzero(allowed & (imbalance==best));index=int(ties[0])
    result=dict(query_index=index,imbalance=best,optimal_board_count=len(ties),candidate_predictions={str(cid):bool(predictions[j,index]) for j,cid in enumerate(request['candidates'])})
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
