"""Reserve uniform unique N15 superqueen partial inputs before upward-transfer fitting."""
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3o';OUT=ROOT/'artifacts/research/20261114/loki/h3o';BASE=ROOT/'artifacts/research/20261114/loki/h3n'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not(OUT/'dataset.json').exists();train=dict(np.load(BASE/'training.npz'));target=np.load(OUT/'solutions-n15.npy')
    assert json.loads((OUT/'feasibility.json').read_text())['status']=='validated_before_fitting'
    positions=np.array([np.flatnonzero(row) for row in target]);counts=np.zeros(226**3,dtype=np.int32)
    for clues in [1,2,3]:
        subsets=np.array(list(itertools.combinations(range(15),clues)))
        for lo in range(0,len(positions),2048):
            cells=positions[lo:lo+2048,subsets]+1;keys=sum(cells[:,:,a].astype(np.int64)*226**(2-a) for a in range(clues)).ravel()
            np.add.at(counts,keys,1)
    universe=np.flatnonzero(counts);keys=np.sort(np.random.default_rng(2026112506).permutation(universe)[:4096]);inputs=np.zeros((4096,225),bool)
    for power in [2,1,0]:
        cells=keys//226**power%226;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    selected_counts=counts[keys]
    for given,count in zip(inputs,selected_counts):assert target[:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'input-universe.npz',keys=universe,counts=counts[universe])
    np.savez_compressed(OUT/'training.npz',n14_all=train['n14_all'])
    np.savez_compressed(OUT/'data.npz',all_solutions=target,test_keys=keys,test_inputs=inputs,test_solution_counts=selected_counts,prior_target_test_keys=np.zeros(0,np.int64))
    report=dict(status='reserved_before_fitting',utc=dt.datetime.now(dt.timezone.utc).isoformat(),task='superqueens:queen+knight attacks',
        transfer='N14 to N15 upward transfer; source-global zero offsets retained without pruning.',source_counts=dict(n14_full=len(train['n14_all'])),
        target_n=15,target_solution_universe=len(target),target_unique_inputs=4096,target_input_seed=2026112506,target_eligible_pool=len(universe),
        target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(selected_counts.sum()),prior_target_input_overlap=0,
        target_distribution='Uniform unique valid1/2/3clue N15 superqueen inputs. This is the first N15 evaluation cohort.',
        data_sha256=sha(OUT/'data.npz'),training_sha256=sha(OUT/'training.npz'),source_sha256=sha(__file__),parent_training_sha256=sha(BASE/'training.npz'),
        target_solutions_sha256=sha(OUT/'solutions-n15.npy'),input_universe_sha256=sha(OUT/'input-universe.npz'))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
