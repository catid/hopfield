"""Reserve inputs with no compatible training-bank pattern before any model fit."""
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'research/20261114/loki/h3g'
OUT=ROOT/'artifacts/research/20261114/loki/h3g'
BASE=ROOT/'artifacts/research/20261114/loki/h3e'


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def decode(keys):
    result=np.zeros((len(keys),144),bool)
    for power in [2,1,0]:
        cells=keys//145**power%145;present=cells>0;result[np.flatnonzero(present),cells[present]-1]=True
    return result


def training_rows(solutions):
    positions=np.array([np.flatnonzero(row) for row in solutions]);keys=[];ids=[]
    for k in [1,2,3]:
        subsets=np.array(list(itertools.combinations(range(12),k)))
        cells=positions[:,subsets]+1
        keys.append(sum(cells[:,:,j].astype(np.int64)*145**(2-j) for j in range(k)).ravel())
        ids.append(np.repeat(np.arange(len(solutions)),len(subsets)))
    return np.concatenate(keys),np.concatenate(ids)


def main():
    assert not(OUT/'dataset.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    base=dict(np.load(BASE/'data.npz'));all_solutions=base['solutions']
    order=np.random.default_rng(2026111413).permutation(len(all_solutions))[:512]
    largest_keys,_=training_rows(all_solutions[order]);covered=np.unique(largest_keys)
    universe=base['full_universe_keys'];eligible=~np.isin(universe,covered)&~np.isin(universe,base['test_keys'])
    pool=universe[eligible];clues=base['full_universe_clues'][eligible]
    assert len(pool)==111781 and {k:int((clues==k).sum()) for k in [1,2,3]}=={1:0,2:221,3:111560}
    rng=np.random.default_rng(2026111414)
    selected=np.concatenate([rng.permutation(pool[clues==2])[:16],rng.permutation(pool[clues==3])[:8176]])
    selected=np.sort(selected);inputs=decode(selected);assert not np.isin(selected,covered).any()
    assert not np.isin(selected,base['test_keys']).any()
    in_largest=np.zeros(len(all_solutions),bool);in_largest[order]=True
    full_counts=[];outside_counts=[]
    for given in inputs:
        compatible=all_solutions[:,np.flatnonzero(given)].all(1)
        assert not compatible[in_largest].any() and compatible.any()
        full_counts.append(int(compatible.sum()));outside_counts.append(int(compatible[~in_largest].sum()))
    assert full_counts==outside_counts
    arrays=dict(all_solutions=all_solutions,test_keys=selected,test_inputs=inputs,test_solution_counts=np.array(full_counts),
        largest_bank_indices=order,source_h3e_test_keys=base['test_keys'])
    banks={}
    for n in [32,128,512]:
        ix=order[:n];solutions=all_solutions[ix];keys,ids=training_rows(solutions)
        counts=np.full((n,144),298,np.int64)
        for power in [2,1,0]:
            cells=keys//145**power%145;present=cells>0
            np.add.at(counts,(ids[present],cells[present]-1),-1)
        assert np.array_equal(counts,np.where(solutions,231,298))
        assert not np.isin(selected,keys).any()
        arrays[f'bank{n}_indices']=ix;arrays[f'bank{n}_free_counts']=counts
        arrays[f'bank{n}_expanded_keys']=keys;arrays[f'bank{n}_expanded_label_indices']=ids
        banks[str(n)]=dict(complete_patterns=n,expanded_rows=len(keys),unique_partial_inputs=len(np.unique(keys)),
                          label_multiplicity_each=298,free_cell_weights=int(counts.sum()))
    np.savez_compressed(OUT/'data.npz',**arrays)
    report=dict(status='reserved_and_validated_before_fits',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        n=12,complete_solution_universe=14200,banks=banks,pattern_seed=2026111413,input_seed=2026111414,
        eligible_pool=111781,eligible_by_clues={'1':0,'2':221,'3':111560},
        test_unique=8192,test_by_clues={'1':0,'2':16,'3':8176},largest_bank_compatible_test_inputs=0,
        test_completions_in_largest_bank=0,prior_h3e_test_input_overlap=0,
        conditioning='Test inputs are sampled proportionally from 1–3clue inputs incompatible with every512training-bank pattern, excluding every priorH3e evaluation input. This changes the distribution and removes all one-clue inputs.',
        labels='Each bank pattern contributes all298 one/two/three-clue masks; each such(input,label) row occurs once. Ambiguous inputs may have several labels within the restricted bank; labels outside the selected bank never enter fitting or calibration.',
        checks=['Nested32/128/512 pattern banks','Every test input has at least one valid full-universe completion',
                'Every valid completion of every test input lies outside largest512bank','No test partial input occurs in any training expansion',
                'No test partial input equals any priorH3e test input','Exact free-cell counts reconstructed from every expanded training row'],
        source_dataset_sha256=sha(BASE/'data.npz'),data_sha256=sha(OUT/'data.npz'),source_sha256=sha(__file__),no_fit_or_search_outcomes=True)
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
