"""Reserve a fresh raw bank, its dihedral closure, and union-incompatible inputs."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3i';OUT=ROOT/'artifacts/research/20261114/loki/h3i'
spec=importlib.util.spec_from_file_location('patterns',SRC.parent/'h3g/prepare.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not(OUT/'dataset.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    epath=ROOT/'artifacts/research/20261114/loki/h3e/data.npz';gpath=ROOT/'artifacts/research/20261114/loki/h3g/data.npz'
    e=dict(np.load(epath));g=dict(np.load(gpath));solutions=e['solutions'];lookup={row.tobytes():i for i,row in enumerate(solutions)}
    available=np.setdiff1d(np.arange(len(solutions)),g['largest_bank_indices'])
    raw=np.random.default_rng(2026111415).permutation(available)[:1024];small=raw[:128]
    images=[];image_indices=[]
    for k in range(4):
        for mirror in [False,True]:
            boards=solutions[small].reshape(-1,12,12)
            if mirror:boards=boards[:,:,::-1]
            transformed=np.rot90(boards,k,axes=(1,2)).reshape(-1,144).copy()
            ids=np.array([lookup[row.tobytes()] for row in transformed]);images.append(transformed);image_indices.append(ids)
            assert np.array_equal(transformed,solutions[ids])
    augmentation=np.unique(np.concatenate(image_indices));union=np.union1d(raw,augmentation)
    assert np.isin(small,augmentation).all() and not np.isin(raw,g['largest_bank_indices']).any()
    covered,_=P.training_rows(solutions[union]);prior=np.union1d(e['test_keys'],g['test_keys'])
    universe=e['full_universe_keys'];eligible=~np.isin(universe,covered)&~np.isin(universe,prior)
    pool=universe[eligible];count=min(4096,len(pool));assert count>0
    selected=np.sort(np.random.default_rng(2026111416).permutation(pool)[:count]);inputs=P.decode(selected)
    full_counts=[]
    for given in inputs:
        compatible=solutions[:,given].all(1);assert compatible.any() and not compatible[union].any()
        full_counts.append(int(compatible.sum()))
    assert not np.isin(selected,covered).any() and not np.isin(selected,prior).any()
    np.savez_compressed(OUT/'data.npz',all_solutions=solutions,raw128_indices=small,raw1024_indices=raw,
        augmented128_indices=augmentation,dihedral_image_indices=np.stack(image_indices),union_indices=union,
        test_keys=selected,test_inputs=inputs,test_solution_counts=np.array(full_counts),prior_test_keys=prior,
        prior_training_bank_indices=g['largest_bank_indices'])
    report=dict(status='reserved_and_validated_before_construction',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        pattern_seed=2026111415,input_seed=2026111416,raw128_patterns=128,raw1024_patterns=1024,
        augmented_patterns_before_deduplication=1024,augmented_unique_patterns=len(augmentation),union_unique_patterns=len(union),
        raw_training_pattern_overlap_with_h3g_bank=0,eligible_pool=len(pool),requested_inputs=4096,test_unique=count,
        test_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},
        union_compatible_inputs=0,prior_h3e_h3g_input_overlap=0,all_compatible_completions=int(sum(full_counts)),
        distribution='Uniform sample of unique 1/2/3-clue inputs incompatible with the union of all four arms training patterns; excludes prior H3e and H3g test inputs.',
        independent_replication='One new raw bank draw, disjoint from the previous H3g512 bank; three later seeds repeat search only. Earlier studies used the full solution universe.',
        no_model_or_search_outcomes=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SRC.parent/'h3g/prepare.py',epath,gpath]},
        data_sha256=sha(OUT/'data.npz'))
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
