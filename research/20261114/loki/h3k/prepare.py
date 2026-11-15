"""Reserve independent source-bank draws and a new N13 behavior cohort."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3k';OUT=ROOT/'artifacts/research/20261114/loki/h3k'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not(OUT/'dataset.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    gp=ROOT/'artifacts/research/20261114/loki/h3g/data.npz';ip=ROOT/'artifacts/research/20261114/loki/h3i/data.npz';jp=ROOT/'artifacts/research/20261114/loki/h3j/data.npz'
    g=dict(np.load(gp));i=dict(np.load(ip));j=dict(np.load(jp))
    excluded=np.union1d(g['largest_bank_indices'],i['union_indices']);available=np.setdiff1d(np.arange(14200),excluded)
    seeds=2026111500+np.arange(512);draws=np.stack([np.random.default_rng(int(seed)).choice(available,16,replace=False) for seed in seeds])
    assert not np.isin(draws,excluded).any() and all(len(np.unique(row))==16 for row in draws)
    universe=j['full_universe_keys'];eligible=universe[~np.isin(universe,j['test_keys'])]
    test_keys=np.sort(np.random.default_rng(2026112500).permutation(eligible)[:2048]);inputs=np.zeros((len(test_keys),169),bool)
    for power in [2,1,0]:
        cells=test_keys//170**power%170;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    counts=j['full_universe_counts'][np.searchsorted(universe,test_keys)]
    for given,count in zip(inputs,counts):assert j['all_solutions'][:,given].all(1).sum()==count>0
    np.savez_compressed(OUT/'data.npz',source_solutions=i['all_solutions'],source_excluded_indices=excluded,source_available_indices=available,
        source_draw_seeds=seeds,source_draw_indices=draws,all_solutions=j['all_solutions'],test_keys=test_keys,test_inputs=inputs,
        test_solution_counts=counts,prior_target_test_keys=j['test_keys'])
    report=dict(status='reserved_before_source_fits_or_target_search',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_n=12,target_n=13,independent_source_draws=512,source_sizes=[1,2,4,8,16],source_seed_first=int(seeds[0]),source_seed_last=int(seeds[-1]),
        excluded_prior_restricted_patterns=len(excluded),available_source_patterns=len(available),
        source_sampling='Each draw independently samples16 distinct patterns without replacement from the fixed eligible pool. Sizes1/2/4/8/16 are prefixes within a draw; overlap across independent draws is allowed.',
        exclusion_limit='H3g512 and all H3i union patterns are excluded. Earlier full-universe studies used all N12 patterns, making never-seen-pattern exclusion impossible; no such novelty claim is made.',
        behavior_selection='Exactly the first6 source draw seeds at sizes1 and8, all12 logical models, regardless of success/failure. Three search seeds per distinct solver energy with explicit identity mappings.',
        target_input_seed=2026112500,target_unique_inputs=2048,target_eligible_pool=len(eligible),target_prior_input_overlap=0,
        target_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_target_completions=int(counts.sum()),
        target_distribution='Uniform unique valid1/2/3clue N13 inputs excluding H3j evaluation inputs. Target labels are evaluation-only.',
        data_sha256=sha(OUT/'data.npz'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),gp,ip,jp]})
    (OUT/'dataset.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
