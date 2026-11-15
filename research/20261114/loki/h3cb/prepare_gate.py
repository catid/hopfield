"""Run the full source preparation with all prospective RNG draws forbidden."""
import argparse
from pathlib import Path
import time
import numpy as np
from common import ROOT,SRC,read,save,sha
import prepare
from audit_sources import verify_prior


def run(folder):
    folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter();attempts=checks=0
    prior=read(SRC/'prior-gate.json');seeds=read(SRC/'seed-audit.json')
    previous=read(SRC.parent/'h3bz/protocol.json')
    p=dict(normalizers=previous['normalizers'],candidate_prior=previous['candidate_prior'],truths=prior['prior_rows'],
           fitted_prior_rows=prior['fitted_prior_rows'],historical_protocol_sha256=seeds['historical_protocol_sha256'],
           seed_bases=dict(permutation=2027700000,source=2027800000))
    for name in ['prior-gate.json','fitted-weight-gate.json','two-score-gate.json','seed-audit.json']:
        save(folder/'source-gates'/name,read(SRC/name))
    def deny(*args,**kwargs):
        nonlocal attempts
        attempts+=1;raise AssertionError('Prospective RNG forbidden in preparation gate')
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    original,previous=np.random.default_rng,prepare.OUT
    try:
        np.random.default_rng=deny;prepare.OUT=folder
        ordinary,pred,pop,boards,bits,scores,history=prepare.run(p)
        independent,prediction,cumulative=verify_prior(ROOT,folder,p,read,sha,check)
        check('prepared_source_universe_and_masks',np.array_equal(ordinary,independent) and np.array_equal(pred,prediction))
        check('prepared_role_files',all(read(folder/a/'selector-scores.json')==scores['H'] and
              read(folder/a/'tree-scores.json')==scores['T'] for a in ['B','E']))
        check('prepared_all_source_populations',all(np.array_equal(pop[c],ordinary[pred[c]]) for c in range(1541)))
        expected={}
        for row in read(ROOT/'results/research/loki/h3bm/source-pairs.json'):
            expected.setdefault(row['unordered_sample_sha256']['8'],[]).append(dict(study='H3bm',case=row['case'],candidate=row['candidate']))
        for study in ['h3bq','h3br','h3bx','h3by','h3bz','h3ca']:
            for row in read(ROOT/f'results/research/loki/{study}/source-banks.json'):
                expected.setdefault(row['unordered_sample_sha256'],[]).append(dict(study='H'+study[1:],case=row['case'],candidate=row['candidate']))
        check('complete_history_including_H3ca',history==expected==read(folder/'historical-eight-banks.json'))
        check('no_RNG_attempts',attempts==0)
    finally:np.random.default_rng=original;prepare.OUT=previous
    result=dict(status='passed_full_preparation_with_RNG_forbidden',checks=checks,RNG_API_attempts=attempts,
                fresh_truth_draws=0,fresh_source_draws=0,actual_oracle_calls=0,
                source_sha256={n:sha(SRC/n) for n in ['prepare.py','prepare_gate.py','audit_sources.py','source_gate.py']},
                elapsed_seconds=time.perf_counter()-started)
    save(folder/'summary.json',result);print(result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);run(parser.parse_args().folder)
