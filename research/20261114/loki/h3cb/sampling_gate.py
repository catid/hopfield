"""Exercise full permutation/bank accounting with RNG replaced by fixed stubs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sampling
from common import SRC,read,save,sha,load
from ledger import Ledger


def run(folder):
    folder.mkdir(parents=True,exist_ok=False)
    ordinary=np.arange(8,dtype=np.uint8)[:,None]
    truth_rows=[dict(candidate=c,eligible=c<1522) for c in range(1541)]
    populations={c:ordinary.copy() for c in range(1522)}
    pred=np.zeros((1541,8),bool);pred[:1522]=True
    scores={a:['1' if c<1522 else None for c in range(1541)] for a in ['H','T']}
    p=dict(truths=truth_rows,budget=dict(planned_cases=1522),seed_bases=dict(permutation=11,source=100))
    calls=[];original_rng=np.random.default_rng
    class Fixed:
        def __init__(self,seed):self.seed=seed
        def permutation(self,eligible):
            assert self.seed==11 and eligible==list(range(1522));calls.append(('permutation',self.seed));return np.array(eligible[::-1])
        def choice(self,M,count,replace):
            assert M==count==8 and replace is False;calls.append(('choice',self.seed));return np.roll(np.arange(8),self.seed%8)
    def factory(seed):return Fixed(seed)
    statuses=[dict(case=i,source='pending',B='pending',E='pending') for i in range(1522)]
    ledger=Ledger(folder/'query-ledger.jsonl');retained=[]
    try:
        np.random.default_rng=factory
        sources,frequencies,boards=sampling.draw(p,ordinary,pred,populations,scores,{},ledger,folder,statuses,sources=retained)
    finally:np.random.default_rng=original_rng;ledger.close()
    chain=load(SRC.parent/'h3bj/verify_public.py','h3cb_sampling_gate_chain').chain
    events=chain((folder/'query-ledger.jsonl').read_bytes());checks=0
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    schedule=events[0]
    check('schedule_once_oracle_only_before_all_sources',schedule['kind']=='truth_schedule' and schedule['data']==dict(
        permutation_seed=11,eligible_candidates=list(range(1522)),permutation=list(range(1521,-1,-1)),cases=1522,
        truth_permutation_draws=1,individual_truth_draws=0,without_replacement=True,policy_access=False))
    check('fixed_RNG_stubs_only',calls==[('permutation',11)]+[('choice',100+i) for i in range(1522)])
    digest=hashlib.sha256(b''.join(bytes(row) for row in ordinary)).hexdigest()
    for case,(source,event) in enumerate(zip(sources,events[1:-1])):
        chosen=np.roll(np.arange(8),(100+case)%8).tolist()
        data=dict(case=case,candidate=1521-case,bank=0,truth_permutation_position=case,truth_schedule_record_sha256=schedule['sha256'],
            seed=100+case,source_population=8,indices=chosen,columns_zero_based=ordinary[chosen].tolist(),initial_candidates=list(range(1522)),
            unordered_sample_sha256=digest,prior_same_study_cases=list(range(case)),historical_exact8_matches=[])
        check('full_source_indices_permutation_coverage_and_duplicates',event['kind']=='source_bank' and event['data']==data and
            source==dict(data,source_sequence=event['sequence'],source_record_sha256=event['sha256']))
    check('all_sources_before_any_query_and_exact_accounting',len(events)==1524 and events[-1]['kind']=='all_source_banks_drawn' and events[-1]['data']==dict(
        cases=1522,truth_permutation_draws=1,individual_truth_draws=0,new_source_draws=1522,source_positive_union_appearances=12176,
        positive_arm_appearances=24352,distinct_N12_boards=8,source_banks_sha256=sha(folder/'source-banks.json')) and
        len(retained)==1522 and frequencies=={c:1 for c in range(1522)} and len(boards)==8 and all(r['source']=='complete' for r in statuses))
    # A technical exception after one bank preserves that bank and all pending statuses.
    failed=folder/'injected-failure';failed.mkdir();fledger=Ledger(failed/'query-ledger.jsonl');partial=[]
    fstatuses=[dict(case=i,source='pending') for i in range(1522)]
    class Failing(Fixed):
        def choice(self,M,count,replace):
            if self.seed==101:raise RuntimeError('Fixed injected source interruption')
            return super().choice(M,count,replace)
    try:
        np.random.default_rng=lambda seed:Failing(seed)
        try:sampling.draw(p,ordinary,pred,populations,scores,{},fledger,failed,fstatuses,sources=partial)
        except RuntimeError as error:failure=str(error)
        else:raise AssertionError('Fixed interruption escaped')
    finally:np.random.default_rng=original_rng;fledger.close()
    stopped=chain((failed/'query-ledger.jsonl').read_bytes())
    check('interruption_retains_paid_source_and_pending_statuses',len(partial)==1 and len(stopped)==2 and
        fstatuses[0]['source']=='complete' and all(r['source']=='pending' for r in fstatuses[1:]) and not (failed/'source-banks.json').exists())
    save(failed/'receipt.json',dict(error=failure,completed_sources=partial,statuses=fstatuses,whole_study_incomplete=True))
    result=dict(status='passed_full_fixed_permutation_and_source_accounting_gate',checks=checks,synthetic_truth_schedule_count=1,
        synthetic_source_banks=1522,synthetic_duplicate_bank_cases=1521,RNG_backend_calls=0,fresh_truth_draws=0,fresh_source_draws=0,actual_oracle_calls=0,
        partial_source_retention_verified=True,source_sha256={n:sha(SRC/n) for n in ['sampling.py','sampling_gate.py','ledger.py']})
    save(folder/'summary.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);run(parser.parse_args().folder)
