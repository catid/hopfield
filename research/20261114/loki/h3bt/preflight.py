"""Restriction-lemma and online-cache fixtures before archived-cohort intervention."""
import tempfile
import time
from pathlib import Path
import numpy as np
from common import ROOT,SRC,read,save,sha,now
from compiler import Budget
import synthetic_gate
import policy
from restriction import gate as restriction_gate
from cost_gate import run as cost_gate


def main():
    start=time.perf_counter();assert not(SRC/'protocol.json').exists();budget=Budget();synthetic_gate.OUT=SRC;toy=synthetic_gate.run(budget);(SRC/'synthetic-gate.json').rename(SRC/'preflight-synthetic-gate.json');restriction=restriction_gate(toy['fixtures'])
    with tempfile.TemporaryDirectory(prefix='h3bt-prefreeze-') as temporary:
        policy.OUT=Path(temporary);pred=np.array([[bool(signature&(1<<cid)) for signature in [0,5,10,3,12,15]] for cid in range(4)]);compiler=policy.OnlineCompiler(pred,['1','2','3','4'],'a'*64,budget,100000);row,proof=compiler.get([0,1,2,3]);states=budget.states;splits=budget.splits;again=compiler.get([0,1,2,3]);assert again[0]==row and again[1]==proof and budget.states==states and budget.splits==splits and len(compiler.rows)==1 and proof['optimal_cost']=='50' and not pred.flags.writeable
        for local,cid in enumerate(proof['candidates']):
            key=proof['root_state'];depth=0
            while proof['states'][key]['reason']!='singleton_leaf':
                node=proof['states'][key];index=node['chosen_split']['board_index'];key=node['positive_state'] if pred[cid,index] else node['negative_state'];depth+=1
            assert proof['states'][key]['mask']==1<<local and depth==2
        cache_fixture=dict(candidates=[0,1,2,3],scores=['1','2','3','4'],optimal_cost=proof['optimal_cost'],weights=proof['weights'],repeated_identical_inputs_add_zero_states_and_splits=True,predictions_readonly=True,compiler_input=read(Path(temporary)/'group0/compiler-input.json'))
    with tempfile.TemporaryDirectory(prefix='h3bt-bound-') as temporary:
        policy.OUT=Path(temporary);compiler=policy.OnlineCompiler(np.zeros((17,2),bool),['1']*17,'b'*64,budget,100000);row,proof=compiler.get(list(range(17)));assert not proof['feasible'] and row['method']=='binary_information_bound' and len(proof['states'])==1 and row['memo_states']==1 and row['canonical_split_attempts']==0
    with tempfile.TemporaryDirectory(prefix='h3bt-cost-') as temporary:
        cost_fixtures=cost_gate(Path(temporary),budget,keep_file_manifest=False)
    save(SRC/'preflight-cost-gate.json',cost_fixtures)
    old=read(ROOT/'results/research/loki/h3br/case-results.json');sources=read(ROOT/'results/research/loki/h3br/source-banks.json');summary=read(ROOT/'results/research/loki/h3br/summary.json');assert [r['case'] for r in old]==[r['case'] for r in sources]==list(range(4096)) and sum(r['arms']['posterior']['policy_calls'] for r in old)==10355 and sum(r['arms']['posterior']['exact_true_singleton'] for r in old)==4083 and sum(r['compiler_feasible'] is False for r in old)==13 and summary['audit']['exact_conditioned_prior_and_all4096_truth_draws_verified']
    result=dict(utc=now(),weighted_synthetic_gate_sha256=sha(SRC/'preflight-synthetic-gate.json'),restriction_lemma_sha256=sha(SRC/'LEMMA.md'),guard_lemma_sha256=sha(SRC/'GUARD_LEMMA.md'),cost_gate_sha256=sha(SRC/'preflight-cost-gate.json'),counterfactual_parity_and_accept_reject_tie_gates_passed=cost_fixtures['all_candidate_rollouts_match'] and cost_fixtures['accept_reject_and_tie_cases_verified'],recursive_selector_calls=cost_fixtures['recursive_selector_calls'],naive_selector_calls=cost_fixtures['naive_selector_calls'],restriction_gate=restriction,online_compiler_cache_fixture=cache_fixture,seventeen_candidate_information_bound_verified=True,all4096_archived_cases_retained=True,baseline_policy_calls=10355,baseline_exact_cases=4083,baseline_infeasible_cases=13,baseline_summary_sha256=sha(ROOT/'results/research/loki/h3br/summary.json'),baseline_case_results_sha256=sha(ROOT/'results/research/loki/h3br/case-results.json'),new_truth_draws=0,new_source_draws=0,new_membership_calls=0,elapsed_seconds=time.perf_counter()-start);save(SRC/'preflight.json',result);print(result)


if __name__=='__main__':main()
