"""Source-only synthetic parity and declared archive identity checks before freeze."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
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
from pruning_gate import run as pruning_gate
from cache_gate import run as cache_gate


def main():
    start=time.perf_counter();assert not(SRC/'protocol.json').exists();budget=Budget();synthetic_gate.OUT=SRC;toy=synthetic_gate.run(budget);(SRC/'synthetic-gate.json').rename(SRC/'preflight-synthetic-gate.json');restriction=restriction_gate(toy['fixtures'])
    with tempfile.TemporaryDirectory(prefix='h3bv-cache-') as temporary:
        policy.OUT=Path(temporary);pred=np.array([[bool(signature&(1<<cid)) for signature in [0,5,10,3,12,15]] for cid in range(4)]);compiler=policy.OnlineCompiler(pred,['1','2','3','4'],'a'*64,budget,100000);rows=[]
        for depth in [4,8]:
            row,proof=compiler.get([0,1,2,3],depth);states=budget.states;splits=budget.splits;again=compiler.get([0,1,2,3],depth);assert again[0]==row and again[1]==proof and budget.states==states and budget.splits==splits and proof['optimal_cost']=='50' and proof['depth_limit']==depth and not pred.flags.writeable;rows.append(row)
        assert len(compiler.rows)==2 and rows[0]['cache_key']!=rows[1]['cache_key'];cache_fixture=dict(candidates=[0,1,2,3],scores=['1','2','3','4'],optimal_cost='50',weights=proof['weights'],requested_depths=[4,8],cache_keys=[r['cache_key'] for r in rows],repeated_identical_inputs_add_zero_states_and_splits=True,predictions_readonly=True,requested_depths_distinct_in_cache=True)
    with tempfile.TemporaryDirectory(prefix='h3bv-limit-') as temporary:
        policy.OUT=Path(temporary);compiler=policy.OnlineCompiler(np.zeros((17,2),bool),['1']*17,'b'*64,budget,100000);states=budget.states;splits=budget.splits
        try:compiler.get(list(range(17)),8)
        except AssertionError:pass
        else:raise AssertionError('Compiler silently accepted ineligible n>16')
        assert budget.states==states and budget.splits==splits and not compiler.rows
    with tempfile.TemporaryDirectory(prefix='h3bv-pruning-') as temporary:pruning_fixtures=pruning_gate(Path(temporary)/'gate',budget)
    with tempfile.TemporaryDirectory(prefix='h3bv-cachegate-') as temporary:cache_fixtures=cache_gate(Path(temporary)/'gate',budget)
    with tempfile.TemporaryDirectory(prefix='h3bv-cost-') as temporary:cost_fixtures=cost_gate(Path(temporary),budget,keep_file_manifest=False)
    save(SRC/'preflight-pruning-gate.json',pruning_fixtures);save(SRC/'preflight-cache-gate.json',cache_fixtures);save(SRC/'preflight-cost-gate.json',cost_fixtures)
    old=read(ROOT/'results/research/loki/h3br/case-results.json');sources=read(ROOT/'results/research/loki/h3br/source-banks.json');summary=read(ROOT/'results/research/loki/h3br/summary.json');assert [r['case'] for r in old]==[r['case'] for r in sources]==list(range(4096)) and sum(r['arms']['posterior']['policy_calls'] for r in old)==10355 and sum(r['arms']['posterior']['exact_true_singleton'] for r in old)==4083 and sum(r['compiler_feasible'] is False for r in old)==13 and summary['audit']['exact_conditioned_prior_and_all4096_truth_draws_verified']
    result=dict(utc=now(),weighted_synthetic_gate_sha256=sha(SRC/'preflight-synthetic-gate.json'),restriction_lemma_sha256=sha(SRC/'LEMMA.md'),guard_lemma_sha256=sha(SRC/'GUARD_LEMMA.md'),depth_lemma_sha256=sha(SRC/'DEPTH_LEMMA.md'),cost_gate_sha256=sha(SRC/'preflight-cost-gate.json'),pruning_gate_sha256=sha(SRC/'preflight-pruning-gate.json'),cache_gate_sha256=sha(SRC/'preflight-cache-gate.json'),bound_lemma_sha256=sha(SRC/'BOUND_LEMMA.md'),counterfactual_cost_and_exact_parity_passed=cost_fixtures['all_candidate_cost_and_exact_rollouts_match'],all_inadmissible_branches_retained=cost_fixtures['inadmissible_branches_all_retained'],pruning_parity_passed=pruning_fixtures['all_feasibility_costs_and_lex_paths_identical'],all_independent_bound_certificates_passed=pruning_fixtures['all_bound_merge_and_tie_certificates_verified'],cache_reuse_and_complete_logical_proofs_passed=cache_fixtures['all_complete_proofs_and_source_reuse_hashes_verified'],recursive_selector_calls=cost_fixtures['recursive_selector_calls'],naive_selector_calls=cost_fixtures['naive_selector_calls'],restriction_gate=restriction,online_compiler_cache_fixture=cache_fixture,seventeen_candidate_ineligibility_not_false_infeasibility_verified=True,all4096_archived_cases_retained=True,baseline_policy_calls=10355,baseline_exact_cases=4083,baseline_infeasible_cases=13,baseline_summary_sha256=sha(ROOT/'results/research/loki/h3br/summary.json'),baseline_case_results_sha256=sha(ROOT/'results/research/loki/h3br/case-results.json'),new_truth_draws=0,new_source_draws=0,new_membership_calls=0,elapsed_seconds=time.perf_counter()-start);save(SRC/'preflight.json',result);print(result)


if __name__=='__main__':main()
