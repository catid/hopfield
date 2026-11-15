"""Fresh uniform-truth paired deeper policies with independent fitted weights."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import json
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import SRC,OUT,frozen,read,save,sha,now
from compiler import Budget
from ledger import Ledger
from ledger_gate import run as ledger_gate
from synthetic_gate import run as synthetic_gate
from restriction import gate as restriction_gate
from pruning_gate import run as pruning_gate
from cache_gate import run as cache_gate
from cost_gate import run as cost_gate
from baseline_gate import run as baseline_gate
from wrapper_gate import run as wrapper_gate
from namespace_gate import run as namespace_gate
from source_gate import run as source_gate
from prepare import run as prepare
from sampling import draw as draw_sources
from fallback import ScanBudget
from fallback_gate import run as fallback_gate
from recovery_namespace_gate import run as recovery_gate
from arm import ArmRunner
from oracle import PaidOracle
from recovery_endpoint import summarize
from recovery_pairing import pair as pair_results
from cap_gate import run as cap_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    started,wall=cpu(),time.perf_counter();p=frozen()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'study-result.json').exists()
    budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_split_evaluations_total'])
    scan=ScanBudget(p['budget']['max_fallback_candidate_board_cells'])
    sources,pairs,runners=[],[],{}
    worker=ledger=oracle=None
    statuses=[dict(case=i,source='pending',B='pending',E='pending') for i in range(1522)]
    def expired(*_):raise TimeoutError('H3cb600CPU/900wall gate; whole study incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        ordinary,sourcepred,sourcepop,boards,bits,score_tables,history=prepare(p)
        sg=source_gate(OUT/'source-gates',p['historical_protocol_sha256'])
        assert sg['prior_rows']==p['truths'] and sg['fitted_prior_rows']==p['fitted_prior_rows']
        assert sg['H_scores']==score_tables['H'] and sg['T_scores']==score_tables['T']
        worker=subprocess.Popen([sys.executable,str(SRC/'selector.py')],cwd=OUT,
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def fixed_select(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
            line=worker.stdout.readline();assert line
            return json.loads(line)
        ledger_gate(p,fixed_select)
        worker.stdin.close();assert worker.wait()==0;worker=None
        toy=synthetic_gate(budget)
        save(OUT/'restriction-gate.json',dict(utc=now(),**restriction_gate(toy['fixtures'])))
        for name,func in [('pruning',pruning_gate),('cache',cache_gate),('cost',cost_gate),
                          ('baseline',baseline_gate),('wrapper',wrapper_gate)]:
            save(OUT/(name+'-gate.json'),func(OUT/(name+'-gate'),budget))
        namespace=namespace_gate(OUT/'namespace-gate',budget)
        save(OUT/'namespace-gate.json',namespace)
        save(OUT/'cap-gate.json',cap_gate(OUT/'cap-gate',budget))
        save(OUT/'fallback-gate.json',fallback_gate(OUT/'fallback-gate',scan))
        save(OUT/'recovery-gate.json',recovery_gate(OUT/'recovery-gate',budget,scan))
        gate_states,gate_work=budget.states,budget.splits
        gate_cells,gate_requests=scan.cells,scan.requests
        prep=dict(utc=now(),protocol_sha256=sha(SRC/'protocol.json'),
                  runtime_gate_memo_states=gate_states,runtime_gate_work_steps=gate_work,
                  runtime_gate_fallback_cells=gate_cells,runtime_gate_fallback_requests=gate_requests,
                  preflight_sha256=sha(SRC/'preflight.json'),source_gate=sg['seed_audit'],
                  source_prior_gate_sha256=sg['prior_gate_sha256'],fitted_weight_gate_sha256=sg['fitted_weight_gate_sha256'],
                  two_score_gate_sha256=sg['two_score_gate_sha256'],all_runtime_gates_before_prospective_draws=True)
        for name in ['population-parity','ledger-gate','synthetic-gate','restriction-gate','pruning-gate',
                     'cache-gate','cost-gate','baseline-gate','wrapper-gate','namespace-gate','cap-gate','fallback-gate','recovery-gate']:
            prep[name+'_sha256']=sha(OUT/(name+'.json'))
        save(OUT/'preparation.json',prep)
        ledger=Ledger(OUT/'query-ledger.jsonl')
        ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json')))
        predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool)
        sources,truth_counts,global_boards=draw_sources(p,ordinary,sourcepred,sourcepop,score_tables,history,ledger,OUT,statuses,sources=sources)
        oracle=PaidOracle(ledger,boards,bits,p['budget']['membership_calls'],p['budget']['membership_calls_per_arm'],arms=('B','E'))
        for arm in ['B','E']:
            runner=ArmRunner(arm,OUT/arm,predictions,score_tables['H'],score_tables['T'],budget,p['budget']['max_integer_bits'],len(boards),scan_budget=scan)
            runners[arm]=runner;runner.run(sources,ledger,oracle.ask,statuses)
        for h,u in zip(runners['B'].results,runners['E'].results):
            pair=pair_results(h,u)
            pairs.append(pair);ledger.append('paired_case_result',pair)
        resources={}
        for arm,runner in runners.items():
            rows=runner.compiler.rows;cost=runner.cost_model
            resources[arm]=dict(tree_memo_states=sum(r['memo_states'] for r in rows),
                compiler_work_steps=sum(r['work_steps'] for r in rows),huffman_merge_operations=sum(r['huffman_merge_operations'] for r in rows),
                canonical_computation_reuse_hits=sum(r['canonical_computation_reuse'] is not None for r in rows),
                counterfactual_memo_states=cost.memo_states,counterfactual_MAP_partitions=cost.MAP_partitions,
                distinct_compiled_inputs=len(rows))
        outcome=dict(**summarize(pairs),cases=1522,MAP_calls={a:sum(r['map_calls'] for r in t.results) for a,t in runners.items()},
                     tree_calls={a:sum(r['tree_calls'] for r in t.results) for a,t in runners.items()},
                     truth_permutation_draws=1,individual_truth_draws=0,new_source_draws=1522,source_positive_union_appearances=12176,positive_arm_appearances=24352,
                     distinct_N12_boards=len(global_boards),source_cases_with_prior_same_study_duplicates=sum(bool(r['prior_same_study_cases']) for r in sources),
                     source_cases_matching_archived_eight_banks=sum(bool(r['historical_exact8_matches']) for r in sources),
                     independently_paid_cross_arm_reacquisitions=sum(r['independently_paid_cross_arm_reacquisitions'] for r in pairs),
                     truth_frequencies=[dict(candidate=r['candidate'],eligible=r['eligible'],sampling_integer_weight=r['sampling_integer_weight'],
                                            cases=truth_counts.get(r['candidate'],0)) for r in p['truths']],
                     handoff_MAP_call_histogram={a:{str(i):sum(r['handoff_MAP_calls']==i for r in t.results) for i in range(5)} for a,t in runners.items()},
                     cases_without_tree_handoff={a:sum(r['handoff_MAP_calls'] is None for r in t.results) for a,t in runners.items()},
                     resources_by_arm=resources,runtime_gate_memo_states=gate_states,runtime_gate_work_steps=gate_work,
                  runtime_gate_fallback_cells=gate_cells,runtime_gate_fallback_requests=gate_requests,
                     global_memo_states=budget.states,global_split_evaluations=budget.splits,
                     fallback_calls={a:sum(r['fallback']['additional_calls'] for r in t.results) for a,t in runners.items()},
                     runtime_gate_fallback_cells=gate_cells,runtime_gate_fallback_requests=gate_requests,
                     fallback_candidate_board_cells=scan.cells,fallback_selection_requests=scan.requests,
                     fallback_resources_by_arm={a:dict(candidate_board_cells=sum(r['fallback']['candidate_board_cells'] for r in t.results),
                       selection_requests=sum(r['fallback']['selection_requests'] for r in t.results),
                       CPU_seconds=sum(r['fallback']['CPU_seconds'] for r in t.results),wall_seconds=sum(r['fallback']['wall_seconds'] for r in t.results),
                       maximum_active_set=max([len(r['original_policy_result']['remaining_candidates']) for r in t.results if r['fallback']['entered']]+[0])) for a,t in runners.items()})
        assert outcome['actual_membership_calls']==oracle.calls and outcome['B_calls']==oracle.arm_calls['B'] and outcome['E_calls']==oracle.arm_calls['E']
        assert budget.states==gate_states+sum(r['tree_memo_states']+r['counterfactual_memo_states'] for r in resources.values())
        assert budget.splits==gate_work+sum(r['compiler_work_steps']+r['counterfactual_MAP_partitions'] for r in resources.values())
        save(OUT/'case-results.json',pairs);save(OUT/'case-statuses.json',statuses)
        ledger.append('study_complete',outcome);ledger.close()
        seals={name.replace('-','_').replace('/','_')+'_sha256':sha(OUT/(name+'.json')) for name in
               ['case-results','case-statuses','source-banks','truth-schedule','preparation']+[a+'/'+n for a in ['B','E'] for n in ['arm-results','groups','counterfactual-groups']]}
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),**outcome,**seals,
             ledger_sha256=sha(OUT/'query-ledger.jsonl'),ledger_records=ledger.sequence,ledger_last_record_sha256=ledger.previous,
             stage_cpu_seconds=cpu()-started,stage_wall_seconds=time.perf_counter()-wall,
             protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as error:
        if worker is not None:worker.terminate();worker.wait()
        if ledger is not None:ledger.close()
        save(OUT/'case-statuses.json',statuses)
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(error),completed_sources=sources,
             completed_pairs=pairs,arm_snapshots={a:t.snapshot() for a,t in runners.items()},
             actual_membership_calls=oracle.calls if oracle else 0,arm_calls=oracle.arm_calls if oracle else dict(B=0,E=0),
             global_memo_states=budget.states,global_split_evaluations=budget.splits,fallback_scan_budget=scan.snapshot(),partial_results_usable=False,cap_is_not_infeasibility=True,
             stage_cpu_seconds=cpu()-started,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')))
        raise


if __name__=='__main__':main()
