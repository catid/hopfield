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
from prior import draw_truth
from arm import ArmRunner
from oracle import PaidOracle
from endpoint import summarize
from pairing import pair as pair_results
from cap_gate import run as cap_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    started,wall=cpu(),time.perf_counter();p=frozen()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'study-result.json').exists()
    budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_split_evaluations_total'])
    sources,pairs,runners=[],[],{}
    worker=ledger=oracle=None
    statuses=[dict(case=i,source='pending',H='pending',T='pending') for i in range(1024)]
    def expired(*_):raise TimeoutError('H3ca600CPU/900wall gate; whole study incomplete')
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
        gate_states,gate_work=budget.states,budget.splits
        prep=dict(utc=now(),protocol_sha256=sha(SRC/'protocol.json'),
                  runtime_gate_memo_states=gate_states,runtime_gate_work_steps=gate_work,
                  preflight_sha256=sha(SRC/'preflight.json'),source_gate=sg['seed_audit'],
                  source_prior_gate_sha256=sg['prior_gate_sha256'],fitted_weight_gate_sha256=sg['fitted_weight_gate_sha256'],
                  two_score_gate_sha256=sg['two_score_gate_sha256'],all_runtime_gates_before_prospective_draws=True)
        for name in ['population-parity','ledger-gate','synthetic-gate','restriction-gate','pruning-gate',
                     'cache-gate','cost-gate','baseline-gate','wrapper-gate','namespace-gate','cap-gate']:
            prep[name+'_sha256']=sha(OUT/(name+'.json'))
        save(OUT/'preparation.json',prep)
        ledger=Ledger(OUT/'query-ledger.jsonl')
        ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json')))
        predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool)
        lookup={bytes(row):i for i,row in enumerate(ordinary)}
        fresh,truth_counts,global_boards={},{},set()
        for case in range(1024):
            truth_seed,source_seed=p['seed_bases']['truth']+case,p['seed_bases']['source']+case
            truth,ticket=draw_truth(p['truths'],truth_seed)
            population=sourcepop[truth];bank=truth_counts.get(truth,0);truth_counts[truth]=bank+1
            indices=np.random.default_rng(source_seed).choice(len(population),8,replace=False)
            sample=population[indices]
            initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist()
            digest=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest()
            assert truth in initial and all(score_tables[arm][c] is not None for arm in ['H','T'] for c in initial)
            data=dict(case=case,candidate=truth,bank=bank,truth_seed=truth_seed,truth_ticket=ticket,seed=source_seed,
                      source_population=len(population),indices=indices.tolist(),columns_zero_based=sample.tolist(),
                      initial_candidates=initial,unordered_sample_sha256=digest,prior_same_study_cases=list(fresh.get(digest,[])),
                      historical_exact8_matches=history.get(digest,[]))
            event=ledger.append('source_bank',data)
            sources.append(dict(data,source_sequence=event['sequence'],source_record_sha256=event['sha256']))
            fresh.setdefault(digest,[]).append(case);global_boards.update(map(bytes,sample));statuses[case]['source']='complete'
        save(OUT/'source-banks.json',sources)
        ledger.append('all_source_banks_drawn',dict(cases=1024,new_truth_draws=1024,new_source_draws=1024,
                      source_positive_union_appearances=8192,positive_arm_appearances=16384,distinct_N12_boards=len(global_boards),
                      source_banks_sha256=sha(OUT/'source-banks.json')))
        oracle=PaidOracle(ledger,boards,bits,p['budget']['membership_calls'],p['budget']['membership_calls_per_arm'])
        for arm in ['H','T']:
            runner=ArmRunner(arm,OUT/arm,predictions,score_tables['H'],score_tables[arm],budget,p['budget']['max_integer_bits'],len(boards))
            runners[arm]=runner;runner.run(sources,ledger,oracle.ask,statuses)
        for h,u in zip(runners['H'].results,runners['T'].results):
            pair=pair_results(h,u,score_tables['T'],[runners['H'].handoff_proof(h),runners['T'].handoff_proof(u)])
            pairs.append(pair);ledger.append('paired_case_result',pair)
        resources={}
        for arm,runner in runners.items():
            rows=runner.compiler.rows;cost=runner.cost_model
            resources[arm]=dict(tree_memo_states=sum(r['memo_states'] for r in rows),
                compiler_work_steps=sum(r['work_steps'] for r in rows),huffman_merge_operations=sum(r['huffman_merge_operations'] for r in rows),
                canonical_computation_reuse_hits=sum(r['canonical_computation_reuse'] is not None for r in rows),
                counterfactual_memo_states=cost.memo_states,counterfactual_MAP_partitions=cost.MAP_partitions,
                distinct_compiled_inputs=len(rows))
        outcome=dict(**summarize(pairs),cases=1024,MAP_calls={a:sum(r['map_calls'] for r in t.results) for a,t in runners.items()},
                     tree_calls={a:sum(r['tree_calls'] for r in t.results) for a,t in runners.items()},
                     new_truth_draws=1024,new_source_draws=1024,source_positive_union_appearances=8192,positive_arm_appearances=16384,
                     distinct_N12_boards=len(global_boards),source_cases_with_prior_same_study_duplicates=sum(bool(r['prior_same_study_cases']) for r in sources),
                     source_cases_matching_archived_eight_banks=sum(bool(r['historical_exact8_matches']) for r in sources),
                     independently_paid_cross_arm_reacquisitions=sum(r['independently_paid_cross_arm_reacquisitions'] for r in pairs),
                     truth_frequencies=[dict(candidate=r['candidate'],eligible=r['eligible'],sampling_integer_weight=r['sampling_integer_weight'],
                                            cases=truth_counts.get(r['candidate'],0)) for r in p['truths']],
                     handoff_MAP_call_histogram={a:{str(i):sum(r['handoff_MAP_calls']==i for r in t.results) for i in range(5)} for a,t in runners.items()},
                     cases_without_tree_handoff={a:sum(r['handoff_MAP_calls'] is None for r in t.results) for a,t in runners.items()},
                     resources_by_arm=resources,runtime_gate_memo_states=gate_states,runtime_gate_work_steps=gate_work,
                     global_memo_states=budget.states,global_split_evaluations=budget.splits)
        assert outcome['actual_membership_calls']==oracle.calls and outcome['H_calls']==oracle.arm_calls['H'] and outcome['T_calls']==oracle.arm_calls['T']
        assert budget.states==gate_states+sum(r['tree_memo_states']+r['counterfactual_memo_states'] for r in resources.values())
        assert budget.splits==gate_work+sum(r['compiler_work_steps']+r['counterfactual_MAP_partitions'] for r in resources.values())
        save(OUT/'case-results.json',pairs);save(OUT/'case-statuses.json',statuses)
        ledger.append('study_complete',outcome);ledger.close()
        seals={name.replace('-','_').replace('/','_')+'_sha256':sha(OUT/(name+'.json')) for name in
               ['case-results','case-statuses','source-banks','preparation']+[a+'/'+n for a in ['H','T'] for n in ['arm-results','groups','counterfactual-groups']]}
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
             actual_membership_calls=oracle.calls if oracle else 0,arm_calls=oracle.arm_calls if oracle else dict(H=0,T=0),
             global_memo_states=budget.states,global_split_evaluations=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,
             stage_cpu_seconds=cpu()-started,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')))
        raise


if __name__=='__main__':main()
