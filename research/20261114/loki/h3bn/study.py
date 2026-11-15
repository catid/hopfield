"""Paid weighted-tree execution from every archived eight-positive checkpoint."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from fractions import Fraction
import gzip
import hashlib
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from compiler import Budget,compile_tree
from weights import integer_weights
from synthetic_gate import run as synthetic_gate
from proof_gate import require_ready_proof
from prepare import run as prepare
from ledger import Ledger,audit_bytes


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def fraction(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator),display_float=float(x))


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_canonical_split_attempts_total']);compiled=[];results=[];calls=0;statuses=[dict(case=i,status='pending') for i in range(1408)]
    def expired(*_):raise TimeoutError('H3bn600CPU900wall; whole-study incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        boards,bits,scores=prepare(p);synthetic_gate(budget);M=ROOT/'results/research/loki/h3bm';prior=read(M/'summary.json');cases=read(M/'case-results.json');oldgroups=read(M/'groups.json');parent_raw=gzip.decompress((M/'query-ledger.jsonl.gz').read_bytes());parent_events=audit_bytes(parent_raw);assert hashlib.sha256(parent_raw).hexdigest()==prior['ledger_sha256'] and len(parent_events)==prior['ledger_records'] and prior['audit']['all_membership_calls_independently_replayed']==15004;predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);parent_queries={};parent_arm_results={}
        for event in parent_events:
            if event['kind']=='query':parent_queries.setdefault((event['data']['case'],event['data']['selection']['query_index']),[]).append(event['sequence'])
            if event['kind']=='arm_result' and event['data']['arm']=='positive8':parent_arm_results[event['data']['case']]=event
        checkpoints=[]
        for case in cases:
            truth=case['candidate'];arm=case['arms']['8'];candidates=list(arm['initial_candidates']);queried=[]
            for step in arm['map_steps']:
                q=parent_events[step['query_sequence']];a=parent_events[step['answer_sequence']];index=step['query_index'];label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label];assert q['sha256']==step['query_sha256'] and a['sha256']==step['answer_sha256'] and q['data']['arm']=='positive8' and q['data']['phase']=='map' and a['data']['label']==step['label']==label and step['candidates_before']==candidates and step['candidates_after']==remaining and index not in queried;candidates=remaining;queried.append(index)
            assert candidates==arm['checkpoint_candidates'];event=parent_arm_results[case['case']];assert event['data']['result']==arm
            if arm['compiler_feasible'] is False:assert len(candidates)>16 and not arm['tree_steps']
            if arm['tree_group'] is None:assert candidates==[truth] and not arm['tree_steps']
            checkpoints.append(dict(case=case['case'],candidate=truth,bank=case['bank'],source_record_sha256=case['source_record_sha256'],parent_arm_result_sequence=event['sequence'],parent_arm_result_sha256=event['sha256'],remaining_candidates=candidates,queried_board_indices=queried,inherited_MAP_calls=len(queried),old_tree_continuation_calls=arm['tree_calls'],old_exact=arm['exact_true_singleton'],original_tree_group=arm['tree_group'],original_compiler_feasible=arm['compiler_feasible']))
        assert len(checkpoints)==1408 and sum(r['original_compiler_feasible'] is True for r in checkpoints)==1067 and sum(r['original_compiler_feasible'] is False for r in checkpoints)==54 and sum(r['original_tree_group'] is None for r in checkpoints)==287;save(OUT/'checkpoints.json',dict(utc=now(),cases=checkpoints,parent_ledger_raw_sha256=hashlib.sha256(parent_raw).hexdigest(),parent_summary_sha256=sha(M/'summary.json'),all1408_retained=True,protocol_sha256=sha(SRC/'protocol.json')));ids=sorted({r['original_tree_group'] for r in checkpoints if r['original_compiler_feasible']});assert ids==p['optimized_parent_group_ids'];aggregate_old=Fraction(0);aggregate_new=Fraction(0)
        for group in ids:
            original=oldgroups[group];candidates=original['candidates'];folder=OUT/f'group{group}';folder.mkdir();before_states=budget.states;before_splits=budget.splits;begin=cpu();beginwall=time.perf_counter();weights,receipt=integer_weights(candidates,scores,p['budget']['max_LCM_or_weight_bits']);save(folder/'weights.json',receipt);codes=np.zeros(len(boards),np.uint32)
            for local,cid in enumerate(candidates):codes|=predictions[cid].astype(np.uint32)<<local
            signatures,first,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(code),lexfirst_board_index=int(index),board_count=int(count)) for code,index,count in zip(signatures,first,counts)];stored=dict(candidates=candidates,universe_boards=len(boards),signatures=catalog);assert stored==read(M/f'group{group}/catalog.json');save(folder/'catalog.json',stored);save(folder/'compiler-input.json',dict(candidates=candidates,weights=[str(w) for w in weights],catalog_sha256=sha(folder/'catalog.json'),depth=4,allowed_inputs_only=True));proof=compile_tree(candidates,weights,catalog,4,budget);save(folder/'proof.json',proof);proof=require_ready_proof(folder/'proof.json',sha(folder/'proof.json'),candidates)
            for local,cid in enumerate(candidates):
                key=proof['root_state'];used=[]
                while proof['states'][key]['reason']!='singleton_leaf':
                    node=proof['states'][key];index=node['chosen_split']['board_index'];assert index not in used;used.append(index);key=node['positive_state'] if predictions[cid,index] else node['negative_state']
                assert proof['states'][key]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4
            oldproof=read(M/f'group{group}/proof.json');oldcost=sum(w*oldproof['leaf_depths'][str(cid)] for cid,w in zip(candidates,weights));newcost=int(proof['optimal_cost']);assert newcost<=oldcost;case_ids=[r['case'] for r in checkpoints if r['original_tree_group']==group];oldexpected=Fraction(oldcost,sum(weights));newexpected=Fraction(newcost,sum(weights));aggregate_old+=len(case_ids)*oldexpected;aggregate_new+=len(case_ids)*newexpected;row=dict(parent_group=group,candidates=candidates,case_ids=case_ids,case_multiplicity=len(case_ids),old_weighted_depth_cost=str(oldcost),optimal_weighted_depth_cost=str(newcost),weight_sum=str(sum(weights)),old_expected=fraction(oldexpected),optimal_expected=fraction(newexpected),memo_states=budget.states-before_states,canonical_split_attempts=budget.splits-before_splits,CPU_seconds=cpu()-begin,wall_seconds=time.perf_counter()-beginwall,weights_sha256=sha(folder/'weights.json'),catalog_sha256=sha(folder/'catalog.json'),compiler_input_sha256=sha(folder/'compiler-input.json'),proof_sha256=sha(folder/'proof.json'));save(folder/'result.json',row);compiled.append(row)
            if len(compiled)%50==0:print('COMPILED',len(compiled),'states',budget.states,'splits',budget.splits,flush=True)
        save(OUT/'groups.json',compiled);prep=dict(utc=now(),all439_compiled_before_new_labels=True,groups_sha256=sha(OUT/'groups.json'),checkpoints_sha256=sha(OUT/'checkpoints.json'),population_parity_sha256=sha(OUT/'population-parity.json'),synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),preflight_sha256=sha(SRC/'preflight.json'),parent_ledger_raw_sha256=hashlib.sha256(parent_raw).hexdigest(),protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'preparation.json',prep);ledger=Ledger(OUT/'query-ledger.jsonl');ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json'),parent_ledger_raw_sha256=prep['parent_ledger_raw_sha256'],checkpoints_sha256=sha(OUT/'checkpoints.json'),compiled_groups_sha256=sha(OUT/'groups.json')))
        for checkpoint in checkpoints:
            case=checkpoint['case'];truth=checkpoint['candidate'];candidates=list(checkpoint['remaining_candidates']);queried=list(checkpoint['queried_board_indices']);group=checkpoint['original_tree_group'];event=ledger.append('archived_checkpoint',checkpoint);steps=[]
            if checkpoint['original_compiler_feasible']:
                folder=OUT/f'group{group}';row=read(folder/'result.json');proof=require_ready_proof(folder/'proof.json',row['proof_sha256'],candidates);key=proof['root_state'];ledger.append('compiled_tree_bound',dict(case=case,parent_group=group,proof_sha256=row['proof_sha256'],result_sha256=sha(folder/'result.json'),checkpoint_sequence=event['sequence'],checkpoint_sha256=event['sha256']))
                while proof['states'][key]['reason']!='singleton_leaf':
                    node=proof['states'][key];index=node['chosen_split']['board_index'];assert index not in queried and calls<p['budget']['new_membership_calls'];request=dict(parent_group=group,proof_sha256=row['proof_sha256'],state=key,candidates=candidates,queried=queried);selection=dict(query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates});old_queries=parent_queries.get((case,index),[]);query=ledger.append('query',dict(case=case,step=checkpoint['inherited_MAP_calls']+len(steps),request=request,selection=selection,board_zero_based=boards[index].tolist(),archived_same_case_query_sequences=old_queries));label=bool((bits[truth,index//8]>>(index%8))&1);calls+=1;remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label];answer=ledger.append('answer',dict(case=case,query_sha256=query['sha256'],label=label,candidates_before=candidates,candidates_after=remaining));assert truth in remaining;steps.append(dict(step=checkpoint['inherited_MAP_calls']+len(steps),query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=query['sequence'],query_sha256=query['sha256'],answer_sequence=answer['sequence'],answer_sha256=answer['sha256'],archived_same_case_query_sequences=old_queries));candidates=remaining;queried=queried+[index];key=node['positive_state'] if label else node['negative_state']
                assert candidates==[truth] and len(steps)<=4
            result=dict(case=case,candidate=truth,bank=checkpoint['bank'],original_tree_group=group,old_exact=checkpoint['old_exact'],new_exact=candidates==[truth],remaining_candidates=candidates,inherited_MAP_calls=checkpoint['inherited_MAP_calls'],old_tree_continuation_calls=checkpoint['old_tree_continuation_calls'],new_tree_continuation_calls=len(steps),new_policy_cumulative_calls=checkpoint['inherited_MAP_calls']+len(steps),archived_query_reacquisitions=sum(bool(r['archived_same_case_query_sequences']) for r in steps),new_query_boards_absent_from_parent_case=sum(not r['archived_same_case_query_sequences'] for r in steps),checkpoint_sequence=event['sequence'],checkpoint_sha256=event['sha256'],steps=steps);assert result['new_exact']==result['old_exact'] and result['new_policy_cumulative_calls']<=8;ledger.append('case_result',result);results.append(result);statuses[case]['status']='complete'
            if checkpoint['bank']==63:print('TRUTH',truth,'newcalls',sum(r['new_tree_continuation_calls'] for r in results[-64:]),'oldcalls',sum(r['old_tree_continuation_calls'] for r in results[-64:]),'totalnew',calls,flush=True)
        oldcalls=sum(r['old_tree_continuation_calls'] for r in results);assert oldcalls==2725 and sum(r['new_exact'] for r in results)==1354;truth_results=[]
        for cfg in p['truths']:
            rows=[r for r in results if r['candidate']==cfg['candidate']];truth_results.append(dict(candidate=cfg['candidate'],cases=64,exact_cases=sum(r['new_exact'] for r in rows),old_continuation_calls=sum(r['old_tree_continuation_calls'] for r in rows),new_continuation_calls=sum(r['new_tree_continuation_calls'] for r in rows),inherited_MAP_calls=sum(r['inherited_MAP_calls'] for r in rows)))
        outcome=dict(decision='supported' if calls<=2588 else 'falsified',actual_continuation_threshold=2588,old_continuation_calls=oldcalls,new_continuation_calls=calls,exact_integer_test=dict(left=100*calls,right=95*oldcalls,passed=100*calls<=95*oldcalls),cases=1408,exact_cases=1354,known_infeasible_cases=54,prior_success_cases=287,optimized_feasible_checkpoints=1067,distinct_optimized_sets=439,no_new_failures_correctness_invariant=True,truth_results=truth_results,inherited_MAP_calls=sum(r['inherited_MAP_calls'] for r in results),new_policy_cumulative_calls=sum(r['new_policy_cumulative_calls'] for r in results),archived_query_reacquisitions=sum(r['archived_query_reacquisitions'] for r in results),new_query_boards_absent_from_parent_case=sum(r['new_query_boards_absent_from_parent_case'] for r in results),new_source_draws=0,new_source_positive_labels=0,historical_eight_positive_appearances=11264,descriptive_case_weighted_expected_old_sum=fraction(aggregate_old),descriptive_case_weighted_expected_optimal_sum=fraction(aggregate_new),descriptive_expected_relative_reduction=fraction(1-aggregate_new/aggregate_old),global_memo_states=budget.states,global_canonical_split_attempts=budget.splits);ledger.append('study_complete',outcome);ledger.close();save(OUT/'case-results.json',results);save(OUT/'case-statuses.json',statuses);save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),**outcome,ledger_sha256=sha(OUT/'query-ledger.jsonl'),ledger_records=ledger.sequence,ledger_last_record_sha256=ledger.previous,case_results_sha256=sha(OUT/'case-results.json'),case_statuses_sha256=sha(OUT/'case-statuses.json'),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'case-statuses.json',statuses);save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,case_statuses=statuses,completed_groups=compiled,new_membership_calls=calls,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
