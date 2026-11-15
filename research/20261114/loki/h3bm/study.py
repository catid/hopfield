"""Paired eight-versus-sixteen source positives under one unchanged query policy."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from compiler import Budget,compile_tree
from synthetic_gate import run as synthetic_gate
from evidence_gate import run as evidence_gate
from evidence import draw_pair,bank_hash
from prepare import run as prepare
from ledger import Ledger
from ledger_gate import run as ledger_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();results=[];worker=None;budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_canonical_split_attempts_total']);calls=0;groups=[];statuses=[dict(case=i,source='pending',positive8='pending',positive16='pending') for i in range(1408)]
    def expired(*_):raise TimeoutError('H3bm600CPU900wall gate; whole-study incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        ordinary,sourcepred,sourcepop,boards,bits,scores,history=prepare(p);worker=subprocess.Popen([sys.executable,str(SRC/'selector.py')],cwd=OUT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def select(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();line=worker.stdout.readline();assert line,'Persistent selector exited';return json.loads(line)
        ledger_gate(p,select);synthetic_gate(budget);evidence_gate();prep=dict(utc=now(),population_parity_sha256=sha(OUT/'population-parity.json'),ledger_gate_sha256=sha(OUT/'ledger-gate.json'),compiler_synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),evidence_gate_sha256=sha(OUT/'evidence-gate.json'),protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'preparation.json',prep);lookup={bytes(row):i for i,row in enumerate(ordinary)};predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);ledger=Ledger(OUT/'query-ledger.jsonl');ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json')));sources=[];fresh={8:{},16:{}};global_boards=set()
        for cfg in p['truths']:
            truth=cfg['candidate'];population=sourcepop[truth]
            for bank,(seed16,seed8) in enumerate(zip(cfg['seeds16'],cfg['subset_seeds8'])):
                case=len(sources);indices,subset,sample16,sample8=draw_pair(population,seed16,seed8);initial={};hashes={};prior_duplicates={}
                for m,sample in [(8,sample8),(16,sample16)]:
                    initial[str(m)]=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();hashes[str(m)]=bank_hash(sample);prior_duplicates[str(m)]=list(fresh[m].get(hashes[str(m)],[]));fresh[m].setdefault(hashes[str(m)],[]).append(case);assert truth in initial[str(m)] and all(scores[m][cid] is not None for cid in initial[str(m)])
                assert set(initial['16'])<=set(initial['8']);global_boards.update(map(bytes,sample16));data=dict(case=case,candidate=truth,bank=bank,seed16=seed16,subset_seed8=seed8,source_population=len(population),indices16=indices.tolist(),subset_indices8=subset.tolist(),columns16_zero_based=sample16.tolist(),columns8_zero_based=sample8.tolist(),initial_candidates=initial,unordered_sample_sha256=hashes,prior_same_study_cases=prior_duplicates,historical_exact16_matches=history.get(hashes['16'],[]),source_positive_union=16,positive_arm_appearances=24);event=ledger.append('source_pair',data);sources.append(dict(data,source_sequence=event['sequence'],source_record_sha256=event['sha256']));statuses[case]['source']='complete'
        save(OUT/'source-pairs.json',sources);ledger.append('all_source_pairs_drawn',dict(pairs=1408,positive_union_appearances=22528,positive_arm_appearances=33792,distinct_N12_boards_across_all_pairs=len(global_boards),source_pairs_sha256=sha(OUT/'source-pairs.json')));call_history={};cache={}
        def ask(case,truth,arm,phase,step,candidates,queried,request,selection):
            nonlocal calls
            index=selection['query_index'];assert index is not None and index not in queried and calls<p['budget']['membership_calls'];previous=list(call_history.get((case,index),[]));query=ledger.append('query',dict(case=case,arm=arm,phase=phase,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),previous_same_case_query_sequences=previous));label=bool((bits[truth,index//8]>>(index%8))&1);calls+=1;remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label];answer=ledger.append('answer',dict(case=case,arm=arm,phase=phase,step=step,query_sha256=query['sha256'],label=label,candidates_before=candidates,candidates_after=remaining));assert truth in remaining;call_history.setdefault((case,index),[]).append(query['sequence']);return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=query['sequence'],query_sha256=query['sha256'],answer_sequence=answer['sequence'],answer_sha256=answer['sha256'],previous_same_case_query_sequences=previous)
        def compile_checkpoint(candidates):
            key=tuple(candidates)
            if key in cache:return cache[key]
            group=len(groups);folder=OUT/f'group{group}';folder.mkdir();before_states=budget.states;before_splits=budget.splits;begin=cpu();beginwall=time.perf_counter()
            if len(candidates)>16:
                budget.state();full=(1<<len(candidates))-1;root=f'{full}:4';proof=dict(candidates=candidates,depth_limit=4,root_state=root,feasible=False,states={root:dict(mask=full,depth_remaining=4,models=len(candidates),feasible=False,reason='binary_information_bound',attempts=[])},leaf_depths={},worst_leaf_depth=None);catalog=None;catalog_sha=None;save(folder/'compiler-input.json',dict(candidates=candidates,depth=4,cardinality_information_bound_only=True,allowed_inputs_only=True))
            else:
                codes=np.zeros(len(boards),np.uint32)
                for local,cid in enumerate(candidates):codes|=predictions[cid].astype(np.uint32)<<local
                signatures,first,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(code),lexfirst_board_index=int(index),board_count=int(count)) for code,index,count in zip(signatures,first,counts)];save(folder/'catalog.json',dict(candidates=candidates,universe_boards=len(boards),signatures=catalog));catalog_sha=sha(folder/'catalog.json');save(folder/'compiler-input.json',dict(candidates=candidates,catalog_sha256=catalog_sha,depth=4,allowed_inputs_only=True));proof=compile_tree(candidates,catalog,4,budget)
            save(folder/'proof.json',proof)
            if proof['feasible']:
                for local,cid in enumerate(candidates):
                    key=proof['root_state'];used=[]
                    while proof['states'][key]['reason']!='singleton_leaf':
                        node=proof['states'][key];index=node['chosen_split']['board_index'];assert index not in used;used.append(index);key=node['positive_state'] if predictions[cid,index] else node['negative_state']
                    assert proof['states'][key]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4
            row=dict(group=group,candidates=candidates,feasible=proof['feasible'],worst_leaf_depth=proof['worst_leaf_depth'],signature_count=len(catalog) if catalog is not None else None,memo_states=budget.states-before_states,canonical_split_attempts=budget.splits-before_splits,CPU_seconds=cpu()-begin,wall_seconds=time.perf_counter()-beginwall,catalog_sha256=catalog_sha,compiler_input_sha256=sha(folder/'compiler-input.json'),proof_sha256=sha(folder/'proof.json'));save(folder/'result.json',row);groups.append(row);cache[tuple(candidates)]=(row,proof);return row,proof
        for source in sources:
            case=source['case'];truth=source['candidate'];arms={};all_queries=[]
            for m in [8,16]:
                arm=f'positive{m}';initial=source['initial_candidates'][str(m)];candidates=list(initial);queried=[];map_steps=[];terminal=None
                for step in range(4):
                    if len(candidates)==1:break
                    request=dict(bits_path='predictions-bits.npy',scores_path=f'score-values{m}.json',board_count=len(boards),candidates=candidates,queried=queried);selection=select(request)
                    if selection['query_index'] is None:
                        event=ledger.append('no_informative_query',dict(case=case,arm=arm,phase='map',step=step,request=request,selection=selection));terminal=dict(sequence=event['sequence'],sha256=event['sha256']);break
                    candidates,row=ask(case,truth,arm,'map',step,candidates,queried,request,selection);map_steps.append(row);queried.append(row['query_index'])
                checkpoint=list(candidates);checkpoint_event=ledger.append('arm_checkpoint',dict(case=case,arm=arm,positive_count=m,candidates=checkpoint,queried=list(queried),map_calls=len(map_steps),source_record_sha256=source['source_record_sha256']));group=None;proof=None;decision_event=None;tree_steps=[]
                if len(checkpoint)>1:
                    group,proof=compile_checkpoint(checkpoint);decision_event=ledger.append('compiler_decision',dict(case=case,arm=arm,group=group['group'],result_sha256=sha(OUT/f'group{group["group"]}/result.json'),feasible=group['feasible'],checkpoint_sequence=checkpoint_event['sequence'],checkpoint_sha256=checkpoint_event['sha256']))
                    if proof['feasible']:
                        key=proof['root_state']
                        while proof['states'][key]['reason']!='singleton_leaf':
                            node=proof['states'][key];index=node['chosen_split']['board_index'];request=dict(group=group['group'],proof_sha256=group['proof_sha256'],state=key,candidates=candidates,queried=queried);selection=dict(query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates});candidates,row=ask(case,truth,arm,'tree',len(map_steps)+len(tree_steps),candidates,queried,request,selection);tree_steps.append(row);queried=queried+[index];key=node['positive_state'] if row['label'] else node['negative_state']
                result=dict(positive_count=m,initial_candidates=initial,map_steps=map_steps,terminal_no_information=terminal,checkpoint_candidates=checkpoint,checkpoint_sequence=checkpoint_event['sequence'],checkpoint_sha256=checkpoint_event['sha256'],tree_group=group['group'] if group else None,compiler_feasible=group['feasible'] if group else None,compiler_decision_sequence=decision_event['sequence'] if decision_event else None,compiler_decision_sha256=decision_event['sha256'] if decision_event else None,tree_steps=tree_steps,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],map_calls=len(map_steps),tree_calls=len(tree_steps),membership_calls=len(map_steps)+len(tree_steps),reacquired_cross_arm_calls=sum(bool(r['previous_same_case_query_sequences']) for r in map_steps+tree_steps));assert result['membership_calls']<=8 and len(tree_steps)<=4;ledger.append('arm_result',dict(case=case,arm=arm,result=result));arms[str(m)]=result;all_queries.extend(queried);statuses[case][arm]='complete'
            result=dict(case=case,candidate=truth,bank=source['bank'],seed16=source['seed16'],subset_seed8=source['subset_seed8'],source_sequence=source['source_sequence'],source_record_sha256=source['source_record_sha256'],arms=arms,source_positive_union=16,positive_arm_appearances=24,actual_membership_calls=sum(r['membership_calls'] for r in arms.values()),reacquired_cross_arm_calls=sum(r['reacquired_cross_arm_calls'] for r in arms.values()),unique_same_case_query_boards=len(set(all_queries)));assert result['actual_membership_calls']-result['reacquired_cross_arm_calls']==result['unique_same_case_query_boards'];ledger.append('pair_result',result);results.append(result)
            if source['bank']==63:print('TRUTH',truth,'eight',sum(r['arms']['8']['exact_true_singleton'] for r in results[-64:]),'sixteen',sum(r['arms']['16']['exact_true_singleton'] for r in results[-64:]),'calls',calls,'states',budget.states,'splits',budget.splits,flush=True)
        worker.stdin.close();assert worker.wait()==0;worker=None;truth_results=[]
        for cfg in p['truths']:
            rows=[r for r in results if r['candidate']==cfg['candidate']];eight=sum(r['arms']['8']['exact_true_singleton'] for r in rows);truth_results.append(dict(candidate=cfg['candidate'],orbits=cfg['orbits'],banks=64,eight_positive_exact=eight,sixteen_positive_exact=sum(r['arms']['16']['exact_true_singleton'] for r in rows),eight_primary_required=61,primary_decision='supported' if eight>=61 else 'falsified',eight_membership_calls=sum(r['arms']['8']['membership_calls'] for r in rows),sixteen_membership_calls=sum(r['arms']['16']['membership_calls'] for r in rows),eight_infeasible_checkpoints=sum(r['arms']['8']['compiler_feasible'] is False for r in rows),sixteen_infeasible_checkpoints=sum(r['arms']['16']['compiler_feasible'] is False for r in rows)))
        arm_totals={}
        for m in ['8','16']:
            rows=[r['arms'][m] for r in results];unresolved=[r for r in rows if len(r['checkpoint_candidates'])>1];arm_totals[m]=dict(exact_cases=sum(r['exact_true_singleton'] for r in rows),map_calls=sum(r['map_calls'] for r in rows),tree_calls=sum(r['tree_calls'] for r in rows),membership_calls=sum(r['membership_calls'] for r in rows),unresolved_checkpoints=len(unresolved),feasible_checkpoint_cases=sum(r['compiler_feasible'] for r in unresolved),infeasible_checkpoint_cases=sum(r['compiler_feasible'] is False for r in unresolved),positive_arm_appearances=1408*int(m),source_cases_with_prior_duplicate=sum(bool(r['prior_same_study_cases'][m]) for r in sources))
        outcome=dict(decision='supported' if all(r['primary_decision']=='supported' for r in truth_results) else 'falsified',truth_results=truth_results,cases=1408,arm_totals=arm_totals,paired=dict(both_exact=sum(r['arms']['8']['exact_true_singleton'] and r['arms']['16']['exact_true_singleton'] for r in results),eight_only=sum(r['arms']['8']['exact_true_singleton'] and not r['arms']['16']['exact_true_singleton'] for r in results),sixteen_only=sum(not r['arms']['8']['exact_true_singleton'] and r['arms']['16']['exact_true_singleton'] for r in results),neither=sum(not r['arms']['8']['exact_true_singleton'] and not r['arms']['16']['exact_true_singleton'] for r in results)),actual_membership_calls=calls,reacquired_cross_arm_calls=sum(r['reacquired_cross_arm_calls'] for r in results),unique_same_case_query_boards=sum(r['unique_same_case_query_boards'] for r in results),positive_union_appearances=22528,positive_arm_appearances=33792,distinct_N12_boards_across_all_pairs=len(global_boards),source16_cases_matching_history=sum(bool(r['historical_exact16_matches']) for r in sources),distinct_compiled_sets=len(groups),global_memo_states=budget.states,global_canonical_split_attempts=budget.splits)
        ledger.append('study_complete',outcome);ledger.close();save(OUT/'case-results.json',results);save(OUT/'groups.json',groups);save(OUT/'case-statuses.json',statuses);save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),**outcome,ledger_sha256=sha(OUT/'query-ledger.jsonl'),ledger_records=ledger.sequence,ledger_last_record_sha256=ledger.previous,case_results_sha256=sha(OUT/'case-results.json'),groups_sha256=sha(OUT/'groups.json'),case_statuses_sha256=sha(OUT/'case-statuses.json'),source_pairs_sha256=sha(OUT/'source-pairs.json'),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        if worker is not None:worker.terminate();worker.wait()
        save(OUT/'case-statuses.json',statuses);save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,case_statuses=statuses,completed_groups=groups,actual_membership_calls=calls,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
