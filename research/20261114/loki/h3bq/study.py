"""Fresh paired three-policy continuation with all-case accounting and paid labels."""
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
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from compiler import Budget,compile_tree as optimal_tree
from firstlex import compile_tree as firstlex_tree
from weights import integer_weights
from synthetic_gate import run as weighted_gate
from firstlex_gate import run as firstlex_gate
from ledger_gate import run as ledger_gate
from ledger import Ledger,canonical
from prepare import run as prepare
POLICIES=['firstlex','posterior','uniform']


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def bank_hash(sample):return hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest()


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists()
    budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_canonical_split_attempts_total']);worker=None;results=[];groups=[];calls=0
    statuses=[dict(case=i,source='pending',map='pending',firstlex='pending',posterior='pending',uniform='pending') for i in range(1408)]
    def expired(*_):raise TimeoutError('H3bq600CPU900wall gate; whole-study incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        ordinary,sourcepred,sourcepop,boards,bits,scores,history=prepare(p)
        worker=subprocess.Popen([sys.executable,str(SRC/'selector.py')],cwd=OUT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def select(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();line=worker.stdout.readline();assert line,'Persistent selector exited';return json.loads(line)
        ledger_gate(p,select);weighted_gate(budget);firstlex_gate(budget)
        prep=dict(utc=now(),protocol_sha256=sha(SRC/'protocol.json'),population_parity_sha256=sha(OUT/'population-parity.json'),ledger_gate_sha256=sha(OUT/'ledger-gate.json'),weighted_gate_sha256=sha(OUT/'synthetic-gate.json'),firstlex_gate_sha256=sha(OUT/'firstlex-synthetic-gate.json'),preflight_sha256=sha(SRC/'preflight.json'))
        save(OUT/'preparation.json',prep);lookup={bytes(row):i for i,row in enumerate(ordinary)};predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool)
        ledger=Ledger(OUT/'query-ledger.jsonl');ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json')))
        sources=[];fresh={};global_boards=set()
        for cfg in p['truths']:
            truth=cfg['candidate'];population=sourcepop[truth]
            for bank,seed in enumerate(cfg['seeds']):
                case=len(sources);indices=np.random.default_rng(seed).choice(len(population),8,replace=False);sample=population[indices]
                initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();digest=bank_hash(sample);assert truth in initial and all(scores[c] is not None for c in initial)
                data=dict(case=case,candidate=truth,bank=bank,seed=seed,source_population=len(population),indices=indices.tolist(),columns_zero_based=sample.tolist(),initial_candidates=initial,unordered_sample_sha256=digest,prior_same_study_cases=list(fresh.get(digest,[])),historical_exact8_matches=history.get(digest,[]))
                event=ledger.append('source_bank',data);sources.append(dict(data,source_sequence=event['sequence'],source_record_sha256=event['sha256']));fresh.setdefault(digest,[]).append(case);global_boards.update(map(bytes,sample));statuses[case]['source']='complete'
        save(OUT/'source-banks.json',sources);ledger.append('all_source_banks_drawn',dict(cases=1408,positive_union_appearances=11264,positive_arm_appearances=33792,distinct_N12_boards=len(global_boards),source_banks_sha256=sha(OUT/'source-banks.json')))
        call_history={}
        def ask(case,truth,policy,phase,step,candidates,queried,request,selection):
            nonlocal calls
            index=selection['query_index'];assert index is not None and index not in queried and calls<p['budget']['membership_calls']
            previous=list(call_history.get((case,index),[]));query=ledger.append('query',dict(case=case,policy=policy,phase=phase,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),previous_same_case_query_sequences=previous))
            label=bool((bits[truth,index//8]>>(index%8))&1);calls+=1;remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label]
            answer=ledger.append('answer',dict(case=case,policy=policy,phase=phase,step=step,query_sha256=query['sha256'],label=label,candidates_before=candidates,candidates_after=remaining));assert truth in remaining
            call_history.setdefault((case,index),[]).append(query['sequence'])
            return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=query['sequence'],query_sha256=query['sha256'],answer_sequence=answer['sequence'],answer_sha256=answer['sha256'],previous_same_case_query_sequences=previous)
        checkpoints=[]
        for source in sources:
            case=source['case'];truth=source['candidate'];candidates=list(source['initial_candidates']);queried=[];steps=[];terminal=None
            for step in range(4):
                if len(candidates)==1:break
                request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=len(boards),candidates=candidates,queried=queried);selection=select(request)
                if selection['query_index'] is None:
                    event=ledger.append('no_informative_query',dict(case=case,request=request,selection=selection));terminal=dict(sequence=event['sequence'],sha256=event['sha256']);break
                candidates,row=ask(case,truth,'shared','map',step,candidates,queried,request,selection);steps.append(row);queried.append(row['query_index'])
            data=dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=source['initial_candidates'],candidates=candidates,queried=queried,map_steps=steps,map_calls=len(steps),terminal_no_information=terminal)
            event=ledger.append('shared_checkpoint',data);checkpoints.append(dict(data,checkpoint_sequence=event['sequence'],checkpoint_sha256=event['sha256']));statuses[case]['map']='complete'
        worker.stdin.close();assert worker.wait()==0;worker=None
        save(OUT/'checkpoints.json',checkpoints);ledger.append('all_shared_checkpoints_complete',dict(cases=1408,shared_MAP_calls=calls,checkpoints_sha256=sha(OUT/'checkpoints.json')))
        mapping={};proofs={};cachekeys=set();bits_sha=sha(OUT/'predictions-bits.npy')
        for checkpoint in checkpoints:
            candidates=checkpoint['candidates'];identity=tuple(candidates)
            if len(candidates)==1 or identity in mapping:continue
            group=len(groups);mapping[identity]=group;folder=OUT/f'group{group}';folder.mkdir();before_states=budget.states;before_splits=budget.splits;begin=cpu();beginwall=time.perf_counter()
            group_cases=[r for r in checkpoints if tuple(r['candidates'])==identity];queried_union={i for r in group_cases for i in r['queried']}
            catalog=None;catalog_sha=None
            if len(candidates)>16:
                budget.state();full=(1<<len(candidates))-1;root=f'{full}:4';first=dict(candidates=candidates,depth_limit=4,root_state=root,feasible=False,states={root:dict(mask=full,depth_remaining=4,models=len(candidates),feasible=False,reason='binary_information_bound',attempts=[])},leaf_depths={},worst_leaf_depth=None)
            else:
                codes=np.zeros(len(boards),np.uint32)
                for local,cid in enumerate(candidates):codes|=predictions[cid].astype(np.uint32)<<local
                signatures,indices,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(i),board_count=int(n)) for s,i,n in zip(signatures,indices,counts)]
                save(folder/'catalog.json',dict(candidates=candidates,universe_boards=len(boards),signatures=catalog));catalog_sha=sha(folder/'catalog.json');first=firstlex_tree(candidates,catalog,4,budget)
            policy_data={};group_proofs={}
            for policy in POLICIES:
                sub=folder/policy;sub.mkdir();weights=None;receipt=None
                if policy=='firstlex':proof=first
                elif not first['feasible']:proof=first
                else:
                    if policy=='posterior':weights,receipt=integer_weights(candidates,scores,p['budget']['max_integer_bits'])
                    else:weights=[1]*len(candidates);receipt=dict(mode='uniform_over_all_remaining_candidates',candidates=candidates,integer_weights=['1']*len(candidates),weight_sum=len(candidates),no_candidate_excluded=True)
                    save(sub/'weights.json',receipt);proof=optimal_tree(candidates,weights,catalog,4,budget);assert proof['feasible'],'Previously feasible set returned infeasible'
                compiler_file='firstlex.py' if policy=='firstlex' or not first['feasible'] else 'compiler.py'
                inputs=dict(policy=policy,candidates=candidates,weights=list(map(str,weights)) if weights is not None else None,depth=4,catalog_sha256=catalog_sha,predictions_sha256=bits_sha,compiler_sha256=sha(SRC/compiler_file),shared_infeasibility_certificate=policy!='firstlex' and not first['feasible'],allowed_inputs_only=True)
                cachekey=hashlib.sha256(canonical(inputs)).hexdigest();assert cachekey not in cachekeys;cachekeys.add(cachekey);save(sub/'compiler-input.json',inputs);save(sub/'proof.json',proof)
                if proof['feasible']:
                    for local,cid in enumerate(candidates):
                        key=proof['root_state'];used=[]
                        while proof['states'][key]['reason']!='singleton_leaf':
                            node=proof['states'][key];index=node['chosen_split']['board_index'];assert index not in used and index not in queried_union;used.append(index);key=node['positive_state'] if predictions[cid,index] else node['negative_state']
                        assert proof['states'][key]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4
                    if weights is not None:assert int(proof['optimal_cost'])<=sum(w*first['leaf_depths'][str(c)] for c,w in zip(candidates,weights))
                policy_data[policy]=dict(proof_sha256=sha(sub/'proof.json'),compiler_input_sha256=sha(sub/'compiler-input.json'),weights_sha256=sha(sub/'weights.json') if receipt else None,cache_key=cachekey,feasible=proof['feasible']);group_proofs[policy]=proof
            row=dict(group=group,candidates=candidates,case_ids=[r['case'] for r in group_cases],feasible=first['feasible'],catalog_sha256=catalog_sha,policies=policy_data,memo_states=budget.states-before_states,canonical_split_attempts=budget.splits-before_splits,CPU_seconds=cpu()-begin,wall_seconds=time.perf_counter()-beginwall)
            save(folder/'result.json',row);groups.append(row);proofs[group]=group_proofs
        save(OUT/'groups.json',groups);ledger.append('all_trees_bound',dict(groups=len(groups),groups_sha256=sha(OUT/'groups.json'),checkpoints_sha256=sha(OUT/'checkpoints.json'),all_policies_compiled_before_continuation=True))
        for checkpoint in checkpoints:
            case=checkpoint['case'];truth=checkpoint['candidate'];group=mapping.get(tuple(checkpoint['candidates']));arms={}
            for policy in POLICIES:
                candidates=list(checkpoint['candidates']);queried=list(checkpoint['queried']);steps=[]
                if group is not None:
                    proofpath=OUT/f'group{group}/{policy}/proof.json';proofsha=sha(proofpath);assert proofsha==groups[group]['policies'][policy]['proof_sha256'];proof=read(proofpath);assert proof['candidates']==candidates and proof['depth_limit']==4 and proof['feasible']==groups[group]['feasible']
                    ledger.append('compiled_policy_bound',dict(case=case,policy=policy,group=group,proof_sha256=proofsha,group_result_sha256=sha(OUT/f'group{group}/result.json'),checkpoint_sha256=checkpoint['checkpoint_sha256']))
                    if proof['feasible']:
                        key=proof['root_state']
                        while proof['states'][key]['reason']!='singleton_leaf':
                            node=proof['states'][key];index=node['chosen_split']['board_index'];request=dict(group=group,policy=policy,proof_sha256=proofsha,state=key,candidates=candidates,queried=queried);selection=dict(query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates})
                            candidates,row=ask(case,truth,policy,'tree',checkpoint['map_calls']+len(steps),candidates,queried,request,selection);steps.append(row);queried=queried+[index];key=node['positive_state'] if row['label'] else node['negative_state']
                result=dict(tree_steps=steps,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],continuation_calls=len(steps),policy_calls=checkpoint['map_calls']+len(steps),cross_policy_reacquisitions=sum(bool(r['previous_same_case_query_sequences']) for r in steps))
                assert len(steps)<=4 and result['policy_calls']<=8;ledger.append('policy_result',dict(case=case,policy=policy,result=result));arms[policy]=result;statuses[case][policy]='complete'
            assert len({r['exact_true_singleton'] for r in arms.values()})==1
            row=dict(case=case,candidate=truth,bank=checkpoint['bank'],group=group,compiler_feasible=groups[group]['feasible'] if group is not None else None,shared_MAP_calls=checkpoint['map_calls'],arms=arms,actual_membership_calls=checkpoint['map_calls']+sum(r['continuation_calls'] for r in arms.values()),cross_policy_reacquisitions=sum(r['cross_policy_reacquisitions'] for r in arms.values()))
            row['unique_same_case_query_boards']=row['actual_membership_calls']-row['cross_policy_reacquisitions'];assert row['unique_same_case_query_boards']==sum(key[0]==case for key in call_history)
            ledger.append('case_result',row);results.append(row)
            if checkpoint['bank']==63:print('TRUTH',truth,'calls',calls,'states',budget.states,'splits',budget.splits,flush=True)
        shared=sum(r['shared_MAP_calls'] for r in results);totals={policy:sum(r['arms'][policy]['continuation_calls'] for r in results) for policy in POLICIES};F,P,U=[totals[k] for k in POLICIES]
        decisions=dict(uniform_vs_firstlex=dict(left=100*U,right=98*F,decision='inconclusive_zero_comparator' if F==0 else 'supported' if 100*U<=98*F else 'falsified',threshold_percent=2),uniform_vs_posterior=dict(left=100*U,right=95*P,decision='inconclusive_zero_comparator' if P==0 else 'supported' if 100*U<=95*P else 'falsified',threshold_percent=5))
        truth_results=[]
        for cfg in p['truths']:
            rows=[r for r in results if r['candidate']==cfg['candidate']];truth_results.append(dict(candidate=cfg['candidate'],cases=64,exact_cases=sum(r['arms']['firstlex']['exact_true_singleton'] for r in rows),infeasible_cases=sum(r['compiler_feasible'] is False for r in rows),prior_success_cases=sum(r['group'] is None for r in rows),shared_MAP_calls=sum(r['shared_MAP_calls'] for r in rows),continuation_calls={policy:sum(r['arms'][policy]['continuation_calls'] for r in rows) for policy in POLICIES}))
        outcome=dict(decisions=decisions,cases=1408,exact_cases=sum(r['arms']['firstlex']['exact_true_singleton'] for r in results),all_percase_exact_flags_identical_correctness_invariant=True,prior_success_cases=sum(r['group'] is None for r in results),feasible_checkpoint_cases=sum(r['compiler_feasible'] is True for r in results),infeasible_checkpoint_cases=sum(r['compiler_feasible'] is False for r in results),distinct_compiled_sets=len(groups),distinct_feasible_sets=sum(r['feasible'] for r in groups),shared_MAP_calls=shared,continuation_calls=totals,policy_calls={policy:shared+totals[policy] for policy in POLICIES},actual_membership_calls=calls,cross_policy_reacquisitions=sum(r['cross_policy_reacquisitions'] for r in results),unique_same_case_query_boards=sum(r['unique_same_case_query_boards'] for r in results),source_positive_union_appearances=11264,positive_arm_appearances=33792,distinct_N12_boards=len(global_boards),source_cases_with_prior_same_study_duplicates=sum(bool(r['prior_same_study_cases']) for r in sources),source_cases_matching_H3bm_eight_banks=sum(bool(r['historical_exact8_matches']) for r in sources),truth_results=truth_results,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits)
        assert calls==shared+sum(totals.values());ledger.append('study_complete',outcome);ledger.close();save(OUT/'case-results.json',results);save(OUT/'case-statuses.json',statuses)
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),**outcome,ledger_sha256=sha(OUT/'query-ledger.jsonl'),ledger_records=ledger.sequence,ledger_last_record_sha256=ledger.previous,case_results_sha256=sha(OUT/'case-results.json'),groups_sha256=sha(OUT/'groups.json'),case_statuses_sha256=sha(OUT/'case-statuses.json'),source_banks_sha256=sha(OUT/'source-banks.json'),checkpoints_sha256=sha(OUT/'checkpoints.json'),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        if worker is not None:worker.terminate();worker.wait()
        save(OUT/'case-statuses.json',statuses);save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,completed_groups=groups,actual_membership_calls=calls,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
