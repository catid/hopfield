"""Independent source reconstruction, all-alternative proofs and full paid replay."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys
sys.set_int_max_str_digits(0)
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
POLICIES=['firstlex','posterior','uniform']


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify(root,require_summary=True):
    started=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bq';P=R/'results/research/loki/h3bq';B=R/'results/research/loki/h3bc'
    read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    deps=[R/'research/20261114/loki/h3bi/verify_public.py',R/'research/20261114/loki/h3bj/verify_public.py',R/'research/20261114/loki/h3bl/verify_public.py'];helper=load(deps[0],'h3bq_independent_firstlex');chain=load(deps[1],'h3bq_independent_chain').chain;weighted=load(deps[2],'h3bq_independent_weighted')
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');parity=read(P/'population-parity.json');sources=read(P/'source-banks.json');checkpoints=read(P/'checkpoints.json');groups=read(P/'groups.json');cases=read(P/'case-results.json');pre=read(S/'preflight.json');seedaudit=read(S/'seed-audit.json');ledger_bytes=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());events=chain(ledger_bytes);export=read(P/'ledger-export.json')
    check('whole_study_budget',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and g['global_memo_states']<=2000000 and g['global_canonical_split_attempts']<=50000000 and g['actual_membership_calls']<=22528 and p['budget']['GPUs']==0)
    check('stage_seals',all(g[k]==raw[k] for k in raw if k!='utc') and pre['utc']<p['frozen_utc']<prep['utc']<events[0]['utc']<raw['utc'] and g['protocol_sha256']==prep['protocol_sha256']==sha(S/'protocol.json') and g['source_sha256']==sha(S/'study.py') and g['preparation_sha256']==sha(P/'preparation.json'))
    for field,name in [('case_results_sha256','case-results.json'),('groups_sha256','groups.json'),('case_statuses_sha256','case-statuses.json'),('source_banks_sha256','source-banks.json'),('checkpoints_sha256','checkpoints.json')]:check('seal_'+name,g[field]==sha(P/name))
    for field,name in [('population_parity_sha256','population-parity.json'),('ledger_gate_sha256','ledger-gate.json'),('weighted_gate_sha256','synthetic-gate.json'),('firstlex_gate_sha256','firstlex-synthetic-gate.json')]:check('prep_'+name,prep[field]==sha(P/name))
    check('ledger_export',hashlib.sha256(ledger_bytes).hexdigest()==g['ledger_sha256']==export['raw_sha256'] and len(ledger_bytes)==export['raw_bytes'] and sha(P/'query-ledger.jsonl.gz')==export['gzip_sha256'] and len(events)==g['ledger_records'] and events[-1]['sha256']==g['ledger_last_record_sha256'])
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    check('unchanged_compilers',sha(S/'firstlex.py')==sha(R/'research/20261114/loki/h3bm/compiler.py') and sha(S/'compiler.py')==sha(R/'research/20261114/loki/h3bl/compiler.py') and sha(S/'weights.py')==sha(R/'research/20261114/loki/h3bl/weights.py'))
    for name in ['selector.py','ledger.py']:check('unchanged_'+name,sha(S/name)==sha(R/'research/20261114/loki/h3bm'/name))
    check('prefreeze_fixtures',pre['all_toy_gates_passed'] and pre['new_source_draws']==pre['new_target_calls']==0 and pre['large_integer_decimal_serialization_digits']==5001 and pre['weighted_gate_sha256']==sha(S/'preflight-weighted-gate.json') and pre['firstlex_gate_sha256']==sha(S/'preflight-firstlex-gate.json') and pre['seed_audit_sha256']==sha(S/'seed-audit.json') and prep['preflight_sha256']==sha(S/'preflight.json'))
    for row in pre['eight_likelihood_fixtures']:
        M=row['population'];k=row['orbit_count'];count=sum(1 for _ in itertools.combinations(range(M),8));prior=math.comb(55,k);check('eight_likelihood_'+str(M),row['eight_subsets']==count and row['prior_subsets']==prior and row['score']==(str(count*prior) if M>=8 else None))
    check('direct_draw_toy',pre['eight_direct_draw_fixture']==dict(seed=813,population=20,indices=np.random.default_rng(813).choice(20,8,replace=False).tolist()))
    historic_seeds=set()
    def walk(x,context=''):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,context+'/'+k)
        elif isinstance(x,list):
            for v in x:walk(v,context)
        elif isinstance(x,int) and 'seed' in context.lower():historic_seeds.add(x)
    for rel,digest in seedaudit['historical_protocol_sha256'].items():check('old_seed_protocol '+rel,sha(R/rel)==digest);walk(read(R/rel))
    seeds=[2026600000+64*r['candidate']+b for r in p['truths'] for b in range(64)]
    check('new_seed_registry',len(seeds)==len(set(seeds))==1408 and seeds==seedaudit['new_seed_values']==[s for r in p['truths'] for s in r['seeds']] and sorted(historic_seeds)==seedaudit['historical_explicit_seed_values'] and not set(seeds)&historic_seeds and not seedaudit['collisions'])
    total_states=0;total_splits=0
    for filename,prefilename,mode in [('synthetic-gate.json','preflight-weighted-gate.json','weighted'),('firstlex-synthetic-gate.json','preflight-firstlex-gate.json','firstlex')]:
        toy=read(P/filename);oldtoy=read(S/prefilename);total_states+=1
        for row in toy['fixtures']:
            cat=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(row['signatures'])];proof=row['result']
            if mode=='weighted':
                cost=weighted.brute((1<<len(row['weights']))-1,row['depth'],row['signatures'],row['weights']);check('weighted_toy '+row['name'],proof['optimal_cost']==row['expected_cost']==(str(cost) if cost is not None else None));ns,nt,_=weighted.weighted_certificate(proof,cat,check,'toy_weighted_'+row['name'])
            else:
                check('firstlex_toy '+row['name'],proof['feasible']==row['expected']==bool(helper.brute((1<<len(proof['candidates']))-1,row['depth'],row['signatures'])));ns,nt,_=helper.certificate(proof,cat,check,'toy_firstlex_'+row['name'])
            total_states+=ns;total_splits+=nt
        check('toy_global_accounting_'+mode,len(toy['fixtures'])==8 and all(toy[k]==oldtoy[k] for k in toy if k!='utc') and toy['global_states_after_gate']==total_states and toy['global_split_attempts_after_gate']==total_splits)
    gate=read(P/'ledger-gate.json');synthetic_bytes=(P/'synthetic-ledger.jsonl').read_bytes();synthetic=chain(synthetic_bytes)
    check('durability_gate',gate['all8_prior_format_logical_queries_and_answers_identical'] and gate['oracle_callbacks_observed_durable_query_before_label']==8 and len(synthetic)==16 and sha(P/'synthetic-ledger.jsonl')==gate['synthetic_ledger_sha256'] and all(gate['tamper_cases_rejected'].values()))
    for i,(qr,ar) in enumerate(p['ledger_gate']['archival_query_answer_paths']):
        q=read(R/qr);a=read(R/ar);sq=synthetic[2*i];sa=synthetic[2*i+1];check('durability_fixture'+str(i),sq['data']==dict(index=i,request=q['request'],selection=q['selection'],board_zero_based=q['board_zero_based']) and sa['data']==dict(index=i,label=a['label'],candidates_before=a['candidates_before'],candidates_after=a['candidates_after'],query_sha256=sq['sha256']) and gate['paired_format_parity'][i]['oracle_callback_monotonic_ns']>sq['monotonic_ns'])
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];bits=np.load(B/'predictions-bits.npy');old=read(R/p['normalizers']['selection_path']);oldprep=read(B/'preparation.json');h12=helper.hits(ordinary,orbits);h13=helper.hits(boards,orbits);base12,extra12=helper.parts(12,orbits);base13,extra13=helper.parts(13,orbits);sourcepop={};sourcepred=[];scores=[];score16=[]
    for cid,chosen in enumerate(choices):
        m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[m12];sourcepop[cid]=columns;sourcepred.append(m12);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
        check(f'candidate{cid}_source',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid]);packed=np.packbits(m13,bitorder='little');record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
        check(f'candidate{cid}_target',np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]);scores.append(str(math.comb(55,len(chosen))*math.comb(len(columns),8)) if len(columns)>=8 else None);score16.append(str(math.comb(55,len(chosen))*math.comb(len(columns),16)) if len(columns)>=16 else None)
    check('fresh_parity_and_scores',scores==read(P/'score-values8.json') and score16==read(P/'score-values.json') and parity['all1541_source_target_filters_and_scores_verified'] and parity['census_summary_sha256']==sha(B/'summary.json') and parity['score8_sha256']==sha(P/'score-values8.json') and parity['score16_sha256']==sha(P/'score-values.json') and read(B/'summary.json')['audit']['all1541N12_masks_and_N13_literal_populations_verified'])
    predictions=np.unpackbits(bits,axis=1,count=73712,bitorder='little').astype(bool);sourcepred=np.array(sourcepred,bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};history={}
    for r in read(R/'results/research/loki/h3bm/source-pairs.json'):
        population=sourcepop[r['candidate']];i=np.random.default_rng(r['seed16']).choice(len(population),16,replace=False);j=np.random.default_rng(r['subset_seed8']).choice(16,8,replace=False);sample=population[i[j]];digest=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest();check('historic_bank_'+str(r['case']),r['columns8_zero_based']==sample.tolist() and r['unordered_sample_sha256']['8']==digest);history.setdefault(digest,[]).append(dict(study='H3bm',case=r['case'],candidate=r['candidate']))
    check('history_seal',history==read(P/'historical-eight-banks.json') and parity['historical_eight_banks_sha256']==sha(P/'historical-eight-banks.json'))
    at=0;paid=0;call_history={}
    def take(kind):
        nonlocal at
        e=events[at];at+=1;check('event_kind'+str(at-1),e['kind']==kind);return e
    check('start_binding',take('study_start')['data']==dict(protocol_sha256=sha(S/'protocol.json'),preparation_sha256=sha(P/'preparation.json')))
    expected_sources=[];fresh={};global_boards=set()
    for cfg in p['truths']:
        truth=cfg['candidate'];population=sourcepop[truth]
        for bank,seed in enumerate(cfg['seeds']):
            case=len(expected_sources);indices=np.random.default_rng(seed).choice(len(population),8,replace=False);sample=population[indices];initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();digest=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest()
            data=dict(case=case,candidate=truth,bank=bank,seed=seed,source_population=len(population),indices=indices.tolist(),columns_zero_based=sample.tolist(),initial_candidates=initial,unordered_sample_sha256=digest,prior_same_study_cases=list(fresh.get(digest,[])),historical_exact8_matches=history.get(digest,[]));event=take('source_bank');row=dict(data,source_sequence=event['sequence'],source_record_sha256=event['sha256'])
            check('fresh_source'+str(case),event['data']==data and row==sources[case] and len(set(indices.tolist()))==8 and truth in initial and all(scores[c] is not None for c in initial));expected_sources.append(row);fresh.setdefault(digest,[]).append(case);global_boards.update(map(bytes,sample))
    check('all_sources_before_MAP',take('all_source_banks_drawn')['data']==dict(cases=1408,positive_union_appearances=11264,positive_arm_appearances=33792,distinct_N12_boards=len(global_boards),source_banks_sha256=sha(P/'source-banks.json')))
    selection_cache={}
    def choose(candidates,queried):
        identity=(tuple(candidates),tuple(queried))
        if identity in selection_cache:return selection_cache[identity]
        best=min(int(scores[c]) for c in candidates);ties=[c for c in candidates if int(scores[c])==best];guide=min(ties);counts=np.count_nonzero(predictions[candidates],axis=0);dis=np.where(predictions[guide],len(candidates)-counts,counts);dis[queried]=-1;maximum=int(dis.max());bt=np.flatnonzero(dis==maximum);index=int(bt[0]) if maximum>0 else None
        result=dict(MAP_tied_candidates=ties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=maximum,optimal_board_count=len(bt),query_index=index,candidate_predictions={str(c):bool(predictions[c,index]) for c in candidates} if index is not None else None,stop_reason=None if index is not None else 'no_informative_unqueried_board_remaining_identity_unresolved');selection_cache[identity]=result;return result
    def replay(case,truth,policy,phase,step,candidates,queried,request,selection):
        nonlocal paid
        index=selection['query_index'];previous=list(call_history.get((case,index),[]));q=take('query');a=take('answer');label=bool(predictions[truth,index]);remaining=[c for c in candidates if bool(predictions[c,index])==label]
        check('paid_query'+str(q['sequence']),q['data']==dict(case=case,policy=policy,phase=phase,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),previous_same_case_query_sequences=previous) and a['data']==dict(case=case,policy=policy,phase=phase,step=step,query_sha256=q['sha256'],label=label,candidates_before=candidates,candidates_after=remaining) and q['sequence']+1==a['sequence'] and q['monotonic_ns']<a['monotonic_ns'] and index not in queried and truth in remaining)
        paid+=1;call_history.setdefault((case,index),[]).append(q['sequence']);return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=q['sequence'],query_sha256=q['sha256'],answer_sequence=a['sequence'],answer_sha256=a['sha256'],previous_same_case_query_sequences=previous)
    expected_checkpoints=[]
    for source in expected_sources:
        case=source['case'];truth=source['candidate'];candidates=list(source['initial_candidates']);queried=[];steps=[];terminal=None
        for step in range(4):
            if len(candidates)==1:break
            request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=73712,candidates=candidates,queried=queried);selection=choose(candidates,queried)
            if selection['query_index'] is None:
                event=take('no_informative_query');check('no_information'+str(case),event['data']==dict(case=case,request=request,selection=selection));terminal=dict(sequence=event['sequence'],sha256=event['sha256']);break
            candidates,row=replay(case,truth,'shared','map',step,candidates,queried,request,selection);steps.append(row);queried.append(row['query_index'])
        data=dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=source['initial_candidates'],candidates=candidates,queried=queried,map_steps=steps,map_calls=len(steps),terminal_no_information=terminal);event=take('shared_checkpoint');row=dict(data,checkpoint_sequence=event['sequence'],checkpoint_sha256=event['sha256']);check('checkpoint'+str(case),event['data']==data and row==checkpoints[case]);expected_checkpoints.append(row)
    shared=paid;check('all_MAP_before_compilation',take('all_shared_checkpoints_complete')['data']==dict(cases=1408,shared_MAP_calls=shared,checkpoints_sha256=sha(P/'checkpoints.json')))
    expected_groups={};proofs={};real_states=0;real_splits=0;branches=0;maxLCM=0;maxweight=0;cachekeys=set()
    for row in expected_checkpoints:
        key=tuple(row['candidates'])
        if len(key)>1 and key not in expected_groups:expected_groups[key]=len(expected_groups)
    check('all_group_sets',len(expected_groups)==len(groups) and all(row['group']==i and expected_groups[tuple(row['candidates'])]==i for i,row in enumerate(groups)))
    for row in groups:
        group=row['group'];candidates=row['candidates'];folder=P/f'group{group}';catalog=None;catalog_sha=None;caseids=[r['case'] for r in checkpoints if r['candidates']==candidates];before_states=real_states;before_splits=real_splits;proofs[group]={}
        if len(candidates)<=16:
            codes=np.sum(predictions[candidates].astype(np.uint32)*(1<<np.arange(len(candidates),dtype=np.uint32))[:,None],axis=0,dtype=np.uint32);signatures,counts=np.unique(codes,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(np.flatnonzero(codes==s)[0]),board_count=int(n)) for s,n in zip(signatures,counts)];catalog_sha=sha(folder/'catalog.json');check('catalog'+str(group),read(folder/'catalog.json')==dict(candidates=candidates,universe_boards=73712,signatures=catalog))
        first=read(folder/'firstlex/proof.json');ns,nt,_=helper.certificate(first,catalog or [],check,f'g{group}_firstlex_');real_states+=ns;real_splits+=nt;check('first_feasibility'+str(group),first['candidates']==candidates and first['depth_limit']==4 and first['feasible']==row['feasible']);expected_policy={}
        for policy in POLICIES:
            sub=folder/policy;proof=read(sub/'proof.json');weights=None;receipt=None
            if policy=='firstlex' or not first['feasible']:check(f'g{group}_{policy}_shared_proof',proof==first)
            else:
                if policy=='posterior':
                    denominators=[int(scores[c]) for c in candidates];L=1;lengths=[]
                    for d in denominators:L=math.lcm(L,d);lengths.append(L.bit_length())
                    weights=[L//d for d in denominators];receipt=dict(score_denominators=list(map(str,denominators)),common_denominator=str(L),LCM_intermediate_bit_lengths=lengths,integer_weights=list(map(str,weights)),weight_bit_lengths=[w.bit_length() for w in weights],max_bits=100000);maxLCM=max(maxLCM,max(lengths));maxweight=max(maxweight,max(w.bit_length() for w in weights))
                else:weights=[1]*len(candidates);receipt=dict(mode='uniform_over_all_remaining_candidates',candidates=candidates,integer_weights=['1']*len(candidates),weight_sum=len(candidates),no_candidate_excluded=True)
                check(f'g{group}_{policy}_weights',read(sub/'weights.json')==receipt and proof['weights']==list(map(str,weights)) and proof['candidates']==candidates and proof['depth_limit']==4);ns,nt,_=weighted.weighted_certificate(proof,catalog,check,f'g{group}_{policy}_');real_states+=ns;real_splits+=nt
                check(f'g{group}_{policy}_nonincrease',proof['feasible'] and int(proof['optimal_cost'])<=sum(w*first['leaf_depths'][str(c)] for c,w in zip(candidates,weights)))
            compiler='firstlex.py' if policy=='firstlex' or not first['feasible'] else 'compiler.py';inputs=dict(policy=policy,candidates=candidates,weights=list(map(str,weights)) if weights is not None else None,depth=4,catalog_sha256=catalog_sha,predictions_sha256=sha(B/'predictions-bits.npy'),compiler_sha256=sha(S/compiler),shared_infeasibility_certificate=policy!='firstlex' and not first['feasible'],allowed_inputs_only=True);key=hashlib.sha256(canonical(inputs)).hexdigest()
            check(f'g{group}_{policy}_full_blind_inputs',read(sub/'compiler-input.json')==inputs and key not in cachekeys);cachekeys.add(key);expected_policy[policy]=dict(proof_sha256=sha(sub/'proof.json'),compiler_input_sha256=sha(sub/'compiler-input.json'),weights_sha256=sha(sub/'weights.json') if receipt else None,cache_key=key,feasible=proof['feasible']);proofs[group][policy]=proof
            if proof['feasible']:
                for local,cid in enumerate(candidates):
                    state=proof['root_state'];used=[]
                    while proof['states'][state]['reason']!='singleton_leaf':
                        node=proof['states'][state];index=node['chosen_split']['board_index'];check(f'g{group}_{policy}_candidate{cid}_step{len(used)}',index not in used and all(index not in checkpoints[c]['queried'] for c in caseids) and bool(predictions[cid,index])==bool(node['chosen_split']['positive_mask']&(1<<local)));used.append(index);state=node['positive_state'] if predictions[cid,index] else node['negative_state']
                    check(f'g{group}_{policy}_candidate{cid}_leaf',proof['states'][state]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4);branches+=1
        check('group_receipt'+str(group),row==read(folder/'result.json') and row['case_ids']==caseids and row['catalog_sha256']==catalog_sha and row['policies']==expected_policy and row['memo_states']==real_states-before_states and row['canonical_split_attempts']==real_splits-before_splits)
    total_states+=real_states;total_splits+=real_splits;check('all_trees_before_first_continuation',take('all_trees_bound')['data']==dict(groups=len(groups),groups_sha256=sha(P/'groups.json'),checkpoints_sha256=sha(P/'checkpoints.json'),all_policies_compiled_before_continuation=True))
    computed=[]
    for checkpoint in checkpoints:
        case=checkpoint['case'];truth=checkpoint['candidate'];group=expected_groups.get(tuple(checkpoint['candidates']));arms={}
        for policy in POLICIES:
            candidates=list(checkpoint['candidates']);queried=list(checkpoint['queried']);steps=[]
            if group is not None:
                proof=proofs[group][policy];proofsha=sha(P/f'group{group}/{policy}/proof.json');check(f'case{case}_{policy}_bound',take('compiled_policy_bound')['data']==dict(case=case,policy=policy,group=group,proof_sha256=proofsha,group_result_sha256=sha(P/f'group{group}/result.json'),checkpoint_sha256=checkpoint['checkpoint_sha256']))
                if proof['feasible']:
                    state=proof['root_state']
                    while proof['states'][state]['reason']!='singleton_leaf':
                        node=proof['states'][state];index=node['chosen_split']['board_index'];request=dict(group=group,policy=policy,proof_sha256=proofsha,state=state,candidates=candidates,queried=queried);selection=dict(query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates});candidates,row=replay(case,truth,policy,'tree',checkpoint['map_calls']+len(steps),candidates,queried,request,selection);steps.append(row);queried=queried+[index];state=node['positive_state'] if row['label'] else node['negative_state']
            result=dict(tree_steps=steps,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],continuation_calls=len(steps),policy_calls=checkpoint['map_calls']+len(steps),cross_policy_reacquisitions=sum(bool(r['previous_same_case_query_sequences']) for r in steps))
            check(f'case{case}_{policy}_result',take('policy_result')['data']==dict(case=case,policy=policy,result=result) and len(steps)<=4 and result['policy_calls']<=8 and (group is not None and groups[group]['feasible'] or not steps));arms[policy]=result
        row=dict(case=case,candidate=truth,bank=checkpoint['bank'],group=group,compiler_feasible=groups[group]['feasible'] if group is not None else None,shared_MAP_calls=checkpoint['map_calls'],arms=arms,actual_membership_calls=checkpoint['map_calls']+sum(r['continuation_calls'] for r in arms.values()),cross_policy_reacquisitions=sum(r['cross_policy_reacquisitions'] for r in arms.values()));row['unique_same_case_query_boards']=row['actual_membership_calls']-row['cross_policy_reacquisitions']
        check('case_complete'+str(case),row==cases[case]==take('case_result')['data'] and row['unique_same_case_query_boards']==sum(key[0]==case for key in call_history) and len({r['exact_true_singleton'] for r in arms.values()})==1);computed.append(row)
    totals={policy:sum(r['arms'][policy]['continuation_calls'] for r in computed) for policy in POLICIES};F,Post,U=[totals[k] for k in POLICIES];decisions=dict(uniform_vs_firstlex=dict(left=100*U,right=98*F,decision='inconclusive_zero_comparator' if F==0 else 'supported' if 100*U<=98*F else 'falsified',threshold_percent=2),uniform_vs_posterior=dict(left=100*U,right=95*Post,decision='inconclusive_zero_comparator' if Post==0 else 'supported' if 100*U<=95*Post else 'falsified',threshold_percent=5));truth_results=[]
    for cfg in p['truths']:
        rows=[r for r in computed if r['candidate']==cfg['candidate']];truth_results.append(dict(candidate=cfg['candidate'],cases=64,exact_cases=sum(r['arms']['firstlex']['exact_true_singleton'] for r in rows),infeasible_cases=sum(r['compiler_feasible'] is False for r in rows),prior_success_cases=sum(r['group'] is None for r in rows),shared_MAP_calls=sum(r['shared_MAP_calls'] for r in rows),continuation_calls={policy:sum(r['arms'][policy]['continuation_calls'] for r in rows) for policy in POLICIES}))
    outcome=dict(decisions=decisions,cases=1408,exact_cases=sum(r['arms']['firstlex']['exact_true_singleton'] for r in computed),all_percase_exact_flags_identical_correctness_invariant=True,prior_success_cases=sum(r['group'] is None for r in computed),feasible_checkpoint_cases=sum(r['compiler_feasible'] is True for r in computed),infeasible_checkpoint_cases=sum(r['compiler_feasible'] is False for r in computed),distinct_compiled_sets=len(groups),distinct_feasible_sets=sum(r['feasible'] for r in groups),shared_MAP_calls=shared,continuation_calls=totals,policy_calls={policy:shared+totals[policy] for policy in POLICIES},actual_membership_calls=paid,cross_policy_reacquisitions=sum(r['cross_policy_reacquisitions'] for r in computed),unique_same_case_query_boards=sum(r['unique_same_case_query_boards'] for r in computed),source_positive_union_appearances=11264,positive_arm_appearances=33792,distinct_N12_boards=len(global_boards),source_cases_with_prior_same_study_duplicates=sum(bool(r['prior_same_study_cases']) for r in sources),source_cases_matching_H3bm_eight_banks=sum(bool(r['historical_exact8_matches']) for r in sources),truth_results=truth_results,global_memo_states=total_states,global_canonical_split_attempts=total_splits)
    check('all_final_endpoints_and_counters',take('study_complete')['data']==outcome and at==len(events) and all(g[k]==v for k,v in outcome.items()) and paid==shared+sum(totals.values()) and shared<=5632 and all(v<=5632 for v in totals.values()) and maxLCM<=100000 and maxweight<=100000 and read(P/'case-statuses.json')==[dict(case=i,source='complete',map='complete',firstlex='complete',posterior='complete',uniform='complete') for i in range(1408)])
    if require_summary:
        s=read(P/'summary.json');check('summary_identity',s['status']=='validated' and all(s[k]==v for k,v in outcome.items()) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py') and s['verifier_dependency_sha256']=={str(path.relative_to(R)):sha(path) for path in deps})
        for rel,digest in s['public_sha256'].items():check('public_export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('plot_provenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all1541_source_target_filters_and_scores_verified=True,all1408_fresh_direct_eight_banks_reconstructed=True,all1408_archived_eight_banks_reconstructed_for_duplicate_accounting=True,all_seed_registry_values_verified=True,all_shared_MAP_decisions_replayed=shared,all_real_compiler_proof_states_verified=real_states,all_real_canonical_split_attempts_verified=real_splits,all_remaining_candidate_policy_branches_verified=branches,all_new_paid_query_answer_calls_replayed=paid,all_percase_accuracy_flags_identical=True,all_infeasible_and_prior_success_cases_retained=True,all_two_integer_cost_endpoints_separately_verified=True,all_cache_identities_include_policy_weights_depth_catalog=True,all_compilation_before_continuation_labels=True,all_cross_policy_reacquisitions_charged=True,maximum_LCM_bits=maxLCM,maximum_integer_weight_bits=maxweight,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
