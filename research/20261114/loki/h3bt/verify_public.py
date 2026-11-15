"""Independent online-policy replay, exact proofs, archive accounting and lemma audit."""
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
from audit_costs import GraphAudit,exact_guard,fixtures


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def restricted(proof,candidates):
    def visit(state,active):
        if len(active)==1:return dict(candidate=active[0])
        node=proof['states'][state];split=node['chosen_split'];positive=[c for c in active if split['positive_mask']&(1<<proof['candidates'].index(c))];negative=[c for c in active if c not in positive]
        if not positive:return visit(node['negative_state'],negative)
        if not negative:return visit(node['positive_state'],positive)
        return dict(board_index=split['board_index'],negative=visit(node['negative_state'],negative),positive=visit(node['positive_state'],positive))
    return visit(proof['root_state'],list(candidates))


def verify(root,require_summary=True):
    started=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bt';P=R/'results/research/loki/h3bt';B=R/'results/research/loki/h3bc';A=R/'results/research/loki/h3br';canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode();checks=0
    adapter=load(S.parent/'export_layer.py','h3bt_export_adapter');hashes={}
    def read(path):
        data=adapter.logical_bytes(path);hashes[str(path)]=hashlib.sha256(data).hexdigest();return json.loads(data)
    def sha(path):
        key=str(path)
        if key not in hashes:hashes[key]=adapter.logical_sha256(path)
        return hashes[key]
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    deps=[R/'research/20261114/loki/h3bi/verify_public.py',R/'research/20261114/loki/h3bj/verify_public.py',R/'research/20261114/loki/h3bl/verify_public.py',S/'audit_costs.py',S.parent/'export_layer.py'];helper=load(deps[0],'h3bt_independent_unweighted');chain=load(deps[1],'h3bt_independent_chain').chain;weighted=load(deps[2],'h3bt_independent_weighted')
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');parity=read(P/'population-parity.json');groups=read(P/'groups.json');cases=read(P/'case-results.json');comparisons=read(P/'baseline-comparison.json');pre=read(S/'preflight.json');prior=read(A/'summary.json');sources=read(R/p['archive']['sources']);oldcases=read(R/p['archive']['cases']);oldcheckpoints=read(R/p['archive']['checkpoints']);parent_bytes=gzip.decompress((R/p['archive']['ledger']).read_bytes());parent=chain(parent_bytes);ledger_bytes=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());events=chain(ledger_bytes);export=read(P/'ledger-export.json');prior_protocol=read(R/'research/20261114/loki/h3br/protocol.json')
    check('whole_budget',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and g['global_memo_states']<=2000000 and g['global_split_evaluations']<=50000000 and g['new_policy_calls']<=32768 and p['budget']['GPUs']==0 and g['new_truth_draws']==g['new_source_draws']==g['new_source_positive_labels']==0)
    check('stage_seals',all(g[k]==raw[k] for k in raw if k!='utc') and pre['utc']<p['frozen_utc']<prep['utc']<events[0]['utc']<raw['utc'] and g['protocol_sha256']==prep['protocol_sha256']==sha(S/'protocol.json') and g['source_sha256']==sha(S/'study.py') and g['preparation_sha256']==sha(P/'preparation.json'))
    for field,name in [('case_results_sha256','case-results.json'),('groups_sha256','groups.json'),('case_statuses_sha256','case-statuses.json'),('baseline_comparison_sha256','baseline-comparison.json'),('counterfactual_registry_sha256','counterfactual-groups.json')]:check('seal_'+name,g[field]==sha(P/name))
    for field,name in [('population_parity_sha256','population-parity.json'),('ledger_gate_sha256','ledger-gate.json'),('synthetic_gate_sha256','synthetic-gate.json'),('restriction_gate_sha256','restriction-gate.json'),('cost_gate_sha256','cost-gate.json')]:check('prep_'+name,prep[field]==sha(P/name))
    check('both_ledger_chains',hashlib.sha256(ledger_bytes).hexdigest()==g['ledger_sha256']==export['raw_sha256'] and len(ledger_bytes)==export['raw_bytes'] and sha(P/'query-ledger.jsonl.gz')==export['gzip_sha256'] and len(events)==g['ledger_records'] and events[-1]['sha256']==g['ledger_last_record_sha256'] and hashlib.sha256(parent_bytes).hexdigest()==p['archive']['raw_ledger_sha256']==prior['ledger_sha256'] and len(parent)==prior['ledger_records'])
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    check('unchanged_algorithms',sha(S/'compiler.py')==sha(R/'research/20261114/loki/h3bl/compiler.py') and sha(S/'weights.py')==sha(R/'research/20261114/loki/h3bl/weights.py') and sha(S/'selector.py')==sha(R/'research/20261114/loki/h3bm/selector.py') and sha(S/'ledger.py')==sha(R/'research/20261114/loki/h3bm/ledger.py'))
    check('prefreeze_baseline',pre['all4096_archived_cases_retained'] and pre['baseline_policy_calls']==10355 and pre['baseline_exact_cases']==4083 and pre['baseline_infeasible_cases']==13 and pre['new_truth_draws']==pre['new_source_draws']==pre['new_membership_calls']==0 and pre['baseline_summary_sha256']==sha(A/'summary.json') and pre['baseline_case_results_sha256']==sha(A/'case-results.json') and pre['restriction_lemma_sha256']==sha(S/'LEMMA.md') and prep['preflight_sha256']==sha(S/'preflight.json'))
    toy=read(P/'synthetic-gate.json');pretoy=read(S/'preflight-synthetic-gate.json');total_states=1;total_splits=0;subset_count=0;subset_paths=0
    for row in toy['fixtures']:
        cat=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(row['signatures'])];proof=row['result'];cost=weighted.brute((1<<len(row['weights']))-1,row['depth'],row['signatures'],row['weights']);check('toy_'+row['name'],proof['optimal_cost']==row['expected_cost']==(str(cost) if cost is not None else None));ns,nt,_=weighted.weighted_certificate(proof,cat,check,'toy_'+row['name']);total_states+=ns;total_splits+=nt
        if proof['feasible']:
            for mask in range(1,1<<len(proof['candidates'])):
                chosen=[c for i,c in enumerate(proof['candidates']) if mask&(1<<i)];tree=restricted(proof,chosen);subset_count+=1
                for cid in chosen:
                    node=tree;used=[];local=proof['candidates'].index(cid)
                    while 'candidate' not in node:
                        index=node['board_index'];check('toy_restricted_query',index not in used);used.append(index);node=node['positive'] if row['signatures'][index]&(1<<local) else node['negative']
                    check('toy_restricted_leaf',node['candidate']==cid and len(used)<=proof['leaf_depths'][str(cid)]);subset_paths+=1
    restriction=dict(all_nonempty_toy_subsets_verified=subset_count,all_subset_candidate_paths_verified=subset_paths,depth_never_increases=True);rg=read(P/'restriction-gate.json');check('toy_counters_and_lemma',len(toy['fixtures'])==8 and all(toy[k]==pretoy[k] for k in toy if k!='utc') and total_states==toy['global_states_after_gate'] and total_splits==toy['global_split_attempts_after_gate'] and pre['restriction_gate']==restriction and all(rg[k]==v for k,v in restriction.items()) and pre['weighted_synthetic_gate_sha256']==sha(S/'preflight-synthetic-gate.json'))
    cost_gate=fixtures(P,S,read,sha,canonical,weighted,check);total_states+=cost_gate['states'];total_splits+=cost_gate['splits'];runtime_cost_gate=read(P/'cost-gate.json');check('combined_runtime_gate_counts',total_states==prep['runtime_gate_memo_states']==runtime_cost_gate['global_states_after_gate']==274 and total_splits==prep['runtime_gate_split_evaluations']==runtime_cost_gate['global_split_evaluations_after_gate']==314 and pre['cost_gate_sha256']==sha(S/'preflight-cost-gate.json') and pre['guard_lemma_sha256']==sha(S/'GUARD_LEMMA.md') and pre['counterfactual_parity_and_accept_reject_tie_gates_passed'])
    proof_export_audit=adapter.validate_export(P);export_provenance=read(P/'export-provenance.json');check('lossless_export_sources',export_provenance['export_source_sha256']==sha(S/'export.py') and export_provenance['adapter_source_sha256']==sha(S.parent/'export_layer.py'))
    cache=pre['online_compiler_cache_fixture'];check('prefreeze_online_cache',cache['optimal_cost']=='50' and cache['weights']==['12','6','4','3'] and cache['repeated_identical_inputs_add_zero_states_and_splits'] and cache['predictions_readonly'] and pre['seventeen_candidate_information_bound_verified'])
    gate=read(P/'ledger-gate.json');synthetic=chain((P/'synthetic-ledger.jsonl').read_bytes());check('durable_gate',len(synthetic)==16 and gate['all8_prior_format_logical_queries_and_answers_identical'] and gate['oracle_callbacks_observed_durable_query_before_label']==8 and all(gate['tamper_cases_rejected'].values()) and gate['synthetic_ledger_sha256']==sha(P/'synthetic-ledger.jsonl'))
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];bits=np.load(B/'predictions-bits.npy');old=read(R/p['normalizers']['selection_path']);oldprep=read(B/'preparation.json');bits_sha=sha(B/'predictions-bits.npy');h12=helper.hits(ordinary,orbits);h13=helper.hits(boards,orbits);base12,extra12=helper.parts(12,orbits);base13,extra13=helper.parts(13,orbits);sourcepop={};sourcepred=[];scores=[];score16=[]
    for cid,chosen in enumerate(choices):
        m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[m12];sourcepop[cid]=columns;sourcepred.append(m12);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());check(f'candidate{cid}_source',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid]);packed=np.packbits(m13,bitorder='little');record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());check(f'candidate{cid}_target',np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]);scores.append(str(math.comb(55,len(chosen))*math.comb(len(columns),8)) if len(columns)>=8 else None);score16.append(str(math.comb(55,len(chosen))*math.comb(len(columns),16)) if len(columns)>=16 else None)
    check('all_population_parity',scores==read(P/'score-values8.json') and score16==read(P/'score-values.json') and parity['all1541_source_target_filters_and_scores_verified'] and parity['score8_sha256']==sha(P/'score-values8.json') and parity['score16_sha256']==sha(P/'score-values.json') and parity['census_summary_sha256']==sha(B/'summary.json') and read(B/'summary.json')['audit']['all1541N12_masks_and_N13_literal_populations_verified'])
    predictions=np.unpackbits(bits,axis=1,count=73712,bitorder='little').astype(bool);sourcepred=np.array(sourcepred,bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};prior_weights=[1485//math.comb(55,len(chosen)) if len(sourcepop[cid])>=8 else 0 for cid,chosen in enumerate(choices)];tickets=[cid for cid,w in enumerate(prior_weights) for _ in range(w)];check('archived_conditional_prior',len(tickets)==4436 and prior['audit']['exact_conditioned_prior_and_all4096_truth_draws_verified'])
    archived_sources={e['data']['case']:e for e in parent if e['kind']=='source_bank'};references={}
    for e in parent:
        if e['kind']=='query':
            d=e['data'];references.setdefault((d['case'],d['selection']['query_index']),[]).append(dict(sequence=e['sequence'],sha256=e['sha256'],policy=d['policy'],phase=d['phase']))
    check('archive_metadata_scope',sum(len(v) for v in references.values())==11344 and prep['archived_query_metadata_records']==11344 and prep['archived_answers_decoded_for_routing']==0 and prep['archived_sources_sha256']==sha(A/'source-banks.json') and prep['all4096_source_filters_reconstructed_before_new_labels'])
    initials=[];truth_counts={}
    for case,source in enumerate(sources):
        seed=2026700000+case;ticket=int(np.random.default_rng(seed).integers(4436));truth=tickets[ticket];bank=truth_counts.get(truth,0);truth_counts[truth]=bank+1;population=sourcepop[truth];indices=np.random.default_rng(2026800000+case).choice(len(population),8,replace=False);sample=population[indices];initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();event=archived_sources[case]
        check('archived_source'+str(case),source['case']==case and source['candidate']==truth and source['bank']==bank and source['truth_seed']==seed and source['truth_ticket']==ticket and source['seed']==2026800000+case and source['indices']==indices.tolist() and source['columns_zero_based']==sample.tolist() and source['initial_candidates']==initial and source['source_sequence']==event['sequence'] and source['source_record_sha256']==event['sha256'] and all(source[k]==v for k,v in event['data'].items()));initials.append(initial)
    selection_cache={}
    def choose(candidates,queried):
        key=(tuple(candidates),tuple(queried))
        if key in selection_cache:return selection_cache[key]
        best=min(int(scores[c]) for c in candidates);ties=[c for c in candidates if int(scores[c])==best];guide=min(ties);counts=np.count_nonzero(predictions[candidates],axis=0);dis=np.where(predictions[guide],len(candidates)-counts,counts);dis[queried]=-1;maximum=int(dis.max());bt=np.flatnonzero(dis==maximum);index=int(bt[0]) if maximum>0 else None;value=dict(MAP_tied_candidates=ties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=maximum,optimal_board_count=len(bt),query_index=index,candidate_predictions={str(c):bool(predictions[c,index]) for c in candidates} if index is not None else None,stop_reason=None if index is not None else 'no_informative_unqueried_board_remaining_identity_unresolved');selection_cache[key]=value;return value
    group_lookup={tuple(row['candidates']):row['group'] for row in groups};check('unique_tree_sets',len(group_lookup)==len(groups) and [row['group'] for row in groups]==list(range(len(groups))) and {int(path.name[5:]) for path in P.glob('group*') if path.is_dir() and path.name[5:].isdigit()}==set(range(len(groups))));at=0;paid=0;proofs={};seen_groups={};real_states=0;real_splits=0;branches=0;maxLCM=0;maxweight=0
    def take(kind):
        nonlocal at
        e=events[at];at+=1;check('event_kind'+str(at-1),e['kind']==kind);return e
    check('start_binding',take('study_start')['data']==dict(protocol_sha256=sha(S/'protocol.json'),preparation_sha256=sha(P/'preparation.json'),archived_ledger_raw_sha256=p['archive']['raw_ledger_sha256'],archived_source_banks_sha256=sha(A/'source-banks.json')))
    def group_proof(candidates,bound):
        nonlocal real_states,real_splits,branches,maxLCM,maxweight
        identity=tuple(candidates);group=group_lookup[identity];row=groups[group];check('tree_completed_before_dependency',p['frozen_utc']<row['utc']<=bound['utc'])
        if identity in seen_groups:return row,proofs[group]
        seen_groups[identity]=group;folder=P/f'group{group}';proof=read(folder/'proof.json');weights=None;receipt=None;catalog_sha=None
        if len(candidates)>16:
            check('large_information_bound'+str(group),row['method']=='binary_information_bound');ns,nt,_=helper.certificate(proof,[],check,f'g{group}_bound_');method='binary_information_bound';compiler_sha=sha(S/'policy.py')
        else:
            codes=np.sum(predictions[candidates].astype(np.uint32)*(1<<np.arange(len(candidates),dtype=np.uint32))[:,None],axis=0,dtype=np.uint32);signatures,counts=np.unique(codes,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(np.flatnonzero(codes==s)[0]),board_count=int(n)) for s,n in zip(signatures,counts)];catalog_sha=sha(folder/'catalog.json');check('catalog'+str(group),read(folder/'catalog.json')==dict(candidates=candidates,universe_boards=73712,signatures=catalog));denominators=[int(scores[c]) for c in candidates];L=1;lengths=[]
            for d in denominators:L=math.lcm(L,d);lengths.append(L.bit_length())
            weights=[L//d for d in denominators];receipt=dict(score_denominators=list(map(str,denominators)),common_denominator=str(L),LCM_intermediate_bit_lengths=lengths,integer_weights=list(map(str,weights)),weight_bit_lengths=[w.bit_length() for w in weights],max_bits=100000);maxLCM=max(maxLCM,max(lengths));maxweight=max(maxweight,max(w.bit_length() for w in weights));check('posterior_weights'+str(group),read(folder/'weights.json')==receipt and proof['weights']==list(map(str,weights)));ns,nt,_=weighted.weighted_certificate(proof,catalog,check,f'g{group}_weighted_');method='posterior_expected_depth_optimum';compiler_sha=sha(S/'compiler.py')
        real_states+=ns;real_splits+=nt;inputs=dict(policy='early_handoff_posterior',method=method,candidates=candidates,weights=list(map(str,weights)) if weights is not None else None,depth=4,catalog_sha256=catalog_sha,predictions_sha256=bits_sha,compiler_sha256=compiler_sha,allowed_inputs_only=True)
        check('online_group_identity'+str(group),row==read(folder/'result.json') and row['group']==group and row['candidates']==proof['candidates']==candidates and proof['depth_limit']==4 and row['feasible']==proof['feasible'] and row['method']==method and row['catalog_sha256']==catalog_sha and row['weights_sha256']==(sha(folder/'weights.json') if receipt else None) and row['compiler_input_sha256']==sha(folder/'compiler-input.json') and read(folder/'compiler-input.json')==inputs and row['cache_key']==hashlib.sha256(canonical(inputs)).hexdigest() and row['proof_sha256']==sha(folder/'proof.json') and row['memo_states']==ns and row['canonical_split_attempts']==nt and p['frozen_utc']<row['utc']<=bound['utc']);
        if proof['feasible']:
            for local,cid in enumerate(candidates):
                state=proof['root_state'];used=[]
                while proof['states'][state]['reason']!='singleton_leaf':
                    node=proof['states'][state];index=node['chosen_split']['board_index'];check(f'g{group}_candidate{cid}_step{len(used)}',index not in used and bool(predictions[cid,index])==bool(node['chosen_split']['positive_mask']&(1<<local)));used.append(index);state=node['positive_state'] if predictions[cid,index] else node['negative_state']
                check(f'g{group}_candidate{cid}_leaf',proof['states'][state]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4);branches+=1
        # Every alternative was checked above; retain only chosen states for later paths.
        chosen_states={}
        def retain(state):
            node=proof['states'][state];chosen_states[state]={k:v for k,v in node.items() if k not in ['alternatives','attempts']}
            if node.get('reason')=='all_alternatives_optimized':
                retain(node['negative_state']);retain(node['positive_state'])
        if proof['feasible']:retain(proof['root_state'])
        proof['states']=chosen_states;proofs[group]=proof
        return row,proof
    cost_registry=read(P/'counterfactual-groups.json');cost_audit=GraphAudit(P,cost_registry,predictions,choose,group_proof,dict(predictions_sha256=bits_sha,scores_sha256=sha(P/'score-values8.json'),selector_sha256=sha(S/'selector.py'),tree_compiler_sha256=sha(S/'compiler.py')),check,read,sha,canonical)
    def replay(case,truth,phase,step,candidates,queried,request,selection):
        nonlocal paid
        index=selection['query_index'];q=take('query');a=take('answer');archived=references.get((case,index),[]);label=bool(predictions[truth,index]);remaining=[c for c in candidates if bool(predictions[c,index])==label]
        check('new_paid_query'+str(q['sequence']),q['data']==dict(case=case,phase=phase,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),archived_query_references=archived) and a['data']==dict(case=case,phase=phase,step=step,query_sha256=q['sha256'],label=label,candidates_before=candidates,candidates_after=remaining) and q['sequence']+1==a['sequence'] and q['monotonic_ns']<a['monotonic_ns'] and index not in queried and truth in remaining);paid+=1
        return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=q['sequence'],query_sha256=q['sha256'],answer_sequence=a['sequence'],answer_sha256=a['sha256'],archived_query_references=archived)
    computed=[]
    for source,initial in zip(sources,initials):
        case=source['case'];truth=source['candidate'];candidates=list(initial);queried=[];map_steps=[];tree_steps=[];routes=[];guards=[];handoff=None;handoff_candidates=None;terminal=None
        check('new_source_bound'+str(case),take('archived_source_bound')['data']==dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=initial))
        while len(candidates)>1:
            bound=take('online_compiled_tree_bound');row,proof=group_proof(candidates,bound);group=row['group'];check('online_binding'+str(bound['sequence']),bound['data']==dict(case=case,MAP_calls=len(map_steps),candidates=candidates,queried=queried,group=group,feasible=proof['feasible'],proof_sha256=row['proof_sha256'],group_result_sha256=sha(P/f'group{group}/result.json'),cache_key=row['cache_key']));routes.append(dict(MAP_calls=len(map_steps),candidates=candidates,group=group,feasible=proof['feasible'],bound_sequence=bound['sequence'],bound_sha256=bound['sha256']))
            decision=None
            if proof['feasible']:
                event=take('posterior_cost_guard_bound');baseline=cost_audit.verify(event['data']['baseline_cost_group'],candidates,queried,4-len(map_steps),event);decision=exact_guard(proof,baseline);data=dict(case=case,MAP_calls=len(map_steps),tree_group=group,baseline_cost_group=baseline['group'],baseline_result_sha256=sha(P/f'cost{baseline["group"]}/result.json'),baseline_cache_key=baseline['cache_key'],**decision);check('guard_before_actual_route',event['data']==data and event['monotonic_ns']>bound['monotonic_ns']);guards.append(dict(data,sequence=event['sequence'],sha256=event['sha256']))
                if len(map_steps)==4:check('MAP4_exact_tie',decision['handoff'] and decision['tie'])
            if proof['feasible'] and decision['handoff']:
                handoff=len(map_steps);handoff_candidates=list(candidates);state=proof['root_state']
                while proof['states'][state]['reason']!='singleton_leaf':
                    node=proof['states'][state];index=node['chosen_split']['board_index'];request=dict(group=group,proof_sha256=row['proof_sha256'],state=state,candidates=candidates,queried=queried);selection=dict(query_index=index,candidate_predictions={str(c):bool(predictions[c,index]) for c in candidates});candidates,step=replay(case,truth,'tree',len(queried),candidates,queried,request,selection);tree_steps.append(step);queried=queried+[index];state=node['positive_state'] if step['label'] else node['negative_state']
                break
            if len(map_steps)==4:break
            request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=73712,candidates=candidates,queried=queried);selection=choose(candidates,queried)
            if selection['query_index'] is None:
                event=take('no_informative_query');check('no_information'+str(case),event['data']==dict(case=case,request=request,selection=selection));terminal=dict(sequence=event['sequence'],sha256=event['sha256']);break
            candidates,step=replay(case,truth,'map',len(queried),candidates,queried,request,selection);map_steps.append(step);queried=queried+[step['query_index']]
        result=dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=initial,routing=routes,cost_guards=guards,map_steps=map_steps,tree_steps=tree_steps,handoff_MAP_calls=handoff,handoff_candidates=handoff_candidates,terminal_no_information=terminal,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],map_calls=len(map_steps),tree_calls=len(tree_steps),membership_calls=len(queried),archived_query_reacquisitions=sum(bool(r['archived_query_references']) for r in map_steps+tree_steps),new_query_boards_absent_from_archived_case=sum(not r['archived_query_references'] for r in map_steps+tree_steps));check('new_case_result'+str(case),result==cases[case]==take('case_result')['data'] and len(map_steps)<=4 and len(tree_steps)<=4 and len(queried)<=8);computed.append(result)
    check('all_current_runs_before_old_outcome_evaluation',take('all_new_policy_executions_complete')['data']==dict(cases=4096,actual_membership_calls=paid,case_results_sha256=sha(P/'case-results.json'),groups_sha256=sha(P/'groups.json'),counterfactual_registry_sha256=sha(P/'counterfactual-groups.json')) and len(seen_groups)==len(groups))
    counterfactual_nodes,counterfactual_partitions=cost_audit.finish()
    expected_comparisons=[];restricted_paths=0;baseline_calls=0
    for new,old,checkpoint in zip(computed,oldcases,oldcheckpoints):
        case=new['case'];truth=new['candidate'];candidates=list(initials[case]);queried=[]
        for step in checkpoint['map_steps']:
            selection=choose(candidates,queried);index=selection['query_index'];label=bool(predictions[truth,index]);remaining=[c for c in candidates if bool(predictions[c,index])==label];q=parent[step['query_sequence']];a=parent[step['answer_sequence']];check('baseline_MAP'+str(q['sequence']),q['data']['selection']==selection and q['sha256']==step['query_sha256'] and a['sha256']==step['answer_sha256'] and a['data']['label']==step['label']==label and step['candidates_before']==candidates and step['candidates_after']==remaining);candidates=remaining;queried.append(index)
        check('baseline_checkpoint'+str(case),checkpoint['candidates']==candidates and checkpoint['queried']==queried and old['case']==checkpoint['case']==case and old['candidate']==checkpoint['candidate']==truth)
        for i,step in enumerate(new['map_steps']):check('same_MAP_prefix'+str(case)+'_'+str(i),step['query_index']==checkpoint['map_steps'][i]['query_index'] and step['label']==checkpoint['map_steps'][i]['label'])
        for step in old['arms']['posterior']['tree_steps']:
            index=step['query_index'];label=bool(predictions[truth,index]);remaining=[c for c in candidates if bool(predictions[c,index])==label];q=parent[step['query_sequence']];a=parent[step['answer_sequence']];check('baseline_tree_call'+str(q['sequence']),q['sha256']==step['query_sha256'] and a['sha256']==step['answer_sha256'] and q['data']['policy']=='posterior' and q['data']['selection']['query_index']==index and a['data']['label']==step['label']==label and step['candidates_before']==candidates and step['candidates_after']==remaining and index not in queried);candidates=remaining;queried.append(index)
        check('baseline_exact_and_cost'+str(case),old['arms']['posterior']['exact_true_singleton']==(candidates==[truth])==new['exact_true_singleton'] and old['arms']['posterior']['policy_calls']==len(queried));baseline_calls+=len(queried);tree=None
        if new['handoff_MAP_calls'] is not None:
            check('later_subset'+str(case),set(checkpoint['candidates'])<=set(new['handoff_candidates']));proof=proofs[seen_groups[tuple(new['handoff_candidates'])]];tree=restricted(proof,checkpoint['candidates'])
            for cid in checkpoint['candidates']:
                node=tree;depth=0
                while 'candidate' not in node:node=node['positive'] if predictions[cid,node['board_index']] else node['negative'];depth+=1
                check('restricted_actual_candidate'+str(case)+'_'+str(cid),node['candidate']==cid and depth<=4);restricted_paths+=1
        expected_comparisons.append(dict(case=case,candidate=truth,old_exact=old['arms']['posterior']['exact_true_singleton'],new_exact=new['exact_true_singleton'],old_policy_calls=old['arms']['posterior']['policy_calls'],new_policy_calls=new['membership_calls'],old_checkpoint_subset_of_handoff=new['handoff_MAP_calls'] is not None,restricted_tree=tree))
    check('all_comparisons',expected_comparisons==comparisons and baseline_calls==10355 and take('postexecution_baseline_evaluation')['data']==dict(baseline_case_results_sha256=sha(A/'case-results.json'),baseline_checkpoints_sha256=sha(A/'checkpoints.json'),comparisons_sha256=sha(P/'baseline-comparison.json'),all4096_exact_flags_identical=True,all_early_feasible_later_subsets_separable=True))
    outcome=dict(decision='supported' if 100*paid<=98*10355 else 'falsified',primary=dict(left=100*paid,right=98*10355,passing_maximum=10147),cases=4096,old_policy_calls=10355,new_policy_calls=paid,new_MAP_calls=sum(r['map_calls'] for r in computed),new_tree_calls=sum(r['tree_calls'] for r in computed),exact_cases=sum(r['exact_true_singleton'] for r in computed),all4096_exact_flags_identical_correctness_invariant=True,baseline_infeasible_cases_retained=13,new_truth_draws=0,new_source_draws=0,new_source_positive_labels=0,historical_positive_appearances_reused=32768,handoff_MAP_call_histogram={str(i):sum(r['handoff_MAP_calls']==i for r in computed) for i in range(5)},cases_without_tree_handoff=sum(r['handoff_MAP_calls'] is None for r in computed),archived_query_reacquisitions=sum(r['archived_query_reacquisitions'] for r in computed),new_query_boards_absent_from_archived_case=sum(r['new_query_boards_absent_from_archived_case'] for r in computed),distinct_compiled_sets=len(groups),distinct_feasible_sets=sum(r['feasible'] for r in groups),tree_memo_states=real_states,tree_canonical_split_attempts=real_splits,counterfactual_memo_states=counterfactual_nodes,counterfactual_MAP_partitions=counterfactual_partitions,actual_cost_guard_evaluations=sum(len(r['cost_guards']) for r in computed),rejected_cost_guards=sum(not item['handoff'] for r in computed for item in r['cost_guards']),accepted_cost_guards=sum(item['handoff'] for r in computed for item in r['cost_guards']),accepted_tied_cost_guards=sum(item['tie'] for r in computed for item in r['cost_guards'] if item['handoff']),all_accepted_guards_expected_nonincreasing=True,runtime_gate_memo_states=total_states,runtime_gate_split_evaluations=total_splits,global_memo_states=total_states+real_states+counterfactual_nodes,global_split_evaluations=total_splits+real_splits+counterfactual_partitions)
    check('primary_all_cases_counters',take('study_complete')['data']==outcome and at==len(events) and all(g[k]==v for k,v in outcome.items()) and outcome['exact_cases']==4083 and paid==outcome['new_MAP_calls']+outcome['new_tree_calls'] and maxLCM<=100000 and maxweight<=100000 and read(P/'case-statuses.json')==[dict(case=i,status='complete') for i in range(4096)])
    if require_summary:
        s=read(P/'summary.json');check('summary_identity',s['status']=='validated' and all(s[k]==v for k,v in outcome.items()) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py') and s['verifier_dependency_sha256']=={str(path.relative_to(R)):sha(path) for path in deps})
        for rel,digest in s['public_sha256'].items():check('public_export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figure_provenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=checks,all1541_population_filters_and_scores_verified=True,all4096_archived_truth_source_banks_reconstructed=True,all4096_new_online_routes_and_paid_calls_replayed=True,new_paid_calls=paid,baseline_MAP_and_posterior_calls_replayed=baseline_calls,all4096_exact_flags_identical=True,all13_baseline_failures_preserved=True,all_tree_groups_referenced_by_actual_or_counterfactual_route=True,all_counterfactual_nodes_reachable_from_actual_guards=True,counterfactual_nodes_verified=counterfactual_nodes,counterfactual_MAP_partitions_verified=counterfactual_partitions,counterfactual_candidate_costs_verified=cost_audit.candidate_paths,all_guard_accept_reject_tie_and_MAP4_decisions_verified=True,lossless_proof_export=proof_export_audit,all_online_proofs_bound_before_queries=True,real_proof_states=real_states,real_split_attempts=real_splits,all_remaining_candidate_branches_verified=branches,restricted_later_checkpoint_candidate_paths_verified=restricted_paths,toy_nonempty_subsets_verified=subset_count,all_total_cost_primary_and_reacquisition_charges_verified=True,maximum_LCM_bits=maxLCM,maximum_integer_weight_bits=maxweight,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
