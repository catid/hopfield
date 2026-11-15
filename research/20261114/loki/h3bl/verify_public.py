"""Independently replay exact weighted optimality and representation-only repair."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys
sys.set_int_max_str_digits(0)
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np


def weighted_certificate(proof,catalog,check,prefix):
    candidates=proof['candidates'];weights=list(map(int,proof['weights']));full=(1<<len(candidates))-1;depth=proof['depth_limit'];states=proof['states'];root=f'{full}:{depth}';seen=set()
    check(prefix+'root',proof['root_state']==root and candidates==sorted(set(candidates)) and len(weights)==len(candidates) and all(w>0 for w in weights) and proof['weight_sum']==str(sum(weights)))
    def inspect(key):
        if key in seen:return int(states[key]['optimal_cost']) if states[key]['feasible'] else None
        seen.add(key);node=states[key];mask,d=map(int,key.split(':'));mass=sum(w for i,w in enumerate(weights) if mask&(1<<i));count=mask.bit_count();attempts=node['attempts'];check(prefix+key+'identity',node['mask']==mask and node['depth_remaining']==d and node['models']==count and node['weight_sum']==str(mass) and d>=0 and mask>0 and mask&full==mask)
        if count==1:check(prefix+key+'singleton',node['feasible'] and node['optimal_cost']=='0' and node['reason']=='singleton_leaf' and not attempts);return 0
        if count>2**d:check(prefix+key+'information',not node['feasible'] and node['optimal_cost'] is None and node['reason']=='binary_information_bound' and not attempts);return None
        partitions={}
        for row in catalog:
            pos=mask&row['signature'];neg=mask^pos
            if not pos or not neg:continue
            unordered=tuple(sorted((pos,neg)));proposal=(row['lexfirst_board_index'],pos,neg)
            if unordered not in partitions or proposal[0]<partitions[unordered][0]:partitions[unordered]=proposal
        options=[dict(canonical_side=min(pos,neg),positive_mask=pos,negative_mask=neg,board_index=index) for index,pos,neg in sorted(partitions.values())];check(prefix+key+'everyalternative',node['canonical_partition_count']==len(options)==len(attempts));possible=[]
        for i,(option,attempt) in enumerate(zip(options,attempts)):
            a=option['negative_mask'];b=option['positive_mask'];expected=dict(option)
            if max(a.bit_count(),b.bit_count())>2**(d-1):expected['result']='child_binary_information_bound'
            else:
                negative=f'{a}:{d-1}';expected['negative_state']=negative;left=inspect(negative)
                if left is None:expected['result']='negative_child_infeasible'
                else:
                    positive=f'{b}:{d-1}';expected['positive_state']=positive;right=inspect(positive)
                    if right is None:expected['result']='positive_child_infeasible'
                    else:
                        cost=mass+left+right;expected.update(result='both_children_feasible',candidate_cost=str(cost));possible.append((cost,option['board_index'],i))
            check(prefix+key+f'alternative{i}',attempt==expected)
        if not possible:check(prefix+key+'infeasibleexhaustion',not node['feasible'] and node['optimal_cost'] is None and node['reason']=='all_nontrivial_partitions_exhausted');return None
        cost,_,i=min(possible);check(prefix+key+'exactminimumandlexfirsttie',node['feasible'] and node['optimal_cost']==str(cost) and node['reason']=='all_alternatives_optimized' and node['chosen_split']==options[i] and node['negative_state']==attempts[i]['negative_state'] and node['positive_state']==attempts[i]['positive_state']);return cost
    cost=inspect(root);check(prefix+'allvisitedstates',seen==set(states) and proof['feasible']==(cost is not None) and proof['optimal_cost']==(str(cost) if cost is not None else None));depths={};queries=set()
    if cost is not None:
        for i,cid in enumerate(candidates):
            key=root;used=[]
            while states[key]['reason']!='singleton_leaf':
                node=states[key];index=node['chosen_split']['board_index'];check(prefix+f'branch{cid}norepeat{len(used)}',index not in used);used.append(index);queries.add(index);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
            check(prefix+f'branch{cid}correctleaf',states[key]['mask']==1<<i and len(used)<=depth);depths[str(cid)]=len(used)
        check(prefix+'weightedleafidentity',cost==sum(w*depths[str(cid)] for cid,w in zip(candidates,weights)))
    check(prefix+'allleafdepths',proof['leaf_depths']==depths and proof['worst_leaf_depth']==(max(depths.values()) if depths else None));return len(states),sum(len(n['attempts']) for n in states.values()),queries


def brute(mask,depth,signatures,weights):
    if mask.bit_count()==1:return 0
    if depth==0:return None
    scores=[]
    for signature in signatures:
        a=mask&signature;b=mask^a
        if not a or not b:continue
        left=brute(a,depth-1,signatures,weights);right=brute(b,depth-1,signatures,weights)
        if left is not None and right is not None:scores.append(sum(w for i,w in enumerate(weights) if mask&(1<<i))+left+right)
    return min(scores) if scores else None


def encode_fraction(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator),display_float=float(x))


def verify(root,require_summary=True):
    started=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bl';P=R/'results/research/loki/h3bl';J=R/'results/research/loki/h3bj';K=R/'results/research/loki/h3bk';B=R/'results/research/loki/h3bc';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    dependency=R/'research/20261114/loki/h3bi/verify_public.py';spec=importlib.util.spec_from_file_location('independent_h3bi_helpers',dependency);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    p=read(S/'protocol.json');failedp=read(R/'research/20261114/loki/h3bk/protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');failed=read(K/'capacity-gate.json');mapping=read(P/'checkpoint-mappings.json');toy=read(P/'synthetic-gate.json');cases=read(J/'case-results.json');groups=read(J/'groups.json');prior=read(J/'summary.json');census=read(B/'summary.json');oldprep=read(B/'preparation.json');old=read(R/p['normalizers']['selection_path'])
    check('wholebudget',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and g['global_memo_states']<=2000000 and g['global_canonical_split_attempts']<=50000000 and p['budget']['GPUs']==0 and g['new_source_draws']==g['new_oracle_labels']==0)
    check('stagehashes',all(g[k]==raw[k] for k in raw if k!='utc') and p['frozen_utc']<mapping['utc']<raw['utc'] and g['protocol_sha256']==mapping['protocol_sha256']==sha(S/'protocol.json') and g['source_sha256']==sha(S/'study.py') and g['checkpoint_mappings_sha256']==sha(P/'checkpoint-mappings.json') and g['retrospective_truth_costs_sha256']==sha(P/'retrospective-truth-costs.json') and g['synthetic_gate_sha256']==sha(P/'synthetic-gate.json'))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    check('unchangedoriginalscientificendpoint',all(p[k]==v for k,v in failedp.items() if k not in ['id','frozen_utc','history','source_sha256','source_data_sha256']) and failed['status']=='failed_or_incomplete' and '4300 digits' in failed['error'] and read(K/'summary.json')['primary_decision'] is None)
    for name in ['compiler.py','weights.py','synthetic_gate.py']:check('unchanged '+name,sha(S/name)==sha(R/'research/20261114/loki/h3bk'/name))
    for name in ['study.py','gate.py']:check('scopepathsonly '+name,(S/name).read_text()==(R/'research/20261114/loki/h3bk'/name).read_text().replace('h3bk','h3bl').replace('H3bk','H3bl'))
    check('inheritedcensusandcheckpointaudit',census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and prior['audit']['all1408_fresh_source_seeds_and_duplicate_checks_replayed'] and prior['audit']['all_groups_complete_catalogs_and_certificates_verified']==125 and prior['public_sha256']['case-results.json']==sha(J/'case-results.json'))
    total_states=1;total_splits=0
    for row in toy['fixtures']:
        catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(row['signatures'])];proof=row['result'];exact=brute((1<<len(row['weights']))-1,row['depth'],row['signatures'],row['weights']);check('fixture '+row['name'],proof['weights']==list(map(str,row['weights'])) and proof['depth_limit']==row['depth'] and proof['optimal_cost']==row['expected_cost']==(str(exact) if exact is not None else None));ns,nt,_=weighted_certificate(proof,catalog,check,'fixture '+row['name']);total_states+=ns;total_splits+=nt
    check('syntheticcapsandweights',len(toy['fixtures'])==8 and toy['all_independent_brute_force_costs_identical'] and toy['resource_caps_raise_incomplete']==dict(state_cap=True,split_cap=True,weight_bit_cap=True) and toy['global_states_after_gate']==total_states and toy['global_split_attempts_after_gate']==total_splits and toy['weight_conversion_fixture']==dict(score_denominators=['6','10','15'],common_denominator='30',LCM_intermediate_bit_lengths=[3,5,5],integer_weights=['5','3','2'],weight_bit_lengths=[3,2,2],max_bits=100000))
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];bits=np.load(B/'predictions-bits.npy');h12=helper.hits(ordinary,orbits);h13=helper.hits(boards,orbits);base12,extra12=helper.parts(12,orbits);base13,extra13=helper.parts(13,orbits);scores=[]
    for cid,chosen in enumerate(choices):
        m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[m12];record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());check(f'candidate{cid}source',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid]);packed=np.packbits(m13,bitorder='little');record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());check(f'candidate{cid}target',np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]);scores.append(str(math.comb(55,len(chosen))*math.comb(len(columns),16)) if len(columns)>=16 else None)
    check('scores',scores==read(J/'score-values.json'));predictions=np.unpackbits(bits,axis=1,count=73712,bitorder='little').astype(bool);expected_mapping=[dict(case=r['case'],original_tree_group=r['tree_group'],checkpoint_candidates=r['checkpoint_candidates'],known_four_query_success=len(r['checkpoint_candidates'])==1) for r in cases];check('all1408case701unresolvedmappings',mapping['cases']==expected_mapping and len(cases)==1408 and sum(not r['known_four_query_success'] for r in expected_mapping)==701 and mapping['all1408_retained'] and mapping['unresolved701_retained'] and p['groups']==[dict(group=r['group'],candidates=r['candidates']) for r in groups] and len(groups)==125)
    aggregate_old=Fraction(0);aggregate_new=Fraction(0);computed=[];proofs={};real_states=0;real_splits=0;branches=0;max_weight_bits=0;max_LCM_bits=0
    for original in groups:
        group=original['group'];folder=P/f'group{group}';candidates=original['candidates'];row=read(folder/'result.json');proof=read(folder/'proof.json');proofs[group]=proof;receipt=read(folder/'weights.json');denominators=list(map(lambda c:int(scores[c]),candidates));L=1;intermediate=[]
        for denominator in denominators:L=L*denominator//math.gcd(L,denominator);intermediate.append(L.bit_length())
        weights=[L//d for d in denominators];expected=dict(score_denominators=list(map(str,denominators)),common_denominator=str(L),LCM_intermediate_bit_lengths=intermediate,integer_weights=list(map(str,weights)),weight_bit_lengths=[w.bit_length() for w in weights],max_bits=100000);check(f'group{group}exactweightsandcaps',receipt==expected and max(intermediate)<=100000 and max(w.bit_length() for w in weights)<=100000 and all(Fraction(w,weights[0])==Fraction(denominators[0],d) for w,d in zip(weights,denominators)));max_weight_bits=max(max_weight_bits,max(w.bit_length() for w in weights));max_LCM_bits=max(max_LCM_bits,max(intermediate))
        codes=np.sum(predictions[candidates].astype(np.uint32)*(1<<np.arange(len(candidates),dtype=np.uint32))[:,None],axis=0,dtype=np.uint32);signatures,counts=np.unique(codes,return_counts=True);catalog=[dict(signature=int(s),lexfirst_board_index=int(np.flatnonzero(codes==s)[0]),board_count=int(n)) for s,n in zip(signatures,counts)];stored=dict(candidates=candidates,universe_boards=73712,signatures=catalog);check(f'group{group}completecatalogandblindinput',read(folder/'catalog.json')==read(J/f'group{group}/catalog.json')==stored and read(folder/'compiler-input.json')==dict(candidates=candidates,weights=list(map(str,weights)),catalog_sha256=sha(folder/'catalog.json'),depth=4,allowed_inputs_only=True) and proof['candidates']==candidates and proof['weights']==list(map(str,weights)) and proof['depth_limit']==4)
        ns,nt,queries=weighted_certificate(proof,catalog,check,f'group{group}');total_states+=ns;total_splits+=nt;real_states+=ns;real_splits+=nt;oldproof=read(J/f'group{group}/proof.json');helper.certificate(oldproof,catalog,check,f'group{group}oldfeasible');oldcost=sum(w*oldproof['leaf_depths'][str(cid)] for cid,w in zip(candidates,weights));newcost=int(proof['optimal_cost']);old_expected=Fraction(oldcost,sum(weights));new_expected=Fraction(newcost,sum(weights));case_ids=[r['case'] for r in cases if r['tree_group']==group];aggregate_old+=len(case_ids)*old_expected;aggregate_new+=len(case_ids)*new_expected;check(f'group{group}mappingnorepeats',all(cases[i]['checkpoint_candidates']==candidates and not queries.intersection(r['query_index'] for r in cases[i]['shared_steps']) for i in case_ids))
        for local,cid in enumerate(candidates):
            state=proof['root_state'];used=[]
            while proof['states'][state]['reason']!='singleton_leaf':
                node=proof['states'][state];index=node['chosen_split']['board_index'];check(f'group{group}actualmodel{cid}step{len(used)}',bool(predictions[cid,index])==bool(node['chosen_split']['positive_mask']&(1<<local)) and index not in used);used.append(index);state=node['positive_state'] if predictions[cid,index] else node['negative_state']
            check(f'group{group}actualmodel{cid}leaf',proof['states'][state]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4);branches+=1
        expected_fields=dict(group=group,candidates=candidates,case_ids=case_ids,case_multiplicity=len(case_ids),old_weighted_depth_cost=str(oldcost),optimal_weighted_depth_cost=str(newcost),weight_sum=str(sum(weights)),old_expected_additional_queries=encode_fraction(old_expected),optimal_expected_additional_queries=encode_fraction(new_expected),strict_expected_improvement=newcost<oldcost,old_worst_leaf_depth=oldproof['worst_leaf_depth'],optimal_worst_leaf_depth=proof['worst_leaf_depth'],signature_count=len(catalog),memo_states=ns,canonical_split_attempts=nt,weights_sha256=sha(folder/'weights.json'),catalog_sha256=sha(folder/'catalog.json'),compiler_input_sha256=sha(folder/'compiler-input.json'),proof_sha256=sha(folder/'proof.json'));check(f'group{group}receiptsandnonincrease',all(row[k]==v for k,v in expected_fields.items()) and newcost<=oldcost);check(f'group{group}exactfailedrunparity',all(sha(folder/f'{name}.json')==sha(K/f'group{group}/{name}.json') for name in ['catalog','weights','compiler-input','proof']) and all(row[k]==failed['completed_groups'][group][k] for k in expected_fields));computed.append(row)
    descriptive=[]
    for case in cases:
        group=case['tree_group'];truth=case['candidate'];new=0 if group is None else proofs[group]['leaf_depths'][str(truth)];descriptive.append(dict(case=case['case'],candidate=truth,tree_group=group,shared_calls=case['shared_membership_calls'],old_tree_continuation_calls=case['hybrid_additional_calls'],optimized_tree_realized_truth_depth=new,baseline_MAP_continuation_calls=case['baseline_additional_calls'],old_hybrid_policy_cost=case['hybrid_policy_label_cost'],optimized_hybrid_realized_truth_cost=case['shared_membership_calls']+new,baseline_MAP_policy_cost=case['baseline_policy_label_cost'],retrospective_no_new_label_acquisition=True))
    check('retrospectivecostsnotnewlabels',descriptive==read(P/'retrospective-truth-costs.json'));decision='supported' if 10*aggregate_new<=9*aggregate_old else 'falsified';outcome=dict(decision=decision,distinct_checkpoint_sets=125,unresolved_cases=701,known_four_query_success_cases=707,case_weighted_expected_old_sum=encode_fraction(aggregate_old),case_weighted_expected_optimal_sum=encode_fraction(aggregate_new),mean_expected_old_continuation=encode_fraction(aggregate_old/701),mean_expected_optimal_continuation=encode_fraction(aggregate_new/701),relative_expected_reduction=encode_fraction(1-aggregate_new/aggregate_old),exact_threshold_comparison=dict(left_10_times_optimal=encode_fraction(10*aggregate_new),right_9_times_old=encode_fraction(9*aggregate_old),passed=10*aggregate_new<=9*aggregate_old),strictly_improved_sets=sum(r['strict_expected_improvement'] for r in computed),strictly_improved_checkpoint_cases=sum(r['case_multiplicity'] for r in computed if r['strict_expected_improvement']),guaranteed_nonincrease_is_correctness_not_hypothesis=True,descriptive_old_continuation_cost=sum(r['old_tree_continuation_calls'] for r in descriptive),descriptive_optimized_continuation_cost=sum(r['optimized_tree_realized_truth_depth'] for r in descriptive),descriptive_baseline_MAP_continuation_cost=sum(r['baseline_MAP_continuation_calls'] for r in descriptive),global_memo_states=total_states,global_canonical_split_attempts=total_splits,new_source_draws=0,new_oracle_labels=0);check('exactcaseweightedprimaryandwholeaccounting',g['groups']==computed and all(g[k]==v for k,v in outcome.items()) and total_states==failed['global_memo_states'] and total_splits==failed['global_canonical_split_attempts'])
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==v for k,v in outcome.items()) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py') and s['verifier_dependency_sha256']=={str(dependency.relative_to(R)):sha(dependency)})
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all1541_source_target_filters_and_exact_scores_verified=True,all125_complete_signature_catalogs_and_original_witnesses_verified=True,all701_unresolved_and707_success_mappings_retained=True,all_exact_integer_weights_and_bit_caps_verified=True,maximum_LCM_bits=max_LCM_bits,maximum_integer_weight_bits=max_weight_bits,every_alternative_cost_and_lex_tie_optimality_certificate_verified=True,real_proof_states=real_states,real_split_attempts=real_splits,all_remaining_candidate_branches_verified=branches,exact_rational_10percent_threshold_and_case_multiplicity_verified=True,all125_compiler_inputs_catalogs_weights_and_proofs_byte_identical_to_failed_run=True,representation_only_repair_not_independent_confirmation=True,retrospective_realized_costs_not_new_queries=True,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
