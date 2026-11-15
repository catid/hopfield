"""Independent counterfactual graph and synthetic-fixture audit.

Does not import the scientific compiler, selector, cost evaluator or restriction.
"""
import hashlib
import math
import numpy as np


def choose(predictions,scores,candidates,queried):
    best=min(int(scores[c]) for c in candidates);ties=[c for c in candidates if int(scores[c])==best];guide=min(ties);counts=np.count_nonzero(predictions[candidates],axis=0);dis=np.where(predictions[guide],len(candidates)-counts,counts);dis[queried]=-1;maximum=int(dis.max());bt=np.flatnonzero(dis==maximum);index=int(bt[0]) if maximum>0 else None
    return dict(MAP_tied_candidates=ties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=maximum,optimal_board_count=len(bt),query_index=index,candidate_predictions={str(c):bool(predictions[c,index]) for c in candidates} if index is not None else None,stop_reason=None if index is not None else 'no_informative_unqueried_board_remaining_identity_unresolved')


def exact_guard(proof,baseline):
    tree_cost=sum(int(w)*proof['leaf_depths'][str(c)] for c,w in zip(proof['candidates'],proof['weights']));assert tree_cost==int(proof['optimal_cost']);baseline_cost=sum(int(w)*baseline['remaining_calls_by_candidate'][str(c)] for c,w in zip(proof['candidates'],proof['weights']))
    return dict(tree_weighted_cost=str(tree_cost),baseline_weighted_cost=str(baseline_cost),weight_sum=proof['weight_sum'],handoff=tree_cost<=baseline_cost,tie=tree_cost==baseline_cost,comparison='exact integer tree_cost <= baseline_cost',expected_nonincrease_if_handoff=tree_cost<=baseline_cost)


class GraphAudit:
    def __init__(self,folder,registry,predictions,select,tree,identities,check,read,sha,canonical):
        self.folder=folder;self.registry=registry;self.predictions=predictions;self.select=select;self.tree=tree;self.identities=identities;self.check=check;self.read=read;self.sha=sha;self.canonical=canonical;self.rows={};self.keys={};self.MAP_partitions=0;self.visiting=set();self.candidate_paths=0

    def verify(self,group,candidates,queried,remaining_MAP,bound):
        check=self.check;queried=sorted(set(queried));folder=self.folder/f'cost{group}';inputs=dict(policy='fixed_remaining_MAP_then_posterior_depth4',candidates=candidates,remaining_MAP=remaining_MAP,queried=queried,depth=4,**self.identities);key=hashlib.sha256(self.canonical(inputs)).hexdigest()
        check('cost_context',candidates==sorted(set(candidates)) and candidates and 0<=remaining_MAP<=4 and all(all(bool(self.predictions[c,index])==bool(self.predictions[candidates[0],index]) for c in candidates) for index in queried))
        if group in self.rows:
            row=self.rows[group];check('cost_cache_reuse',row['cache_key']==key and row['utc']<=bound['utc']);return row
        check('cost_acyclic_unique_cache',group not in self.visiting and key not in self.keys);self.visiting.add(group);row=self.read(folder/'result.json');check('cost_full_input_identity',self.read(folder/'input.json')==inputs and row['input_sha256']==self.sha(folder/'input.json') and row['group']==group and row['cache_key']==key and row['candidates']==candidates and row['queried']==queried and row['remaining_MAP']==remaining_MAP and row['utc']<=bound['utc'] and row['all_candidate_branches_retained'] and row['actual_oracle_calls']==0 and row['CPU_seconds']>=0 and row['wall_seconds']>=0)
        if self.registry is not None:check('cost_registry_identity',self.registry[group]==dict(group=group,cache_key=key,input_sha256=self.sha(folder/'input.json'),result_sha256=self.sha(folder/'result.json')))
        if len(candidates)==1:
            expected={str(candidates[0]):0};check('cost_singleton',row['reason']=='singleton' and row['request'] is None and row['selection'] is None and row['tree'] is None and row['children']==[])
        elif remaining_MAP==0:
            tree_row,proof=self.tree(candidates,row);expected={str(c):proof['leaf_depths'][str(c)] for c in candidates};check('cost_terminal_tree',row['reason']=='posterior_tree' and proof['feasible'] and row['request'] is None and row['selection'] is None and row['children']==[] and row['tree']==dict(group=tree_row['group'],proof_sha256=tree_row['proof_sha256'],group_result_sha256=self.sha(self.folder/f'group{tree_row["group"]}/result.json')))
        else:
            selection=self.select(candidates,queried);index=selection['query_index'];request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=self.predictions.shape[1],candidates=candidates,queried=queried);check('cost_MAP_selection',row['reason']=='MAP_partition' and row['request']==request and row['selection']==selection and row['tree'] is None and index is not None and index not in queried and len(row['children'])==2);expected={};self.MAP_partitions+=1
            for label,ref in zip([False,True],row['children']):
                subset=[c for c in candidates if bool(self.predictions[c,index])==label];check('cost_both_nonempty_branches',subset and ref['label'] is label);child=self.verify(ref['group'],subset,queried+[index],remaining_MAP-1,row);check('cost_child_hash',ref==dict(label=label,group=child['group'],result_sha256=self.sha(self.folder/f'cost{child["group"]}/result.json'),cache_key=child['cache_key']));expected.update({str(c):1+child['remaining_calls_by_candidate'][str(c)] for c in subset})
        check('cost_every_candidate_cost',row['remaining_calls_by_candidate']==expected and all(type(v) is int and 0<=v<=remaining_MAP+4 for v in expected.values()));self.candidate_paths+=len(expected);self.rows[group]=row;self.keys[key]=group;self.visiting.remove(group);return row

    def finish(self):
        actual={int(path.name[4:]) for path in self.folder.glob('cost*') if path.is_dir() and path.name[4:].isdigit()};self.check('cost_no_unreferenced_work',actual==set(self.rows) and actual==set(range(len(actual))) and (self.registry is None or len(self.registry)==len(actual)));return len(self.rows),self.MAP_partitions


def fixtures(P,S,read,sha,canonical,weighted,check):
    gate=read(P/'cost-gate.json');before_states=0;before_splits=0;total_naive=0;total_selector=0;total_paths=0
    for fixture in gate['fixtures']:
        folder=P/'cost-gate'/fixture['name'];scores=fixture['scores'];n=len(scores);pred=np.array([[bool(sig&(1<<c)) for sig in fixture['signatures']] for c in range(n)]);identities=dict(predictions_sha256=hashlib.sha256(np.packbits(pred,axis=1,bitorder='little').tobytes()).hexdigest(),scores_sha256=hashlib.sha256(canonical(scores)).hexdigest(),selector_sha256=sha(S/'selector.py'),tree_compiler_sha256=sha(S/'compiler.py'));tree_rows={};proofs={};tree_groups={};states=0;splits=0
        for path in folder.glob('group*/result.json'):
            row=read(path);check('fixture_tree_unique',tuple(row['candidates']) not in tree_groups);tree_groups[tuple(row['candidates'])]=row['group'];tree_rows[row['group']]=row
        used=set()
        def tree(candidates,bound):
            nonlocal states,splits
            group=tree_groups[tuple(candidates)];row=tree_rows[group];check('fixture_tree_before_bound',row['utc']<=bound['utc']);used.add(group)
            if group in proofs:return row,proofs[group]
            directory=folder/f'group{group}';proof=read(directory/'proof.json');codes=np.sum(pred[candidates].astype(np.uint32)*(1<<np.arange(len(candidates),dtype=np.uint32))[:,None],axis=0,dtype=np.uint32);signatures,counts=np.unique(codes,return_counts=True);catalog=[dict(signature=int(sig),lexfirst_board_index=int(np.flatnonzero(codes==sig)[0]),board_count=int(count)) for sig,count in zip(signatures,counts)];denominators=[int(scores[c]) for c in candidates];L=1;lengths=[]
            for d in denominators:L=math.lcm(L,d);lengths.append(L.bit_length())
            weights=[L//d for d in denominators];receipt=dict(score_denominators=list(map(str,denominators)),common_denominator=str(L),LCM_intermediate_bit_lengths=lengths,integer_weights=list(map(str,weights)),weight_bit_lengths=[w.bit_length() for w in weights],max_bits=100000);inputs=dict(policy='early_handoff_posterior',method='posterior_expected_depth_optimum',candidates=candidates,weights=list(map(str,weights)),depth=4,catalog_sha256=sha(directory/'catalog.json'),predictions_sha256=identities['predictions_sha256'],compiler_sha256=sha(S/'compiler.py'),allowed_inputs_only=True)
            check('fixture_tree_inputs',read(directory/'catalog.json')==dict(candidates=candidates,universe_boards=pred.shape[1],signatures=catalog) and read(directory/'weights.json')==receipt and proof['weights']==list(map(str,weights)) and read(directory/'compiler-input.json')==inputs and row['compiler_input_sha256']==sha(directory/'compiler-input.json') and row['cache_key']==hashlib.sha256(canonical(inputs)).hexdigest() and row['catalog_sha256']==sha(directory/'catalog.json') and row['weights_sha256']==sha(directory/'weights.json') and row['proof_sha256']==sha(directory/'proof.json') and proof['candidates']==candidates and proof['depth_limit']==4 and row['feasible']==proof['feasible']);ns,nt,_=weighted.weighted_certificate(proof,catalog,check,'cost_fixture_');check('fixture_tree_counters',row['memo_states']==ns and row['canonical_split_attempts']==nt);states+=ns;splits+=nt;proofs[group]=proof;return row,proof
        select=lambda candidates,queried:choose(pred,scores,candidates,queried);audit=GraphAudit(folder,None,pred,select,tree,identities,check,read,sha,canonical);_,root=tree(list(range(n)),gate);naive_calls=0
        for run in fixture['runs']:
            remaining=run['remaining_MAP'];baseline=audit.verify(run['baseline_group'],list(range(n)),[],remaining,gate);check('fixture_run_guard',run['root_result_sha256']==sha(folder/f'cost{run["baseline_group"]}/result.json') and run['guard']==exact_guard(root,baseline) and run['calls_by_candidate']==baseline['remaining_calls_by_candidate'] and run['repeat_cache_zero_work']);naive={}
            for truth in range(n):
                active=list(range(n));queried=[];paid=0
                for _ in range(remaining):
                    if len(active)==1:break
                    selection=select(active,queried);index=selection['query_index'];naive_calls+=1;active=[c for c in active if bool(pred[c,index])==bool(pred[truth,index])];queried.append(index);paid+=1
                if len(active)>1:
                    _,terminal=tree(active,gate);paid+=terminal['leaf_depths'][str(truth)]
                naive[str(truth)]=paid
            check('fixture_independent_per_candidate',naive==run['calls_by_candidate'])
            if remaining==0:check('fixture_zero_tie',run['guard']['tie'] and run['guard']['handoff'])
            if remaining==1:check('fixture_frozen_costs',int(run['guard']['tree_weighted_cost'])==fixture['one_MAP_expected_tree_cost'] and int(run['guard']['baseline_weighted_cost'])==fixture['one_MAP_expected_baseline_cost'])
        nc,np_=audit.finish();check('fixture_complete_counts',len(tree_rows)==len(used)==fixture['tree_groups'] and nc==fixture['baseline_groups'] and np_==fixture['recursive_selector_calls'] and naive_calls==fixture['naive_selector_calls'] and fixture['all_branches_identical']);before_states+=states+nc;before_splits+=splits+np_;total_naive+=naive_calls;total_selector+=np_;total_paths+=audit.candidate_paths
    check('cost_gate_accounting',gate['compiler_and_baseline_memo_states']==before_states and gate['compiler_and_baseline_split_evaluations']==before_splits and gate['recursive_selector_calls']==total_selector==16 and gate['naive_selector_calls']==total_naive==61 and gate['remaining_MAP_values']==[0,1,2,4] and gate['actual_oracle_calls']==0 and gate['all_candidate_rollouts_match'] and gate['accept_reject_and_tie_cases_verified'])
    for rel,digest in gate['artifact_sha256'].items():check('cost_gate_artifact_hash',sha(P/'cost-gate'/rel)==digest)
    return dict(states=before_states,splits=before_splits,candidate_paths=total_paths,selector_calls=total_selector,naive_selector_calls=total_naive)
