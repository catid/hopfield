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
    admissible=baseline['all_candidates_identified'];handoff=admissible and tree_cost<=baseline_cost
    return dict(tree_weighted_cost=str(tree_cost),baseline_weighted_cost=str(baseline_cost),weight_sum=proof['weight_sum'],baseline_admissible=admissible,infeasible_baseline_candidates=[c for c in proof['candidates'] if not baseline['exact_identification_by_candidate'][str(c)]],cost_comparison_passes=tree_cost<=baseline_cost,handoff=handoff,tie=tree_cost==baseline_cost,reason='baseline_inadmissible' if not admissible else ('cost_accept' if handoff else 'cost_reject'),comparison='all baseline candidates identified AND exact integer tree_cost <= baseline_cost',expected_nonincrease_when_accepted=handoff)


class GraphAudit:
    def __init__(self,folder,registry,predictions,select,tree,identities,check,read,sha,canonical,cost_groups):
        self.folder=folder;self.registry=registry;self.predictions=predictions;self.select=select;self.tree=tree;self.identities=identities;self.check=check;self.read=read;self.sha=sha;self.canonical=canonical;self.rows={};self.keys={};self.MAP_partitions=0;self.visiting=set();self.candidate_paths=0;self.cost_groups=cost_groups

    def verify(self,group,candidates,queried,remaining_MAP,bound):
        check=self.check;queried=sorted(set(queried));folder=self.folder/f'cost{group}';inputs=dict(policy='fixed_remaining_MAP_then_posterior_depth4',candidates=candidates,remaining_MAP=remaining_MAP,queried=queried,depth=4,**self.identities);key=hashlib.sha256(self.canonical(inputs)).hexdigest()
        check('cost_context',candidates==sorted(set(candidates)) and candidates and 0<=remaining_MAP<=4 and all(all(bool(self.predictions[c,index])==bool(self.predictions[candidates[0],index]) for c in candidates) for index in queried))
        if group in self.rows:
            row=self.rows[group];check('cost_cache_reuse',row['cache_key']==key and row['utc']<=bound['utc']);return row
        check('cost_acyclic_unique_cache',group not in self.visiting and key not in self.keys);self.visiting.add(group);row=self.read(folder/'result.json');check('cost_full_input_identity',self.read(folder/'input.json')==inputs and row['input_sha256']==self.sha(folder/'input.json') and row['group']==group and row['cache_key']==key and row['candidates']==candidates and row['queried']==queried and row['remaining_MAP']==remaining_MAP and row['utc']<=bound['utc'] and row['all_candidate_branches_retained'] and row['actual_oracle_calls']==0 and row['CPU_seconds']>=0 and row['wall_seconds']>=0)
        if self.registry is not None:check('cost_registry_identity',self.registry[group]==dict(group=group,cache_key=key,input_sha256=self.sha(folder/'input.json'),result_sha256=self.sha(folder/'result.json')))
        if len(candidates)==1:
            expected={str(candidates[0]):0};exact={str(candidates[0]):True};check('cost_singleton',row['reason']=='singleton' and row['request'] is None and row['selection'] is None and row['tree'] is None and row['children']==[])
        elif remaining_MAP==0:
            tree_row,proof=self.tree(candidates,row,4);expected={str(c):proof['leaf_depths'][str(c)] if proof['feasible'] else 0 for c in candidates};exact={str(c):proof['feasible'] for c in candidates};check('cost_terminal_tree',row['reason']==('posterior_tree' if proof['feasible'] else 'posterior_tree_infeasible') and row['request'] is None and row['selection'] is None and row['children']==[] and row['tree']==dict(group=tree_row['group'],proof_sha256=tree_row['proof_sha256'],group_result_sha256=self.sha(self.folder/f'group{tree_row["group"]}/result.json')))
        else:
            selection=self.select(candidates,queried);index=selection['query_index'];request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=self.predictions.shape[1],candidates=candidates,queried=queried);check('cost_MAP_selection',row['reason']=='MAP_partition' and row['request']==request and row['selection']==selection and row['tree'] is None and index is not None and index not in queried and len(row['children'])==2);expected={};exact={};self.MAP_partitions+=1
            for label,ref in zip([False,True],row['children']):
                subset=[c for c in candidates if bool(self.predictions[c,index])==label];check('cost_both_nonempty_branches',subset and ref['label'] is label);child=self.verify(ref['group'],subset,queried+[index],remaining_MAP-1,row);check('cost_child_hash',ref==dict(label=label,group=child['group'],result_sha256=self.sha(self.folder/f'cost{child["group"]}/result.json'),cache_key=child['cache_key']));expected.update({str(c):1+child['remaining_calls_by_candidate'][str(c)] for c in subset});exact.update({str(c):child['exact_identification_by_candidate'][str(c)] for c in subset})
        check('cost_every_candidate_cost',row['remaining_calls_by_candidate']==expected and row['exact_identification_by_candidate']==exact and all(type(value) is bool for value in exact.values()) and row['all_candidates_identified']==all(exact.values()) and all(type(v) is int and 0<=v<=remaining_MAP+4 for v in expected.values()));self.candidate_paths+=len(expected);self.rows[group]=row;self.keys[key]=group;self.visiting.remove(group);return row

    def finish(self):
        actual=self.cost_groups(self.folder);self.check('cost_no_unreferenced_work',actual==set(self.rows) and actual==set(range(len(actual))) and (self.registry is None or len(self.registry)==len(actual)));return len(self.rows),self.MAP_partitions
