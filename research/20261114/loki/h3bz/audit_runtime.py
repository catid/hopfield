"""Independent complete runtime-gate replay; no scientific compiler is invoked."""
import hashlib
import numpy as np
from audit_costs import GraphAudit, choose, exact_guard
from audit_trees import TreeAudit
from audit_original import OriginalAudit
from verify_tree import weighted_certificate
from verify_gates import verify as verify_serialized_gates
from audit_baseline_gate import verify as verify_baseline_gate


def verify(P,S,prep,check,archive,canonical):
    read,sha=archive.read,archive.sha
    gate_states=1;gate_splits=0;toy=read(P/'synthetic-gate.json')
    for fixture in toy['fixtures']:
        catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(fixture['signatures'])];proof=fixture['result'];ns,nt,_=weighted_certificate(proof,catalog,check,'weighted_gate_');check('weighted_gate_exact_cost',proof['optimal_cost']==fixture['expected_cost']);gate_states+=ns;gate_splits+=nt
    check('weighted_gate_counts',gate_states==toy['global_states_after_gate'] and gate_splits==toy['global_split_attempts_after_gate'])
    pruning=read(P/'pruning-gate.json');cache_gate=read(P/'cache-gate.json');adapter_paths={'pruning':P/'pruning-gate','cache':P/'cache-gate'}
    # The serialized gate audit accepts physical gzip ledgers and logical proof reads.
    gate_audit=verify_serialized_gates(P,S,pruning_name='pruning-gate',cache_name='cache-gate',read_json=read,hash_file=sha)
    gate_states+=gate_audit['charged_memo_states'];gate_splits+=gate_audit['charged_work_steps'];check('pruning_cache_gate_global_counts',cache_gate['global_memo_states']==gate_states and cache_gate['global_work_steps']==gate_splits)
    cost_gate=read(P/'cost-gate.json');cost_gate_states=cost_gate_splits=cost_gate_paths=0;gate_selector=gate_naive=0
    for fixture in cost_gate['fixtures']:
        folder=P/'cost-gate'/fixture['name'];n=len(fixture['scores']);pred=np.array([[bool(sig&(1<<c)) for sig in fixture['signatures']] for c in range(n)]);ids=dict(predictions_sha256=hashlib.sha256(np.packbits(pred,axis=1,bitorder='little').tobytes()).hexdigest(),scores_sha256=hashlib.sha256(canonical(fixture['scores'])).hexdigest(),selector_sha256=sha(S/'selector.py'),tree_compiler_sha256=sha(S/'compiler.py'),tree_wrapper_sha256=sha(S/'policy.py'));rows=[read(folder/f'group{i}/result.json') for i in range(fixture['tree_groups'])];trees=TreeAudit(folder,S,rows,pred,fixture['scores'],ids['predictions_sha256'],check,read,sha,canonical);select=lambda candidates,queried:choose(pred,fixture['scores'],candidates,queried);costs=GraphAudit(folder,None,pred,select,trees.get,ids,check,read,sha,canonical,archive.logical_cost_groups);naive_calls=0
        for run in fixture['runs']:
            _,tree=trees.get(list(range(n)),cost_gate,run['requested_depth']);baseline=costs.verify(run['baseline_group'],list(range(n)),[],run['remaining_MAP'],cost_gate);decision=exact_guard(tree,baseline);check('cost_gate_guard',decision==run['guard'] and run['root_result_sha256']==sha(folder/f'cost{run["baseline_group"]}/result.json') and run['calls_by_candidate']==baseline['remaining_calls_by_candidate'] and run['exact_by_candidate']==baseline['exact_identification_by_candidate'] and (not decision['baseline_admissible'] or decision['cost_comparison_passes']));naive={};exact={}
            for truth in range(n):
                active=list(range(n));queried=[];used=0
                for _ in range(run['remaining_MAP']):
                    if len(active)==1:break
                    selection=select(active,queried);index=selection['query_index'];naive_calls+=1;active=[c for c in active if bool(pred[c,index])==bool(pred[truth,index])];queried.append(index);used+=1
                if len(active)>1:
                    _,terminal=trees.get(active,cost_gate,4);used+=terminal['leaf_depths'][str(truth)] if terminal['feasible'] else 0;exact[str(truth)]=terminal['feasible']
                else:exact[str(truth)]=True
                naive[str(truth)]=used
            check('cost_gate_independent_rollouts',naive==run['calls_by_candidate'] and exact==run['exact_by_candidate'])
            if 'remaining_only' in fixture:check('six_failed_branches_retained',sum(not value for value in exact.values())==6 and decision['reason']=='baseline_inadmissible' and not decision['handoff'])
        ns,nt=trees.finish();nc,np_=costs.finish();check('cost_gate_node_counts',nc==fixture['baseline_groups'] and np_==fixture['recursive_selector_calls'] and naive_calls==fixture['naive_selector_calls']);cost_gate_states+=ns+nc;cost_gate_splits+=nt+np_;cost_gate_paths+=costs.candidate_paths;gate_selector+=np_;gate_naive+=naive_calls
    check('cost_gate_total_counts',cost_gate_states==cost_gate['compiler_and_baseline_memo_states'] and cost_gate_splits==cost_gate['compiler_and_baseline_split_evaluations'] and gate_selector==cost_gate['recursive_selector_calls']==30 and gate_naive==cost_gate['naive_selector_calls']==152)
    for rel,digest in cost_gate['artifact_sha256'].items():check('cost_fixture_identity',sha(P/'cost-gate'/rel)==digest)
    gate_states+=cost_gate_states;gate_splits+=cost_gate_splits;check('cost_gate_global_work',gate_states==cost_gate['global_states_after_gate'] and gate_splits==cost_gate['global_split_evaluations_after_gate'])
    original_gate=verify_baseline_gate(P/'baseline-gate',S);original_summary=read(P/'baseline-gate.json');check('baseline_gate_receipt',original_summary==read(P/'baseline-gate/summary.json') and original_gate['charged_memo_states']==original_summary['memo_states_added'] and original_gate['charged_work_steps']==original_summary['work_steps_added']);gate_states+=original_gate['charged_memo_states'];gate_splits+=original_gate['charged_work_steps']
    wrapper=read(P/'wrapper-gate.json');check('wrapper_receipt',wrapper==read(P/'wrapper-gate/summary.json') and len(wrapper['fixtures'])==4);wrapper_states=wrapper_work=0
    for fixture in wrapper['fixtures']:
        cfg=fixture['fixture'];n=cfg['n'];pred=np.array([[bool(sig&(1<<cid)) for sig in cfg['signatures']] for cid in range(n)]);folder=P/'wrapper-gate'/cfg['name'];auditor=OriginalAudit(folder,S,[fixture['result']],pred,cfg['scores'],'a'*64,check,read,sha,canonical);_,proof=auditor.get(list(range(n)),prep);check('wrapper_declared_feasibility',proof['feasible']==cfg['feasible']);ns,nt=auditor.finish();wrapper_states+=ns;wrapper_work+=nt
    for name,digest in wrapper['source_sha256'].items():check('wrapper_source_identity',sha(S/name)==digest)
    check('wrapper_work_counts',wrapper_states==wrapper['memo_states_added'] and wrapper_work==wrapper['work_steps_added']);gate_states+=wrapper_states;gate_splits+=wrapper_work
    check('complete_runtime_gate_counts',gate_states==prep['runtime_gate_memo_states'] and gate_splits==prep['runtime_gate_work_steps'])
    return dict(memo_states=gate_states,work_steps=gate_splits,inline_artifact_checks=gate_audit['checks'],baseline_artifact_checks=original_gate['checks'],wrapper_states=wrapper_states,wrapper_work=wrapper_work)
