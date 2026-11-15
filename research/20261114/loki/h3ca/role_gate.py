"""Reject wrong actual/counterfactual score roles and cache identities independently."""
import copy
import json
from common import SRC,read,sha
from audit_io import AuditIO
from audit_namespace_gate import verify
from ledger import canonical


def run(folder,states,work,archive_root=None):
    folder=folder.resolve()
    audit_root=folder if archive_root is None else archive_root.resolve()
    def initial_check(name,value):assert value,name
    base=AuditIO(audit_root,SRC,initial_check)
    read=base.read
    rows=read(folder/'T/arm-results.json')
    root=next(r['cost_guards'][0]['baseline_cost_group'] for r in rows if r['cost_guards'] and
              read(folder/f"T/cost{r['cost_guards'][0]['baseline_cost_group']}/result.json")['reason']=='MAP_partition')
    treegroup=rows[2]['routing'][0]['group']
    mutations={}
    modes={
        'actual_MAP_uses_tree_score_file':'complete_query_record_binding',
        'counterfactual_MAP_uses_tree_score_file':'cost_MAP_selection',
        'T_selector_file_contains_tree_scores':'fixed_fixture_source_score_identity',
        'T_tree_file_contains_selector_scores':'fixed_fixture_source_score_identity',
        'tree_input_selector_identity_swapped':'tree_full_input_identity',
        'tree_input_tree_identity_swapped':'tree_full_input_identity',
        'counterfactual_selector_identity_swapped':'cost_full_input_identity',
        'counterfactual_tree_identity_swapped':'cost_full_input_identity',
        'tree_cache_key_from_other_role':'tree_full_input_identity',
        'namespace_selector_identity_swapped':'arm_namespace_identity',
    }
    for mode,expected in modes.items():
        checks=0
        def check(name,value):
            nonlocal checks
            assert value,name;checks+=1
        io=AuditIO(audit_root,SRC,check);original_read,original_data=io.read,io.data
        def changed_read(path):
            value=original_read(path);relative=str(path.relative_to(folder))
            if mode=='T_selector_file_contains_tree_scores' and relative=='T/selector-scores.json':
                return read(folder/'T/tree-scores.json')
            if mode=='T_tree_file_contains_selector_scores' and relative=='T/tree-scores.json':
                return read(folder/'T/selector-scores.json')
            value=copy.deepcopy(value)
            if mode=='counterfactual_MAP_uses_tree_score_file' and relative==f'T/cost{root}/result.json':
                value['request']['scores_path']='tree-scores.json'
            if mode.startswith('tree_input_') and relative==f'T/group{treegroup}/compiler-input.json':
                key='selector_scores_sha256' if 'selector' in mode else 'tree_scores_sha256'
                value[key]='0'*64
            if mode.startswith('counterfactual_') and mode.endswith('_swapped') and relative==f'T/cost{root}/input.json':
                key='selector_scores_sha256' if 'selector' in mode else 'tree_scores_sha256'
                value[key]='0'*64
            if mode=='tree_cache_key_from_other_role':
                wrong=read(folder/f'H/group{treegroup}/result.json')['cache_key']
                if relative=='T/groups.json':value[treegroup]['cache_key']=wrong
                if relative==f'T/group{treegroup}/result.json':value['cache_key']=wrong
            return value
        def changed_data(path):
            raw=original_data(path)
            if path!=folder/'query-ledger.jsonl' or mode not in ['actual_MAP_uses_tree_score_file','namespace_selector_identity_swapped']:return raw
            rows=[json.loads(line) for line in raw.splitlines()];changed=False;previous='0'*64
            for event in rows:
                if not changed and event['data'].get('arm')=='T':
                    if mode=='actual_MAP_uses_tree_score_file' and event['kind']=='query' and event['data']['phase']=='map':
                        event['data']['request']['scores_path']='tree-scores.json';changed=True
                    elif mode=='namespace_selector_identity_swapped' and event['kind']=='arm_namespace_bound':
                        event['data']['selector_scores_sha256']='0'*64;changed=True
                event['previous_sha256']=previous
                if event['kind']=='answer':event['data']['query_sha256']=previous
                import hashlib
                event['sha256']=hashlib.sha256(canonical({k:v for k,v in event.items() if k!='sha256'})).hexdigest()
                previous=event['sha256']
            assert changed
            return b''.join(canonical(event)+b'\n' for event in rows)
        io.read=changed_read;io.data=changed_data
        try:verify(folder,io,check,states,work)
        except AssertionError as error:
            assert str(error)==expected,(mode,str(error),expected)
            mutations[mode]=dict(rejected_at=expected,checks_before_rejection=checks)
        else:raise AssertionError('Score role mutation accepted: '+mode)
        finally:
            if io.archive:io.archive.close()
    if base.archive:base.archive.close()
    return dict(status='passed_independent_score_role_and_cache_identity_mutations',mutations=mutations,
                no_artifact_edits=True,fresh_truth_draws=0,fresh_source_draws=0,actual_oracle_calls=0,
                source_sha256={name:sha(SRC/name) for name in ['role_gate.py','audit_namespace_gate.py','audit_arm.py','audit_trees.py','audit_costs.py']})
