"""Validate H1bj public manifest, all proofs and resource/query ceilings."""
import json
from extra_coverage_study import ROOT,PUBLIC,sha
from audit_extra_coverage import public_audit


def main():
    path=ROOT/'results/research/ripper/extra_coverage_public_manifest.json';m=json.loads(path.read_text())
    for name,digest in m['sha256'].items():assert sha(ROOT/name)==digest,name
    s=json.loads((PUBLIC/'summary.json').read_text());assert sha(PUBLIC/'summary.json')==m['raw_audit_sha256'];public_audit()
    runtime=json.loads((PUBLIC/'runtime.json').read_text());assert runtime==s['runtime'] and runtime['status']=='validated' and runtime['wall_s']<180 and runtime['aggregate_cpu_s']<900
    commands=runtime['commands'];assert len(commands)==66 and all(c['returncode']==0 for c in commands)
    assert all(c['action']=='predraw' for c in commands[:32]) and commands[32]['action']=='seal' and all(c['action']=='bank' for c in commands[33:65]) and commands[-1]['action']=='summary'
    q=s['query_accounting'];assert q['actual_increment']<=403200 and q['aligned_added_to_own']==q['permuted_added_to_own']==201600
    assert q['new_union_total']==q['old_union']+q['actual_increment'] and q['h1az_through_present_sequence_union']==5148181+q['actual_increment']
    assert s['alternative_query_accounting']['each_arm_complete_deployment']==q['aligned_total']+240000==q['permuted_total']+240000
    print(json.dumps(dict(status='passed',manifest_paths=len(m['sha256'])+1,all_masks_sealed_before_extra_label_commands=True,runtime_and_query_caps_passed=True)))


if __name__=='__main__':main()
