"""Portable public H1bn adapter; frozen scientific files remain unchanged."""
import json
import audit_corruption_law as audit
from corruption_law_study import ROOT,PUBLIC,REGISTRY,PROTOCOL,sha


def portable_verify(raw=False):
    assert not raw,'Original-label audits require original raw artifacts'
    p=json.loads(PROTOCOL.read_text());mapping={str((REGISTRY/name).relative_to(ROOT)):PUBLIC/name for name in ['excluded_ids.npz','ledger.json']}
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():
        path=mapping.get(rel,ROOT/rel);assert sha(path)==digest,rel
    return p


def portable_job(seed):
    audit.verify=portable_verify
    return audit.audit_bank(seed,True)


def main():
    manifest=ROOT/'results/research/ripper/corruption_law_public_manifest.json';m=json.loads(manifest.read_text())
    for rel,digest in m['sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(PUBLIC/'summary.json')==m['raw_audit_sha256'];audit.verify=portable_verify;audit.public_job=portable_job;audit.public_audit()
    s=json.loads((PUBLIC/'summary.json').read_text());runtime=json.loads((PUBLIC/'runtime.json').read_text());assert runtime==s['runtime'] and runtime['status']=='validated' and runtime['wall_s']<180 and runtime['aggregate_cpu_s']<900
    commands=runtime['commands'];assert len(commands)==68 and all(c['returncode']==0 for c in commands)
    assert commands[0]['action']=='select' and all(c['action']=='predraw' for c in commands[1:33]) and commands[33]['action']=='materialize' and commands[34]['action']=='source-audit' and all(c['action']=='bank' for c in commands[35:67]) and commands[-1]['action']=='summary'
    q=s['query_accounting'];assert q['new_union_queries']==857088 and q['shared_noisy_queries']==0 and all(q[arm+'_queries']==285696 for arm in ['bursty','iid5','iid_pair'])
    assert q['h1az_through_present_sequence_union']==9047097 and q['clean_unique_oracle_cells']==81*s['source_audit']['new_unique_source_ids']<=321408 and q['clean_materialized_cell_instances']==321408
    assert s['alternative_query_accounting']['each_alternative_deployment']==285696
    print(json.dumps(dict(status='passed',manifest_paths=len(m['sha256'])+1,portable_registry_mappings=2,all_IDs_and32_common_masks_sealed_before_label_commands=True,resource_and_query_caps_passed=True)))


if __name__=='__main__':main()
