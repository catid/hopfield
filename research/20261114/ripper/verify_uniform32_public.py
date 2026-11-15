"""Portable public H1bk adapter; frozen scientific files remain unchanged."""
import json
import audit_uniform32 as audit
from uniform32_study import ROOT,PUBLIC,REGISTRY,PROTOCOL,sha


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
    manifest=ROOT/'results/research/ripper/uniform32_public_manifest.json';m=json.loads(manifest.read_text())
    for rel,digest in m['sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(PUBLIC/'summary.json')==m['raw_audit_sha256'];audit.verify=portable_verify;audit.public_job=portable_job;audit.public_audit()
    s=json.loads((PUBLIC/'summary.json').read_text());runtime=json.loads((PUBLIC/'runtime.json').read_text());assert runtime==s['runtime'] and runtime['status']=='validated' and runtime['wall_s']<180 and runtime['aggregate_cpu_s']<900
    commands=runtime['commands'];assert len(commands)==68 and all(c['returncode']==0 for c in commands)
    assert commands[0]['action']=='select' and all(c['action']=='predraw' for c in commands[1:33]) and commands[33]['action']=='materialize' and commands[34]['action']=='source-audit' and all(c['action']=='bank' for c in commands[35:67]) and commands[-1]['action']=='summary'
    q=s['query_accounting'];assert q['new_union_queries']<=1843200 and q['uniform_queries']==q['donor_queries']==921600 and q['new_union_queries']==1843200-q['shared_queries']
    assert s['alternative_query_accounting']['uniform_complete_deployment']==921600 and s['alternative_query_accounting']['donor_complete_deployment']==921600+240000
    print(json.dumps(dict(status='passed',manifest_paths=len(m['sha256'])+1,portable_registry_mappings=2,all_IDs_and_masks_sealed_before_label_commands=True,resource_and_query_caps_passed=True)))


if __name__=='__main__':main()
