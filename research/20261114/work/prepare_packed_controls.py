"""Freeze H2bt source packet after H2bs full preservation audit, never cost filtering."""
import ast
import json
import study as S
DEST=S.OUT/'bp_first_packed_controls'
APPROVED='8e10fe77ae11e3b934eb52b24e7edb962b756e74a0409cf92f85b14cdaaa18d7'


def modules():
    names={p.name for p in S.HERE.glob('*packed_controls*.py')};pending=list(names)
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse((S.HERE/name).read_text())):
            imports=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in imports:
                file=(module or '').split('.')[0]+'.py'
                if file not in names and (S.HERE/file).exists():names.add(file);pending.append(file)
    return sorted(names)


def main():
    target=DEST/'source-packet.json';assert not target.exists();proposal=S.HERE/'bp-first-packed-controls-proposal.json';assert S.sha(proposal)==APPROVED
    design=json.loads(proposal.read_text());previous=S.OUT/'bp_first_packed_dispatch';parent=json.loads((previous/'protocol.json').read_text());audit=json.loads((previous/'audit.json').read_text())
    assert audit['status']=='passed' and audit['timed_model_pipeline_copies']==147456 and S.sha(previous/'protocol.json')==design['source']['H2bs_protocol_sha256']
    assert S.sha(previous/'sealed-replays.json')==audit['seal_sha256']
    for key in ['source_sha256','source_evidence_sha256']:
        for name,digest in parent[key].items():assert S.sha(S.ROOT/name)==digest,name
    evidence=[proposal,DEST/'initial-design.json']
    for name in ['study-synthetic-gate.json','readiness-synthetic-gate.json','cost-synthetic-gate.json','mapping-synthetic-gate.json','global-audit-synthetic-gate.json']:
        file=DEST/name;gate=json.loads(file.read_text());assert gate['status']=='passed' and gate['source_outcomes_read'] is False and gate['gpu_executions']==0
        for source,digest in gate['source_sha256'].items():assert S.sha(S.ROOT/source)==digest,source
        evidence.append(file)
    for name in ['protocol.json','audit.json','audit-worker.json','sealed-replays.json']:evidence.append(previous/name)
    for name in ['queue-synthetic-gate.json','accounting-synthetic-gate.json','source-mapping-synthetic-gate.json']:
        file=previous/name;gate=json.loads(file.read_text());assert gate['status']=='passed'
        for source,digest in gate['source_sha256'].items():assert S.sha(S.ROOT/source)==digest,source
        evidence.append(file)
    source=parent['source'];base=S.ROOT/source['directory'];assert S.sha(base/'audit.json')==design['source']['H2br_audit_sha256']==source['audit_sha256']
    assert S.sha(base/'sealed-replays.json')==design['source']['H2br_seal_sha256']==source['seal_sha256']
    packing={'directory':str(previous.relative_to(S.ROOT)),'protocol_sha256':S.sha(previous/'protocol.json'),'audit_sha256':S.sha(previous/'audit.json'),'seal_sha256':S.sha(previous/'sealed-replays.json')}
    p={'status':'frozen_before_changed_source_gates','approved_proposal_sha256':APPROVED,'created_utc':S.utc(),'models':parent['models'],'source':source,'packing_source':packing,'budgets':design['proposed_budgets'],'timing_schedule':design['timing_schedule'],
       'source_sha256':{**parent['source_sha256'],**{str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in modules()}},
       'source_evidence_sha256':{**parent['source_evidence_sha256'],**{str(file.relative_to(S.ROOT)):S.sha(file) for file in evidence}},
       'source_gate':{'indices':'First64 inputs in H2br shards0and15, identities0..63 and3840..3903.','model_inputs_per_arm_per_gpu':384,'actual_modes':['LP96','BP_geometry_LP96','packed_dispatch'],'order_both_gpus':['LP96','BP_geometry_LP96','packed_dispatch']},
       'accounting':design['accounting'],'scope':'Source-gate packet only; independent source audit eligibility never depends on H2bs cost outcome. Parent packet review is required before changed GPU gates. No timing authorization from packet creation.'}
    S.save(target,p);print(json.dumps({'packet_sha256':S.sha(target),'source_bindings':len(p['source_sha256']),'evidence_bindings':len(p['source_evidence_sha256'])},indent=2))


if __name__=='__main__':main()
