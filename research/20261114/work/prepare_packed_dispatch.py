"""Freeze H2bs source packet only after H2br's complete independent audit."""
import ast
import json
from pathlib import Path
import study as S
DEST=S.OUT/'bp_first_packed_dispatch'
APPROVED='75dea18de381d852ba9f7e2ba2b05539d90eac5b5e21e6d1354ceef22e39182d'


def modules():
    names={p.name for p in S.HERE.glob('*packed_dispatch*.py')};pending=list(names)
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse((S.HERE/name).read_text())):
            imports=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in imports:
                file=(module or '').split('.')[0]+'.py'
                if file not in names and (S.HERE/file).exists():names.add(file);pending.append(file)
    return sorted(names)


def main():
    target=DEST/'source-packet.json';assert not target.exists();proposal=S.HERE/'bp-first-packed-dispatch-proposal.json';assert S.sha(proposal)==APPROVED
    design=json.loads(proposal.read_text());base=S.OUT/'bp_first_endpoint_dispatch';parent=json.loads((base/'protocol.json').read_text())
    # Outcome-dependent source artifacts are opened only after the passed audit.
    audit=json.loads((base/'audit.json').read_text());assert audit['status']=='passed' and audit['protocol_sha256']==design['source']['protocol_sha256']
    assert S.sha(base/'sealed-replays.json')==audit['sealed_replays_sha256'] and audit['fresh_measured_replay_model_pipeline_copies']==221184
    for key in ['source_sha256','source_evidence_sha256']:
        for name,digest in parent[key].items():assert S.sha(S.ROOT/name)==digest,name
    evidence=[proposal,DEST/'initial-design.json',DEST/'source-development-notes.json']
    for name in ['queue-synthetic-gate.json','accounting-synthetic-gate.json','study-synthetic-gate.json','source-mapping-synthetic-gate.json']:
        path=DEST/name;gate=json.loads(path.read_text());assert gate['status']=='passed' and gate['source_outcomes_read'] is False and gate['gpu_executions']==0
        for source,digest in gate['source_sha256'].items():assert S.sha(S.ROOT/source)==digest,source
        evidence.append(path)
    for name in ['protocol.json','audit.json','audit-worker.json','sealed-replays.json','datasets.json','independent-input-audit.json']:evidence.append(base/name)
    datasets=json.loads((base/'datasets.json').read_text())['datasets'];assert len(datasets)==16
    for ds in datasets:assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    receipts={}
    for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
        path=base/'runs'/f'block0-{mode}'/'result.json';receipts[mode]=S.sha(path);evidence.append(path)
    source={'directory':str(base.relative_to(S.ROOT)),'protocol_sha256':S.sha(base/'protocol.json'),'audit_sha256':S.sha(base/'audit.json'),'seal_sha256':S.sha(base/'sealed-replays.json'),'datasets':datasets,'cohort_receipts_sha256':receipts}
    p={'status':'frozen_before_changed_source_gates','approved_proposal_sha256':APPROVED,'created_utc':S.utc(),'models':parent['models'],'source':source,'budgets':design['proposed_budgets'],'timing_schedule':design['timing_schedule'],
       'source_sha256':{**parent['source_sha256'],**{str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in modules()}},
       'source_evidence_sha256':{**parent['source_evidence_sha256'],**{str(path.relative_to(S.ROOT)):S.sha(path) for path in evidence}},
       'source_gate':{'indices':'First64 inputs in H2br shards0and15, original global identities0..63 and3840..3903.','model_inputs_per_arm_per_gpu':384,'actual_modes':['original_order','packed'],'order_both_gpus':['original_order','packed']},
       'accounting':design['accounting'],'scope':'Source-gate packet only. Approval of this packet is required before either changed GPU gate. No timing execution is authorized by creating this file.'}
    S.save(target,p);print(json.dumps({'packet_sha256':S.sha(target),'source_bindings':len(p['source_sha256']),'evidence_bindings':len(p['source_evidence_sha256'])},indent=2))


if __name__=='__main__':main()
