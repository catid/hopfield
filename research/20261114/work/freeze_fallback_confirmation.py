"""Freeze H2bk source, statistics, cohort and budgets before mandatory gates/draws."""
import copy
import json
import study as S
from fallback_confirmation_study import PP,DEST,BASE,REPORTED


def freeze():
    assert not PP.exists();source=S.OUT/'loop80_confirmation';parent=json.loads((source/'protocol.json').read_text())
    proposal=S.HERE/'fallback-confirmation-proposal.json';draft=json.loads(proposal.read_text())
    registry_path=DEST/'seed-registry.json';registry=json.loads(registry_path.read_text());assert registry['status']=='passed' and registry['fresh_inputs_drawn']==0 and registry['proposed_seed_base']==BASE
    stat_path=DEST/'statistics-gate.json';stat=json.loads(stat_path.read_text());assert stat['status']=='passed' and not stat['fresh_outcomes_read']
    for name,h in stat['source_sha256'].items():assert S.sha(S.HERE/name)==h
    for r in registry['protocol_registry']:assert S.sha(S.ROOT/r['path'])==r['sha256']
    h2bj=S.OUT/'certificate_fallback/audit.json';j=json.loads(h2bj.read_text());assert j['status']=='passed' and j['execution_decision']==j['cross_certificate_decision']=='supported'
    h2bi=S.OUT/'artifact_overlap/audit.json';assert json.loads(h2bi.read_text())['status']=='passed'
    names=['fallback_confirmation_study.py','fallback_confirmation_batch.py','fallback_confirmation_job.py',
        'fallback_confirmation_statistics.py','preflight_fallback_confirmation_statistics.py',
        'freeze_fallback_confirmation.py','preflight_fallback_confirmation.py','audit_fallback_confirmation.py',
        'audit_fallback_confirmation_inputs.py','audit_fallback_confirmation_seed_registry.py',
        'certificate_fallback_policy.py','bounded_archive_writer.py','artifact_overlap_pipeline.py']
    p=copy.deepcopy(draft);p.update(status='frozen_before_mandatory_source_gates_and_fresh_draws',frozen_utc=S.utc(),
        arms=parent['arms'],reported_arms=list(REPORTED),models=parent['models'],
        jobs=[{'id':f'shard{shard:02d}-s{seed}','shard':shard,'seed':seed,'gpu':shard%2} for shard in range(16) for seed in range(3)],
        cohort={'base_seed':BASE,'n':4096,'size':180,'loop_probability':.8,'prior_registry':registry['prior_data_manifests']})
    p['source_sha256']={**parent['source_sha256'],**{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names}}
    evidence=[proposal,registry_path,stat_path,source/'protocol.json',source/'audit.json',source/'datasets.json',
        source/'model-extraction.json',h2bj,h2bi,S.OUT/'certificate_fallback/protocol.json',
        S.OUT/'certificate_fallback/routing.json',S.OUT/'artifact_overlap/protocol.json',
        S.OUT/'artifact_overlap_preflight/synthetic-gate.json']
    evidence.extend(source/f'model-seed{s}-{a}.npz' for s in range(3) for a in ['pl','old_binary'])
    p['source_evidence_sha256']={str(q.relative_to(S.ROOT)):S.sha(q) for q in evidence}
    p['source_gate_files']={}
    for seed in range(3):
        for q in (source/'runs'/f'shard00-s{seed}').glob('*.npz'):p['source_gate_files'][str(q.relative_to(S.ROOT))]=S.sha(q)
        for q in (source/'runs'/f'shard15-s{seed}').glob('batch07*.npz'):p['source_gate_files'][str(q.relative_to(S.ROOT))]=S.sha(q)
        for folder in ['fallback','diagnostics']:
            q=S.OUT/'certificate_fallback'/folder/f'i4078-s{seed}.npz';p['source_gate_files'][str(q.relative_to(S.ROOT))]=S.sha(q)
    S.save(PP,p);S.save(DEST/'protocol.json',p);S.emit('fallback_confirmation_frozen',sha256=S.sha(PP))


if __name__=='__main__':freeze()
