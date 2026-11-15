"""Bind complete archived cohort and queue/timing policy before any replay."""
import copy
import json
import study as S
from artifact_overlap_study import PP,DEST


def main():
    assert not PP.exists();source=S.OUT/'loop80_confirmation';a=json.loads((source/'audit.json').read_text());assert a['status']=='passed' and a['all_final_states']==98304;proposal=S.HERE/'artifact-overlap-proposal.json';p=copy.deepcopy(json.loads(proposal.read_text()));sg=S.OUT/'artifact_overlap_preflight/synthetic-gate.json';g=json.loads(sg.read_text());assert g['status']=='passed' and g['byte_identical_files']==24 and g['metrics']['async']['maximum_pending']==2 and g['metrics']['async']['backpressure_seconds']>0
    for path,h in g['source_sha256'].items():assert S.sha(S.ROOT/path)==h
    sp=json.loads((source/'protocol.json').read_text());ds=json.loads((source/'datasets.json').read_text())['datasets'][0];jobs=[]
    for seed in range(3):
        job=next(j for j in sp['jobs'] if j['shard']==0 and j['seed']==seed);run=source/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert receipt['status']=='complete' and receipt['protocol_sha256']==S.sha(source/'protocol.json');expected={'outputs.npz',*[f'batch{i:02d}.npz' for i in range(8)],*[r['path'] for r in receipt['trace_archives']]};files={path.name:S.sha(path) for path in sorted(run.glob('*.npz'))};assert set(files)==expected and len(files)==57;assert files['outputs.npz']==receipt['outputs_sha256'];assert all(files[r['path']]==r['sha256'] for r in receipt['trace_archives']);jobs.append({'id':job['id'],'source_job':job,'source_receipt_sha256':S.sha(run/'result.json'),'files_sha256':files})
    rows={r['job']['id']:r for r in a['rows']};joint={arm:int(__import__('numpy').array([rows[f'shard00-s{seed}']['success'][v] for seed in range(3)]).all(0).sum()) for v,arm in enumerate(sp['arms'])}
    p.update(status='frozen_before_mandatory_source_gates_and_timing',frozen_utc=S.utc(),source_cohort={'source_protocol_sha256':S.sha(source/'protocol.json'),'source_audit_sha256':S.sha(source/'audit.json'),'dataset':ds,'jobs':jobs,'independent_source_inputs':256,'model_inputs':768,'source_joint_success_by_internal_arm':joint,'source_full_study_joint96':4095,'source_full_study_failures_retained_in_original_study':'Input4078/all3caps remains in H2bh; deterministic shard00 speed cohort does not include it. No sourceoutcome-based cohort filtering.'},paired_order=[['serialized','async'],['async','serialized'],['async','serialized'],['serialized','async']])
    names=['bounded_archive_writer.py','preflight_archive_writer.py','artifact_overlap_pipeline.py','artifact_overlap_study.py','freeze_artifact_overlap.py','audit_artifact_overlap.py','fast_npz.py'];p['source_sha256']={**sp['source_sha256'],**{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names}};evidence=[proposal,sg,source/'protocol.json',source/'audit.json',source/'datasets.json',source/'model-extraction.json',source/'timing-profile.json'];p['source_evidence_sha256']={str(f.relative_to(S.ROOT)):S.sha(f) for f in evidence};p['models']=sp['models'];p['source_internal_arms']=sp['arms'];p['limits']['independent_cpu_audit_wall_seconds']=120;S.save(PP,p);S.save(DEST/'protocol.json',p);S.emit('artifact_overlap_frozen',sha256=S.sha(PP),source_npz_files=sum(len(j['files_sha256']) for j in jobs))


if __name__=='__main__':main()
