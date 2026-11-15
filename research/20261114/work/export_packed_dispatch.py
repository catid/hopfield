"""H2bs public evidence and zero-new-observation performance accounting."""
import json
from pathlib import Path
import shutil
import sys
import study as S
DEST=S.OUT/'bp_first_packed_dispatch'
PUBLIC=S.ROOT/'results/research/work/bp-first-packed-dispatch'


def evidence():
    assert not PUBLIC.exists() and json.loads((DEST/'public-staging/audit.json').read_text())['status']=='passed'
    shutil.copytree(DEST/'public-staging',PUBLIC)
    candidate=DEST/'candidate-protocol.json';assert candidate.exists();shutil.copyfile(candidate,PUBLIC/'candidate-protocol.json')
    for name in ['initial-design.json','queue-synthetic-gate.json','accounting-synthetic-gate.json','study-synthetic-gate.json','source-mapping-synthetic-gate.json','source-development-notes.json']:shutil.copyfile(DEST/name,PUBLIC/name)


def publish():
    a=json.loads((PUBLIC/'audit.json').read_text());v=json.loads((PUBLIC/'public-verification.json').read_text());f=json.loads((PUBLIC/'figure-determinism.json').read_text());assert a['status']==v['status']==f['status']=='passed' and f['visually_inspected']
    assert a['fresh_scientific_inputs']==a['new_scientific_jobs']==a['new_scientific_evaluations']==0 and a['timed_model_pipeline_copies']==147456
    root=PUBLIC.parent;summary=json.loads((root/'summary.json').read_text());assert 'bp_first_packed_dispatch' not in summary
    summary['bp_first_packed_dispatch']={'hypothesis':'H2bs','source_cohort':'H2br','fresh_inputs':0,'new_scientific_jobs':0,'new_scientific_evaluations':0,'timed_model_pipeline_copies':147456,'source_preservation':a['source_preservation'],'cost_endpoint':a['cost_endpoint'],'scope':'Performance-only examined-source packing comparison with newly executed original-order control. All host queue/packing/scatter charged; no new accuracy or population claim and no historical timing substitution.'}
    summary['hypotheses'].append({'campaign':'bp_first_packed_dispatch','hypothesis':'H2bs_complete_cost_saving10','outcome':a['cost_endpoint']})
    p=json.loads((PUBLIC/'protocol.json').read_text());summary['protocols']['bp-first-packed-dispatch-protocol']={'sha256':S.sha(PUBLIC/'protocol.json'),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bs','status':'completed'}
    summary['status']='Ongoing research; H2bs source-preserving offline packing cost independently audited';summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc()
    S.save(root/'bp-first-packed-dispatch-summary.json',a);S.save(root/'summary.json',summary);S.save(root/'manifest.json',{'source_sha256':S.sha(Path(__file__)),'files':{str(q.relative_to(root)):S.sha(q) for q in sorted(root.rglob('*')) if q.is_file() and q!=root/'manifest.json'}})


if __name__=='__main__':publish() if '--publish' in sys.argv else evidence()
