"""Export complete H2bg intervention evidence; zero new fresh evaluations."""
import json
import shutil
import study as S

DEST=S.OUT/'loop80_continuation'
PUBLIC=S.ROOT/'results/research/work/continuation'


def evidence():
    a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';PUBLIC.mkdir(exist_ok=True)
    for name in ['protocol.json','audit.json','inputs.json','synthetic-gate.json','source-gate-gpu0.json','source-gate-gpu1.json','worker0.json','worker1.json','live-gpu-counters.txt','prelaunch-missing-union-input.log','prelaunch-missing-union-gpu0.log','prelaunch-missing-union-gpu1.log']:shutil.copyfile(DEST/name,PUBLIC/name)
    for name in ['inputs','runs','audit-rows']:shutil.copytree(DEST/name,PUBLIC/name,dirs_exist_ok=True)
    diag=PUBLIC/'source-diagnosis';diag.mkdir(exist_ok=True)
    for name in ['union.json','geometry-progress.json']:shutil.copyfile(S.OUT/'loop80_residual_diagnosis'/name,diag/name)
    S.save(PUBLIC/'evidence-manifest.json',{'files':{str(f.relative_to(PUBLIC)):S.sha(f) for f in sorted(PUBLIC.rglob('*')) if f.is_file() and f.name not in ['evidence-manifest.json','public-verification.json','figure-determinism.json','continuation.png','continuation.svg']},'source_sha256':S.sha(__file__)})


def publish():
    a=json.loads((PUBLIC/'audit.json').read_text());v=json.loads((PUBLIC/'public-verification.json').read_text());assert v['status']=='passed';root=PUBLIC.parent;S.save(root/'continuation-summary.json',a);summary=json.loads((root/'summary.json').read_text());assert 'loop80_continuation' not in summary;summary['loop80_continuation']={'hypothesis':'H2bg','fresh_inputs':0,'new_scientific_evaluations':0,'source_intervention_cases':148,'source_all_input_denominator':4096,'joint_success':a['joint_success'],'model_rescues':a['model_rescues'],'primary_all148_decision':a['primary_all148_rescued_by96_limit256']['decision'],'size512_contrast_decision':a['separate512_no_changed_choices_or_outcomes']['decision'],'scope':'Archived H2be cap continuation only. All148 cases retained; source BP control and4096-row denominator preserved. Total96 repairs all source failures; fresh confirmation required.'};summary['hypotheses'].append({'campaign':'loop80_continuation','hypothesis':'H2bg source96rounds repairs all148caps; separate512 changes no choices/outcomes','outcome':{'all148_rescued':a['primary_all148_rescued_by96_limit256']['decision'],'size512_no_change':a['separate512_no_changed_choices_or_outcomes']['decision']}});p=json.loads((PUBLIC/'protocol.json').read_text());summary['protocols']['continuation-protocol']={'sha256':S.sha(PUBLIC/'protocol.json'),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bg','status':'completed_source_intervention'};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc();S.save(root/'summary.json',summary);S.save(root/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(f.relative_to(root)):S.sha(f) for f in sorted(root.rglob('*')) if f.is_file() and f!=root/'manifest.json'}})


if __name__=='__main__':
    import sys
    publish() if '--publish' in sys.argv else evidence()
