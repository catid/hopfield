"""H2bp post-audit public copy and exact fresh scientific accounting."""
import json
from pathlib import Path
import shutil
import sys
import study as S
DEST=S.OUT/'bp_first_endpoint_shift'
PUBLIC=S.ROOT/'results/research/work/bp-first-endpoint-shift'


def evidence():
    assert not PUBLIC.exists()
    assert json.loads((DEST/'public-staging/audit.json').read_text())['status']=='passed'
    shutil.copytree(DEST/'public-staging',PUBLIC)
    shutil.copyfile(S.OUT/'bp_first_fresh_confirmation/portable-statistics-preflight.json',PUBLIC/'inherited-portable-statistics-preflight.json')


def publish():
    a=json.loads((PUBLIC/'audit.json').read_text());v=json.loads((PUBLIC/'public-verification.json').read_text());f=json.loads((PUBLIC/'figure-determinism.json').read_text())
    assert a['status']==v['status']==f['status']=='passed' and f['visually_inspected']
    assert a['planned_jobs']==48 and a['primary_model_maze_arm_evaluations']==36864 and a['fresh_independent_inputs']==4096
    root=PUBLIC.parent;summary=json.loads((root/'summary.json').read_text());assert 'bp_first_endpoint_shift' not in summary
    summary['bp_first_endpoint_shift']={'hypothesis':'H2bp','fresh_inputs':4096,'new_scientific_jobs':48,'new_scientific_evaluations':36864,
        'primary_accuracy_pair':0,'fixed_model_seeds':[0,1,2],'joint_success':a['joint_success'],'model_success':a['model_success'],
        'primary_endpoints':a['primary_endpoints'],'cost_endpoint':a['cost_endpoint'],'failure_accounting':a['failure_accounting'],'distance_strata':json.loads((PUBLIC/'distance-strata.json').read_text())['bins'],
        'scope':'Independent4096-maze uniform-endpoint shift with unchanged BP-first geometry policy; three logical arms, fixed models. Pair0 is the predetermined accuracy copy. Source gates,warmups,LP internal prefixes and repeated timing pairs add no scientific counts.'}
    for name,value in {**a['primary_endpoints'],'complete_answer_cost_saving10':a['cost_endpoint']}.items():
        summary['hypotheses'].append({'campaign':'bp_first_endpoint_shift','hypothesis':'H2bp_'+name,'outcome':value})
    for key,increment in [('jobs',48),('primary_jobs',48),('run_arm_rows',144),('model_maze_evaluations',36864),('primary_model_maze_evaluations',36864)]:summary[key]+=increment
    p=json.loads((PUBLIC/'protocol.json').read_text());summary['protocols']['bp-first-endpoint-shift-protocol']={'sha256':S.sha(PUBLIC/'protocol.json'),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bp','status':'completed'}
    summary['status']='Ongoing research; H2bp uniform-endpoint shift independently audited'
    summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc()
    S.save(root/'bp-first-endpoint-shift-summary.json',a);S.save(root/'summary.json',summary)
    S.save(root/'manifest.json',{'source_sha256':S.sha(Path(__file__)),'files':{str(q.relative_to(root)):S.sha(q) for q in sorted(root.rglob('*')) if q.is_file() and q!=root/'manifest.json'}})


if __name__=='__main__':publish() if '--publish' in sys.argv else evidence()
