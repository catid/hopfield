"""Publish H2bj without incrementing scientific totals."""
import json
import shutil
import study as S
from certificate_fallback_study import PP, DEST

PUBLIC=S.ROOT/'results/research/work/certificate-fallback'


def evidence():
    PUBLIC.mkdir(exist_ok=True)
    for name in ['protocol.json','audit.json','routing.json','synthetic-gate.json','worker0.json','worker1.json',
                 'gate-worker0.json','gate-worker1.json','live-progress.json','per-input.csv','per-model-input.csv']:
        shutil.copyfile(DEST/name,PUBLIC/name)
    shutil.copytree(DEST/'public-fixtures',PUBLIC/'public-fixtures',dirs_exist_ok=True)


def publish():
    a=json.loads((PUBLIC/'audit.json').read_text());v=json.loads((PUBLIC/'public-verification.json').read_text())
    assert a['status']==v['status']=='passed' and v['model_input_routes']==12288
    root=PUBLIC.parent;S.save(root/'certificate-fallback-summary.json',a)
    s=json.loads((root/'summary.json').read_text());assert 'certificate_fallback' not in s
    s['certificate_fallback']={'hypothesis':'H2bj','fresh_inputs':0,'new_scientific_evaluations':0,
        'execution_decision':a['execution_decision'],'cross_certificate_decision':a['cross_certificate_decision'],
        'joint_success':a['joint_success'],'model_success':a['model_success'],
        'accepted_lp_model_inputs':12285,'actual_routed_bp_model_inputs':3,
        'scope':'Complete archived H2bh cohort. The BP rescue was known; actual subset execution is verified. New finite diagnostic finds all three BP outputs satisfy the original LP local certificate. No graph/global proof follows from that certificate. Zero fresh evidence; H2bh absolute99.9% remains inconclusive.'}
    for name in ['execution','cross_certificate']:
        s['hypotheses'].append({'campaign':'certificate_fallback','hypothesis':'H2bj_'+name,'outcome':a[name+'_decision']})
    p=json.loads(PP.read_text());s['protocols']['certificate-fallback-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bj','status':'completed'}
    s['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');s['updated_utc']=S.utc();S.save(root/'summary.json',s)
    S.save(root/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(f.relative_to(root)):S.sha(f) for f in sorted(root.rglob('*')) if f.is_file() and f!=root/'manifest.json'}})


if __name__=='__main__':
    import sys
    publish() if '--publish' in sys.argv else evidence()
