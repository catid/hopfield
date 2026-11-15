"""Publish the complete H2bl negative result with zero fresh-count increment."""
import json
import shutil
import study as S
from cross_certificate_screen_study import PP,DEST

PUBLIC=S.ROOT/'results/research/work/cross-certificate-screen'


def evidence():
    PUBLIC.mkdir(exist_ok=True)
    for name in ['protocol.json','audit.json','sealed-margins.json','synthetic-gate.json','gate-gpu0.json','gate-gpu1.json','worker0.json','worker1.json','live-progress.json','per-model-input.csv']:
        shutil.copyfile(DEST/name,PUBLIC/name)
    shutil.copytree(DEST/'public-fixtures',PUBLIC/'public-fixtures',dirs_exist_ok=True)
    rows=[json.loads(q.read_text()) for q in sorted((DEST/'runs').glob('*/result.json'))]
    S.save(PUBLIC/'timing-accounting.json',{'rows':rows,'aggregate_seconds':{k:sum(r[k] for r in rows) for k in ['source_io_seconds','transfer_seconds','margin_seconds','sealing_seconds']},'scope':'Source margin measurement components, not inference latency or a speed endpoint.'})


def publish():
    a=json.loads((PUBLIC/'audit.json').read_text());v=json.loads((PUBLIC/'public-verification.json').read_text());assert a['status']==v['status']=='passed' and v['model_input_scalar_rows']==12288
    root=PUBLIC.parent;S.save(root/'cross-certificate-screen-summary.json',a);s=json.loads((root/'summary.json').read_text());assert 'cross_certificate_screen' not in s
    s['cross_certificate_screen']={'hypothesis':'H2bl','fresh_inputs':0,'new_scientific_evaluations':0,'source_model_inputs':12288,
        'primary_endpoints':a['primary_endpoints'],'control_confusion_matrices':a['control_confusion_matrices'],
        'scope':'All H2bh BP source boards, margins sealed before labels. The dual screen retains all11778 shortest boards but rejects only1/510 graph failures, falsifying90% rejection. No fresh accuracy or certificate-only cascade claim.'}
    for name,r in a['primary_endpoints'].items():s['hypotheses'].append({'campaign':'cross_certificate_screen','hypothesis':'H2bl_'+name,'outcome':r})
    p=json.loads(PP.read_text());s['protocols']['cross-certificate-screen-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bl','status':'completed'}
    s['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');s['updated_utc']=S.utc();S.save(root/'summary.json',s)
    S.save(root/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(q.relative_to(root)):S.sha(q) for q in sorted(root.rglob('*')) if q.is_file() and q!=root/'manifest.json'}})


if __name__=='__main__':
    import sys
    publish() if '--publish' in sys.argv else evidence()
