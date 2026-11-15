"""Publish H2ac inference evidence without re-exporting earlier campaigns."""
import json
import shutil
import study as S
from hard_negative_transfer import PP,DEST


def main():
    out=S.ROOT/'results/research/work';target=out/'hard-negative-transfer';target.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='validated'
    S.save(out/'hard-negative-transfer-summary.json',a)
    for name in ['runs.csv','per-model-input.csv','per-input.csv','structures.csv','landscape.csv']:shutil.copyfile(DEST/name,out/f'hard-negative-transfer-{name}')
    for name in ['cast-gate.json','numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json','independent-input-audit.json','live-inventory.jsonl','structures.json','state-import-isolation.json']:shutil.copyfile(DEST/name,target/name)
    shutil.copyfile(PP,target/'protocol.json')
    for seed in range(3):
        shutil.copyfile(DEST/f'hard-negative-scaled-seed{seed}.npy',target/f'hard-negative-scaled-seed{seed}.npy')
    fixture=target/'state-fixtures';fixture.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixture/path.name)
    for name in ['independent-state-verification.json','diagnosis.json']:shutil.copyfile(DEST/name,out/f'hard-negative-transfer-{name}')
    rechecks=[json.loads((DEST/f'diagnostic-recheck-gpu{g}.json').read_text()) for g in [0,1]];assert all(x['status']=='passed' for x in rechecks)
    S.save(out/'hard-negative-transfer-diagnostic-recheck.json',{'status':'passed','diagnostics_checked':sum(x['model_input_diagnostics'] for x in rechecks),'maximum_error':max(max(r['maximum_errors'].values()) for x in rechecks for r in x['rows']),'new_answer_trajectories':0,'receipts':rechecks})
    for path in (DEST/'public-preview').iterdir():
        if path.is_file():shutil.copyfile(path,out/path.name)
    summary=json.loads((out/'summary.json').read_text());assert 'hard_negative_transfer' not in summary and 'hard_negative_feasibility' in summary
    summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=36864;summary['primary_model_maze_evaluations']+=36864;summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('hard-negative-transfer-runs.csv')
    summary['hypotheses'].append({'campaign':'hard_negative_transfer','hypothesis':'H2ac hard-negative LP joint gain>=10pp versus earlier LP binary and NI<=.1pp versus BP PL','outcome':{'gain':a['gain_primary'],'noninferiority':a['noninferiority_primary'],'conjunction':a['overall_decision']}})
    p=json.loads(PP.read_text());summary['protocols']['hard-negative-transfer-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['hard_negative_transfer']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':36864,'coefficient_source':'H2ab original-training mined negatives only','new_optimizer_updates':0,'label_diagnostics_separate':12288,'source_constraints_verified_positive_after_cast':1699842,'joint_success':a['joint_success'],'scope':'Fixed source fit; unchanged binary decoder, original maximum coefficient norm, no target tuning or graph repair. Native updates and binary sweeps separate.'};summary['updated_utc']=S.utc();summary['status']='Twenty-five planned inference/training studies, selected-case diagnostics, objective/norm certificates and source-only feasibility screens complete; ongoing research continues separately';S.save(out/'summary.json',summary)
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(path.relative_to(out)):S.sha(path) for path in sorted(out.rglob('*')) if path.is_file() and path!=out/'manifest.json'}})
    S.emit('hard_negative_transfer_export',jobs=summary['jobs'],evaluations=summary['model_maze_evaluations'])


if __name__=='__main__':main()
