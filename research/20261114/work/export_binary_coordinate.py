"""Public H2aa export; source-only subsequent studies remain separate."""
import json
import shutil
import study as S
from binary_coordinate_study import PP,DEST


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_lp_relaxation.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_lp_relaxation
    export_lp_relaxation.main();out=S.ROOT/'results/research/work';a=json.loads((DEST/'audit.json').read_text());assert a['status']=='validated';S.save(out/'binary-coordinate-summary.json',a)
    for name in ['runs.csv','per-model-input.csv','per-input.csv','structures.csv','landscape.csv']:shutil.copyfile(DEST/name,out/f'binary-coordinate-{name}')
    target=out/'binary-coordinate';target.mkdir(exist_ok=True)
    for name in ['numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json','independent-input-audit.json','live-inventory.jsonl','source-counter-verification.json','structures.json']:shutil.copyfile(DEST/name,target/name)
    fixtures=target/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in sorted((DEST/'state-fixtures').iterdir()):
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    verified=json.loads((DEST/'independent-state-verification.json').read_text());assert verified['status']=='passed';S.save(out/'binary-coordinate-independent-state-verification.json',verified)
    diagnosis=json.loads((DEST/'diagnosis.json').read_text());assert diagnosis['status']=='complete';S.save(out/'binary-coordinate-diagnosis.json',diagnosis)
    rechecks=[json.loads((DEST/f'diagnostic-recheck-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' for r in rechecks);S.save(out/'binary-coordinate-diagnostic-recheck.json',{'status':'passed','diagnostics_checked':sum(r['model_input_diagnostics'] for r in rechecks),'maximum_error':max(max(x['maximum_errors'].values()) for r in rechecks for x in r['rows']),'new_answer_trajectories':0,'receipts':rechecks})
    summary=json.loads((out/'summary.json').read_text());summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=36864;summary['primary_model_maze_evaluations']+=36864;summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('binary-coordinate-runs.csv');summary['hypotheses'].append({'campaign':'binary_coordinate','hypothesis':'H2aa fourcolorbinarydescent jointgain>=90pp versusLPnative andNI<=.1pp versusBPPL','outcome':{'gain':a['gain_primary'],'noninferiority':a['noninferiority_primary'],'conjunction':a['overall_decision']}});p=json.loads(PP.read_text());summary['protocols']['binary-coordinate-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['binary_coordinate_descent']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':36864,'binary_arithmetic':'float64promotion','sweep_cap':512,'new_training_updates':0,'label_diagnostics_separate':12288,'scope':'Objective-onlydecoder; no continuousprefix/graphrepair, labels evaluation-only. Nativeupdates andbinarysweeps notconflated.'};summary['updated_utc']=S.utc();summary['status']='twenty-four planned inference/trainingstudies, two selected-case interventions, exactobjective/norm certificates and source-onlyscreens complete; ongoingresearch continues separately';S.save(out/'summary.json',summary);manifest(out);S.emit('binary_coordinate_public_export',jobs=summary['jobs'],evaluations=summary['model_maze_evaluations'])


if __name__=='__main__':main()
