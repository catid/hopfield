"""Public export of fixed-lambda H2z outcomes, checks and provenance."""
import json
import shutil
import study as S
from lp_relaxation import PP,DEST


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_lp_transfer.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_lp_transfer
    export_lp_transfer.main();out=S.ROOT/'results/research/work';a=json.loads((DEST/'audit.json').read_text());assert a['status']=='validated'
    S.save(out/'lp-relaxation-summary.json',a)
    for name in ['runs.csv','per-model-input.csv','per-input.csv']:shutil.copyfile(DEST/name,out/f'lp-relaxation-{name}')
    target=out/'lp-relaxation';target.mkdir(exist_ok=True)
    for name in ['numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json','independent-input-audit.json','live-inventory.jsonl','output-identities.json','scalar-counterexample.json']:shutil.copyfile(DEST/name,target/name)
    fixtures=target/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in sorted((DEST/'state-fixtures').iterdir()):
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    direct=json.loads((DEST/'independent-state-verification.json').read_text());assert direct['status']=='passed'
    S.save(out/'lp-relaxation-independent-state-verification.json',direct)
    rechecks=[json.loads((DEST/f'diagnostic-recheck-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' for r in rechecks)
    S.save(out/'lp-relaxation-diagnostic-recheck.json',{'status':'passed','diagnostics_checked':sum(r['model_input_diagnostics'] for r in rechecks),'maximum_error':max(max(x['maximum_errors'].values()) for r in rechecks for x in r['rows']),'new_answer_trajectories':0,'receipts':rechecks})
    summary=json.loads((out/'summary.json').read_text());summary['jobs']+=a['evaluation_jobs'];summary['primary_jobs']+=a['evaluation_jobs'];summary['model_maze_evaluations']+=a['new_scientific_model_maze_evaluations'];summary['primary_model_maze_evaluations']+=a['new_scientific_model_maze_evaluations'];summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('lp-relaxation-runs.csv');summary['hypotheses'].append({'campaign':'lp_relaxation','hypothesis':'H2z fixedbinary-preservinglambda4: jointsolve gain>=90pp versusLPnative andNI<=.1pp versusBPPL','outcome':{'gain':a['gain_primary'],'noninferiority':a['noninferiority_primary'],'conjunction':a['overall_decision']}})
    p=json.loads(PP.read_text());summary['protocols']['lp-relaxation-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['lp_binary_preserving_relaxation']={'new_fresh_inputs':4096,'new_model_maze_arms':36864,'new_optimizer_updates':0,'lambda_fixed':4.,'source_gate_trajectories_separate':384,'scope':'Frozenbinaryenergy withmodifiedcontinuousextension; nativecontrols, no graphfallback orretuning. DifferentLP/BPtrainingobjectives andbudgets explicitlyseparate.'};summary['updated_utc']=S.utc();summary['status']='twenty-three planned inference/trainingstudies, two selected-case interventions, exactobjective/norm certificates and source-onlyscreens complete; ongoingresearch continues separately';S.save(out/'summary.json',summary);manifest(out);S.emit('lp_relaxation_public_export',jobs=summary['jobs'],scientific_evaluations=summary['model_maze_evaluations'])


if __name__=='__main__':main()
