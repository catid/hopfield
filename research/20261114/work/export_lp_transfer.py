"""Publish norm-matched LPfreshcohort comparison, with separate objective/quality gates."""
import json
import shutil
import study as S
from lp_transfer import PP,DEST


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_training_min_norm.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_training_min_norm
    export_training_min_norm.main();out=S.ROOT/'results/research/work';a=json.loads((DEST/'audit.json').read_text());assert a['status']=='validated';S.save(out/'lp-transfer-summary.json',a)
    for name in ['runs.csv','per-model-input.csv','per-input.csv']:shutil.copyfile(DEST/name,out/f'lp-transfer-{name}')
    target=out/'lp-transfer';target.mkdir(exist_ok=True)
    for name in ['cast-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json','independent-input-audit.json']:shutil.copyfile(DEST/name,target/name)
    for seed in range(3):
        for suffix in ['pt','npy']:shutil.copyfile(DEST/f'lp-scaled-seed{seed}.{suffix}',target/f'lp-scaled-seed{seed}.{suffix}')
    dynamics=DEST/'dynamics';diagnosis=json.loads((dynamics/'audit.json').read_text());assert diagnosis['status']=='passed'
    published_dynamics=target/'dynamics';published_dynamics.mkdir(exist_ok=True)
    for path in sorted(dynamics.iterdir()):
        if path.is_file() and path.suffix in ['.json','.csv','.npz']:shutil.copyfile(path,published_dynamics/path.name)
    S.save(out/'lp-transfer-dynamics.json',diagnosis)
    rechecks=[json.loads((DEST/f'diagnostic-recheck-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' for r in rechecks);S.save(out/'lp-transfer-diagnostic-recheck.json',{'status':'passed','diagnostics_checked':sum(r['model_input_diagnostics'] for r in rechecks),'maximum_error':max(max(x['maximum_errors'].values()) for r in rechecks for x in r['rows']),'new_answer_trajectories':0,'receipts':rechecks})
    summary=json.loads((out/'summary.json').read_text());summary['jobs']+=a['evaluation_jobs'];summary['primary_jobs']+=a['evaluation_jobs'];summary['model_maze_evaluations']+=a['new_scientific_model_maze_evaluations'];summary['primary_model_maze_evaluations']+=a['new_scientific_model_maze_evaluations'];summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('lp-transfer-runs.csv');summary['hypotheses'].append({'campaign':'lp_transfer','hypothesis':'H2y norm-matchedLP both-negative ranking gain>=90pp and separate solve noninferiority<=0.1pp','outcome':{'ranking':a['ranking_primary'],'accuracy':a['accuracy_primary'],'conjunction':a['overall_decision']}})
    for name in ['lp-transfer','training-feasibility','training-min-norm']:
        path=S.HERE/f'{name}-protocol.json';p=json.loads(path.read_text());summary['protocols'][f'{name}-protocol']={'sha256':S.sha(path),'frozen_utc':p['frozen_utc']}
    summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['lp_norm_matched_transfer']={'new_fresh_inputs':4096,'new_model_maze_arms':36864,'new_optimizer_updates':0,'positive_source_constraints_after_float32_cast':1698987,'post_outcome_diagnostic_source_states':36864,'post_outcome_native_map_probes':73728,'diagnostic_new_primary_trajectories':0,'scope':'LP andBPdifferentobjectives/computebudgets; nativeinference comparison, no equal-computeoptimizerclaim. Saved-state probes do not extend cap or replace primary outcomes.'};summary['updated_utc']=S.utc();summary['status']='twenty-two planned inference/trainingstudies, two selected-case interventions, exactobjective/norm certificates and source-onlyscreens complete; ongoingresearch continues separately';S.save(out/'summary.json',summary);manifest(out);S.emit('lp_transfer_public_export',jobs=summary['jobs'],scientific_evaluations=summary['model_maze_evaluations'])


if __name__=='__main__':main()
