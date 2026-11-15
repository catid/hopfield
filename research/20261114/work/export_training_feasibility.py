"""Publish source-only feasibility witnesses without evaluation-count inflation."""
import json
import shutil
import study as S


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_contrastive.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_contrastive
    export_contrastive.main();source=S.OUT/'training_feasibility';out=S.ROOT/'results/research/work';target=out/'training-feasibility';target.mkdir(exist_ok=True);verified=json.loads((source/'independent-verification.json').read_text());assert verified['status']=='passed'
    for name in ['protocol.json','solver-summary.json','initial-manifest.json','feature-gate.json','failed-import.log','failed-import-summary.json','environment-recovery.json']:shutil.copyfile(source/name,target/name)
    for seed in range(3):
        shutil.copyfile(source/f'initial-seed{seed}.npz',target/f'initial-seed{seed}.npz');dest=target/f'seed{seed}';dest.mkdir(exist_ok=True)
        for name in ['candidate-theta.npy','numerical-solution.npz','result.json']:shutil.copyfile(source/f'seed{seed}'/name,dest/name)
    shutil.copyfile(source/'independent-verification.json',out/'training-feasibility-verification.json');solver=json.loads((source/'solver-summary.json').read_text());recovery=json.loads((source/'environment-recovery.json').read_text());result={'status':'verified_feasible','hypothesis':'H2w','decision':verified['decision'],'solver':solver,'verification':verified,'aggregate_scientific_cpu_wall_seconds':solver['aggregate_wall_seconds']+recovery['failed_import_seconds_charged'],'dependency_recovery':recovery,'new_independent_inputs':0,'new_model_evaluations':0,'new_gradient_updates':0,'scope':verified['scope'],'publication_files':{str(p.relative_to(out)):S.sha(p) for p in target.rglob('*') if p.is_file()}};S.save(out/'training-feasibility-summary.json',result)
    summary=json.loads((out/'summary.json').read_text());summary['training_constraint_feasibility']={'models':3,'constraints_per_model':566329,'total_independently_verified_constraints':1698987,'cpu_feasibility_solves':3,'decision':verified['decision'],'new_inference_evaluations':0,'new_gradient_updates':0,'public_summary':'training-feasibility-summary.json'};summary['hypotheses'].append({'campaign':'training_feasibility','hypothesis':'H2w source-only unit-margin sparse/dense-negative ranking and all local-label flips are jointly feasible in23real parameters','outcome':{'decision':verified['decision'],'models':3,'scope':'Original1000trainingarrays/frozenreadouts only; no generalization or decoderclaim.'}});summary['updated_utc']=S.utc();S.save(out/'summary.json',summary);manifest(out);S.emit('training_feasibility_export',constraints=1698987,evaluation_count_unchanged=summary['model_maze_evaluations'])


if __name__=='__main__':main()
