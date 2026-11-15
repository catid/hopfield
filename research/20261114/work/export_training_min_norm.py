"""Publish certifiedcoordinate-norm intervals, keeping source-only counts separate."""
import json
import shutil
import study as S


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_training_feasibility.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_training_feasibility
    export_training_feasibility.main();source=S.OUT/'training_min_norm';out=S.ROOT/'results/research/work';target=out/'training-min-norm';target.mkdir(exist_ok=True);verified=json.loads((source/'independent-verification.json').read_text());assert verified['status']=='passed'
    for name in ['protocol.json','solver-summary.json']:shutil.copyfile(source/name,target/name)
    for seed in range(3):
        shutil.copyfile(source/f'initial-seed{seed}.npz',target/f'initial-seed{seed}.npz');dest=target/f'seed{seed}';dest.mkdir(exist_ok=True)
        for path in (source/f'seed{seed}').iterdir():
            if path.is_file():shutil.copyfile(path,dest/path.name)
    shutil.copyfile(source/'independent-verification.json',out/'training-min-norm-verification.json');solver=json.loads((source/'solver-summary.json').read_text());result={'status':'certified_coordinate_norm_intervals','study_id':'H2x','decision':verified['decision'],'solver':solver,'verification':verified,'aggregate_scientific_cpu_wall_seconds':solver['aggregate_wall_seconds'],'new_independent_inputs':0,'new_model_evaluations':0,'new_gradient_updates':0,'scope':verified['scope'],'publication_files':{str(p.relative_to(out)):S.sha(p) for p in target.rglob('*') if p.is_file()}};S.save(out/'training-min-norm-summary.json',result)
    summary=json.loads((out/'summary.json').read_text());summary['training_minimum_coordinate_norm']={'models':3,'constraints_per_model':566329,'total_independently_verified_constraints':1698987,'exact_dual_support_rows':37,'cpu_min_norm_solves':3,'decision':verified['decision'],'new_inference_evaluations':0,'new_gradient_updates':0,'public_summary':'training-min-norm-summary.json'};summary['hypotheses'].append({'campaign':'training_min_norm','hypothesis':'H2x bound the minimum coordinateLinf norm for the fixedunit-margin trainingconstraints within1%relativewidth','outcome':{'decision':verified['decision'],'models':3,'scope':'Original1000trainingarrays/frozenreadouts only; coordinate/gauge/margin dependent.'}});summary['updated_utc']=S.utc();S.save(out/'summary.json',summary);manifest(out);S.emit('training_min_norm_export',exact_dual_rows=37,evaluation_count_unchanged=summary['model_maze_evaluations'])


if __name__=='__main__':main()
