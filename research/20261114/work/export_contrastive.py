"""Publish matched answer-only training evidence with explicit denominators."""
import json
import shutil
import study as S
from contrastive_training import PP,DEST


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_bfs_scope.py'),'files':{str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}})


def main():
    import export_bfs_scope
    export_bfs_scope.main();out=S.ROOT/'results/research/work';a=json.loads((DEST/'audit.json').read_text());assert a['status']=='validated'
    S.save(out/'contrastive-training-summary.json',a)
    for name in ['runs.csv','per-model-input.csv','per-input.csv']:shutil.copyfile(DEST/name,out/f'contrastive-training-{name}')
    target=out/'contrastive-training';target.mkdir(exist_ok=True)
    for name in ['training-arrays.npz','training-manifest.json','numerical-gate.json','performance-gate-gpu0.json','performance-gate-gpu1.json','independent-input-audit.json']:
        shutil.copyfile(DEST/name,target/name)
    for seed in range(3):
        initial=target/'initial-models';initial.mkdir(exist_ok=True);shutil.copyfile(S.OUT/f'models/seed{seed}.pt',initial/f'seed{seed}.pt')
        for name in [f'minibatches-seed{seed}.npz',f'train-readout-seed{seed}.npz',f'train-readout-seed{seed}.json']:shutil.copyfile(DEST/name,target/name)
        for arm in ['pl','contrastive']:
            dest=target/f'fits/s{seed}-{arm}';dest.mkdir(parents=True,exist_ok=True)
            for name in ['checkpoint.pt','optimizer.pt','trace.json','result.json']:shutil.copyfile(DEST/f'fits/s{seed}-{arm}'/name,dest/name)
    rechecks=[json.loads((DEST/f'diagnostic-recheck-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' for r in rechecks)
    S.save(out/'contrastive-training-diagnostic-recheck.json',{'status':'passed','diagnostics_checked':sum(r['model_input_diagnostics'] for r in rechecks),'maximum_error':max(max(x['maximum_errors'].values()) for r in rechecks for x in r['rows']),'new_answer_trajectories':0,'receipts':rechecks})
    landscape=json.loads((DEST/'landscape-summary.json').read_text());assert landscape['status']=='validated'
    for name in ['landscape-summary.json','landscape.csv','structures.csv','dense-case.json']:shutil.copyfile(DEST/name,out/f'contrastive-training-{name}')
    summary=json.loads((out/'summary.json').read_text());jobs=a['evaluation_jobs']+a['fit_jobs'];summary['jobs']+=jobs;summary['primary_jobs']+=jobs;summary['model_maze_evaluations']+=a['new_scientific_model_maze_evaluations'];summary['primary_model_maze_evaluations']+=a['new_scientific_model_maze_evaluations'];summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('contrastive-training-runs.csv')
    summary['hypotheses'].append({'campaign':'contrastive_training','hypothesis':'H2v ranking gain>=90pp and separate solve noninferiority<=0.1pp','outcome':{'ranking':a['ranking_primary'],'accuracy':a['accuracy_primary'],'conjunction':a['overall_decision']}});p=json.loads(PP.read_text());summary['protocols']['contrastive-training-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['answer_only_training']={'fit_jobs':6,'updates':2400,'minibatch_presentations_separate':153600,'new_eval_inputs':4096,'new_eval_model_maze_arms':36864,'original_training_arrays':True,'baseline':'Matched BPAdamWcontinuedPL; unchanged original checkpoint control also retained.'};summary['updated_utc']=S.utc();summary['status']='twenty-one planned studies, two selected-case interventions, exhaustive shortest-answer energies, exact objective certificate and BFSscope control complete; ongoing research continues separately';S.save(out/'summary.json',summary);manifest(out);S.emit('contrastive_public_export',jobs=summary['jobs'],scientific_evaluations=summary['model_maze_evaluations'])


if __name__=='__main__':main()
