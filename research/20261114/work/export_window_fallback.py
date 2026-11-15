"""Public bounded-window cost and latency confirmation, with prior pilot separate."""
import json
import shutil
import numpy as np
import study as S


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_selective_fallback.py'),'files':{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}})


def main():
    import export_selective_fallback
    export_selective_fallback.main();out=S.ROOT/'results/research/work';source=S.OUT/'window_fallback';a=json.loads((source/'audit.json').read_text());assert a['status']=='validated'
    S.save(out/'window-fallback-summary.json',{k:v for k,v in a.items() if k not in ['rows','evidence_sha256']}|{'full_local_audit_sha256':S.sha(source/'audit.json')})
    for name in ['runs','per-model-input','per-input','latencies']:shutil.copyfile(source/f'{name}.csv',out/f'window-fallback-{name}.csv')
    for old,new in [('failure-classification','failure-classification'),('scipy-statistics-verification','statistics-verification')]:shutil.copyfile(source/f'{old}.json',out/f'window-fallback-{new}.json')
    classified=json.loads((source/'failure-classification.json').read_text());cycles=[]
    for index in sorted({r['input_index'] for r in classified['rows'] if not r['endpoint_only'] and r['branch_nodes']==0}):
        shard,i=divmod(index,256)
        with np.load(source/f'shard{shard:02d}.npz') as data:tokens=data['tokens'][i];distance=int(data['shortest_edges'][i])
        rows=[r for r in classified['rows'] if r['input_index']==index];verified=[]
        for row in rows:
            board=np.zeros(tokens.size,bool);board[row['selected_flat_indices']]=True;board=board.reshape(tokens.shape);degree=np.zeros_like(tokens,dtype=np.int8);degree[1:]+=board[:-1];degree[:-1]+=board[1:];degree[:,1:]+=board[:,:-1];degree[:,:-1]+=board[:,1:]
            if row['components']==1 and bool(np.all(degree[board]==2)) and bool(np.all(board[tokens>=2])):
                verified.append(row|{'every_selected_degree':2,'endpoint_degrees':degree[tokens>=2].tolist(),'walls_selected':int((board&(tokens==0)).sum())})
        if verified:cycles.append({'input_index':index,'tokens':tokens.tolist(),'shortest_edges_evaluation_only':distance,'rows':verified})
    S.save(out/'window-fallback-cycle-cases.json',{'selection':'Post-outcome descriptive: every wrong selected board with one component, all selected degree2 and both supplied endpoints selected. No new inference or primary decision change.','cycles':cycles,'source_classification_sha256':S.sha(source/'failure-classification.json')})
    pilots=[json.loads((source/f'prior-cohort-pilot-gpu{g}.json').read_text()) for g in [0,1]];pilotrows=[]
    for width in [256,1024,8192]:
        seconds=[sum(x['fallback_seconds_including_gather_init_and_output'] for r in pilots for x in r['rows'] if x['window']==width and x['repetition']==rep) for rep in range(3)];pilotrows.append({'window':width,'fallback_seconds_by_repetition':seconds,'median_fallback_seconds':float(np.median(seconds))})
    S.save(out/'window-fallback-prior-cohort-pilot.json',{'status':'passed','evidence_role':'Prior H2t timing-development replay, excluded from fresh confirmation. Fixed1024 and unchanged2% target chosen before these outputs;256/8192 are explicit development comparators.','rows':pilotrows,'answer_trajectories_separate':sum(r['answer_trajectories'] for r in pilots),'receipts':pilots})
    rechecks=[json.loads((source/f'diagnostic-recheck-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' for r in rechecks)
    S.save(out/'window-fallback-diagnostic-recheck.json',{'status':'passed','diagnostics_checked':sum(r['diagnostics_checked'] for r in rechecks),'maximum_error':max(max(x['maximum_errors'].values()) for r in rechecks for x in r['rows']),'new_answer_trajectories':0,'receipts':rechecks})
    S.save(out/'window-fallback-timing.json',{'scope':'Each fixed1024/cohort-end policy independently executes the entire native prefix in3timing repetitions; all validator/setup/queue/output costs included. Component sums exclude archive/evidence I/O; raw wall delays include intervening evidence work.','repetitions':a['timing'],'latency_summary':a['latency_summary'],'source_receipts':[json.loads((source/f'timing-gpu{g}.json').read_text()) for g in [0,1]],'timing_repeat_trajectories':a['timing_repeat_trajectories']})
    summary=json.loads((out/'summary.json').read_text());jobs=a['main_jobs']+a['fallback_jobs'];summary['jobs']+=jobs;summary['primary_jobs']+=jobs;summary['model_maze_evaluations']+=a['new_scientific_trajectories'];summary['primary_model_maze_evaluations']+=a['new_scientific_trajectories'];summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('window-fallback-runs.csv');summary['window_fallback_reuse']={'exact_native_output_reuses':a['exact_native_output_reuses'],'logical_scientific_outcomes':a['logical_scientific_outcomes'],'new_scientific_trajectories':a['new_scientific_trajectories'],'timing_repeat_trajectories_separate':a['timing_repeat_trajectories'],'prior_cohort_pilot_trajectories_separate':sum(r['answer_trajectories'] for r in pilots)}
    summary['hypotheses'].append({'campaign':'window_fallback','hypothesis':'H2u fixed1024 graph-informed fallback gains>=0.05pp with<=2% measured answer overhead','outcome':a['primary']});pp=S.HERE/'window_fallback-protocol.json';p=json.loads(pp.read_text());summary['protocols']['window_fallback-protocol']={'sha256':S.sha(pp),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((source/'audit.json').relative_to(S.ROOT))]=S.sha(source/'audit.json');summary['updated_utc']=S.utc();summary['status']='twenty planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately'
    S.save(out/'summary.json',summary);manifest(out);S.emit('window_public_export',jobs=summary['jobs'],scientific_evaluations=summary['model_maze_evaluations'],separate_timing_trajectories=a['timing_repeat_trajectories'])


if __name__=='__main__':main()
