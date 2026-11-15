"""Export graph-informed selective-decoder results, including all timing costs."""
import json
import shutil
import study as S


def manifest(out):
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'base_export_sha256':S.sha(S.HERE/'export_larger_relaxation.py'),'files':{p.name:S.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}})


def main():
    import export_larger_relaxation
    export_larger_relaxation.main();out=S.ROOT/'results/research/work';source=S.OUT/'selective_fallback';a=json.loads((source/'audit.json').read_text());assert a['status']=='validated'
    S.save(out/'selective-fallback-summary.json',{k:v for k,v in a.items() if k not in ['rows','evidence_sha256']}|{'full_local_audit_sha256':S.sha(source/'audit.json')})
    for name in ['runs','per-model-input','per-input']:shutil.copyfile(source/f'{name}.csv',out/f'selective-fallback-{name}.csv')
    for old,new in [('failure-classification','failure-classification'),('scipy-statistics-verification','statistics-verification'),('validator-gate','validator-gate'),('validator-benchmark','validator-benchmark')]:shutil.copyfile(source/f'{old}.json',out/f'selective-fallback-{new}.json')
    rechecks=[json.loads((source/f'diagnostic-recheck-gpu{g}.json').read_text()) for g in [0,1]];assert all(r['status']=='passed' for r in rechecks)
    S.save(out/'selective-fallback-diagnostic-recheck.json',{'status':'passed','diagnostics_checked':sum(r['diagnostics_checked'] for r in rechecks),'maximum_error':max(max(x['maximum_errors'].values()) for r in rechecks for x in r['rows']),'new_answer_trajectories':0,'receipts':rechecks})
    S.save(out/'selective-fallback-timing.json',{'scope':'Offline graph-informed decoder with deferred fallback batching. All validator/setup/queue/output costs included; component-summed decoder timings exclude data loading and evidence serialization. No online per-input latency claim.','repetitions':a['timing'],'source_receipts':[json.loads((source/f'timing-gpu{g}.json').read_text()) for g in [0,1]],'timing_repeat_trajectories':a['timing_repeat_trajectories']})
    summary=json.loads((out/'summary.json').read_text());jobs=a['main_jobs']+a['fallback_jobs'];summary['jobs']+=jobs;summary['primary_jobs']+=jobs;summary['model_maze_evaluations']+=a['new_scientific_trajectories'];summary['primary_model_maze_evaluations']+=a['new_scientific_trajectories'];summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('selective-fallback-runs.csv')
    summary['selective_fallback_reuse']={'exact_native_output_reuses':a['exact_native_output_reuses'],'logical_scientific_outcomes':a['logical_scientific_outcomes'],'new_scientific_trajectories':a['new_scientific_trajectories'],'timing_repeat_trajectories_separate':a['timing_repeat_trajectories']}
    summary['hypotheses'].append({'campaign':'selective_fallback','hypothesis':'H2t graph-informed fallback gains >=0.05pp with <=2% measured answer overhead','outcome':a['primary']});pp=S.HERE/'selective_fallback-protocol.json';p=json.loads(pp.read_text());summary['protocols']['selective_fallback-protocol']={'sha256':S.sha(pp),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((source/'audit.json').relative_to(S.ROOT))]=S.sha(source/'audit.json');summary['updated_utc']=S.utc();summary['status']='nineteen planned studies, two selected-case interventions, and exhaustive shortest-answer energies complete; ongoing research continues separately'
    S.save(out/'summary.json',summary);manifest(out);S.emit('selective_public_export',jobs=summary['jobs'],scientific_evaluations=summary['model_maze_evaluations'],separate_timing_trajectories=a['timing_repeat_trajectories'])


if __name__=='__main__':main()
