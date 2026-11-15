"""Public H2bk evidence and separately authorized summary/manifest update."""
import csv
import json
import shutil
import study as S
from fallback_confirmation_study import PP,DEST,REPORTED

PUBLIC=S.ROOT/'results/research/work/fallback-confirmation'


def write_table(path,rows):
    with path.open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evidence():
    PUBLIC.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed'
    for name in ['protocol.json','audit.json','numerical-gate.json','source-gate-gpu0.json','source-gate-gpu1.json',
                 'seed-registry.json','statistics-gate.json','independent-input-audit.json','datasets.json','model-extraction.json',
                 'eval-worker0.json','eval-worker1.json','live-gates.json','live-evaluate.json','per-input.csv','per-model-input.csv']:
        shutil.copyfile(DEST/name,PUBLIC/name)
    for name in ['state-fixtures','routed-failure-fixtures']:shutil.copytree(DEST/name,PUBLIC/name,dirs_exist_ok=True)
    for q in DEST.glob('model-seed*-*.npz'):shutil.copyfile(q,PUBLIC/q.name)
    extra={w['jobs_complete'][0]:w['additional_bp_load_seconds_charged_once'] for w in a['worker_receipts']};runs=[];prefixes=[];timing=[]
    for row in a['rows']:
        job=row['job'];r=json.loads((DEST/'runs'/job['id']/'result.json').read_text())
        timing.append({'job':job,'receipt_sha256':S.sha(DEST/'runs'/job['id']/'result.json'),
            'prefix_batch_timings':r['batch_timings'],'hybrid_batch_timings':r['hybrid_batch_timings'],
            'input_transfer_seconds':r['input_transfer_seconds'],'hidden_seconds':r['hidden_seconds_shared'],
            'diagnostic_seconds':r['diagnostic_seconds'],'archive_seconds':r['archive_seconds'],
            'archive_writer_metrics':r['archive_writer_metrics'],'job_seconds':r['job_seconds'],
            'bp_load_charge_here':extra.get(job['id'],0.)})
        for v,arm in enumerate(REPORTED):
            cost=row['prefix']['outcomes'][0 if v==0 else 7]['answer_seconds'] if v<2 else row['main_charged_seconds' if v==2 else 'strict_charged_seconds']+extra.get(job['id'],0.)
            runs.append({'job':job['id'],'seed':job['seed'],'shard':job['shard'],'arm':arm,'n':256,
                'solved':sum(row['success'][v]),'charged_seconds':cost,'bp_load_charge_here':extra.get(job['id'],0.) if v>=2 else 0.,
                'actual_subset_inputs':len(row['routed']) if v>=2 else 0,'accepted_lp_reuses':row['accepted_lp'] if v>=2 else 0})
        prefixes.extend(r for r in row['prefix']['outcomes'] if r['arm'] not in ['pl','unified96'])
    write_table(PUBLIC/'runs.csv',runs);write_table(PUBLIC/'internal-prefix-runs.csv',prefixes)
    S.save(PUBLIC/'timing-accounting.json',{'status':'passed','protocol_sha256':S.sha(PP),'rows':timing,
        'worker_model_loads':[{k:w[k] for k in ['gpu','model_load_seconds_by_seed','additional_bp_load_seconds_charged_once']} for w in a['worker_receipts']],
        'cost_endpoint':a['cost_endpoint'],'scope':'Full prefix and incremental component receipts; required extra BP model load charged once per GPU to each hybrid arm. Serialization/diagnostics outside charged answer cost, inside bounded worker wall time.'})


def publish():
    a=json.loads((PUBLIC/'audit.json').read_text());stats=json.loads((PUBLIC/'statistics-verification.json').read_text());fixed=json.loads((PUBLIC/'state-verification.json').read_text());supp=json.loads((PUBLIC/'routed-failure-verification.json').read_text())
    assert a['status']==stats['status']==fixed['status']==supp['status']=='passed'
    assert fixed['fixtures']==48 and fixed['reported_state_checks']==192
    assert supp['routed_model_cases']==a['actual_fallback_trajectories']
    failures={(r['job']['shard']*256+i,r['job']['seed']) for r in a['rows'] for i in range(256) if not all(r['success'][v][i] for v in range(4))}
    routed={(r['input_index'],r['model_seed']) for r in a['all_routed_cases']}
    published=json.loads((PUBLIC/'routed-failure-fixtures/manifest.json').read_text())['fixtures']
    assert {(r['input_index'],r['model_seed']) for r in published}==failures|routed
    root=PUBLIC.parent;S.save(root/'fallback-confirmation-summary.json',a);s=json.loads((root/'summary.json').read_text());assert 'fallback_confirmation' not in s
    for key in ['jobs','primary_jobs']:s[key]+=48
    for key in ['model_maze_evaluations','primary_model_maze_evaluations']:s[key]+=49152
    s['run_arm_rows']+=192;s['run_tables'].append('fallback-confirmation/runs.csv')
    s['hypotheses'].append({'campaign':'fallback_confirmation','hypothesis':'H2bk separate main/dual absolute99.9%, BP net NI-0.1pp and measured charged overhead<=10% endpoints.','outcome':{**a['primary_endpoints'],'charged_cost':a['cost_endpoint']}})
    p=json.loads(PP.read_text());s['protocols']['fallback-confirmation-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bk','status':'completed'}
    s['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json')
    s['fallback_confirmation']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':49152,'reported_arms':list(REPORTED),
        'joint_success':a['joint_success'],'model_success':a['model_success'],'primary_endpoints':a['primary_endpoints'],'cost_endpoint':a['cost_endpoint'],
        'actual_fallback_trajectories':a['actual_fallback_trajectories'],'accepted_lp_model_inputs_reused':a['accepted_lp_model_inputs_reused'],
        'full_internal_and_reported_state_checks':122880,
        'scope':'Fresh cohort only, three fixed models joined per maze. Main and dual certificates separately reported; actual BP subset execution, full wrong-certified/failed output inclusion and independently audited original prefixes. No source pooling or retrospective upgrade to H2bh.'}
    s['updated_utc']=S.utc();S.save(root/'summary.json',s);S.save(root/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(q.relative_to(root)):S.sha(q) for q in sorted(root.rglob('*')) if q.is_file() and q!=root/'manifest.json'}})


if __name__=='__main__':
    import sys
    publish() if '--publish' in sys.argv else evidence()
