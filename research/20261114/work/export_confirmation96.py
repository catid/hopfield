"""Publish H2bh with four counted reported arms and separate prefix evidence."""
import csv
import json
import shutil
import study as S
from confirmation96_study import PP,DEST,ARMS,REPORTED


def table(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    public=S.ROOT/'results/research/work';out=public/'confirmation96';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed' and a['all_final_states']==98304 and a['reported_model_maze_arm_evaluations']==49152;assert json.loads((DEST/'statistics-verification.json').read_text())['status']=='passed';assert json.loads((DEST/'independent-state-verification.json').read_text())['final_states']==192;assert json.loads((DEST/'state-import-isolation.json').read_text())['status']=='passed';S.save(public/'confirmation96-summary.json',a)
    for name in ['protocol.json','numerical-gate.json','source-gate-gpu0.json','source-gate-gpu1.json','seed-registry.json','statistics-gate.json','independent-input-audit.json','model-extraction.json','eval-worker0.json','eval-worker1.json','independent-state-verification.json','state-import-isolation.json','live-progress.json','statistics-verification.json','timing-profile.json','timing-accounting.json','failure-verification.json','per-input.csv']:shutil.copyfile(DEST/name,out/name)
    reported=[ARMS[v] for v in REPORTED];runs=[r for r in a['run_rows'] if r['arm'] in reported];prefixes=[r for r in a['run_rows'] if r['arm'] not in reported];assert len(runs)==192 and len(prefixes)==192;table(out/'runs.csv',runs);table(out/'internal-prefix-runs.csv',prefixes);cases=[]
    for r in a['rows']:
        for v in REPORTED:
            for i,success in enumerate(r['success'][v]):cases.append({'input_index':r['job']['shard']*256+i,'model_seed':r['job']['seed'],'arm':ARMS[v],'success':success})
    assert len(cases)==49152;table(out/'per-model-input.csv',cases)
    for path in DEST.glob('model-seed*-*.npz'):shutil.copyfile(path,out/path.name)
    shutil.copytree(DEST/'state-fixtures',out/'state-fixtures',dirs_exist_ok=True);shutil.copytree(DEST/'failure-fixtures',out/'failure-fixtures',dirs_exist_ok=True);fv=json.loads((DEST/'failure-verification.json').read_text());assert fv['status']=='passed' and fv['final_states']==4*(12288-a['model_success']['unified96']);summary=json.loads((public/'summary.json').read_text());assert 'confirmation96' not in summary;summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=49152;summary['primary_model_maze_evaluations']+=49152;summary['run_arm_rows']+=192;summary['run_tables'].append('confirmation96/runs.csv');summary['hypotheses'].append({'campaign':'confirmation96','hypothesis':'H2bh four separate fresh accuracy decisions:96joint>=99.9%,96gain>=0.5ppvs48,96regression<=0.1%vs48,BPnetNI>=-0.1pp; separate measured96chargedoverhead<=10%.','outcome':{**a['primary_endpoints'],'charged_cost':a['cost_endpoint']}});p=json.loads(PP.read_text());summary['protocols']['confirmation96-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bh','status':'completed'};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['confirmation96']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':49152,'new_optimizer_updates':0,'reported_arms':reported,'joint_success':{arm:a['joint_success'][arm] for arm in reported},'primary_endpoints':a['primary_endpoints'],'cost_endpoint':a['cost_endpoint'],'full_independent_internal_and_final_state_checks':98304,'scope':'Fresh4096loop80,3fixedmodels,fourreportedBP/48/64/96policies. Internal binary/veto/16prefixstates retained and independentlyaudited; not additional counted experiments. Four accuracy decisions and costdecision separately labeled;64descriptive. No pooling/sourcegeneralization/familywiseconfidenceclaim.'};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(path.relative_to(public)):S.sha(path) for path in sorted(public.rglob('*')) if path.is_file() and path!=public/'manifest.json'}})


if __name__=='__main__':main()
