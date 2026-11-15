"""Publish fully audited fresh H2ay transfer and update counted study totals."""
import csv
import json
import shutil
import study as S
from density_transfer import PP,DEST,ARMS


def main():
    audit_source=S.OUT/'local_factor_audit';public=S.ROOT/'results/research/work';out=public/'density-transfer';out.mkdir(exist_ok=True);a=json.loads((audit_source/'audit.json').read_text());assert a['status']=='passed' and a['all_final_states']==73728;assert json.loads((audit_source/'density-statistics-verification.json').read_text())['status']=='passed';assert json.loads((DEST/'independent-state-verification.json').read_text())['final_states']==288;S.save(public/'density-transfer-summary.json',a)
    for name in ['runs.csv','per-input.csv']:shutil.copyfile(audit_source/name,public/f'density-transfer-{name}')
    cases=[]
    for r in a['rows']:
        for v,arm in enumerate(ARMS):
            for i,success in enumerate(r['success'][v]):cases.append({'input_index':r['job']['shard']*256+i,'model_seed':r['job']['seed'],'arm':arm,'success':success})
    with (public/'density-transfer-per-model-input.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    for name in ['protocol.json','numerical-gate.json','seed-registry.json','independent-input-audit.json','model-extraction.json','eval-worker0.json','eval-worker1.json','independent-state-verification.json','state-import-isolation.json','live-progress.json','timing-progress.json','audit-progress.json','audit-budget-exhausted.json','audit-profile.json','audit-profile.txt','residual-cap-diagnosis.json']:shutil.copyfile(DEST/name,out/name)
    shutil.copyfile(audit_source/'density-statistics-verification.json',out/'statistics-verification.json')
    shutil.copyfile(S.OUT/'density_transfer_preflight/numerical-gate.json',out/'density-generator-preflight.json')
    for path in DEST.glob('model-seed*-*.npz'):shutil.copyfile(path,out/path.name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    summary=json.loads((public/'summary.json').read_text());assert 'density_transfer' not in summary;summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=73728;summary['primary_model_maze_evaluations']+=73728;summary['run_arm_rows']+=len(a['run_rows']);summary['run_tables'].append('density-transfer-runs.csv');summary['hypotheses'].append({'campaign':'density_transfer','hypothesis':'H2ay lexical512 netgain>=.2pp vs lexical256, lossprobability<=.1pp, netNI>=-.1pp vs BP; separately','outcome':a['primary_endpoints']});p=json.loads(PP.read_text());summary['protocols']['density-transfer-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((audit_source/'audit.json').relative_to(S.ROOT))]=S.sha(audit_source/'audit.json');summary['density_transfer']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':73728,'new_optimizer_updates':0,'verification':'Original300s H2ay audit timed out; all outputs later checked under separately frozen H2az600s local-factor verification, with no inference rerun.','joint_success':a['joint_success'],'primary_endpoints':a['primary_endpoints'],'energy_vs_lexical_descriptive':a['energy_vs_lexical_descriptive'],'full_independent_final_state_checks':73728,'scope':'Fresh180loop50, threefixedmodels,sixfixedpolicies. Explicitenergy-increasingcyclegraphprior. Three primary decisions reported separately. Both512 continuations start from lexical256; energy-ranking comparison descriptive. Finitecohortresult,notuniversalguarantee.'};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
