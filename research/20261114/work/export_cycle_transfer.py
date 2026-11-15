"""Publish fully audited fresh H2au transfer and update counted study totals."""
import csv
import json
import shutil
import study as S
from cycle_transfer import PP,DEST,ARMS


def main():
    public=S.ROOT/'results/research/work';out=public/'cycle-transfer';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed' and a['all_final_states']==122880;assert json.loads((DEST/'statistics-verification.json').read_text())['status']=='passed';assert json.loads((DEST/'independent-state-verification.json').read_text())['final_states']==480;S.save(public/'cycle-transfer-summary.json',a)
    for name in ['runs.csv','per-input.csv']:shutil.copyfile(DEST/name,public/f'cycle-transfer-{name}')
    cases=[]
    for r in a['rows']:
        for v,arm in enumerate(ARMS):
            for i,success in enumerate(r['success'][v]):cases.append({'input_index':r['job']['shard']*256+i,'model_seed':r['job']['seed'],'arm':arm,'success':success})
    with (public/'cycle-transfer-per-model-input.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    for name in ['protocol.json','numerical-gate.json','seed-registry.json','independent-input-audit.json','model-extraction.json','eval-worker0.json','eval-worker1.json','statistics-verification.json','independent-state-verification.json','state-import-isolation.json']:shutil.copyfile(DEST/name,out/name)
    for path in DEST.glob('model-seed*-*.npz'):shutil.copyfile(path,out/path.name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    summary=json.loads((public/'summary.json').read_text());assert 'cycle_transfer' not in summary;summary['jobs']+=96;summary['primary_jobs']+=96;summary['model_maze_evaluations']+=122880;summary['primary_model_maze_evaluations']+=122880;summary['run_arm_rows']+=len(a['run_rows']);summary['run_tables'].append('cycle-transfer-runs.csv');summary['hypotheses'].append({'campaign':'cycle_transfer','hypothesis':'H2au lexicalnetgain>=.2ppvsvetoandnetNIwithin.1ppvsBP,separately','outcome':a['primary_endpoints']});p=json.loads(PP.read_text());summary['protocols']['cycle-transfer-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['cycle_transfer']={'fresh_inputs':8192,'new_model_maze_arm_evaluations':122880,'new_optimizer_updates':0,'joint_success':a['joint_success'],'primary_endpoints':a['primary_endpoints'],'energy_vs_lexical_descriptive':a['energy_vs_lexical_descriptive'],'full_independent_final_state_checks':122880,'scope':'Fresh120loop20, threefixedmodels,fivefixedpolicies. Explicitenergy-increasingcyclegraphprior. LexicalgainandnetNIseparatelysupported; nolearnedrankingbenefitclaim. 8192perfectjointsuccessesarefinitecohortresult,notuniversalguarantee.'};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
