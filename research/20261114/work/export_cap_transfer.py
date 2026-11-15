"""Publish fully audited fresh H2ax transfer and update counted study totals."""
import csv
import json
import shutil
import study as S
from cap_transfer import PP,DEST,ARMS


def main():
    public=S.ROOT/'results/research/work';out=public/'cap-transfer';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed' and a['all_final_states']==73728;assert json.loads((DEST/'statistics-verification.json').read_text())['status']=='passed';assert json.loads((DEST/'independent-state-verification.json').read_text())['final_states']==288;S.save(public/'cap-transfer-summary.json',a)
    for name in ['runs.csv','per-input.csv']:shutil.copyfile(DEST/name,public/f'cap-transfer-{name}')
    cases=[]
    for r in a['rows']:
        for v,arm in enumerate(ARMS):
            for i,success in enumerate(r['success'][v]):cases.append({'input_index':r['job']['shard']*256+i,'model_seed':r['job']['seed'],'arm':arm,'success':success})
    with (public/'cap-transfer-per-model-input.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    for name in ['protocol.json','numerical-gate.json','seed-registry.json','independent-input-audit.json','model-extraction.json','eval-worker0.json','eval-worker1.json','statistics-verification.json','independent-state-verification.json','state-import-isolation.json','live-progress.json']:shutil.copyfile(DEST/name,out/name)
    for path in DEST.glob('model-seed*-*.npz'):shutil.copyfile(path,out/path.name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    summary=json.loads((public/'summary.json').read_text());assert 'cap_transfer' not in summary;summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=73728;summary['primary_model_maze_evaluations']+=73728;summary['run_arm_rows']+=len(a['run_rows']);summary['run_tables'].append('cap-transfer-runs.csv');summary['hypotheses'].append({'campaign':'cap_transfer','hypothesis':'H2ax lexical512 netgain>=.2pp vs lexical256, lossprobability<=.1pp, netNI>=-.1pp vs BP; separately','outcome':a['primary_endpoints']});p=json.loads(PP.read_text());summary['protocols']['cap-transfer-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['cap_transfer']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':73728,'new_optimizer_updates':0,'joint_success':a['joint_success'],'primary_endpoints':a['primary_endpoints'],'energy_vs_lexical_descriptive':a['energy_vs_lexical_descriptive'],'full_independent_final_state_checks':73728,'scope':'Fresh180loop20, threefixedmodels,sixfixedpolicies. Explicitenergy-increasingcyclegraphprior. Three primary decisions reported separately. Both512 continuations start from lexical256; energy-ranking comparison descriptive. Finitecohortresult,notuniversalguarantee.'};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
