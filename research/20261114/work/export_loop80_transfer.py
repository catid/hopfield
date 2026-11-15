"""Publish the fully audited fresh H2be confirmation and counted study totals."""
import csv
import json
import shutil
import study as S
from loop80_transfer import PP,DEST,ARMS


def main():
    public=S.ROOT/'results/research/work';out=public/'loop80-transfer';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed' and a['all_final_states']==73728;assert json.loads((DEST/'statistics-verification.json').read_text())['status']=='passed';assert json.loads((DEST/'independent-state-verification.json').read_text())['final_states']==288;assert json.loads((DEST/'state-import-isolation.json').read_text())['status']=='passed';S.save(public/'loop80-transfer-summary.json',a)
    for name in ['protocol.json','numerical-gate.json','seed-registry.json','independent-input-audit.json','model-extraction.json','eval-worker0.json','eval-worker1.json','independent-state-verification.json','state-import-isolation.json','live-progress.json','statistics-verification.json','timing-profile.json','per-input.csv','runs.csv']:shutil.copyfile(DEST/name,out/name)
    shutil.copyfile(S.OUT/'loop80_transfer_preflight/absolute-statistics-gate.json',out/'absolute-statistics-gate.json')
    cases=[]
    for r in a['rows']:
        for v,arm in enumerate(ARMS):
            for i,success in enumerate(r['success'][v]):cases.append({'input_index':r['job']['shard']*256+i,'model_seed':r['job']['seed'],'arm':arm,'success':success})
    with (out/'per-model-input.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    for path in DEST.glob('model-seed*-*.npz'):shutil.copyfile(path,out/path.name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    summary=json.loads((public/'summary.json').read_text());assert 'loop80_transfer' not in summary;summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=73728;summary['primary_model_maze_evaluations']+=73728;summary['run_arm_rows']+=len(a['run_rows']);summary['run_tables'].append('loop80-transfer/runs.csv');summary['hypotheses'].append({'campaign':'loop80_transfer','hypothesis':'H2be: unified48 gain >=1 pp versus original16, regression probability <=0.1%, net BP NI >=-0.1 pp, extra-budget gain >=0.5 pp versus unified16, absolute joint success >=99.9%; separate decisions.','outcome':a['primary_endpoints']});p=json.loads(PP.read_text());summary['protocols']['loop80-transfer-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2be','status':'completed'};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['loop80_transfer']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':73728,'new_optimizer_updates':0,'joint_success':a['joint_success'],'primary_endpoints':a['primary_endpoints'],'full_independent_final_state_checks':73728,'scope':'Fresh 180x180 loop80 mazes, three fixed models and six fixed policies. Native-stable veto cap eligibility and terminal certification separated from total16 versus48 cycle rounds. Five endpoint decisions are separate; no pooling with source repairs or simultaneous95 claim.'};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
