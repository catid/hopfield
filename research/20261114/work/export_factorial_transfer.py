"""Publish H2aj outcomes, bounded timing, provenance and independent audits."""
import json
import shutil
import study as S
from factorial_transfer import PP,DEST


def main():
    out=S.ROOT/'results/research/work';target=out/'factorial-transfer';target.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='validated';full=json.loads((DEST/'independent-trace-audit.json').read_text());assert full['status']=='passed' and full['all_final_states']==61440
    S.save(out/'factorial-transfer-summary.json',a)
    for name in ['runs.csv','per-model-input.csv','per-input.csv']:shutil.copyfile(DEST/name,out/f'factorial-transfer-{name}')
    for name in ['numerical-gate.json','independent-input-audit.json','independent-trace-audit.json','model-extraction.json','eval-worker0.json','eval-worker1.json','independent-state-verification.json','failure-diagnosis.json','state-import-isolation.json']:shutil.copyfile(DEST/name,target/name)
    shutil.copyfile(PP,target/'protocol.json');shutil.copyfile(S.OUT/'archive_benchmark/result.json',target/'archive-benchmark.json')
    for path in DEST.glob('model-seed*-*.npz'):shutil.copyfile(path,target/path.name)
    fixtures=target/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    summary=json.loads((out/'summary.json').read_text());assert 'factorial_transfer' not in summary
    summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=61440;summary['primary_model_maze_evaluations']+=61440;summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('factorial-transfer-runs.csv');summary['hypotheses'].append({'campaign':'factorial_transfer','hypothesis':'H2aj independent source-fit by block-search factorial','outcome':a['primary_endpoints']});p=json.loads(PP.read_text());summary['protocols']['factorial-transfer-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['factorial_transfer']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':61440,'reused_binary_prefixes':24576,'new_optimizer_updates':0,'joint_success':a['joint_success'],'interaction_pp_descriptive':a['interaction_pp_descriptive'],'source_fit':'Fixed H2ag witnesses via H2ah cast; older H2ae coefficients as control.','full_independent_final_state_checks':61440,'full_independent_literal_candidate_checks':13768,'scope':'Five arms; shared hidden and exact binary prefixes. Three separateendpoint decisions, no opaqueconjunction or simultaneous95claim.'};summary['updated_utc']=S.utc();summary['status']='Ongoing research; H2aj prospective factorial complete with independent state and trace audits';S.save(out/'summary.json',summary)
    S.save(out/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(path.relative_to(out)):S.sha(path) for path in sorted(out.rglob('*')) if path.is_file() and path!=out/'manifest.json'}});S.emit('factorial_export',jobs=summary['jobs'],evaluations=summary['model_maze_evaluations'])


if __name__=='__main__':main()
