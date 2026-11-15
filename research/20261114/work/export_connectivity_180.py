"""Stage H2am public evidence; finalize shared index only after publication release."""
import argparse
import json
import shutil
import study as S
from connectivity_transfer_180 import PP,DEST


def stage():
    public=S.ROOT/'results/research/work';out=public/'connectivity-transfer-180';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());full=json.loads((DEST/'independent-trace-audit.json').read_text());assert a['status']=='validated' and full['status']=='passed' and full['all_final_states']==49152;S.save(public/'connectivity-transfer-180-summary.json',a)
    for name in ['runs.csv','per-model-input.csv','per-input.csv']:shutil.copyfile(DEST/name,public/f'connectivity-transfer-180-{name}')
    for name in ['protocol.json','numerical-gate.json','independent-input-audit.json','independent-trace-audit.json','model-extraction.json','eval-worker0.json','eval-worker1.json','independent-state-verification.json','state-import-isolation.json','attribution-audit.json']:shutil.copyfile(DEST/name,out/name)
    for path in DEST.glob('model-seed*-*.npz'):shutil.copyfile(path,out/path.name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)


def finalize():
    public=S.ROOT/'results/research/work';a=json.loads((DEST/'audit.json').read_text());summary=json.loads((public/'summary.json').read_text());assert 'connectivity_transfer_180' not in summary;summary['jobs']+=48;summary['primary_jobs']+=48;summary['model_maze_evaluations']+=49152;summary['primary_model_maze_evaluations']+=49152;summary['run_arm_rows']+=len(a['rows']);summary['run_tables'].append('connectivity-transfer-180-runs.csv');summary['hypotheses'].append({'campaign':'connectivity_transfer_180','hypothesis':'H2am fresh180veto gain>=1pp overbinary and separate NI<=.1ppvsBP','outcome':a['primary_endpoints']});p=json.loads(PP.read_text());summary['protocols']['connectivity-transfer-180-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['connectivity_transfer_180']={'fresh_inputs':4096,'new_model_maze_arm_evaluations':49152,'reused_binary_prefixes':24576,'new_optimizer_updates':0,'joint_success':a['joint_success'],'skip_descriptive':a['skip_descriptive'],'scope':'Explicit symbolicgraphpolicies, fixedoldLPweights, fresh180loop20. Gainoverbinarysupported; BPnoninferiorityfalsified. No isolatedveto benefit:355minimumfreepathlength exceeds256blocksize.','full_independent_final_state_checks':49152};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stage','finalize']);globals()[p.parse_args().command]()
