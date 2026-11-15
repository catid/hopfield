"""Publish H2ak source screen and retain H2al failed size-invariance gate."""
import json
import shutil
import study as S
from connectivity_screen import PP,DEST


def main():
    public=S.ROOT/'results/research/work';out=public/'connectivity-screen';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';S.save(public/'connectivity-screen-summary.json',a)
    for name in ['protocol.json','numerical-gate.json','skip-control.json','audit.json','per-input.csv','worker0.json','worker1.json','independent-state-verification.json','state-import-isolation.json']:shutil.copyfile(DEST/name,out/name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    failed=public/'connectivity-size-gate';failed.mkdir(exist_ok=True);source=S.OUT/'connectivity_transfer'
    for name in ['failed-gate.json','failed-padding-gate-diagnosis.json','padding-boundary-diagnosis.json']:shutil.copyfile(source/name,failed/name)
    shutil.copyfile(S.HERE/'connectivity-transfer-protocol.json',failed/'protocol.json')
    summary=json.loads((public/'summary.json').read_text());assert 'connectivity_screen' not in summary and 'connectivity_size_gate' not in summary;summary['connectivity_screen']={'hypothesis':'H2ak','fresh_inputs':0,'new_block_interventions':12288,'reused_binary_prefixes':12288,'archived_skip_control_mappings':12288,'additional_baseline_trajectories':0,'targets':a['targets'],'joint_success':a['joint_success'],'native_descent_disconnections':a['native_descent_disconnections'],'scope':'PosthocseenH2ajcohort; explicit symbolicgraphknowledge; separate policytargets; skipzero-loss conditionalidentityproof distinguished from empirical gainretention.'};summary['connectivity_size_gate']={'hypothesis':'H2al','status':'failed_before_fresh_generation','fresh_inputs':0,'diagnostic_model_inputs':192,'cause':'Actualaddedwalls introduce pairtable unaryterms at oldgridboundary; no paddingobjectiveinvariance.'};summary['hypotheses'].append({'campaign':'connectivity_screen','hypothesis':'H2ak zero pairedregressions and retain>=90%of63originalgains, eachpolicy separately','outcome':a['targets']});summary['hypotheses'].append({'campaign':'connectivity_size_gate','hypothesis':'H2al padding size-invariance gate','outcome':'failed; boundaryobjectivechange independentlydiagnosed; nofreshcohort'});summary['protocols']['connectivity-screen-protocol']={'sha256':S.sha(PP),'frozen_utc':json.loads(PP.read_text())['frozen_utc']};failedpp=S.HERE/'connectivity-transfer-protocol.json';summary['protocols']['connectivity-transfer-protocol']={'sha256':S.sha(failedpp),'frozen_utc':json.loads(failedpp.read_text())['frozen_utc']};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
