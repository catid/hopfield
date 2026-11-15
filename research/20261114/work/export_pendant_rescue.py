"""Publish H2ao gate failure and H2ap source intervention without fresh totals."""
import json
import shutil
import study as S
from pendant_rescue_v2 import PP,DEST


def main():
    public=S.ROOT/'results/research/work';out=public/'pendant-rescue';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';S.save(public/'pendant-rescue-summary.json',a)
    for name in ['protocol.json','numerical-gate.json','old-chain-fixed-point-control.json','audit.json','per-input.csv','worker0.json','worker1.json','residual-diagnosis.json','independent-state-verification.json','state-import-isolation.json']:shutil.copyfile(DEST/name,out/name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (DEST/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    failed=public/'pendant-fixture-gate';failed.mkdir(exist_ok=True)
    for name in ['protocol.json','failed-gate.json','old-chain-fixed-point-control.json']:shutil.copyfile(S.OUT/'pendant_rescue'/name,failed/name)
    summary=json.loads((public/'summary.json').read_text());assert 'pendant_rescue' not in summary and 'pendant_fixture_gate' not in summary;summary['pendant_rescue']={'hypothesis':'H2ap','fresh_inputs':0,'included_model_inputs':12288,'reused_veto_prefixes':12288,'new_search_inputs':a['new_search_inputs'],'initial_simple_path_identity_reuses':a['initial_simple_paths'],'new_optimizer_updates':0,'targets':a['targets'],'joint_success':a['joint_success'],'scope':'PosthocseenH2ancohort;explicitarticulation/reachabilityprior.>=11/22rescuesfalsified:0rescues,zeroeligiblependants.Zero-regressionidentityproofseparate. Oldcontrolfixedpointscertified;noextrabudgetconfound.'};summary['pendant_fixture_gate']={'hypothesis':'H2ao','status':'failed_before_intervention','fresh_inputs':0,'new_searches':0,'cause':'Thefirst32exhaustive3x3fixturestatesusedforliteralcostgatehavezeroraw/admittedcandidates;preservedfailedcoverageassertion.'}
    summary['hypotheses'].append({'campaign':'pendant_fixture_gate','hypothesis':'H2ao nonempty literal candidate coverage gate','outcome':'failedbeforeinterventions;correctedseparatelyinH2ap'});summary['hypotheses'].append({'campaign':'pendant_rescue','hypothesis':'H2ap >=11of22jointfailuresrescuedandzerojointregressions,separately','outcome':a['targets']})
    for path in [S.HERE/'pendant-rescue-protocol.json',PP]:summary['protocols'][path.stem]={'sha256':S.sha(path),'frozen_utc':json.loads(path.read_text())['frozen_utc']}
    summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
