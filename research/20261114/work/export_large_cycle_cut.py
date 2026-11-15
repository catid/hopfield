"""Publish H2aw source cap intervention without adding fresh primary totals."""
import json
import shutil
import study as S


def main():
    source=S.OUT/'large_cycle_cut';public=S.ROOT/'results/research/work';out=public/'large-cycle-cut';out.mkdir(exist_ok=True);a=json.loads((source/'audit.json').read_text());assert a['status']=='passed'
    for name in ['protocol.json','numerical-gate.json','old256-fixed-point-control.json','audit.json','per-input.csv','worker0.json','worker1.json','independent-state-verification.json','state-import-isolation.json','verification-environment.json']:
        shutil.copyfile(source/name,out/name)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for path in (source/'state-fixtures').iterdir():
        if path.is_file():shutil.copyfile(path,fixtures/path.name)
    S.save(public/'large-cycle-cut-summary.json',a)
    summary=json.loads((public/'summary.json').read_text());assert 'large_cycle_cut' not in summary
    summary['large_cycle_cut']={'hypothesis':'H2aw','fresh_inputs':0,'included_model_inputs':12288,'new_search_inputs_per_policy':84,'reused_lexical256_prefixes_per_policy':12288,'primary':a['primary'],'joint_success':a['joint_success'],'model_success':a['model_success'],'new_optimizer_updates':0,'scope':'Source intervention from identical archived H2av lexical256 outputs for both policies. Chain cap256 to512; no other candidate geometry or native rule change. All28 joint/84 model failures rescued by each policy, zero regressions; all84 new moves lower original energy. Earlier positive cuts remain in the reused prefix. Zero fresh generalization evidence.'}
    summary['hypotheses'].append({'campaign':'large_cycle_cut','hypothesis':'H2aw','outcome':a['primary']});summary['audit_sha256'][str((source/'audit.json').relative_to(S.ROOT))]=S.sha(source/'audit.json');p=json.loads((source/'protocol.json').read_text());summary['protocols']['large-cycle-cut-protocol']={'sha256':S.sha(source/'protocol.json'),'frozen_utc':p['frozen_utc'],'hypothesis':'H2aw','status':'completed'};summary['updated_utc']=S.utc();S.save(public/'summary.json',summary)
    S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
