"""Publish all-case source terminal/extra-round comparison without fresh totals."""
import json
import shutil
import study as S
from terminal_cycle import PP,DEST


def main():
    public=S.ROOT/'results/research/work';out=public/'terminal-cycle';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';S.save(public/'terminal-cycle-summary.json',a)
    for n in ['protocol.json','numerical-gate.json','audit.json','per-input.csv','worker0.json','worker1.json','independent-state-verification.json','state-import-isolation.json','residual-cap-diagnosis.json']:shutil.copyfile(DEST/n,out/n)
    shutil.copyfile(S.OUT/'terminal_cycle_preflight/preflight-schema-failure.json',out/'preflight-schema-failure.json');fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for p in (DEST/'state-fixtures').iterdir():
        if p.is_file():shutil.copyfile(p,fixtures/p.name)
    summary=json.loads((public/'summary.json').read_text());assert 'terminal_cycle' not in summary;summary['terminal_cycle']={'hypothesis':'H2ba','fresh_inputs':0,'source_model_inputs':12288,'new_search_inputs':167,'known_terminal_bookkeeping_model_cases':27,'inherited_veto_caps_unchanged':10,'primary':a['primary'],'joint_success':a['joint_success'],'model_success':a['model_success'],'scope':'Archived source intervention. Certificate-only10joint rescues known from prior diagnosis; new16rounds52additional joint rescues/zero regressions. All25remainingmodel caps retained. No fresh generalization/confidence claim or evaluation totals.'};summary['hypotheses'].append({'campaign':'terminal_cycle','hypothesis':'H2ba >=31 additional jointrescues beyond terminalonly andzero regressions','outcome':a['primary']});p=json.loads(PP.read_text());summary['protocols']['terminal-cycle-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2ba','status':'completed'};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
