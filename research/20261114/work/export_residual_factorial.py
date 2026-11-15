"""Publish H2bb disjoint source branches; composed arm adds no trajectories."""
import json
import shutil
import study as S
from residual_factorial import PP,DEST


def main():
    public=S.ROOT/'results/research/work';out=public/'residual-factorial';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';S.save(public/'residual-factorial-summary.json',a)
    for n in ['protocol.json','numerical-gate.json','audit.json','per-input.csv','worker0.json','worker1.json','live-sm-progress.txt','independent-state-verification.json','state-import-isolation.json']:shutil.copyfile(DEST/n,out/n)
    fixtures=out/'state-fixtures';fixtures.mkdir(exist_ok=True)
    for p in (DEST/'state-fixtures').iterdir():
        if p.is_file():shutil.copyfile(p,fixtures/p.name)
    summary=json.loads((public/'summary.json').read_text());assert 'residual_factorial' not in summary;summary['residual_factorial']={'hypothesis':'H2bb','fresh_inputs':0,'source_model_inputs':12288,'new_search_inputs':24,'known_terminal_veto_bookkeeping':1,'composed_arm_new_trajectories':0,'primary':a['primary'],'joint_success':a['joint_success'],'model_success':a['model_success'],'scope':'Disjoint five-input cap families; branches evaluated once, joint exact composition/no interaction or extra independent evidence. All4096previouslyseeninputs retained. Zero freshgeneralization claim or evaluationtotals.'};summary['hypotheses'].append({'campaign':'residual_factorial','hypothesis':'H2bb >=3rescues per5input family and>=6combined, eachzero regressions','outcome':a['primary']});p=json.loads(PP.read_text());summary['protocols']['residual-factorial-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bb','status':'completed'};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
