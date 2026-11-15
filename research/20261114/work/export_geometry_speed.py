"""Publish H2bd exact speed experiments without adding scientific samples."""
import json
import shutil
import study as S
from geometry_speed_study import PP,DEST,GATE


def main():
    public=S.ROOT/'results/research/work';out=public/'geometry-speed';out.mkdir(exist_ok=True);a=json.loads((DEST/'audit.json').read_text());v=json.loads((DEST/'public-verification.json').read_text());assert a['status']=='passed' and v['status']=='passed';S.save(public/'geometry-speed-summary.json',a)
    for name in ['protocol.json','audit.json','geometry-benchmark.json','policy-worker0.json','policy-worker1.json','policy-pairs-gpu0.json','policy-pairs-gpu1.json','public-verification.json','live-sm-progress.txt']:shutil.copyfile(DEST/name,out/name)
    for name in ['synthetic-gate.json','archive-gate.json','policy-gate-gpu0.json','policy-gate-gpu1.json']:shutil.copyfile(GATE/name,out/name)
    for path in DEST.glob('gpu*-rep*-*.npz'):shutil.copyfile(path,out/path.name)
    shutil.copytree(DEST/'public-fixtures',out/'public-fixtures',dirs_exist_ok=True)
    summary=json.loads((public/'summary.json').read_text());assert 'geometry_speed' not in summary;summary['geometry_speed']={'hypothesis':'H2bd','fresh_inputs':0,'new_scientific_evaluations':0,'geometry_median_speedup':a['geometry']['median_paired_process_speedup'],'geometry_decision':a['geometry']['decision'],'full_policy_median_speedup':a['policy']['median_paired_wall_speedup'],'full_policy_decision':a['policy']['decision'],'scope':'Exact selected-graph bridge geometry. Four balanced timing pairs each;1024 frozen geometry states and768 source model-input complete-policy cases. All source/pair choices/traces/boards/flags preserved; no new accuracy evidence.'};summary['hypotheses'].append({'campaign':'geometry_speed','hypothesis':'H2bd separate >=2x geometry process and >=1.1x complete-policy wall targets with exact parity','outcome':{'geometry':a['geometry']['decision'],'full_policy':a['policy']['decision']}});p=json.loads(PP.read_text());summary['protocols']['geometry-speed-protocol']={'sha256':S.sha(PP),'frozen_utc':p['frozen_utc'],'hypothesis':'H2bd','status':'completed'};summary['audit_sha256'][str((DEST/'audit.json').relative_to(S.ROOT))]=S.sha(DEST/'audit.json');summary['updated_utc']=S.utc();S.save(public/'summary.json',summary);S.save(public/'manifest.json',{'source_sha256':S.sha(__file__),'files':{str(p.relative_to(public)):S.sha(p) for p in sorted(public.rglob('*')) if p.is_file() and p!=public/'manifest.json'}})


if __name__=='__main__':main()
