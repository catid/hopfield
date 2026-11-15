"""Publish metadata-only successor evidence after a complete independent audit."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3aq'
def main():
    p=frozen(True);source=read(OUT/'source-outcomes.json');files=['capacity-gate.json','study-result.json','source-outcomes.json','construction-safety-seal.json','preflight.json','campaign.log']+[f'b{i}/{arm}/{name}'for i in range(8)for arm in('published_filtered','raw_unfiltered','D4_unfiltered')for name in('model.npz','construction.json')]+[f'family{f}/{name}'for f in(0,1)for name in('input-universe.npz','true-columns.npy','true-F.npy')]+[f"support{r['group']}/{name}"for r in source['supports']for name in('F.npz','capacity.npz')]
    for rel in files:dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest)
    audit=verify(ROOT,False);g=read(PUBLIC/'capacity-gate.json');s=dict(status='validated',utc=now(),hypothesis='H3aq',decision=g['decision'],rows=g['rows'],primary=g['primary'],support_results=g['support_results'],aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],preflight_CPU_seconds=g['preflight_CPU_seconds'],preflight_charged_wall_seconds=g['preflight_charged_wall_seconds'],new_run_CPU_seconds=g['new_run_CPU_seconds'],new_run_wall_seconds=g['new_run_wall_seconds'],GPU_seconds=0,distinct_full_supports=len(source['supports']),safety=g['safety'],scope=p['scope'],technical_predecessor=p['technical_predecessor'],audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel)for rel in files});save(PUBLIC/'summary.json',s)
    with(PUBLIC/'capacity.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(s['rows'][0]));w.writeheader();w.writerows(s['rows'])
    print(dict(decision=s['decision'],primary=[r['capacity_pct']for r in s['primary']],audit=audit))
if __name__=='__main__':main()
