"""Export complete new-fit evidence and audit it using public files only."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3an'
def main():
    p=frozen(True);source=read(OUT/'source-outcomes.json');files=['capacity-gate.json','study-result.json','evaluation.json','study-start.json','source-banks.json','source-fits-seal.json','source-outcomes.json','source-validation.json','distribution-validation.json','live-utilization.json','campaign.log']+[f'fit-worker-g{g}.{ext}'for g in(0,1)for ext in('json','log')]+[f'b{i}/{name}'for i in range(8)for name in('source-design.npz','acquisition-population.npy','acquisition-population-indices.npy','parameters.npz','model.npz','model.json','loss-validation.json','source-patterns.npz')]+[f'family{f}/{name}'for f in(0,1)for name in('input-universe.npz','true-F.npy','true-columns.npy','reference.json')]+[f"support{r['group']}/{name}"for r in source['supports']for name in('F.npz','capacity.npz')]
    for rel in files:dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest)
    audit=verify(ROOT,False);g=read(PUBLIC/'capacity-gate.json');s=dict(status='validated',utc=now(),hypothesis='H3an',decision=g['decision'],rows=g['rows'],support_results=g['support_results'],aggregate_CPU_seconds=g['aggregate_cpu_seconds'],aggregate_GPU_worker_seconds=g['aggregate_GPU_worker_seconds'],wall_seconds=g['wall_seconds'],new_independent_fits=8,distinct_full_supports=len(source['supports']),scope=p['scope'],audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel)for rel in files});save(PUBLIC/'summary.json',s);flat=[{k:v for k,v in r.items()if not isinstance(v,dict)}for r in s['rows']]
    with(PUBLIC/'reliability.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(dict(decision=s['decision'],audit=audit))
if __name__=='__main__':main()
