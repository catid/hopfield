"""Export the full nested grid and independently audit all measured fit gates."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3ao'
def main():
    p=frozen(True);source=read(OUT/'source-outcomes.json');files=['capacity-gate.json','study-result.json','evaluation.json','study-start.json','source-banks.json','source-fits-seal.json','source-outcomes.json','ordinary-filter-seal.json','live-utilization.json','campaign.log']+[f'fit-worker-g{g}.{ext}'for g in(0,1)for ext in('json','log')]+[f'b{i}/{name}'for i in range(8)for name in('ordered64-indices.npy','source-design.npz','parameters.npz','model.npz','model.json','loss-validation.json')]+[f'b{i}/d{dose}/{name}'for i in range(8)for dose in(16,32,64)for name in('source-design.npz','count-model.npz')]+[f'family{f}/{name}'for f in(0,1)for name in('input-universe.npz','true-columns.npy','true-F.npy')]+[f"support{r['group']}/{name}"for r in source['supports']for name in('F.npz','capacity.npz')]
    for rel in files:dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,dest)
    audit=verify(ROOT,False);g=read(PUBLIC/'capacity-gate.json');s=dict(status='validated',utc=now(),hypothesis='H3ao',decision=g['decision'],family_results=g['family_results'],rows=g['rows'],support_results=g['support_results'],aggregate_CPU_seconds=g['aggregate_cpu_seconds'],aggregate_GPU_worker_seconds=g['aggregate_GPU_worker_seconds'],wall_seconds=g['wall_seconds'],new_BP_fits=8,descriptive_unfitted_BP_doses=[16,64],distinct_full_supports=len(source['supports']),audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel)for rel in files});save(PUBLIC/'summary.json',s)
    with(PUBLIC/'nested-results.csv').open('w')as f:
        fields=list(dict.fromkeys(key for row in s['rows']for key in row));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(s['rows'])
    print(dict(decision=s['decision'],families=s['family_results'],audit=audit))
if __name__=='__main__':main()
