"""Publish hypothetical acquisition distributions with zero-mass patterns retained."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3av'


def main():
    p=frozen(True);source=read(OUT/'source-selection.json');files=['capacity-gate.json','study-result.json','source-selection.json','campaign.log']+[f'b{i}/{name}' for i in range(8) for name in ('initial-evidence.npz','acquisition-evidence.npz','exact-distribution.json')]+[f'family{i}/{name}' for i in (0,1) for name in ('source-population.npy','true-columns.npy','true-F.npy','input-universe.npz')]+[f"support{r['group']}/{name}" for r in source['supports'] for name in ('model.npz','capacity.npz')]
    for rel in files:
        path=PUBLIC/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,path)
    audit=verify(ROOT,False);g=read(PUBLIC/'capacity-gate.json');s=dict(status='validated',utc=now(),hypothesis='H3av',**{k:g[k] for k in ('decision','rows','support_results')},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,new_observed_labels=0,distinct_full_supports=len(source['supports']),abstract_masks=sum(r['abstract_masks'] for r in g['rows']),positive_mass_masks=sum(r['positive_mass_masks'] for r in g['rows']),audit=audit,acquisition=p['acquisition'],posterior=p['posterior'],protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    with(PUBLIC/'reliability.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(s['rows'][0]));writer.writeheader();writer.writerows(s['rows'])
    print(dict(decision=s['decision'],probabilities=[r['success_pct'] for r in s['rows']],audit=audit))


if __name__=='__main__':main()
