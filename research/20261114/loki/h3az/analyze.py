"""Export alias prevalence and explicitly label a zero-N14-enumeration result."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3az'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','campaign.log','N12-partition.json','pooled-N12-N13-partition.json','N14-behavior-comparison.json']+[f"N{r['n']}/candidate{r['candidate']}/{name}" for records in g['stage_records'].values() for r in records for name in ('columns.npy','F.npy')]
    for rel in files:
        path=PUBLIC/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,path)
    audit=verify(ROOT,False);s=dict(status='validated',utc=now(),hypothesis='H3az',**{k:g[k] for k in ('decision','family_results','descriptive_all1541','relevant_nontrivial_N12_classes','N13_models','N14_models','pooled_refined_classes','stage_records')},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    keys=list(dict.fromkeys(k for r in s['family_results'] for k in r))
    with(PUBLIC/'family-identifiability.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(s['family_results'])
    print(dict(decision=s['decision'],prevalence=s['descriptive_all1541'],N13=s['N13_models'],N14=s['N14_models'],audit=audit))


if __name__=='__main__':main()
