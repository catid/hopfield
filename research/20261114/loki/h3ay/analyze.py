"""Export full populations, exact pairwise witness fractions and concrete boards."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3ay'


def main():
    frozen();files=['capacity-gate.json','study-result.json','campaign.log','distinguishing-witnesses.json','N12-shared-columns.npy','N13-intersection.npy','N13-A-only.npy','N13-B-only.npy']+[f'{family}/{name}' for family in ('A','B') for name in ('N12-columns.npy','N12-F.npy','N13-columns.npy','N13-F.npy','N13-clue-census.npz')]
    for rel in files:
        path=PUBLIC/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,path)
    audit=verify(ROOT,False);g=read(PUBLIC/'capacity-gate.json');s=dict(status='validated',utc=now(),hypothesis='H3ay',**{k:g[k] for k in ('decision','rows','N12_shared_states','N13_intersection_states','N13_symmetric_difference_states','clue_censuses')},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,new_observed_labels=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    with(PUBLIC/'pairwise-reliability.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(s['rows'][0]));writer.writeheader();writer.writerows(s['rows'])
    print(dict(decision=s['decision'],rows=s['rows'],audit=audit))


if __name__=='__main__':main()
